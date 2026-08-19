"""
WideBind training: streaming from token_stream_{GENRE}.bin files.
"""

import os, sys, math, time, json, glob, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
import numpy as np
from torch.serialization import add_safe_globals

from core import WideBindConfig, WideBindStack, MirrorLRScheduler
try:
    from analyze_checkpoint import generate_report
except ImportError:
    from scripts.analyze_checkpoint import generate_report

add_safe_globals([WideBindConfig])


class TokenStream:
    """Memory-mapped uint16 token stream; converted to torch.long per batch."""
    def __init__(self, path):
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        self.len = len(self.data)
    def get_batch(self, seq_len, batch_size, offset, vocab=50000):
        needed = batch_size * seq_len + 1
        if offset + needed > self.len:
            offset = 0
        chunk = self.data[offset:offset + needed]
        if vocab is not None:
            # uint16-С„Р°Р№Р»С‹ РјРѕРіСѓС‚ СЃРѕРґРµСЂР¶Р°С‚СЊ С‚РѕРєРµРЅС‹ в‰Ґ vocab в†’ device-side assert
            # РІ codes[tokens] (index out of bounds); РєР»РёРїР°РµРј РґРѕ Р±РµР·РѕРїР°СЃРЅРѕСЃС‚Рё.
            chunk = np.clip(chunk, 0, vocab - 1)
        x = torch.from_numpy(chunk[:batch_size * seq_len].reshape(batch_size, seq_len).copy())
        y = torch.from_numpy(chunk[1:batch_size * seq_len + 1].reshape(batch_size, seq_len).copy())
        return x.long(), y.long(), offset + batch_size * seq_len


def create_lr_scheduler(optimizer, warmup, max_steps, lr):
    """Linear warmup + cosine decay (returns multiplier 0..1 for LambdaLR)."""
    def get_lr_mult(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(max_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_mult)


def _restore_optimizer(optimizer, model, ckpt_opt):
    """Restore AdamW state by parameter name.

    Survives params added after the checkpoint was saved (reasoning_gate,
    reasoning_proj): positional load would either fail (group size mismatch)
    or silently shift state onto the wrong params, so we re-map by name.
    Checkpoint param order == current order without the new gate params.
    """
    flat = [(id(p), tuple(p.shape)) for g in optimizer.param_groups for p in g['params']]
    names = {id(p): n for n, p in model.named_parameters()}
    old = ckpt_opt
    try:
        optimizer.load_state_dict(old)
        if all(tuple(sd['exp_avg'].shape) == flat[i][1]
               for i, sd in optimizer.state_dict()['state'].items()):
            return True
    except ValueError:
        pass
    new_sd = optimizer.state_dict()
    new_sd['state'] = {}
    no_gate = [(names[id(p)], p) for g in optimizer.param_groups for p in g['params']
               if not names[id(p)].startswith('reasoning_gate.')]
    pos = {id(p): i for i, (n, p) in enumerate(
        (names[id(p)], p) for g in optimizer.param_groups for p in g['params'])}
    moved = skipped = 0
    for si, (n, p) in enumerate(no_gate):
        st = old['state'].get(si)
        if st is None:
            continue
        if tuple(st['exp_avg'].shape) != tuple(p.shape):
            skipped += 1
            print(f'  Optimizer state shape mismatch at {si} {n}: '
                  f'{tuple(st["exp_avg"].shape)} vs {tuple(p.shape)}')
            continue
        new_sd['state'][pos[id(p)]] = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                                       for k, v in st.items()}
        moved += 1
    for gi in range(min(len(new_sd['param_groups']), len(old['param_groups']))):
        if 'lr' in old['param_groups'][gi]:
            new_sd['param_groups'][gi]['lr'] = old['param_groups'][gi]['lr']
    optimizer.load_state_dict(new_sd)
    if skipped:
        print(f'  WARNING: {skipped} optimizer slots skipped (shape mismatch), {moved} restored')
    return moved > 0


def train(cfg=None, resume_path=None):
    if cfg is None:
        cfg = WideBindConfig()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32  # no AMP for stability
    
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Data
    print(f'Loading data from {cfg.data_dir}')
    stream_files = sorted(glob.glob(os.path.join(cfg.data_dir, 'token_stream_*_clean.bin')))
    if not stream_files:
        stream_files = sorted(glob.glob(os.path.join(cfg.data_dir, 'token_stream_*.bin')))
    if not stream_files:
        raise FileNotFoundError(f'No token_stream_*.bin files in {cfg.data_dir}')
    
    streams = [TokenStream(f) for f in stream_files]
    total_tokens = sum(s.len for s in streams)
    print(f'Found {len(streams)} files, {total_tokens:,} total tokens')
    
    # Model (retry once on OOM вЂ” transient CUDA context cleanup)
    try:
        model = WideBindStack(cfg).to(device)
    except RuntimeError as e:
        if 'out of memory' in str(e) and device == 'cuda':
            print('[WideBind] OOM on first attempt, clearing cache and retrying...')
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(1)
            model = WideBindStack(cfg).to(device)
        else:
            raise
    n_params = model.param_count()
    print(f'Model: {n_params:,} params ({n_params/1e6:.2f}M)')
    if device == 'cuda':
        print(f'  VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB')
    
    # Phase tracking state (EMA-based adaptive threshold)
    model._phase_ratio_ema = [0.0] * cfg.n_layers
    model._phase_ratio_std = [1.0] * cfg.n_layers
    
    # Optimizer
    param_groups = model.param_groups()
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

    # AMP (Automatic Mixed Precision)
    use_amp = getattr(cfg, 'use_amp', False) and device == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        print('  AMP: ON (mixed precision)')

    # Scheduler: mirror-adaptive or cosine
    if cfg.scheduler == 'mirror':
        scheduler = MirrorLRScheduler(model, optimizer, cfg.lr,
            warmup=cfg.warmup_steps, target_var=cfg.target_var,
            mag_threshold=cfg.mag_threshold, lr_min_ratio=cfg.lr_min_ratio,
            max_decay_steps=cfg.max_decay_steps,
            var_min_for_lr_decay=cfg.var_min_for_lr_decay)
        print(f'Scheduler: MirrorLRScheduler (target_var={cfg.target_var}, mag_threshold={cfg.mag_threshold})')
    else:
        scheduler = create_lr_scheduler(optimizer, cfg.warmup_steps, cfg.max_steps, cfg.lr)
        print(f'Scheduler: cosine decay')
    
    # Resume
    start_step = 0
    best_val_loss = float('inf')
    if resume_path == 'auto':
        # Find latest checkpoint: interrupt > step_* > best
        ckpts = sorted(glob.glob(os.path.join(cfg.save_dir, 'interrupt_step_*.pt')))
        if not ckpts:
            ckpts = sorted(glob.glob(os.path.join(cfg.save_dir, 'step_*.pt')))
        if not ckpts:
            ckpts = sorted(glob.glob(os.path.join(cfg.save_dir, 'best.pt')))
        if ckpts:
            resume_path = ckpts[-1]
            print(f'Auto-resuming from latest: {resume_path}')
    if resume_path and os.path.exists(resume_path):
        print(f'Resuming from {resume_path}')
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
        if missing:
            print(f'  Missing keys (new arch): {len(missing)}')
        if unexpected:
            print(f'  Unexpected keys (old arch): {len(unexpected)}')
        if not _restore_optimizer(optimizer, model, ckpt['optimizer']):
            print('  WARNING: optimizer state could not be restored (fresh Adam)')
        if 'scheduler' in ckpt:
            sched_sd = ckpt['scheduler']
            if sched_sd.get('type') == 'MirrorLRScheduler':
                scheduler.load_state_dict(sched_sd)
            elif cfg.scheduler == 'mirror':
                # Switching from cosine to mirror вЂ” use step only
                scheduler._step = ckpt['step']
                print(f'  Switched to MirrorLRScheduler at step {ckpt["step"]}')
            else:
                scheduler.load_state_dict(sched_sd)
        else:
            if cfg.scheduler == 'mirror':
                scheduler._step = ckpt['step']
            else:
                scheduler.last_epoch = ckpt['step'] + 1
                for pg, lr in zip(optimizer.param_groups, scheduler.get_lr()):
                    pg['lr'] = lr
        start_step = ckpt['step']
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
    reasoning_enabled_step = ckpt.get('reasoning_enabled_step', 0) if resume_path and os.path.exists(resume_path) else 0
    
    # State for recurrent layers
    state = None
    gs = None
    rng = torch.Generator().manual_seed(42)
    
    # Training loop
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    
    stream_idx = 0
    offset = 0
    tokens_seen = 0
    t0 = time.time()
    
    print(f'Starting training from step {start_step}')
    print(f'Streams: {len(streams)} ({", ".join(f"{s.len:,}" for s in streams)} tokens)')
    print('Press Ctrl+C to save checkpoint and exit gracefully.')
    try:
        for step in range(start_step, cfg.max_steps):
            model.train()
            if model.explicit_reasoning:
                model.reasoning_enabled_step = reasoning_enabled_step
            
            # в”Ђв”Ђв”Ђ Mixed stream sampling: pick a random position in a random stream в”Ђв”Ђв”Ђ
            # When offset reaches end of current stream, randomly pick next stream
            # This keeps state continuity within a stream while mixing genres
            # at stream boundaries (FANTASY~82%, ADVENTUR~18% of batches)
            if offset == 0:
                stream_idx = torch.randint(0, len(streams), (1,), generator=rng).item()
                offset = 0
                state = None  # reset state on stream switch (document boundary)
                gs = None
                if model.explicit_reasoning:
                    model.reset_reasoning()  # new document: new chain
            
            # в”Ђв”Ђв”Ђ Multi-scale seq curriculum: С‡РµСЂРµРґРѕРІР°РЅРёРµ РґР»РёРЅС‹ Р±Р°С‚С‡Р° РїРѕ РѕРєС‚Р°РІР°Рј П„ в”Ђв”Ђв”Ђ
            # L=64 (П„в‰¤32, РѕРєС‚Р°РІС‹ 0вЂ“13): 7/9 С€Р°РіРѕРІ
            # L=256 (П„в‰¤92, РѕРєС‚Р°РІС‹ 14вЂ“23): 1/9 С€Р°РіРѕРІ
            # L=512 (П„в‰¤149, РѕРєС‚Р°РІС‹ 24вЂ“31): 1/9 С€Р°РіРѕРІ
            seq_pool = [64, 64, 64, 64, 64, 64, 64, 256, 512]
            seq_len = seq_pool[step % len(seq_pool)]
            
            stream = streams[stream_idx]
            x, y, offset = stream.get_batch(seq_len, cfg.batch_size, offset, cfg.vocab)
            if offset == 0:
                continue  # retry with new random stream
            
            x, y = x.to(device), y.to(device)
            
            # в”Ђв”Ђв”Ђ Forward (with optional AMP) в”Ђв”Ђв”Ђ
            with autocast('cuda', enabled=use_amp):
                h = model.embed_tokens(x)
                out, state, gs, _ = model(h, state, global_state=gs, step=step)
                ce_loss, aux_dict = model.compute_losses(out, y, h_emb=h)
            
            # NaN guard
            if torch.isnan(ce_loss) or torch.isinf(ce_loss):
                raise RuntimeError(f'NaN/Inf CE loss at step {step}')
            
            state = tuple(s.detach() for s in state) if state is not None else None
            if gs is not None:
                gs = gs.detach()
            
            # CE gradients (retain graph for aux backward)
            ce_grads = torch.autograd.grad(ce_loss, model.parameters(),
                                           retain_graph=bool(aux_dict), allow_unused=True)

            # Separate bypass losses from spectral alignment
            w_div = getattr(cfg, 'div_weight', 0.0)
            w_rp = getattr(cfg, 'gate_repulse_weight', 0.0)
            w_nv = getattr(cfg, 'alpha_novelty_weight', 0.0)
            bypass_keys = {'div', 'gate_repulse', 'alpha_novelty', 'ranking'}
            bypass_losses = {k: None for k in bypass_keys}
            aligned_list = []
            if aux_dict:
                for k, v in aux_dict.items():
                    w = 1.0
                    if k == 'div': w = w_div
                    if k == 'gate_repulse': w = w_rp
                    if k == 'alpha_novelty': w = w_nv
                    if not (isinstance(v, torch.Tensor) and v.requires_grad):
                        continue
                    if k in bypass_keys and w > 0:
                        bypass_losses[k] = v * w
                    elif w > 0:
                        aligned_list.append(v * w)

            # Bypass gradients: single backward to avoid freeing graph multiple times
            bypass_list = [v for v in bypass_losses.values() if v is not None]
            if bypass_list:
                bypass_total = sum(bypass_list)
                bypass_retain = bool(aligned_list)  # keep graph for aux_total if needed
                bypass_grads = torch.autograd.grad(
                    bypass_total, model.parameters(), retain_graph=bypass_retain, allow_unused=True)
            else:
                bypass_grads = None

            # Spectral gradient alignment for non-bypass aux
            if aligned_list:
                aux_total = sum(aligned_list)
                aux_grads = torch.autograd.grad(aux_total, model.parameters(), allow_unused=True)
                ce_list, aux_list = [], []
                for gce, gaux in zip(ce_grads, aux_grads):
                    if gce is not None and gaux is not None:
                        ce_list.append(gce.flatten())
                        aux_list.append(gaux.flatten())
                if ce_list:
                    ce_flat = torch.cat(ce_list)
                    aux_flat = torch.cat(aux_list)
                    cos_sim = F.cosine_similarity(ce_flat.unsqueeze(0), aux_flat.unsqueeze(0))
                    scale = max(0, min(10.0, cos_sim.item())) * ce_flat.norm() / (aux_flat.norm() + 1e-8)
                else:
                    scale = 0.0
            else:
                aux_grads = None
                scale = 0.0

            # Combine: g = g_CE + scale * g_aligned + bypass_grads
            with torch.no_grad():
                for p, cg in zip(model.parameters(), ce_grads):
                    if cg is not None:
                        p.grad = cg.clone()
                    else:
                        p.grad = None
                if aux_grads is not None and scale > 0:
                    for p, ag in zip(model.parameters(), aux_grads):
                        if p.grad is not None and ag is not None:
                            p.grad.add_(ag, alpha=scale)
                        elif ag is not None:
                            p.grad = ag * scale
                if bypass_grads is not None:
                    for p, bg in zip(model.parameters(), bypass_grads):
                        if bg is not None:
                            if p.grad is not None:
                                p.grad.add_(bg)
                            else:
                                p.grad = bg.clone()
            
            # Adaptive phase scaling: EMA-based mirror/base gradient balance
            phase_scales = []
            for i, layer in enumerate(model.layers):
                mirror_norm = 0.0
                base_norm = 0.0
                for p in layer.mirror_parameters:
                    if p.grad is not None:
                        mirror_norm += p.grad.norm().item() ** 2
                for p in layer.base_parameters:
                    if p.grad is not None:
                        base_norm += p.grad.norm().item() ** 2
                mirror_norm = mirror_norm ** 0.5
                base_norm = base_norm ** 0.5
                ratio = mirror_norm / (base_norm + 1e-8)
                ema = model._phase_ratio_ema[i]
                std = model._phase_ratio_std[i]
                model._phase_ratio_ema[i] = 0.99 * ema + 0.01 * ratio
                model._phase_ratio_std[i] = 0.99 * std + 0.01 * abs(ratio - ema)
                mir_s = max(0.2, min(2.0, 1.0 / (1.0 + math.exp(-(ratio - ema) / (std + 1e-8)))))
                phase_scales.append((mir_s, ratio))
                for p in layer.mirror_parameters:
                    if p.grad is not None:
                        p.grad *= mir_s
            mean_mirror_scale = sum(s[0] for s in phase_scales) / len(phase_scales)
            mean_ratio = sum(s[1] for s in phase_scales) / len(phase_scales)
            tokens_seen += cfg.batch_size * seq_len
            
            # Clip gradients
            if cfg.grad_clip > 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if model.explicit_reasoning:
                reasoning_enabled_step += 1
            
            current_lr = scheduler.get_last_lr()[0]
            
            # в”Ђв”Ђв”Ђ Soft EOS-aware state reset: Р·Р°С‚СѓС…Р°РЅРёРµ РІРјРµСЃС‚Рѕ РѕР±РЅСѓР»РµРЅРёСЏ в”Ђв”Ђв”Ђ
            if (y[:, -1] == 2).any():
                if state is not None:
                    state = tuple(s * 0.1 for s in state)
                if gs is not None:
                    gs = gs * 0.1
            
            # Log
            if step % cfg.log_interval == 0:
                dt = time.time() - t0
                tok_s = tokens_seen / max(dt, 1e-8)
                lc = getattr(model, '_cached_losses', {})
                aux_str = ' '.join(f'{k}={v:.4f}' for k, v in lc.items())
                gate_str = ''
                rg = getattr(model, 'reasoning_gate', None)
                if rg is not None:
                    gates = getattr(model, '_reasoning_gates', None)
                    if gates:
                        gate_str = ' gates=' + str([round(g, 3) for g in gates])
                print(f'  step={step:>6} loss={ce_loss.item():.4f} lr={current_lr:.2e} '
                      f'tok/s={tok_s:.0f} stream={stream_idx} '
                      f'ms={mean_mirror_scale:.3f} mr={mean_ratio:.4f} | {aux_str}{gate_str}')
            
            # Eval
            if step > 0 and step % cfg.eval_interval == 0:
                val_loss = evaluate(model, streams, cfg, device)
                print(f'  EVAL step={step}: val_loss={val_loss:.4f} val_ppl={math.exp(val_loss):.2f}')
                if device == 'cuda':
                    torch.cuda.empty_cache()
                scheduler.report_val_loss(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_path = os.path.join(cfg.save_dir, f'best.pt')
                    torch.save({
                        'step': step,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'cfg': cfg,
                        'reasoning_enabled_step': reasoning_enabled_step,
                    }, save_path)
                    print(f'  Saved best model to {save_path}')
                    generate_report(save_path)
            
            # Save
            if step > 0 and step % cfg.save_interval == 0:
                save_path = os.path.join(cfg.save_dir, f'step_{step}.pt')
                torch.save({
                    'step': step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'cfg': cfg,
                    'reasoning_enabled_step': reasoning_enabled_step,
                }, save_path)
                print(f'  Saved checkpoint to {save_path}')
                generate_report(save_path)
    except KeyboardInterrupt:
        print('\n[WideBind] Ctrl+C detected, saving checkpoint...')
        save_path = os.path.join(cfg.save_dir, f'interrupt_step_{step}.pt')
        torch.save({
            'step': step,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'cfg': cfg,
            'reasoning_enabled_step': reasoning_enabled_step,
        }, save_path)
        print(f'[WideBind] Saved interrupt checkpoint to {save_path}')
        generate_report(save_path)
        print('[WideBind] Exiting gracefully.')
        sys.exit(0)
    
    print('Training complete!')


@torch.no_grad()
def evaluate(model, streams, cfg, device):
    model.eval()
    if getattr(model, 'explicit_reasoning', False):
        model.reset_reasoning()
    total_loss = 0.0
    total_steps = 0
    state = None
    
    # Use first stream for eval (or hold-out)
    stream = streams[0]
    offset = max(len(stream) // 2, cfg.batch_size * cfg.seq_len + 1)
    
    for _ in range(min(100, stream.len // (cfg.batch_size * cfg.seq_len))):
        x, y, offset = stream.get_batch(cfg.seq_len, cfg.batch_size, offset, cfg.vocab)
        if offset == 0:
            break
        x, y = x.to(device), y.to(device)
        h = model.embed_tokens(x)
        out, state, _, _ = model(h, state)
        loss = model.compute_loss(out, y, h_emb=h)
        total_loss += loss.item()
        total_steps += 1
    
    model.train()
    return total_loss / max(total_steps, 1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=24)
    parser.add_argument('--bottleneck', type=int, default=896)
    parser.add_argument('--bind-K', type=int, default=64)
    parser.add_argument('--mlp-groups', type=int, default=8)
    parser.add_argument('--mlp-expand', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max-steps', type=int, default=50000)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--eval-interval', type=int, default=1000)
    parser.add_argument('--save-interval', type=int, default=5000)
    parser.add_argument('--scheduler', type=str, default='mirror', choices=['cosine', 'mirror'])
    parser.add_argument('--head', type=str, default='partitioned', choices=['partitioned', 'codec'],
                        help='LM head: partitioned (softmax-CE) or codec (SignedAmpCodec CE + W_pred + echo)')
    parser.add_argument('--amp-obj', type=str, default='ce', choices=['ce', 'mh'],
                        help='Codec objective: ce = one CE (confirmed recipe), mh = margin+hinge')
    parser.add_argument('--no-amp-pred', action='store_true',
                        help='Disable W_pred transition operator in codec head')
    parser.add_argument('--traj-manifold', action='store_true',
                        help='Trajectory: РјР°РЅРёС„РѕР»Рґ РїРµСЂРµС…РѕРґРѕРІ (TrajectoryManifoldBind)')
    parser.add_argument('--traj-beams', type=int, default=0, help='Manifold: С‡РёСЃР»Рѕ Р»СѓС‡РµР№ (0 = Р°РІС‚Рѕ = ceil(sqrt(buffer)))')
    parser.add_argument('--traj-buffer', type=int, default=1024, help='Manifold: Р±СѓС„РµСЂ РїРµСЂРµС…РѕРґРѕРІ')
    parser.add_argument('--traj-gain', type=float, default=0.05, help='Manifold: РјР°СЃС€С‚Р°Р± РІРєР»Р°РґР°')
    args = parser.parse_args()
    
    cfg = WideBindConfig(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        n_layers=args.n_layers,
        bottleneck=args.bottleneck,
        bind_K=args.bind_K,
        mlp_groups=args.mlp_groups,
        mlp_expand=args.mlp_expand,
        lr=args.lr,
        max_steps=args.max_steps,
        warmup_steps=args.warmup,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        scheduler=args.scheduler,
        amp_codec=(args.head == 'codec'),
        amp_obj=args.amp_obj,
        amp_pred=not args.no_amp_pred,
        traj_manifold=args.traj_manifold,
        traj_beams=args.traj_beams,
        traj_buffer_size=args.traj_buffer,
        traj_gain=args.traj_gain,
    )
    
    train(cfg, resume_path=args.resume)
