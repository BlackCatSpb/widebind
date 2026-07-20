"""
WideBind training: streaming from token_stream_{GENRE}.bin files.
"""

import os, sys, math, time, json, glob, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
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
    def get_batch(self, seq_len, batch_size, offset):
        needed = batch_size * seq_len + 1
        if offset + needed > self.len:
            offset = 0
        chunk = self.data[offset:offset + needed]
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
    
    # Model (retry once on OOM — transient CUDA context cleanup)
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
    
    # Optimizer
    param_groups = model.param_groups()
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
    
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
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            sched_sd = ckpt['scheduler']
            if sched_sd.get('type') == 'MirrorLRScheduler':
                scheduler.load_state_dict(sched_sd)
            elif cfg.scheduler == 'mirror':
                # Switching from cosine to mirror — use step only
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
    
    # State for recurrent layers
    state = None
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
            
            # ─── Mixed stream sampling: pick a random position in a random stream ───
            # When offset reaches end of current stream, randomly pick next stream
            # This keeps state continuity within a stream while mixing genres
            # at stream boundaries (FANTASY~82%, ADVENTUR~18% of batches)
            if offset == 0:
                stream_idx = torch.randint(0, len(streams), (1,), generator=rng).item()
                offset = 0
                state = None  # reset state on stream switch (document boundary)
            
            # ─── Multi-scale seq curriculum: чередование длины батча по октавам τ ───
            # L=64 (τ≤32, октавы 0–13): 7/9 шагов
            # L=256 (τ≤92, октавы 14–23): 1/9 шагов
            # L=512 (τ≤149, октавы 24–31): 1/9 шагов
            seq_pool = [64, 64, 64, 64, 64, 64, 64, 256, 512]
            seq_len = seq_pool[step % len(seq_pool)]
            
            stream = streams[stream_idx]
            x, y, offset = stream.get_batch(seq_len, cfg.batch_size, offset)
            if offset == 0:
                continue  # retry with new random stream
            
            x, y = x.to(device), y.to(device)
            
            # ─── Forward ───
            h = model.embed_tokens(x)
            out, state, _ = model(h, state)
            loss = model.compute_loss(out, y)
            
            # ─── Backward ───
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            
            optimizer.step()
            scheduler.step()
            
            tokens_seen += cfg.batch_size * seq_len
            current_lr = scheduler.get_last_lr()[0]
            
            # ─── EOS-aware state reset: if batch ends with EOS (token 2),
            # reset state so next batch doesn't learn cross-document dependencies ───
            if (y[:, -1] == 2).any():
                state = None
            
            # Log
            if step % cfg.log_interval == 0:
                dt = time.time() - t0
                tok_s = tokens_seen / max(dt, 1e-8)
                print(f'  step={step:>6} loss={loss.item():.4f} lr={current_lr:.2e} '
                      f'tok/s={tok_s:.0f} stream={stream_idx}')
            
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
        }, save_path)
        print(f'[WideBind] Saved interrupt checkpoint to {save_path}')
        generate_report(save_path)
        print('[WideBind] Exiting gracefully.')
        sys.exit(0)
    
    print('Training complete!')


@torch.no_grad()
def evaluate(model, streams, cfg, device):
    model.eval()
    total_loss = 0.0
    total_steps = 0
    state = None
    
    # Use first stream for eval (or hold-out)
    stream = streams[0]
    offset = max(len(stream) // 2, cfg.batch_size * cfg.seq_len + 1)
    
    for _ in range(min(100, stream.len // (cfg.batch_size * cfg.seq_len))):
        x, y, offset = stream.get_batch(cfg.seq_len, cfg.batch_size, offset)
        if offset == 0:
            break
        x, y = x.to(device), y.to(device)
        h = model.embed_tokens(x)
        out, state, _ = model(h, state)
        loss = model.compute_loss(out, y)
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
    parser.add_argument('--bind-K', type=int, default=16)
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
    )
    
    train(cfg, resume_path=args.resume)
