"""
WideBind Cloud Training: Yandex Cloud / any GPU server.

Usage:
    python scripts/cloud_train.py --data-dir /path/to/token_streams --save-dir /path/to/checkpoints

Args:
    --data-dir   Path to directory with token_stream_*.bin files
    --save-dir   Where to save checkpoints and logs (default: ./checkpoints)
    --resume     Path to checkpoint to resume from (optional)
"""
import os, sys, math, time, glob, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np

from core import WideBindConfig, WideBindStack, MirrorLRScheduler, AdaptiveController


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


@torch.no_grad()
def eval_loss(model, streams, cfg, device, n_batches=200):
    model.eval()
    total, count = 0.0, 0
    for stream in streams:
        off = max(stream.len // 4, cfg.batch_size * cfg.seq_len + 1)
        for _ in range(n_batches):
            x, y, off = stream.get_batch(cfg.seq_len, cfg.batch_size, off)
            if off == 0: break
            h = model.embed_tokens(x.to(device))
            out, _, _ = model(h, None)
            total += model.compute_loss(out, y.to(device)).item()
            count += 1
    model.train()
    return total / max(count, 1)


def train(cfg, data_dir, save_dir, resume_path=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if device == 'cuda' else 0
    print(f'Device: {device} ({gpu_name})  VRAM: {vram:.1f} GB')

    # ─── Data ───
    stream_files = sorted(glob.glob(os.path.join(data_dir, 'token_stream_*_clean.bin')))
    if not stream_files:
        stream_files = sorted(glob.glob(os.path.join(data_dir, 'token_stream_*.bin')))
    if not stream_files:
        raise FileNotFoundError(f'No token_stream_*.bin in {data_dir}')
    streams = [TokenStream(f) for f in stream_files]
    total_tokens = sum(s.len for s in streams)
    print(f'Data: {len(streams)} files, {total_tokens:,} tokens')

    # ─── Model ───
    model = WideBindStack(cfg).to(device)
    print(f'Model: {model.param_count():,} params ({model.param_count()/1e6:.2f}M)')
    print(f'  tie_bind={cfg.tie_bind}  tie_mirror_proj={cfg.tie_mirror_proj}')
    print(f'  lambda_d={cfg.lambda_d}  bind_K={cfg.bind_K}  mirror_k={cfg.mirror_k}')

    # ─── Batch size (auto-fit) ───
    for bs in [2, 1]:
        torch.cuda.empty_cache()
        try:
            x = torch.randint(0, cfg.vocab, (bs, cfg.seq_len), device=device)
            h = model.embed_tokens(x)
            out, _, _ = model(h, None)
            out[:, :1].sum().backward()
            model.zero_grad()
            del x, h, out
            cfg.batch_size = bs
            print(f'Batch size: {bs}')
            break
        except RuntimeError:
            model.zero_grad(); torch.cuda.empty_cache()
    else:
        cfg.batch_size = 1
        print('Batch size: 1 (minimum)')

    # ─── Optimizer ───
    optimizer = torch.optim.AdamW(model.param_groups(), betas=(0.9, 0.95))
    scheduler = MirrorLRScheduler(model, optimizer, cfg.lr,
        warmup=cfg.warmup_steps, target_var=cfg.target_var,
        mag_threshold=cfg.mag_threshold, lr_min_ratio=cfg.lr_min_ratio,
        max_decay_steps=cfg.max_decay_steps,
        var_min_for_lr_decay=cfg.var_min_for_lr_decay)

    # ─── Resume ───
    start_step = 0
    best_val = float('inf')
    os.makedirs(save_dir, exist_ok=True)

    if resume_path and os.path.isfile(resume_path):
        print(f'Resuming from {resume_path}')
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_step = ckpt['step']
        best_val = ckpt.get('best_val_loss', float('inf'))
        print(f'  Step {start_step}, best val loss {best_val:.4f}')

    # ─── Training ───
    state = None
    si, off = 0, 0
    tokens_seen = 0
    t0 = time.time()
    log_metrics = {'step': [], 'loss': [], 'idiff': [], 'gate_var': [], 'lr': []}

    print(f'Training: {start_step} -> {cfg.max_steps} steps')
    print(f'  {">":>6}  {"loss":>8}  {"|1-alpha|":>10}  {"gate_var":>8}  {"lr":>10}  {"tok/s":>6}')
    print('  ' + '-' * 58)

    try:
        for step in range(start_step, cfg.max_steps):
            model.train()
            stream = streams[si]
            x, y, off = stream.get_batch(cfg.seq_len, cfg.batch_size, off)
            if off == 0:
                si = (si + 1) % len(streams)
                _, _, off = streams[si].get_batch(cfg.seq_len, cfg.batch_size, 0)
                state = None

            x, y = x.to(device), y.to(device)
            h = model.embed_tokens(x)
            out, state, _ = model(h, state)
            loss = model.compute_loss(out, y, pred_weight=1.0)
            loss.backward()

            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            tokens_seen += cfg.batch_size * cfg.seq_len

            if step % cfg.log_interval == 0:
                dt = time.time() - t0
                tok_s = tokens_seen / max(dt, 1e-8)
                with torch.no_grad():
                    idiff = torch.stack([
                        (1.0 - l.mirror.alpha_diag.data).abs().mean()
                        for l in model.layers
                    ]).mean().item()
                    gv = torch.stack([
                        l.mirror._last_gates.var() for l in model.layers
                    ]).mean().item()
                lr = scheduler.get_last_lr()[0]
                log_metrics['step'].append(step)
                log_metrics['loss'].append(loss.item())
                log_metrics['idiff'].append(idiff)
                log_metrics['gate_var'].append(gv)
                log_metrics['lr'].append(lr)
                print(f'{step:>6}  {loss.item():>8.4f}  {idiff:>10.6f}  {gv:>8.6f}  {lr:>10.2e}  {tok_s:>6.0f}')

            if step > 0 and step % cfg.eval_interval == 0:
                vl = eval_loss(model, streams, cfg, device)
                print(f'  EVAL: val_loss={vl:.4f} ppl={math.exp(vl):.1f}')
                scheduler.report_val_loss(vl)
                if vl < best_val:
                    best_val = vl
                    torch.save({
                        'step': step, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_val_loss': best_val, 'cfg': cfg,
                    }, os.path.join(save_dir, 'best.pt'))
                    print(f'  New best! Saved')

            if step > 0 and step % cfg.save_interval == 0:
                torch.save({
                    'step': step, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_val_loss': best_val, 'cfg': cfg,
                }, os.path.join(save_dir, f'step_{step}.pt'))
                print(f'  Checkpoint saved (step {step})')

    except KeyboardInterrupt:
        print('\nSaving interrupt checkpoint...')
        torch.save({
            'step': step, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': best_val, 'cfg': cfg,
        }, os.path.join(save_dir, f'interrupt_step_{step}.pt'))
        print('Saved.')
        sys.exit(0)

    dt = time.time() - t0
    print(f'\nDone in {dt:.0f}s ({tokens_seen/dt:.0f} tok/s)')
    print(f'Best val loss: {best_val:.4f}')

    # Final alpha analysis
    with torch.no_grad():
        print('\n=== Alpha Analysis ===')
        for i, l in enumerate(model.layers):
            a = l.mirror.alpha_diag.data
            ad = (1.0 - a).abs().mean().item()
            print(f'  L{i}: alpha={a.mean().item():.4f}+/-{a.std().item():.4f}  |1-alpha|={ad:.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WideBind Cloud Training')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory with token_stream_*.bin files')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Directory for checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--D', type=int, default=4096)
    parser.add_argument('--n-layers', type=int, default=32)
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max-steps', type=int, default=500000)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--eval-interval', type=int, default=2000)
    parser.add_argument('--save-interval', type=int, default=5000)
    args = parser.parse_args()

    cfg = WideBindConfig(
        D=args.D, n_layers=args.n_layers,
        seq_len=args.seq_len, lr=args.lr, max_steps=args.max_steps,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        grad_clip=1.0, conv_kernel=48,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        log_dir=os.path.join(args.save_dir, 'logs'),
    )

    train(cfg, args.data_dir, args.save_dir, args.resume)
