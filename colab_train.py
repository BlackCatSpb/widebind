"""
WideBind Colab: D=3584 on T4 with fp16 + gradient checkpointing.
Usage:
    from google.colab import drive
    drive.mount('/content/drive')

    !python colab_train.py --drive-path /content/drive/MyDrive/widebind_data
"""

import os, sys, math, time, glob, argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.serialization import add_safe_globals

from wbconfig import WideBindConfig
from core import WideBindStack, MirrorLRScheduler, AdaptiveController

add_safe_globals([WideBindConfig])


class TokenStream:
    """Memory-efficient stream: uint16 numpy, converted to torch.long per batch."""
    def __init__(self, path):
        self.data = np.fromfile(path, dtype=np.uint16)
        self.len = len(self.data)
    def get_batch(self, seq_len, batch_size, offset):
        needed = batch_size * seq_len + 1
        if offset + needed > self.len:
            offset = 0
        chunk = self.data[offset:offset + needed]
        x = torch.from_numpy(chunk[:batch_size * seq_len].reshape(batch_size, seq_len).copy())
        y = torch.from_numpy(chunk[1:batch_size * seq_len + 1].reshape(batch_size, seq_len).copy())
        return x.long(), y.long(), offset + batch_size * seq_len


def find_best_batch_size(model, seq_len, device, start=8):
    """Binary search for max batch size that fits in VRAM."""
    lo, hi = 1, start
    best = 1
    torch.cuda.empty_cache()
    x = torch.randint(0, 50000, (start, seq_len), device=device)
    try:
        h = model.embed_tokens(x)
        out, _ = model(h, None)
        out[:, :1].sum().backward()
        best = start
    except RuntimeError:
        hi = start
    finally:
        model.zero_grad()
        torch.cuda.empty_cache()

    while lo <= hi and best < hi:
        mid = (lo + hi) // 2
        torch.cuda.empty_cache()
        try:
            x = torch.randint(0, 50000, (mid, seq_len), device=device)
            h = model.embed_tokens(x)
            out, _ = model(h, None)
            out[:, :1].sum().backward()
            best = mid
            lo = mid + 1
        except RuntimeError:
            hi = mid - 1
        finally:
            model.zero_grad()
            torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    return max(1, best)


def train(cfg, drive_path=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'
    print(f'Device: {device} ({gpu_name})')
    if device == 'cuda':
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'VRAM: {mem:.1f} GB')

    # ─── Data ───
    data_dir = drive_path or cfg.data_dir
    print(f'Loading data from {data_dir}')
    stream_files = sorted(glob.glob(os.path.join(data_dir, 'token_stream_*_clean.bin')))
    if not stream_files:
        stream_files = sorted(glob.glob(os.path.join(data_dir, 'token_stream_*.bin')))
    if not stream_files:
        raise FileNotFoundError(f'No token_stream_*.bin in {data_dir}')
    streams = [TokenStream(f) for f in stream_files]
    total_tokens = sum(s.len for s in streams)
    print(f'Found {len(streams)} files, {total_tokens:,} total tokens')
    print(f'RAM for data: {total_tokens * 2 / 1e9:.1f} GB (uint16)')

    # ─── Model ───
    model = WideBindStack(cfg).to(device)
    n_params = model.param_count()
    print(f'Model: {n_params:,} params ({n_params/1e6:.2f}M)')

    # ─── Auto batch ───
    print('Finding optimal batch size...')
    cfg.batch_size = find_best_batch_size(model, cfg.seq_len, device)
    print(f'  Batch size: {cfg.batch_size}')

    # ─── Mixed precision ───
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    if scaler:
        print('  Using fp16 mixed precision + gradient scaling')

    # ─── Optimizer ───
    param_groups = model.param_groups()
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

    # ─── Scheduler ───
    if cfg.scheduler == 'mirror':
        scheduler = MirrorLRScheduler(model, optimizer, cfg.lr, warmup=cfg.warmup_steps)
        print(f'Scheduler: MirrorLRScheduler')
    else:
        def get_lr_mult(step):
            if step < cfg.warmup_steps:
                return step / max(cfg.warmup_steps, 1)
            progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_mult)
        print(f'Scheduler: cosine')

    # ─── Resume ───
    start_step = 0
    best_val_loss = float('inf')
    save_dir = drive_path or cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)

    ckpt_files = sorted(glob.glob(os.path.join(save_dir, 'step_*.pt')),
                         key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0]))
    if ckpt_files:
        latest = ckpt_files[-1]
        print(f'Resuming from {latest}')
        ckpt = torch.load(latest, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
        if missing:
            print(f'  Missing keys: {len(missing)}')
        if unexpected:
            print(f'  Unexpected keys: {len(unexpected)}')
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt and cfg.scheduler == 'mirror':
            scheduler.load_state_dict(ckpt['scheduler'])
        elif 'scheduler' in ckpt:
            scheduler.last_epoch = ckpt['step'] + 1
        else:
            scheduler._step = ckpt['step']
        start_step = ckpt['step']
        best_val_loss = ckpt.get('best_val_loss', float('inf'))

    # ─── Training ───
    state = None
    stream_idx = 0
    offset = 0
    tokens_seen = 0
    t0 = time.time()

    print(f'Starting training from step {start_step}')
    print(f'  Steps: {start_step} → {cfg.max_steps} ({cfg.max_steps - start_step} remaining)')
    print(f'  Tokens per step: {cfg.batch_size * cfg.seq_len}')

    try:
        for step in range(start_step, cfg.max_steps):
            model.train()

            stream = streams[stream_idx]
            x, y, offset = stream.get_batch(cfg.seq_len, cfg.batch_size, offset)
            if offset == 0:
                stream_idx = (stream_idx + 1) % len(streams)
                stream = streams[stream_idx]
                _, _, offset = stream.get_batch(cfg.seq_len, cfg.batch_size, 0)
                state = None

            x, y = x.to(device), y.to(device)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                h = model.embed_tokens(x)
                out, state = model(h, state)
                loss = model.compute_loss(out, y)

            if scaler:
                scaler.scale(loss).backward()
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            tokens_seen += cfg.batch_size * cfg.seq_len
            current_lr = scheduler.get_last_lr()[0]

            if step % cfg.log_interval == 0:
                dt = time.time() - t0
                tok_s = tokens_seen / max(dt, 1e-8)
                mem_gb = torch.cuda.max_memory_allocated() / 1e9 if device == 'cuda' else 0
                print(f'  step={step:>6} loss={loss.item():.4f} lr={current_lr:.2e} '
                      f'tok/s={tok_s:.0f} mem={mem_gb:.1f}/{mem:.1f} GB')

                if device == 'cuda':
                    torch.cuda.reset_peak_memory_stats()

            if step > 0 and step % cfg.eval_interval == 0:
                val_loss = evaluate(model, streams, cfg, device)
                print(f'  EVAL step={step}: val_loss={val_loss:.4f} val_ppl={math.exp(val_loss):.2f}')
                if device == 'cuda':
                    torch.cuda.empty_cache()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_path = os.path.join(save_dir, f'best.pt')
                    torch.save({
                        'step': step, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_val_loss': best_val_loss, 'cfg': cfg,
                    }, save_path)
                    print(f'  Saved best to {save_path}')

            if step > 0 and step % cfg.save_interval == 0:
                save_path = os.path.join(save_dir, f'step_{step}.pt')
                torch.save({
                    'step': step, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_val_loss': best_val_loss, 'cfg': cfg,
                }, save_path)
                print(f'  Saved checkpoint to {save_path}')

    except KeyboardInterrupt:
        print('\nSaving interrupt checkpoint...')
        save_path = os.path.join(save_dir, f'interrupt_step_{step}.pt')
        torch.save({
            'step': step, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': best_val_loss, 'cfg': cfg,
        }, save_path)
        print(f'Saved to {save_path}')
        sys.exit(0)

    print('Training complete!')


@torch.no_grad()
def evaluate(model, streams, cfg, device):
    model.eval()
    total_loss = 0.0
    total_steps = 0
    n_batches = min(500, sum(s.len for s in streams) // (cfg.batch_size * cfg.seq_len) // len(streams))

    for stream in streams:
        offset = max(stream.len // 4, cfg.batch_size * cfg.seq_len + 1)
        for _ in range(n_batches):
            x, y, offset = stream.get_batch(cfg.seq_len, cfg.batch_size, offset)
            if offset == 0:
                break
            x, y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=device=='cuda'):
                h = model.embed_tokens(x)
                out, _ = model(h, None)
                loss = model.compute_loss(out, y)
            total_loss += loss.item()
            total_steps += 1

    model.train()
    return total_loss / max(total_steps, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--drive-path', type=str, default=None,
                        help='Google Drive path containing token_stream_*.bin and checkpoints/')
    parser.add_argument('--D', type=int, default=3584)
    parser.add_argument('--n-layers', type=int, default=24)
    parser.add_argument('--bind-K', type=int, default=16)
    parser.add_argument('--mlp-groups', type=int, default=32)
    parser.add_argument('--mlp-expand', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max-steps', type=int, default=300000)
    parser.add_argument('--warmup', type=int, default=2000)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--eval-interval', type=int, default=1000)
    parser.add_argument('--save-interval', type=int, default=5000)
    parser.add_argument('--scheduler', type=str, default='mirror', choices=['cosine', 'mirror'])
    args = parser.parse_args()

    cfg = WideBindConfig(
        D=args.D,
        n_layers=args.n_layers,
        bottleneck=args.D,      # MLP = D for base bottleneck
        bind_K=args.bind_K,
        mlp_groups=args.mlp_groups,
        mlp_expand=args.mlp_expand,
        seq_len=args.seq_len,
        lr=args.lr,
        max_steps=args.max_steps,
        warmup_steps=args.warmup,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        scheduler=args.scheduler,
        data_dir=args.drive_path or '',
        save_dir=args.drive_path or 'checkpoints',
        log_dir=args.drive_path or 'logs',
        grad_clip=1.0,
        conv_kernel=48,
    )

    train(cfg, drive_path=args.drive_path)
