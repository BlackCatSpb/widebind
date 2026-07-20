"""
Micro architecture test: verify W_pred learning at small scale.

Architecture logic preserved:
  PartitionedEmbed → Bind → Mirror(|pred_error| gate, aux loss) → MLP → Head
  surv=K/G≥1, d/k integer, sparse block codes, grouped MLP, VSA memory.

Config: D=256, L=4, G=4, d=64, K=4, mirror_k=4, expand=4, ~600K params.
Runs in ~15min for 10K steps on T4 (B=8, L=32).
"""

import sys, os, time, math, glob, gc, json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import WideBindStack
from core.config import WideBindConfig


def micro_config(data_dir='.'):
    return WideBindConfig(
        D=256,
        n_layers=4,
        bind_K=4,
        vocab=10000,
        code_dim=16,
        code_sparsity=8,
        mirror_k=4,
        w_pred_scale_init=3.0,
        mlp_groups=4,
        mlp_expand=4,
        conv_kernel=8,
        seq_len=32,
        batch_size=8,
        lr=3e-4,
        max_steps=10000,
        warmup_steps=500,
        weight_decay=0.01,
        grad_clip=1.0,
        scheduler='mirror',
        gate_lr_mult=20.0,
        target_var=0.1,
        mag_threshold=0.3,
        lr_min_ratio=0.05,
        max_decay_steps=10000,
        save_dir=os.path.join(data_dir, 'checkpoints_micro'),
        log_dir=os.path.join(data_dir, 'logs_micro'),
        data_dir=data_dir,
    )


def load_data(data_dir):
    """Load token streams from .bin files, clamp to vocab range."""
    stream_files = sorted(glob.glob(os.path.join(data_dir, 'token_stream_*_clean.bin')))
    if not stream_files:
        stream_files = sorted(glob.glob(os.path.join(data_dir, 'token_stream_*.bin')))
    if not stream_files:
        print(f'No data files found in {data_dir}')
        print('Generating random data (100K tokens)')
        return [np.random.randint(0, 10000, size=100000, dtype=np.uint16)]
    streams = []
    for f in stream_files:
        raw = np.fromfile(f, dtype=np.uint16)
        raw = raw % 10000  # clamp to vocab
        # remove outliers (>99th percentile of frequency to avoid degenerate seqs)
        streams.append(raw.astype(np.uint16))
    total = sum(len(s) for s in streams)
    print(f'Loaded {len(stream_files)} files, {total:,} tokens')
    return streams


def run(data_dir='/content/drive/MyDrive/widebind_data'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"})')

    cfg = micro_config(data_dir)
    streams = load_data(data_dir)
    model = WideBindStack(cfg).to(device)
    print(f'Params: {model.param_count():,}')
    print(f'surv=K/G={cfg.bind_K}/{cfg.mlp_groups}={cfg.bind_K/cfg.mlp_groups:.0f}')
    print(f'd/k={cfg.D//cfg.mlp_groups}/{cfg.mirror_k}={cfg.D//cfg.mlp_groups//cfg.mirror_k}:1')

    os.makedirs(cfg.save_dir, exist_ok=True)

    param_groups = model.param_groups()
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

    # Manual lr scheduler: simple linear warmup + cosine
    def get_lr(step):
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
        return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    B, L = cfg.batch_size, cfg.seq_len
    state = None
    gs = None
    si = 0
    offset = 0
    t0 = time.time()
    tokens_seen = 0
    start_step = 0

    print(f'Training: {cfg.max_steps} steps, B={B}, L={L}')
    print(f'  step{"":>6} loss{"":>8} |I-diff|{"":>8} gate_var{"":>8} var(ls){"":>8} tok/s')
    print('-' * 65)

    for step in range(start_step, cfg.max_steps):
        stream = streams[si]
        if offset + B * L + 1 > len(stream):
            offset = 0
            si = (si + 1) % len(streams)
            stream = streams[si]
            state = None
            gs = None

        chunk = stream[offset:offset + B * L + 1]
        x = torch.from_numpy(chunk[:B * L].copy()).long().view(B, L).to(device)
        y = torch.from_numpy(chunk[1:B * L + 1].copy()).long().view(B, L).to(device)
        offset += B * L

        h = model.embed_tokens(x)
        out, state, gs = model(h, state, global_state=gs)
        loss = model.compute_loss(out, y, pred_weight=1.0)
        del x, y, h, out

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tokens_seen += B * L

        # Manual LR update
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            base = pg.get('lr_base', pg['lr'])
            if step == start_step:
                pg['lr_base'] = pg['lr']
            pg['lr'] = base * lr / cfg.lr if step >= cfg.warmup_steps else lr / cfg.lr * base

        if step % 100 == 0:
            with torch.no_grad():
                alpha_dev = torch.stack([
                    (1.0 - l.mirror.alpha_diag.data).abs().mean()
                    for l in model.layers
                ]).mean().item()
                gv = torch.stack([l.mirror._last_gates.var() for l in model.layers]).mean().item()
                lv = torch.stack([l.mirror.log_scale.data.var() for l in model.layers]).mean().item()
            dt = time.time() - t0
            tok_s = tokens_seen / max(dt, 1e-8)
            print(f'  {step:>6}  {loss.item():.4f}  {alpha_dev:.6f}  {gv:.6f}  {lv:.6f}  {tok_s:.0f}')

        if step > 0 and step % 5000 == 0:
            ckpt = {'step': step, 'model': model.state_dict(), 'cfg': cfg}
            torch.save(ckpt, os.path.join(cfg.save_dir, f'micro_step_{step}.pt'))
            print(f'  Checkpoint saved (step {step})')

    dt = time.time() - t0
    print(f'\nDone in {dt:.0f}s ({tokens_seen/dt:.0f} tok/s)')

    # Final alpha analysis
    with torch.no_grad():
        for i, l in enumerate(model.layers):
            alpha = l.mirror.alpha_diag.data
            dev = (1.0 - alpha).abs().mean().item()
            print(f'  L{i}: |1-alpha|_mean={dev:.6f} alpha_mean={alpha.mean().item():.4f} gate_mean={l.mirror._last_gates.mean().item():.4f} gate_var={l.mirror._last_gates.var().item():.4f}')

    # Summary verdict
    with torch.no_grad():
        final_alpha_dev = torch.stack([
            (1.0 - l.mirror.alpha_diag.data).abs().mean()
            for l in model.layers
        ]).mean().item()

    print(f'\n=== VERDICT ===')
    print(f'Init |1-alpha| ≈ 0.01  Final |1-alpha| = {final_alpha_dev:.6f}')
    if final_alpha_dev > 0.02:
        print('✓ alpha LEARNED (moved >2% from 1.0)')
    elif final_alpha_dev > 0.015:
        print('~ alpha slightly moved (>1.5%)')
    else:
        print('✗ alpha NOT LEARNING (stuck near 1.0)')

    return final_alpha_dev


if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '/content/drive/MyDrive/widebind_data'
    run(data_dir)
