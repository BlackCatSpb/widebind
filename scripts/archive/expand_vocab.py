"""
WideBind: expand vocabulary without losing trained weights.

The new vocab size is NOT picked randomly — it follows the math:

  1. Combinadic capacity (hard ceiling):
        max_V = C(K, S) = C(32, 6) = 906192
     Every token is a unique 6-of-32 code; V cannot exceed this.

  2. uint16 storage ceiling:
     Token streams are stored as np.uint16 arrays in .bin files, so
        V <= 65536
     unless the data is re-serialized to uint32.

  3. Segment-load balance (exact divisibility):
     Each of the K=32 embedding segments serves V·S/K tokens. For a
     perfectly balanced load this must be an integer:
        V·S/K integer  <=>  V multiple of K/gcd(K,S) = 32/2 = 16
     (50000 = 16·3125 satisfies this already.)

  4. Data-driven floor (Zipf tail):
     The largest token id present in the token streams defines the
     minimum viable vocab. Adding a margin above it avoids clashing
     with future data.

     Recommended:  vocab = round_up_to_16(max_token_id + 1 + margin)

  Net formula:
        V* = min( max(V_data_floor, V_current),
                  round_up_to_16(...),
                  max_usable )
     max_usable = min(65536, C(K,S))

Usage:
    python scripts/expand_vocab.py best.pt \
        --streams wb/token_stream_FANTASY_clean.bin \
        --margin 5000 \
        --out expanded.pt
"""

import argparse, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from torch.serialization import add_safe_globals

from core import WideBindConfig, WideBindStack


def comb(n, k):
    return math.comb(n, k)


def round_up_to(v, multiple):
    return ((v + multiple - 1) // multiple) * multiple


def compute_vocab(K, S, data_floor=0, margin=0, current_vocab=0, force=None):
    """Return (recommended_vocab, explanation_lines)."""
    cap_combinadic = comb(K, S)
    cap_uint16 = 65536
    max_usable = min(cap_uint16, cap_combinadic)

    if force is not None:
        assert force <= max_usable, f'vocab {force} exceeds max {max_usable}'
        assert force % 16 == 0, f'vocab {force} not multiple of 16 (segment balance)'
        return force, []

    floor = max(data_floor, current_vocab)
    target = floor + margin
    v = round_up_to(target, 16)
    v = min(v, max_usable)

    expl = [
        f'C(K,S)       = C({K},{S})        = {cap_combinadic:,}   (combinadic capacity)',
        f'uint16 cap   = 65536             (token stream storage)',
        f'max usable   = min(above)        = {max_usable:,}',
        f'data floor   = {data_floor:,}    (max token id + 1 in streams)',
        f'current vocab= {current_vocab:,}',
        f'margin       = {margin:,}',
        f'target       = max(floor,curr)+margin = {target:,}',
        f'round to 16  = {v:,}   (V·S/K = {v*S//K:,} integer)',
    ]
    return v, expl


def max_token_id_in_streams(streams, sample_every=1):
    """Scan token streams and return the largest token id (+1)."""
    if not streams:
        return 0
    mx = 0
    for p in streams:
        arr = np.memmap(p, dtype=np.uint16, mode='r')
        if sample_every > 1:
            arr = arr[::sample_every]
        if arr.size:
            mx = max(mx, int(arr.max()))
        del arr
    return mx + 1


def expand(checkpoint_path, new_vocab, out_path):
    """Load checkpoint, rebuild model with bigger vocab, transfer weights."""
    add_safe_globals([WideBindConfig])
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    old_cfg = ckpt.get('cfg', WideBindConfig())
    old_vocab = old_cfg.vocab
    assert new_vocab >= old_vocab, f'new vocab {new_vocab} < old {old_vocab}'

    # Rebuild config and model
    cfg = old_cfg.clone() if hasattr(old_cfg, 'clone') else old_cfg
    try:
        import copy
        cfg = copy.deepcopy(old_cfg)
    except Exception:
        pass
    cfg.vocab = new_vocab
    model = WideBindStack(cfg)

    # Transfer all shared weights, keep new codes (prefix-identical by construction)
    new_sd = model.state_dict()
    old_sd = ckpt['model']

    # Filter: only load keys with compatible shapes (skip embed.codes, lm_head.codes,
    # lm_head.token_bias which legitimately grew with vocab)
    compat = {k: v for k, v in old_sd.items()
              if k in new_sd and new_sd[k].shape == v.shape}
    skipped = [k for k in old_sd if k not in compat]
    missing, unexpected = model.load_state_dict(compat, strict=False)
    print(f'  loaded compatible weights: {len(compat)}')
    for k in skipped:
        print(f'  skipped (vocab-size change): {k}')
    if missing:
        print(f'  missing: {len(missing)}')
    if unexpected:
        print(f'  unexpected: {len(unexpected)}')

    # Transfer token_bias prefix (first old_vocab entries keep their learned values)
    key = 'lm_head.token_bias'
    if key in old_sd and key in new_sd:
        with torch.no_grad():
            model.lm_head.token_bias[:old_vocab] = old_sd[key]

    # Verify codes prefix-identical (old tokens keep identical sparse codes)
    embed_codes = new_sd['embed.codes']
    old_embed_codes = old_sd['embed.codes']
    assert embed_codes.shape[1] == old_embed_codes.shape[1], 'K changed unexpectedly'
    prefix_ok = bool(torch.equal(embed_codes[:old_vocab], old_embed_codes))
    print(f'  codes prefix identical for old {old_vocab} tokens: {prefix_ok}')
    assert prefix_ok, 'sparse codes prefix mismatch — old token embeddings would change!'

    # Save expanded checkpoint
    ckpt['cfg'] = cfg
    ckpt['model'] = model.state_dict()
    ckpt['vocab_expanded'] = new_vocab
    torch.save(ckpt, out_path)
    print(f'Saved expanded checkpoint -> {out_path}')
    return cfg


def main():
    ap = argparse.ArgumentParser(description='Expand vocab without losing weights')
    ap.add_argument('checkpoint', type=str, help='path to .pt checkpoint')
    ap.add_argument('--out', type=str, default='', help='output path (default: <ckpt>_v<V>.pt)')
    ap.add_argument('--vocab', type=int, default=0, help='force exact vocab (must be mult of 16)')
    ap.add_argument('--streams', type=str, nargs='*', default=[],
                    help='token stream .bin files to scan for max token id')
    ap.add_argument('--margin', type=int, default=5000,
                    help='margin above data floor (default 5000)')
    ap.add_argument('--sample-every', type=int, default=1,
                    help='scan every Nth token (faster on huge files)')
    args = ap.parse_args()

    # Load old cfg to know current vocab and K/S
    add_safe_globals([WideBindConfig])
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ckpt.get('cfg', WideBindConfig())
    K, S = cfg.code_dim, cfg.code_sparsity
    current_vocab = cfg.vocab

    print('=== VOCAB MATH ===')
    data_floor = 0
    if args.streams:
        data_floor = max_token_id_in_streams(args.streams, args.sample_every)
        print(f'  max token id in streams: {data_floor - 1:,}')

    new_vocab, expl = compute_vocab(
        K, S, data_floor=data_floor, margin=args.margin,
        current_vocab=current_vocab, force=args.vocab)
    for line in expl:
        print(f'  {line}')
    print(f'==> recommended vocab = {new_vocab:,}')

    out = args.out or args.checkpoint.replace('.pt', f'_v{new_vocab}.pt')
    expand(args.checkpoint, new_vocab, out)


if __name__ == '__main__':
    main()
