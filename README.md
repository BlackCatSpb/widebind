# WideBind

Hybrid VSA-style language model operating in full D=896 space with:
- **Bind**: D → K (K=16 bottleneck) → bilinear → D — cross-dim mixing through learnable projections
- **Memory**: VSA vector superposition (not covariance matrix) — O(D) instead of O(D²)
- **Gates**: Per-dim element-wise sigmoid/exp with learnable biases
- **Conv**: Depthwise 48-tap 1D convolution
- **Spectral**: DCT basis mixing with learned per-dim scaling
- **MLP**: D → bottleneck → D with SiLU activation

Key advantages over factorized K=24 MemBind:
- 156M params fit in ~2 GB VRAM (vs 1.6M for K=24)
- Full D=896 expressiveness in memory and gates
- VSA memory avoids covariance matrix quadratic cost
- Learnable bind bottleneck (K=16) provides cross-dim mixing without full D×D matrices

## Quick start

```bash
# Train
python train.py --data-dir /path/to/token_streams --batch-size 2 --max-steps 50000

# Generate
python generate.py checkpoints/best.pt --prompt "Hello"
```

## Architecture

```
h → Pre-LN → Conv → Bind(D→K→D) → + VSA_memory → + Mirror → Spectral(DCT) → MLP → h'
```

## Config

See `config.py` for all hyperparameters.
