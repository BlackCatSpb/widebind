# Training Log — GroupedMLP architecture

**Architecture:** WideBind 24L D=896 K=16 GroupedMLP (G=8, expand=8×)
**Params:** 41,246,336 (41.25M)
**Data:** 2,859,675,614 tokens (ACTION 1.1B + DETECT 1.8B)
**GPU:** MX550 2GB, B=2, L=128, fp32

---

## Run 1 — GroupedMLP (8×112→896→112, 8× expansion per group)

| Step | Train Loss | LR | Notes |
|------|-----------|-----|-------|
| 0 | 12.93 | 3e-07 | init, warmup start |
| 100 | 11.62 | 3e-05 | warmup |
| 200 | 6.03 | 6e-05 | warmup |
| 300 | 7.60 | 9e-05 | spike |
| 400 | 4.28 | 1.2e-04 | |
| 500 | 4.26 | 1.5e-04 | |
| 600 | 4.43 | 1.8e-04 | spike |
| 700 | 4.18 | 2.1e-04 | |
| 800 | 2.98 | 2.4e-04 | sharp drop — MLP found useful direction |
| 900 | 1.60 | 2.7e-04 | |
| 1000 | **1.44** | 3.0e-04 | warmup complete, **val_loss=1.99 ppl=7.32** |

### Eval at step 1000

| Metric | Old MLP (step 1000) | GroupedMLP (step 1000) |
|--------|-------------------|------------------------|
| Train loss | ~3.5 | **1.44** |
| Val loss | ~8.22 | **1.99** |
| Val ppl | ~3700 | **7.32** |

### Layer stats at step 1000

| Layer | eff_r(MLP) | λ_k mean | log_scale μ | bind_er |
|-------|-----------|----------|------------|---------|
| L0 | 62.1 / 112 | 0.50 | 0.0000 | 13.1 |
| L10 | 62.5 / 112 | 0.93 | 0.0000 | 12.9 |
| L17 | 58.2 / 112 | 1.24 | −0.0002 | 12.8 |
| L20 | 32.8 / 112 | 1.37 | 0.0005 | 12.7 |
| L23 | 11.8 / 112 | 1.50 | 0.0004 | 10.9 |

- L0-L17 all groups active (eff_rank ~55-62/112)
- Gradual collapse L18→L23 (structural, LM head compression)
- λ_k grows 0.50→1.50 with increasing per-dim variation (std 0.002→0.015)
- Mirror not yet active (log_scale ≈ 0, exp=1)
- Gates frozen at init (b_i=−3.0, b_d=5.0)
- Bind healthy (eff_rank 10.9-13.6/16)

| 1000 | 1.44 | 3.0e-04 | warmup done, **val_loss=1.99 ppl=7.32** |
| 2000 | 1.53 | 3.0e-04 | **val_loss=2.27 ppl=9.67** — plateau at constant LR |
| 2100-2700 | 1.3-1.7 | 3.0e-04 | high variance (B=2), no trend — cosine decay not yet felt |

### Eval at step 2000

| Metric | Old MLP | GroupedMLP |
|--------|---------|------------|
| Train loss | 1.34 | 1.53 |
| Val loss | **5.33** | **2.27** |
| Generalization gap | 3.99 | 0.74 |
| L23 eff_rank | 4.2 / 896 | 11.8 / 112 |

**Key improvement:** Generalization gap reduced 5.4× (3.99 → 0.74). Model generalizes, not memorizes.

### Current status

- Train loss plateau ~1.4 at constant LR=3e-4
- Cosine decay too slow — switching to MirrorLRScheduler at step 3000+
- MirrorLRScheduler: modulates LR from log_scale variance + mirror magnitude
  - var(log_scale) grows → decay factor drops
  - |mirror| shrinks at convergence → mag_factor drops
  - Combined: LR_adapt = base_lr × (1 - var/0.1) × min(1, mag/0.3)
- Expected: adaptive LR breaks plateau by reducing LR when mirror stabilizes

### Comparison with Run 0 (flat MLP 896→896→896)

Flat MLP at step 2000: train_loss=1.34, val_loss=5.33, MLP barely moved (std=0.033), L23 eff_rank=4.2.
GroupedMLP at step 1000: val_loss=1.99, L23 eff_rank=11.8 — **22× better utilization at L23, 4× better val_loss in half the steps.****
