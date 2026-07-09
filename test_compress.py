"""
FCF-CPR test: compress best.pt using lambda_d quantization + Zeckendorf sparse.
"""
import math, os, sys
import torch
import numpy as np

# ─── FCF math utilities ───

def lambda_d(d=2):
    """Generalized golden ratio for order d."""
    # solve x^d = x^{d-1} + ... + x + 1
    # For d=2: (1 + sqrt(5)) / 2
    if d == 2:
        return (1 + 5**0.5) / 2
    x = 2.0
    for _ in range(100):
        x = (x**d - 1) / (x**(d-1) - 1) if x**(d-1) != 1 else x
    return x

def lambda_levels(k_min=-10, k_max=10, d=2):
    """λ_d^{-k} levels for quantization."""
    lam = lambda_d(d)
    return [lam ** (-k) for k in range(k_min, k_max + 1)]

def generalized_fib(n, d=2):
    """Generalized Fibonacci F^d_n."""
    if n < 0:
        return 0
    if n == 0:
        return 1
    fib = [0] * (n + 1)
    fib[0] = 1
    for i in range(1, n + 1):
        s = 0
        for k in range(1, d + 1):
            s += fib[i - k] if i - k >= 0 else 0
        fib[i] = s
    return fib[n]

def zeckendorf_encode(n, fibs=None):
    """Zeckendorf representation: list of Fibonacci indices (no consecutive)."""
    if fibs is None:
        fibs = [1, 2]
        while fibs[-1] <= n:
            fibs.append(fibs[-1] + fibs[-2])
    fibs = [f for f in fibs if f <= n]
    codes = []
    for i in range(len(fibs) - 1, -1, -1):
        if n >= fibs[i]:
            codes.append(i)
            n -= fibs[i]
    return codes

def zeckendorf_decode(indices, fibs=None):
    """Decode Zeckendorf back to integer."""
    if fibs is None:
        fibs = [1, 2]
        max_idx = max(indices) + 1
        while len(fibs) < max_idx:
            fibs.append(fibs[-1] + fibs[-2])
    return sum(fibs[i] for i in indices)


# ─── Compression classes ───

def quantize_lambda(tensor, k_min=-10, k_max=10, d=2):
    """Quantize tensor to nearest λ_d^{-k} level."""
    lam = lambda_d(d)
    levels = torch.tensor([lam ** (-k) for k in range(k_min, k_max + 1)], dtype=torch.float32)
    flat = tensor.flatten()
    indices = torch.zeros(flat.shape, dtype=torch.int16)
    for i, val in enumerate(flat):
        idx = (levels - val).abs().argmin().item()
        indices[i] = idx
    return indices.reshape(tensor.shape), levels

def dequantize_lambda(indices, levels):
    """Restore from quantized indices."""
    return levels[indices].reshape(indices.shape)

def analyze_tensor_group(tensor, name=""):
    """Statistics for determining quantization method."""
    flat = tensor.flatten().float()
    print(f"  {name}: shape={list(tensor.shape)} "
          f"mean={flat.mean():.6f} std={flat.std():.6f} "
          f"min={flat.min():.6f} max={flat.max():.6f} "
          f"nnz={torch.count_nonzero(flat).item()}/{flat.numel()}")


# ─── Test ───

def test_compress():
    ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'best.pt')
    if not os.path.isfile(ckpt_path):
        # Try desktop
        ckpt_path = os.path.expanduser('~/Desktop/best.pt')
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(__file__), '..', '..', 'best.pt')
    if not os.path.isfile(ckpt_path):
        print(f'best.pt not found, using step_10000.pt from checkpoints/')
        ckpt_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'step_10000.pt')
    if not os.path.isfile(ckpt_path):
        print('No checkpoint found!')
        return
    
    print(f'Loading: {ckpt_path}')
    print(f'Size: {os.path.getsize(ckpt_path) / 1e9:.2f} GB')
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(f'Keys: {list(ckpt.keys())}')
    
    sd = ckpt['model']
    total_elems = sum(p.numel() for p in sd.values())
    total_bytes = sum(p.numel() * p.element_size() for p in sd.values())
    print(f'\nModel: {len(sd)} keys, {total_elems:,} elems, {total_bytes/1e6:.1f} MB')
    
    # Classify tensors
    print('\n=== Tensor analysis ===')
    removable_buffers = ['V_dct', 'codes']
    near_constant = []  # b_i, b_d
    log_scale_tensors = []
    main_weights = []
    sparse_convs = []
    lambdas = []
    
    for k, v in sd.items():
        if any(r in k for r in removable_buffers):
            analyze_tensor_group(v, f'[REMOVABLE] {k}')
        elif 'b_i' in k or 'b_d' in k:
            near_constant.append((k, v))
            analyze_tensor_group(v, f'[SCALAR] {k}')
        elif 'log_scale' in k:
            log_scale_tensors.append((k, v))
            analyze_tensor_group(v, f'[LOG_SCALE] {k}')
        elif 'lambda_k' in k:
            lambdas.append((k, v))
            analyze_tensor_group(v, f'[LAMBDA] {k}')
        elif 'pre_ln_w' in k or 'norm_w' in k or 'final_norm_w' in k:
            analyze_tensor_group(v, f'[NORM] {k}')
        elif 'conv.weight' in k and v.ndim >= 2:
            sparse_convs.append((k, v))
            analyze_tensor_group(v, f'[CONV_SPARSE] {k}')
        else:
            main_weights.append((k, v))
    
    # Estimate compression
    print('\n=== Compression estimate ===')
    
    # 1. Removable buffers
    removable_size = sum(v.numel() * v.element_size() for k, v in sd.items() 
                         if any(r in k for r in removable_buffers))
    print(f'Removable buffers: {removable_size/1e6:.1f} MB -> 0 MB')
    
    # 2. Scalar gate biases (b_i, b_d) - store as 1 scalar per layer
    scalar_size = sum(v.numel() * v.element_size() for k, v in near_constant)
    n_layers = len(near_constant) // 2
    scalar_compressed = n_layers * 2 * 4  # 2 biases × 4 bytes fp32
    print(f'Scalar gates ({n_layers} layers): {scalar_size/1e6:.1f} MB -> {scalar_compressed} bytes')
    
    # 3. Lambda levels
    lam_size = sum(v.numel() * v.element_size() for k, v in lambdas)
    lam_compressed = sum(v.numel() for k, v in lambdas)  # indices
    print(f'Lambda levels: {lam_size/1e6:.1f} MB -> {lam_compressed/1e3:.1f} KB (8-bit indices)')
    
    # 4. log_scale (8-bit λ-quant)
    ls_size = sum(v.numel() * v.element_size() for k, v in log_scale_tensors)
    ls_compressed = sum(v.numel() for k, v in log_scale_tensors)  # uint8
    print(f'Log_scale: {ls_size/1e6:.1f} MB -> {ls_compressed/1e3:.1f} KB (8-bit)')
    
    # 5. Norm weights (8-bit near-1.0 quantization)
    norm_size = sum(v.numel() * v.element_size() for k, v in sd.items()
                    if 'norm_w' in k or 'pre_ln_w' in k or 'final_norm_w' in k)
    norm_compressed = sum(v.numel() for k, v in sd.items()
                          if 'norm_w' in k or 'pre_ln_w' in k or 'final_norm_w' in k)
    print(f'Norm weights: {norm_size/1e6:.1f} MB -> {norm_compressed/1e3:.1f} KB (8-bit)')
    
    # 6. Conv sparse (Zeckendorf)
    conv_size = sum(v.numel() * v.element_size() for k, v in sparse_convs)
    conv_nnz = sum(torch.count_nonzero(v).item() for k, v in sparse_convs)
    conv_total = sum(v.numel() for k, v in sparse_convs)
    print(f'Conv sparse: {conv_size/1e3:.1f} KB ({conv_nnz}/{conv_total} non-zero)')
    # Zeckendorf encoding: each index needs ~log2(max_idx) bits
    zeck_bits = conv_nnz * math.ceil(math.log2(conv_total / max(1, len(sparse_convs))))
    print(f'  -> Zeckendorf sparse: ~{zeck_bits/8/1e3:.1f} KB')
    
    # 7. Main weights (8-bit λ-quant)
    main_size = sum(v.numel() * v.element_size() for k, v in main_weights)
    main_compressed = sum(v.numel() for k, v in main_weights)  # uint8
    print(f'Main weights: {main_size/1e3:.1f} MB -> {main_compressed/1e3:.1f} MB (8-bit)')
    
    # Total
    print(f'\n=== Summary ===')
    print(f'Original model: {total_bytes/1e9:.2f} GB')
    compressed = main_compressed + ls_compressed + norm_compressed + lam_compressed + scalar_compressed + zeck_bits//8
    print(f'Compressed model: {compressed/1e9:.2f} GB ({compressed/1e6:.1f} MB)')
    print(f'Ratio: {total_bytes / max(compressed, 1):.1f}×')
    
    # ─── Test quantization accuracy ───
    print('\n=== Quantization accuracy test ===')
    lam = lambda_d(2)
    levels = torch.tensor([lam ** (-k) for k in range(-8, 9)], dtype=torch.float32)
    print(f'Lambda levels (d=2, k=-8..8): {[f"{v:.4f}" for v in levels.tolist()]}')
    
    # Test on W_proj from layer 0
    for k, v in sd.items():
        if 'layers.0.W_proj' in k:
            quantized, lvls = quantize_lambda(v, k_min=-8, k_max=8)
            restored = dequantize_lambda(quantized, lvls)
            mse = ((v - restored) ** 2).mean().item()
            max_err = (v - restored).abs().max().item()
            print(f'  W_proj: MSE={mse:.8f}, max_err={max_err:.6f}')
            print(f'  Indices: unique={len(quantized.unique())}/{len(lvls)} levels')
            break
    
    # Test on log_scale
    for k, v in sd.items():
        if 'log_scale' in k:
            quantized, lvls = quantize_lambda(v, k_min=-10, k_max=10)
            restored = dequantize_lambda(quantized, lvls)
            mse = ((v - restored) ** 2).mean().item()
            max_err = (v - restored).abs().max().item()
            print(f'  log_scale: MSE={mse:.12f}, max_err={max_err:.8f}')
            break
    
    # Test Zeckendorf on conv indices
    for k, v in sd.items():
        if 'conv.weight' in k and 'layers.0' in k:
            flat = v.flatten()
            nnz_idx = torch.nonzero(flat.abs() > 1e-6).flatten()
            print(f'\n  Zeckendorf test: {nnz_idx.numel()} non-zero of {flat.numel()}')
            fibs = [1, 2]
            max_idx = nnz_idx.max().item()
            while fibs[-1] <= max_idx:
                fibs.append(fibs[-1] + fibs[-2])
            # Encode first 10 indices
            for i in range(min(10, nnz_idx.numel())):
                enc = zeckendorf_encode(nnz_idx[i].item(), fibs)
                dec = zeckendorf_decode(enc, fibs)
                assert dec == nnz_idx[i].item(), f'Zeckendorf error: {dec} != {nnz_idx[i]}'
            print(f'  Zeckendorf: OK (encode/decode roundtrip)')
            # Bits per index
            bits_per = math.ceil(math.log2(max_idx + 1))
            zeck_bits_via_fib = len(fibs)  # worst case: each fib index used once
            print(f'  Binary: {bits_per} bits/index, Zeckendorf max: {zeck_bits_via_fib} bits/index')
            break

    print('\nDone.')


if __name__ == '__main__':
    test_compress()
