"""SiTU-GLU (bounded SwiGLU variant, Kimi K3) tests.

SiTU-GLU(x) = (β1·tanh(Wg·x/β1) ⊙ sigmoid(Wg·x)) ⊙ (β2·tanh(Wu·x/β2))
  gate branch capping factor |·| ≤ β1, up branch |·| ≤ β2 → |gate·up| ≤ β1·β2.

Key invariants:
  * Default path is UNCHANGED: situ_glu=False keeps the exact SwiGLU computation.
  * No softmax anywhere (only tanh + sigmoid, matching WideBind's no-softmax rule).
  * Zero extra parameters vs SwiGLU (same W_gate/W_up/W_down).
  * Bounded internal activation → lower overflow/NaN risk (fp32 on T4).
"""

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F

from core.config import WideBindConfig
from core.mlp import GroupedMLP
from core.model import WideBindStack


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _manual_swiglu(mlp, x):
    """Reference SwiGLU computation with the module's own weights."""
    D = mlp.D
    B, L, _ = x.shape
    hb = F.rms_norm(x, (D,), mlp.norm_w).reshape(B, L, mlp.G, mlp.d)
    g = torch.einsum('blgd,gdf->blgf', hb, mlp.W_gate)
    u = torch.einsum('blgd,gdf->blgf', hb, mlp.W_up)
    mid = F.silu(g) * u
    return torch.einsum('blgf,gfd->blgd', mid, mlp.W_down).reshape(B, L, D)


def _manual_situ(mlp, x, beta1=None, beta2=None):
    """Reference SiTU-GLU computation with the module's own weights."""
    D = mlp.D
    B, L, _ = x.shape
    beta1 = mlp.beta1 if beta1 is None else beta1
    beta2 = mlp.beta2 if beta2 is None else beta2
    hb = F.rms_norm(x, (D,), mlp.norm_w).reshape(B, L, mlp.G, mlp.d)
    g = torch.einsum('blgd,gdf->blgf', hb, mlp.W_gate)
    u = torch.einsum('blgd,gdf->blgf', hb, mlp.W_up)
    gate = beta1 * torch.tanh(g / beta1) * torch.sigmoid(g)
    up = beta2 * torch.tanh(u / beta2)
    return torch.einsum('blgf,gfd->blgd', gate * up, mlp.W_down).reshape(B, L, D)


def test_default_path_unchanged():
    """situ_glu=False reproduces the exact SwiGLU computation."""
    torch.manual_seed(0)
    mlp = GroupedMLP(896, expand=4, groups=32, swiglu=True, situ_glu=False)
    x = torch.randn(2, 8, 896)
    out = mlp(x)
    ref = _manual_swiglu(mlp, x)
    assert torch.allclose(out, ref, atol=1e-6), 'default SwiGLU path deviates from reference'
    assert not mlp.situ_glu


def test_config_flag_default_false():
    cfg = WideBindConfig()
    assert cfg.mlp_situ_glu is False, 'mlp_situ_glu must default to False (no behavior change)'
    assert cfg.mlp_swiglu is True


def test_situ_glu_shape():
    torch.manual_seed(1)
    mlp = GroupedMLP(896, expand=4, groups=32, swiglu=False, situ_glu=True)
    x = torch.randn(2, 16, 896)
    out = mlp(x)
    assert out.shape == x.shape, f'Shape: {out.shape}'


def test_situ_glu_same_param_count():
    torch.manual_seed(0)
    a = GroupedMLP(896, expand=4, groups=32, swiglu=True, situ_glu=False)
    torch.manual_seed(0)
    b = GroupedMLP(896, expand=4, groups=32, swiglu=False, situ_glu=True)
    na = sum(p.numel() for p in a.parameters())
    nb = sum(p.numel() for p in b.parameters())
    assert na == nb, f'param count changed: swiglu={na} situ={nb}'
    ka = {k: tuple(v.shape) for k, v in a.state_dict().items()}
    kb = {k: tuple(v.shape) for k, v in b.state_dict().items()}
    assert ka == kb, f'shape mismatch: {set(ka) ^ set(kb)}'


def test_situ_glu_bounded_internal():
    """Both branches capped: |gate|≤β1, |up|≤β2 → |gate·up|≤β1·β2=100."""
    torch.manual_seed(2)
    beta1, beta2 = 4.0, 25.0
    mlp = GroupedMLP(896, expand=4, groups=32, swiglu=False, situ_glu=True,
                     beta1=beta1, beta2=beta2)
    x = torch.randn(4, 8, 896) * 30.0  # force saturation
    B, L, D = x.shape
    hb = F.rms_norm(x, (D,), mlp.norm_w).reshape(B, L, mlp.G, mlp.d)
    g = torch.einsum('blgd,gdf->blgf', hb, mlp.W_gate)
    u = torch.einsum('blgd,gdf->blgf', hb, mlp.W_up)
    gate = beta1 * torch.tanh(g / beta1) * torch.sigmoid(g)
    up = beta2 * torch.tanh(u / beta2)
    assert gate.abs().max().item() <= beta1 + 1e-5, f'gate exceeded β1: {gate.abs().max().item()}'
    assert up.abs().max().item() <= beta2 + 1e-5, f'up exceeded β2: {up.abs().max().item()}'
    mid = gate * up
    assert mid.abs().max().item() <= beta1 * beta2 + 1e-4, f'|gate·up| exceeded β1·β2: {mid.abs().max().item()}'


def test_situ_glu_zero_at_zero():
    """Both branches vanish at x=0 → output is exactly 0 (no softmax bias)."""
    torch.manual_seed(3)
    mlp = GroupedMLP(896, expand=4, groups=32, swiglu=False, situ_glu=True)
    x = torch.zeros(1, 4, 896)
    out = mlp(x)
    assert out.abs().max().item() == 0.0, 'SiTU-GLU must be zero on zero input'


def test_situ_glu_local_response_close_to_swiglu():
    """For near-origin pre-activations SiTU-GLU ≈ SwiGLU (tanh ≈ identity)."""
    torch.manual_seed(0)
    a = GroupedMLP(896, expand=4, groups=32, swiglu=True, situ_glu=False)
    torch.manual_seed(0)
    b = GroupedMLP(896, expand=4, groups=32, swiglu=False, situ_glu=True)
    # shrink gate/up weights so pre-activations are small (rms_norm normalizes input)
    with torch.no_grad():
        a.W_gate.mul_(0.01); a.W_up.mul_(0.01)
        b.W_gate.mul_(0.01); b.W_up.mul_(0.01)
    x = torch.randn(2, 8, 896)
    out_a = a(x)
    out_b = b(x)
    rel = (out_b - out_a).abs().max().item() / (out_a.abs().max().item() + 1e-8)
    assert rel < 0.01, f'SiTU-GLU diverged from SwiGLU near origin: rel={rel:.4f}'


def test_situ_glu_forward_backward_finite():
    torch.manual_seed(4)
    mlp = GroupedMLP(896, expand=4, groups=32, swiglu=False, situ_glu=True)
    x = torch.randn(2, 8, 896, requires_grad=True)
    out = mlp(x)
    loss = out.sum()
    loss.backward()
    assert not torch.isnan(out).any(), 'NaN in forward'
    for n, p in mlp.named_parameters():
        assert p.grad is not None, f'no grad: {n}'
        assert torch.isfinite(p.grad).all(), f'non-finite grad: {n}'
    assert torch.isfinite(x.grad).all(), 'non-finite input grad'


def test_situ_glu_takes_precedence():
    """situ_glu=True must override swiglu=True (weights exist, situ formula runs)."""
    torch.manual_seed(7)
    mlp = GroupedMLP(896, expand=4, groups=32, swiglu=True, situ_glu=True)
    x = torch.randn(2, 8, 896)
    out = mlp(x)
    ref = _manual_situ(mlp, x)
    assert torch.allclose(out, ref, atol=1e-6), 'SiTU-GLU path not executed when swiglu=True'
    # sanity: SiTU output must NOT equal the SwiGLU reference on large inputs
    big = torch.randn(2, 8, 896) * 10.0
    ref_swi = _manual_swiglu(mlp, big)
    ref_situ = _manual_situ(mlp, big)
    assert not torch.allclose(ref_swi, ref_situ, atol=1e-4), 'SiTU and SwiGLU identical on saturated input'


def test_situ_glu_full_stack_trains():
    """Small stack with mlp_situ_glu=True: loss finite, gradients flow, param count sane."""
    torch.manual_seed(5)
    cfg = WideBindConfig(n_layers=2, D=896, mlp_groups=8, mlp_situ_glu=True,
                         code_dim=16, code_sparsity=4, vocab=1000)
    model = WideBindStack(cfg)
    assert model.layers[0].mlp.situ_glu, 'flag not wired to block'
    opt = torch.optim.AdamW(model.param_groups(), lr=1e-3)
    losses = []
    for _ in range(5):
        x = torch.randint(0, 1000, (2, 8))
        h = model.embed_tokens(x)
        out, _, _ = model(h, None)
        loss = model.compute_loss(out, x)
        loss.backward()
        assert model.layers[0].mlp.W_gate.grad is not None, 'no gradient through SiTU-GLU gate'
        opt.step()
        opt.zero_grad()
        assert not torch.isnan(loss).item(), f'NaN loss: {loss.item()}'
        losses.append(loss.item())
    assert all(math.isfinite(l) for l in losses), f'non-finite losses: {losses}'


def test_stack_default_flag_off():
    cfg = WideBindConfig(n_layers=1, D=896, mlp_groups=8)
    model = WideBindStack(cfg)
    assert model.layers[0].mlp.situ_glu is False, 'default must keep situ_glu off'
    assert model.layers[0].mlp.swiglu is True


def test_no_softmax_used():
    """SiTU-GLU must not introduce softmax (only tanh + sigmoid)."""
    import inspect
    src = inspect.getsource(GroupedMLP.forward)
    assert 'softmax' not in src, 'softmax leaked into GroupedMLP.forward'
    assert 'sigmoid' in src and 'tanh' in src


if __name__ == '__main__':
    tests = [fn for fn in dir() if fn.startswith('test_')]
    passed = 0
    failed = 0
    for name in tests:
        try:
            globals()[name]()
            print(f'  PASS  {name}')
            passed += 1
        except Exception as e:
            print(f'  FAIL  {name}: {e}')
            failed += 1
            import traceback
            traceback.print_exc()
    print(f'\n{passed}/{passed + failed} passed')
    sys.exit(0 if failed == 0 else 1)
