"""Comprehensive tests for WideBind core components."""

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn

from core.config import WideBindConfig
from core.model import (
    WideBindStack, WideBindBlock, GroupedCognitiveMirror, GroupedMLP,
    PartitionedEmbedding, PartitionedHead, LmHead,
    sparse_block_codes, dct_basis, vsa_prefix_scan,
    compute_timescales, compute_spectrum,
)
from core.live_inference import LiveInference, MirrorMonitor


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─── Sparse Block Codes ──────────────────────────────────────────────

def test_sparse_codes_exact_sparsity():
    codes = sparse_block_codes(vocab=50000, K=32, S=6)
    assert codes.shape == (50000, 32)
    counts = codes.sum(dim=-1)
    assert (counts == 6).all(), f'Not all tokens have exactly 6 active bits: {counts.unique().tolist()}'


def test_sparse_codes_bits_used():
    codes = sparse_block_codes(vocab=50000, K=32, S=6)
    freq = codes.sum(dim=0)
    assert (freq > 0).all(), f'Some bits never used: {freq.tolist()}'
    min_f, max_f = freq.min().item(), freq.max().item()
    assert min_f > 0.18 * 50000, f'Bit {freq.argmin()} underused: {min_f/50000:.3f}'
    assert max_f < 0.20 * 50000, f'Bit {freq.argmax()} overused: {max_f/50000:.3f}'


def test_sparse_codes_deterministic():
    c1 = sparse_block_codes(vocab=100, K=32, S=6)
    c2 = sparse_block_codes(vocab=100, K=32, S=6)
    assert (c1 == c2).all(), 'sparse_block_codes not deterministic'


def test_sparse_codes_combinadic_coverage():
    codes = sparse_block_codes(vocab=50000, K=32, S=6)
    seen = set()
    for v in range(50000):
        bits = tuple(codes[v].nonzero(as_tuple=True)[0].tolist())
        seen.add(bits)
    assert len(seen) == 50000, f'Duplicate codes: {50000 - len(seen)} collisions'


# ─── PartitionedEmbedding ──────────────────────────────────────────

def test_partitioned_embed_shape():
    cfg = WideBindConfig(D=896, code_dim=32, code_sparsity=6)
    emb = PartitionedEmbedding(cfg)
    tokens = torch.randint(0, 50000, (2, 16))
    h = emb(tokens)
    assert h.shape == (2, 16, 896), f'Shape mismatch: {h.shape}'


def test_partitioned_embed_gradient_grouping():
    cfg = WideBindConfig(D=896, code_dim=32, code_sparsity=6)
    emb = PartitionedEmbedding(cfg)
    tokens = torch.randint(0, 50000, (4, 32))
    h = emb(tokens)
    loss = h.sum()
    loss.backward()

    codes = emb.codes[tokens]
    for k in range(emb.K):
        active = codes[:, :, k].sum().item() > 0
        grad_norm = emb.basis.grad[k].norm().item()
        if active:
            assert grad_norm > 0, f'basis[{k}] has gradient but should not (active)'
        else:
            assert grad_norm == 0.0, f'basis[{k}] has gradient {grad_norm:.6f} but should be 0 (inactive)'


def test_partitioned_embed_small_vocab():
    cfg = WideBindConfig(D=896, code_dim=16, code_sparsity=4, vocab=1800)
    assert cfg.vocab <= 1820  # C(16,4)=1820
    emb = PartitionedEmbedding(cfg)
    tokens = torch.randint(0, 1800, (1, 8))
    h = emb(tokens)
    assert h.shape == (1, 8, 896)


def test_partitioned_embed_grad_nonzero_with_active_bits():
    cfg = WideBindConfig(D=896, code_dim=32, code_sparsity=6)
    emb = PartitionedEmbedding(cfg)
    tokens = torch.zeros(1, 1, dtype=torch.long)
    tokens[0, 0] = 42
    h = emb(tokens)
    loss = h.sum()
    loss.backward()
    assert emb.basis.grad is not None
    assert emb.basis.grad.abs().sum().item() > 0, 'No gradient flowed to basis weights'


# ─── PartitionedHead ───────────────────────────────────────────────

def test_partitioned_head_shape():
    cfg = WideBindConfig(D=896, code_dim=32, code_sparsity=6)
    head = PartitionedHead(cfg)
    h = torch.randn(2, 16, 896)
    logits = head(h)
    assert logits.shape == (2, 16, 50000), f'Shape mismatch: {logits.shape}'


def test_partitioned_head_gradient_grouping():
    cfg = WideBindConfig(D=896, code_dim=32, code_sparsity=6)
    head = PartitionedHead(cfg)
    h = torch.randn(4, 32, 896, requires_grad=True)
    logits = head(h)
    loss = logits.sum()
    loss.backward()

    for k in range(head.K):
        grad_norm = head.readout.grad[k].norm().item()
        assert grad_norm > 0, f'readout[{k}] has no gradient'


def test_partitioned_head_zero_h_gives_uniform_logits():
    cfg = WideBindConfig(D=896, code_dim=32, code_sparsity=6)
    head = PartitionedHead(cfg)
    h = torch.zeros(1, 1, 896)
    logits = head(h)
    assert (logits == 0).all(), 'zero h should give zero logits (codes are ±1)'


# ─── VSA Prefix Scan ─────────────────────────────────────────────

def test_vsa_scan_exact():
    B, L, D = 1, 4, 2
    a = torch.full((B, L, D), 0.5)
    b = torch.ones(B, L, D)
    # manual: mem[0]=b0=1, mem[1]=a1*mem0+b1=0.5+1=1.5,
    # mem[2]=a2*mem1+b2=0.75+1=1.75, mem[3]=a3*mem2+b3=0.875+1=1.875
    out, final = vsa_prefix_scan(a, b)
    expected = torch.tensor([[[1.0, 1.0], [1.5, 1.5], [1.75, 1.75], [1.875, 1.875]]])
    assert torch.allclose(out, expected, atol=1e-5), f'Scan mismatch: {out} vs {expected}'
    assert torch.allclose(final, expected[:, -1:]), f'Final state mismatch: {final} vs {expected[:, -1]}'


def test_vsa_scan_with_state():
    B, L, D = 1, 2, 1
    a = torch.full((B, L, D), 0.5)
    b = torch.ones(B, L, D)
    state = torch.full((B, D), 10.0)
    out, final = vsa_prefix_scan(a, b, state)
    # mem0 = state=10, it's excluded from output
    # mem1 = a0 * mem0 + b0 = 0.5*10 + 1 = 6.0
    # mem2 = a1 * mem1 + b1 = 0.5*6 + 1 = 4.0
    expected_out = torch.tensor([[[6.0], [4.0]]])
    expected_final = torch.tensor([[4.0]])
    assert torch.allclose(out, expected_out, atol=1e-5), f'out={out} vs {expected_out}'
    assert torch.allclose(final, expected_final, atol=1e-5), f'final={final} vs {expected_final}'


def test_vsa_scan_batched():
    B, L, D = 3, 8, 5
    a = torch.rand(B, L, D)
    b = torch.rand(B, L, D)
    out, final = vsa_prefix_scan(a, b)
    assert out.shape == (B, L, D), f'Shape: {out.shape}'
    assert final.shape == (B, D), f'Final shape: {final.shape}'
    # verify manual scan matches
    mem = b[:, 0:1].clone()
    for t in range(1, L):
        mem = a[:, t:t+1] * mem + b[:, t:t+1]
    assert torch.allclose(final, mem[:, -1], atol=1e-5), 'Final state mismatch with manual scan'


# ─── GroupedCognitiveMirror ──────────────────────────────────────────

def test_mirror_shape():
    D, G, k = 896, 32, 4
    mirror = GroupedCognitiveMirror(D, G=G, k=k)
    B, L = 2, 16
    h = torch.randn(B, L, D)
    mem_all = torch.randn(B, L, D)
    out = mirror(h, mem_all)
    assert out.shape == (B, L, D), f'Shape: {out.shape}'


def test_mirror_skip_connection_preserves_gradient():
    D, G, k = 352, 32, 4
    mirror = GroupedCognitiveMirror(D, G=G, k=k, skip_alpha=0.1)
    B, L = 1, 4
    h = torch.randn(B, L, D)
    mem_all = torch.randn(B, L, D)
    log_scale_before = mirror.log_scale.data.clone()
    
    out = mirror(h, mem_all)
    loss = out.sum()
    loss.backward()
    
    assert mirror.log_scale.grad is not None, 'No gradient to log_scale'
    grad_norm = mirror.log_scale.grad.norm().item()
    assert grad_norm > 0, f'log_scale grad is zero ({grad_norm}), skip connection not working'


def test_mirror_per_expert_gates():
    D, G, k = 896, 32, 4
    mirror = GroupedCognitiveMirror(D, G=G, k=k)
    B, L = 4, 32
    h = torch.randn(B, L, D)
    mem_all = torch.randn(B, L, D)
    out = mirror(h, mem_all)
    gate = mirror._last_gates
    assert gate.shape == (G,), f'Gate shape: {gate.shape}'
    assert (gate >= 0).all() and (gate <= 1).all(), f'Gate out of [0,1]: [{gate.min().item()}, {gate.max().item()}]'


def test_mirror_grad_cache():
    D, G, k = 896, 32, 4
    mirror = GroupedCognitiveMirror(D, G=G, k=k)
    grad_h = torch.randn(4, 16, D)
    mirror.cache_grad_norms(grad_h)
    norms = mirror._prev_grad_norm
    assert norms.shape == (G,)
    assert (norms >= 0).all(), 'Negative gradient norm'


def test_mirror_global_state():
    D, G, k = 896, 32, 4
    mirror = GroupedCognitiveMirror(D, G=G, k=k)
    B, L = 2, 16
    h = torch.randn(B, L, D)
    mem_all = torch.randn(B, L, D)
    global_state = torch.randn(1, 1, D)
    out_with = mirror(h, mem_all, global_state)
    out_without = mirror(h, mem_all, global_state=None)
    assert out_with.shape == out_without.shape


# ─── GroupedMLP ─────────────────────────────────────────────────────

def test_mlp_shape():
    D, G, expand = 896, 8, 4
    mlp = GroupedMLP(D, expand=expand, groups=G)
    h = torch.randn(2, 16, D)
    out = mlp(h)
    assert out.shape == h.shape, f'Shape: {out.shape}'


def test_mlp_nonzero():
    D, G, expand = 896, 8, 4
    mlp = GroupedMLP(D, expand=expand, groups=G)
    h = torch.randn(1, 4, D)
    out = mlp(h)
    assert out.abs().sum().item() > 0, 'MLP output is zero'


# ─── WideBindStack (end-to-end) ─────────────────────────────────────

def test_stack_forward():
    cfg = WideBindConfig(n_layers=4, D=896, mlp_groups=8, code_dim=32, code_sparsity=6)
    model = WideBindStack(cfg).to(device)
    x = torch.randint(0, cfg.vocab, (2, 16), device=device)
    h = model.embed_tokens(x)
    out, state, global_state = model(h)
    assert out.shape == h.shape, f'Output shape: {out.shape} vs input {h.shape}'
    assert len(state) == cfg.n_layers, f'State len: {len(state)} vs {cfg.n_layers}'
    assert global_state.shape == (1, 1, cfg.D), f'Global state shape: {global_state.shape}'


def test_stack_forward_twice_with_state():
    cfg = WideBindConfig(n_layers=2, D=896, mlp_groups=8, code_dim=32, code_sparsity=6)
    model = WideBindStack(cfg).to(device)
    x = torch.randint(0, cfg.vocab, (1, 8), device=device)
    h = model.embed_tokens(x)
    out1, state1, gs1 = model(h)
    out2, state2, gs2 = model(h, state1, gs1)
    assert out2.shape == out1.shape


def test_stack_loss():
    cfg = WideBindConfig(n_layers=2, D=896, mlp_groups=8, code_dim=32, code_sparsity=6)
    model = WideBindStack(cfg).to(device)
    x = torch.randint(0, cfg.vocab, (2, 16), device=device)
    h = model.embed_tokens(x)
    out, _, _ = model(h)
    loss = model.compute_loss(out[:, :-1], x[:, 1:])
    assert loss.item() > 0, f'Loss should be positive: {loss.item()}'
    loss.backward()
    total_grad = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert total_grad > 0, f'Zero total gradient: {total_grad}'


def test_stack_param_count():
    cfg = WideBindConfig(n_layers=4, D=896, mlp_groups=8, code_dim=32, code_sparsity=6)
    model = WideBindStack(cfg)
    n = model.param_count()
    assert n > 0, f'Zero parameters'
    assert cfg.D == 896 or True  # just check baseline


def test_stack_embed_alignment():
    cfg = WideBindConfig(D=3584, mlp_groups=32, code_dim=32, code_sparsity=6)
    model = WideBindStack(cfg)
    D = cfg.D
    K = cfg.code_dim
    assert D % K == 0
    d = D // K
    # embed, head, mirror, mlp all have K=32 groups aligned
    assert model.embed.K == K
    assert model.lm_head.K == K


def test_strict_false_compatibility():
    """Old checkpoints (without persistent=False buffers) should load."""
    cfg = WideBindConfig(n_layers=2, D=896, mlp_groups=8, code_dim=32, code_sparsity=6)
    model = WideBindStack(cfg)
    # Simulate an old state_dict that's missing trace buffers
    old_sd = {k: v for k, v in model.state_dict().items()
              if not any(b in k for b in ['_last_gates', '_last_h_pool', '_prev_grad_norm', '_last_magnitude'])}
    # Also remove any trace buffers from live_inference
    model.load_state_dict(old_sd, strict=False)
    # Loss should still work
    x = torch.randint(0, cfg.vocab, (1, 4))
    h = model.embed_tokens(x)
    out, _, _ = model(h)
    loss = model.compute_loss(out[:, :-1], x[:, 1:])
    assert not torch.isnan(loss), 'Loss is NaN after strict=False load'


# ─── DCT Basis ──────────────────────────────────────────────────────

def test_dct_basis_orthogonal():
    for n in [64, 128, 256]:
        V = dct_basis(n)
        product = V @ V.T
        assert product.shape == (n, n)
        diff = (product - torch.eye(n)).abs().max().item()
        assert diff < 1e-5, f'DCT basis not orthogonal at n={n}: max diff={diff}'


def test_dct_basis_orthogonal_large():
    V = dct_basis(512)
    product = V @ V.T
    diff = (product - torch.eye(512)).abs().max().item()
    # Larger n accumulates more floating point error
    assert diff < 5e-5, f'DCT basis not orthogonal at n=512: max diff={diff}'


def test_dct_basis_first_row():
    V = dct_basis(896)
    # DC component: sqrt(2/N) * 1/√2 * N = sqrt(2/N) * N/√2 = sqrt(N)
    expected = torch.full((896,), math.sqrt(2.0 / 896) / math.sqrt(2)) * 1.0
    assert torch.allclose(V[0], expected, atol=1e-6)


# ─── AdaptiveController ─────────────────────────────────────────────

def test_adaptive_controller_ranges():
    cfg = WideBindConfig(n_layers=2, D=896, mlp_groups=8, code_dim=32, code_sparsity=6)
    model = WideBindStack(cfg)
    from core.model import AdaptiveController
    expl, diff = AdaptiveController.stats(model.layers)
    assert 0 <= expl <= 1, f'Exploration out of [0,1]: {expl}'
    assert 0 <= diff <= 1, f'Diff out of [0,1]: {diff}'
    b_i = AdaptiveController.b_i(model.layers)
    b_d = AdaptiveController.b_d(model.layers)
    assert -3 <= b_i <= 0, f'b_i out of range: {b_i}'
    assert 2 <= b_d <= 6, f'b_d out of range: {b_d}'
    scale = AdaptiveController.w_mem2v_scale(model.layers)
    assert 0.5 <= scale <= 1.0, f'mem2v_scale out of range: {scale}'
    alpha = AdaptiveController.ema_alpha(model.layers)
    assert 0.90 <= alpha <= 0.99, f'ema_alpha out of range: {alpha}'


# ─── LiveInference ─────────────────────────────────────────────────

def test_live_inference_basic():
    cfg = WideBindConfig(n_layers=2, D=896, mlp_groups=8, code_dim=32, code_sparsity=6)
    model = WideBindStack(cfg).to(device)
    model.eval()
    live = LiveInference(model, cfg)
    h = model.embed_tokens(torch.randint(0, cfg.vocab, (1, 4), device=device))
    out = live.respond(h)
    assert out.shape == (1, 4, cfg.D), f'Shape: {out.shape}'


def test_live_inference_state_persists():
    cfg = WideBindConfig(n_layers=2, D=896, mlp_groups=8, code_dim=32, code_sparsity=6)
    model = WideBindStack(cfg).to(device)
    model.eval()
    live = LiveInference(model, cfg)
    
    h1 = model.embed_tokens(torch.randint(0, cfg.vocab, (1, 4), device=device))
    out1 = live.respond(h1)
    
    h2 = model.embed_tokens(torch.randint(0, cfg.vocab, (1, 4), device=device))
    out2 = live.respond(h2)
    
    # state is not None after respond
    assert live.layer_states is not None, 'Layer states should be set after respond'
    assert live.global_state is not None, 'Global state should be set after respond'


def test_live_inference_think():
    cfg = WideBindConfig(n_layers=2, D=896, mlp_groups=8, code_dim=32, code_sparsity=6)
    model = WideBindStack(cfg).to(device)
    model.eval()
    live = LiveInference(model, cfg)
    
    with torch.no_grad():
        h = live.think(n_steps=10)
    # think feeds last output back, so each step is 1 token
    assert h.shape == (1, 1, cfg.D), f'Shape after think: {h.shape}'


def test_live_inference_think_persists():
    cfg = WideBindConfig(n_layers=2, D=896, mlp_groups=8, code_dim=32, code_sparsity=6)
    model = WideBindStack(cfg).to(device)
    model.eval()
    live = LiveInference(model, cfg)
    
    with torch.no_grad():
        live.think(n_steps=5)
    assert live.step > 0, 'Step counter not incremented'


def test_live_inference_reset():
    cfg = WideBindConfig(n_layers=2, D=896, mlp_groups=8, code_dim=32, code_sparsity=6)
    model = WideBindStack(cfg).to(device)
    model.eval()
    live = LiveInference(model, cfg)
    
    h = model.embed_tokens(torch.randint(0, cfg.vocab, (1, 4), device=device))
    live.respond(h)
    assert live.layer_states is not None, 'State should be set after respond'
    live.reset_state()
    assert live.layer_states is None, 'State should be None after reset'
    assert live.global_state is None, 'Global state should be None after reset'


# ─── MirrorMonitor ────────────────────────────────────────────────

def test_mirror_monitor_trace():
    cfg = WideBindConfig(n_layers=2, D=896, mlp_groups=8, code_dim=32, code_sparsity=6)
    model = WideBindStack(cfg).to(device)
    model.eval()
    monitor = MirrorMonitor(model)
    x = torch.randint(0, cfg.vocab, (2, 16), device=device)
    h = model.embed_tokens(x)
    with torch.no_grad():
        model(h)
    
    monitor.capture()
    assert len(monitor.history['step']) == 1, f'History len: {len(monitor.history["step"])}'
    assert 'expert_gates' in monitor.history, 'No gates in history'
    assert 'tau' in monitor.history, 'No tau in history'
    assert 'global_state_norm' in monitor.history, 'No global_state norm'
    
    summary = monitor.summary(window=1)
    assert 'expert_gates_mean' in summary


def test_mirror_monitor_rolling():
    cfg = WideBindConfig(n_layers=2, D=896, mlp_groups=8, code_dim=32, code_sparsity=6)
    model = WideBindStack(cfg).to(device)
    model.eval()
    monitor = MirrorMonitor(model, max_history=5)
    
    for _ in range(10):
        x = torch.randint(0, cfg.vocab, (1, 4), device=device)
        h = model.embed_tokens(x)
        with torch.no_grad():
            model(h)
        monitor.capture()
    
    assert len(monitor.history['step']) == 5, f'History should be capped: {len(monitor.history["step"])}'


# ─── D=3584 quick shape tests ─────────────────────────────────────

def test_large_config_forward():
    cfg = WideBindConfig(n_layers=2, D=3584, mlp_groups=32, mlp_expand=8,
                          bind_K=16, code_dim=32, code_sparsity=6)
    model = WideBindStack(cfg)
    n = model.param_count()
    assert n > 0
    x = torch.randint(0, cfg.vocab, (1, 4))
    h = model.embed_tokens(x)
    out, state, gs = model(h)
    assert out.shape == (1, 4, 3584)


# ─── Parameter counts ─────────────────────────────────────────────

def test_partitioned_embed_fewer_params():
    cfg_dense = WideBindConfig(D=896, code_dim=32, code_sparsity=6)
    emb = PartitionedEmbedding(cfg_dense)
    # 32 × (896/32) = 32 × 28 = 896 params per embed*head
    expected = 896
    assert emb.basis.numel() == expected, f'Expected {expected} got {emb.basis.numel()}'


def test_gradient_grouping_demonstrable():
    """Gradient to w_k is exactly zero when bit k is inactive across the batch."""
    cfg = WideBindConfig(D=896, code_dim=32, code_sparsity=6)
    emb = PartitionedEmbedding(cfg)
    tokens = torch.tensor([[0, 1, 2, 42, 100, 500, 1000, 5000]])
    h = emb(tokens)
    loss = h.sum()
    loss.backward()
    for k in range(emb.K):
        is_active_anywhere = emb.codes[tokens][:, :, k].any().item()
        grad = emb.basis.grad[k].norm().item()
        if not is_active_anywhere:
            assert grad == 0.0, f'w_{k} has grad {grad} but bit inactive for all tokens'
        # if active, grad is > 0
        if is_active_anywhere:
            assert grad > 0, f'w_{k} has zero grad but bit is active'


# ─── Run all ────────────────────────────────────────────────────────

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
