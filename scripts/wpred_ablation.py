"""
A/B/C ablation: W_pred gradient analysis for all three approaches.

Tests on both random data and structured synthetic data.
Measures gradient norms through every path to W_pred.

Approach A: Remove k_lo/k_hi split — pred_error flows to all k dims
Approach B: Scalar alpha per expert — W_pred (G,k,k) -> alpha (G,)
Approach C: InfoNCE aux loss — replace MSE with contrastive

Usage:
  python scripts/wpred_ablation.py          # Runs all tests
  python scripts/wpred_ablation.py --quick   # Only gradient analysis (no training)
  python scripts/wpred_ablation.py --approach A  # Only approach A
"""

import sys, os, time, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from core.model import GroupedCognitiveMirror, WideBindStack, WideBindBlock
from core.config import WideBindConfig


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

B, L = 8, 32       # batch, seq len
D = 256             # model dim
G = 4               # groups
k = 4               # K-space dim per expert
n_layers = 2        # layers for stack tests


def make_cfg():
    return WideBindConfig(
        D=D, n_layers=n_layers,         bind_K=8,
        vocab=50000, code_dim=32, code_sparsity=6,
        mirror_k=k, w_pred_scale_init=3.0,
        mlp_groups=G, mlp_expand=4,
        conv_kernel=8, seq_len=L, batch_size=B,
        lr=3e-4, weight_decay=0.01,
        gate_lr_mult=5.0,
    )


def structured_data(B, L, vocab=1000):
    """Synthetic structured data: sine wave pattern in token space."""
    t = torch.arange(L).float() / L
    phase = torch.rand(B) * 2 * math.pi
    freq = 2.0 + torch.rand(B) * 4.0
    tokens = (phase.unsqueeze(1) + freq.unsqueeze(1) * t.unsqueeze(0)).sin()
    tokens = ((tokens + 1) * 0.5 * (vocab - 1)).long().clamp(0, vocab - 1)
    return tokens


# ═══════════════════════════════════════════════════════════════════════
# Approach implementations (patched forward methods)
# ═══════════════════════════════════════════════════════════════════════

class BaselineMirror(GroupedCognitiveMirror):
    """Original implementation (reference)."""
    pass


class ApproachAMirror(GroupedCognitiveMirror):
    """Approach A: Remove k_lo/k_hi split. pred_error flows to all k dims."""

    def forward(self, h, mem_all, global_state=None, diff=None):
        B, L, D = h.shape
        G, d, k = self.G, self.d, self.k

        h_g = h.reshape(B, L, G, d)
        mem_g = mem_all.reshape(B, L, G, d)
        mc_g = mem_g.mean(dim=1, keepdim=True)

        hp = torch.einsum('blgd,gdk->blgk', h_g, self.W_proj)
        if hp.requires_grad:
            hp.register_hook(lambda g: (
                self._prev_grad_norm.copy_(g.detach().norm(dim=-1).mean(dim=(0, 1))),
                None
            )[1])
        mc_k = torch.einsum('b l gd,gdk->b l gk', mc_g, self.W_proj)

        hp_prev = torch.cat([torch.zeros_like(hp[:, 0:1]), hp[:, :-1]], dim=1)

        # Slow signals
        temp_k = (hp - mc_k) * self.w_temp
        if global_state is not None:
            gs_k = torch.einsum('b l gd,gdk->b l gk',
                                global_state.reshape(1, 1, G, d), self.W_proj)
            temp_k = temp_k + (hp - gs_k) * self.w_global

        # Predictive
        pred_k = hp_prev * self.alpha.view(1, 1, G, 1)
        pred_error = (hp - pred_k) * self.w_pred_scale
        self._cached_pred_k = pred_k
        self._cached_hp = hp

        # Fast signals
        hp_perm = hp.permute(0, 2, 3, 1).reshape(B, G * k, L)
        hp_smooth = self.conv_smooth(hp_perm)[:, :, :L]
        hp_smooth = hp_smooth.reshape(B, G, k, L).permute(0, 3, 1, 2)
        smooth_k = hp - hp_smooth

        sym_k = (hp * self.w_sym_u) * (hp_prev * self.w_sym_v)

        # ─── APPROACH A: NO lo/hi split ───
        delta = temp_k + pred_error + smooth_k + sym_k  # all k dims

        delta = F.rms_norm(delta, (delta.shape[-1],))
        delta = delta + self.tanh_bias

        linear = torch.einsum('blgk,gkd->blgd', delta, self.W_out)
        skip_alpha = torch.exp(self.log_skip_alpha).view(1, 1, G, 1)
        mirror = torch.tanh(linear) + skip_alpha * linear
        mirror = mirror * torch.exp(self.log_scale)

        # Gate
        gate_signal = torch.abs(pred_error)
        gate_logits = torch.einsum('blgk,gk->blg', gate_signal, self.w_gate) + self.b_gate
        grad_mod = torch.exp(self.log_grad_mod_scale) * torch.tanh(self._prev_grad_norm + self.grad_mod_bias)
        gate_logits = gate_logits + grad_mod.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            dvar = delta.var(dim=(0, 1), unbiased=False).mean(dim=-1)
            if diff is not None:
                ema_alpha = self._delta_var_ema_min + diff * (self._delta_var_ema_max - self._delta_var_ema_min)
            else:
                ema_alpha = 0.9
            self._delta_var.mul_(ema_alpha).add_(dvar * (1.0 - ema_alpha))
        dvar_mod = torch.exp(self.log_dvar_mod_scale) * torch.tanh(self._delta_var + self.dvar_mod_bias)
        gate_logits = gate_logits + dvar_mod.unsqueeze(0).unsqueeze(0)

        expert_gate = torch.sigmoid(gate_logits)
        mirror = mirror * expert_gate.unsqueeze(-1)
        mirror = mirror.reshape(B, L, D)

        self._last_magnitude.fill_(mirror.abs().mean().item())
        self._last_gates.copy_(expert_gate.detach().mean(dim=(0, 1)))
        self._last_h_pool.copy_(h_g.detach().mean(dim=(0, 1)))

        return mirror


class ApproachBMirror(GroupedCognitiveMirror):
    """Approach B: Scalar alpha per expert.
    Now the native implementation in GroupedCognitiveMirror.
    """
    pass


class ApproachCStack(WideBindStack):
    """Approach C: InfoNCE aux loss instead of MSE."""
    
    def compute_loss(self, h, targets, pred_weight=0.1, info_temperature=0.1):
        logits = self.lm_head(h)
        ce_loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab),
                                   targets.reshape(-1), reduction='mean')
        
        pred_loss = 0.0
        n_pred = 0
        cache = getattr(self, '_pred_cache', [])
        for pred_k, hp in cache:
            # pred_k, hp: (B, L, G, k)
            B2, L2, G2, k2 = pred_k.shape
            
            # InfoNCE: treat each (position, expert) as a separate "sample"
            # pos: pred_k[b,l,g,:] should match hp[b,l,g,:]
            # neg: all other hp in the batch
            
            # Flatten to (B*L*G, k)
            pos = pred_k.reshape(-1, k2)         # (N, k)
            targets_hp = hp.reshape(-1, k2)       # (N, k)
            
            # Cosine similarities: (N, N)
            pos_norm = F.normalize(pos, dim=-1)
            target_norm = F.normalize(targets_hp, dim=-1)
            sim = pos_norm @ target_norm.T / info_temperature  # (N, N)
            
            # Labels: diagonal is positive
            labels = torch.arange(sim.shape[0], device=sim.device)
            
            loss_info = F.cross_entropy(sim, labels, reduction='mean')
            pred_loss = pred_loss + loss_info
            n_pred = n_pred + 1
        
        if n_pred > 0:
            pred_loss = pred_loss / n_pred
        
        return ce_loss + pred_weight * pred_loss


# ═══════════════════════════════════════════════════════════════════════
# Gradient analysis
# ═══════════════════════════════════════════════════════════════════════

def analyze_gradients(model, desc, data_random=True):
    """Run one forward-backward and measure gradient through W_pred paths."""
    cfg = model.cfg
    
    if data_random:
        x = torch.randint(0, cfg.vocab, (B, L), device=device)
        y = torch.randint(0, cfg.vocab, (B, L), device=device)
    else:
        x = structured_data(B, L, cfg.vocab).to(device)
        y = structured_data(B, L, cfg.vocab).to(device)
    
    model.zero_grad()
    model.train()
    state, gs = None, None
    h = model.embed_tokens(x)
    out, state, gs = model(h, state, global_state=gs)
    loss = model.compute_loss(out, y, pred_weight=1.0)
    loss.backward()
    
    results = {}
    results['loss'] = loss.item()
    
    for i, layer in enumerate(model.layers):
        mir = layer.mirror
        k_local = mir.k
        
        # Measure |I-diff|
        if hasattr(mir, 'W_pred'):
            eye = torch.eye(k_local, device=mir.W_pred.device).unsqueeze(0)
            idiff = (mir.W_pred.data - eye).abs().mean().item()
            results[f'L{i}_idiff'] = idiff
        elif hasattr(mir, 'alpha'):
            alpha_mean = mir.alpha.data.mean().item()
            results[f'L{i}_idiff'] = abs(alpha_mean - 1.0)
        else:
            results[f'L{i}_idiff'] = -1.0
        
        # Check gradient norms
        if hasattr(mir, 'W_pred') and mir.W_pred.grad is not None:
            results[f'L{i}_W_pred_grad'] = mir.W_pred.grad.norm().item()
        if hasattr(mir, 'alpha') and mir.alpha.grad is not None:
            results[f'L{i}_alpha_grad'] = mir.alpha.grad.norm().item()
        
        # Gradient through pred_error in delta
        hp = mir._cached_hp
        pred_k = mir._cached_pred_k
        if hp is not None and pred_k is not None:
            with torch.no_grad():
                pred_error = (hp - pred_k) * mir.w_pred_scale
                results[f'L{i}_pred_error_mean'] = pred_error.abs().mean().item()
                results[f'L{i}_pred_error_std'] = pred_error.std().item()
    
    model.zero_grad()
    return results


def run_gradient_analysis():
    """Compare gradient flow for all approaches."""
    print('\n' + '=' * 70)
    print('GRADIENT ANALYSIS')
    print('=' * 70)
    
    cfg = make_cfg()
    
    for name, mirror_cls in [
        ('BASELINE', GroupedCognitiveMirror),
        ('APPROACH_A', ApproachAMirror),
        ('APPROACH_B', ApproachBMirror),
    ]:
        print(f'\n--- {name} ---')
        
        model = WideBindStack(cfg).to(device)
        for layer in model.layers:
            old = layer.mirror
            new = mirror_cls(D, G=G, k=k, w_pred_scale_init=cfg.w_pred_scale_init).to(device)
            new.load_state_dict(old.state_dict(), strict=False)
            layer.mirror = new
            del old
        
        res = analyze_gradients(model, name)
        print(f'  Loss: {res["loss"]:.4f}')
        for i in range(n_layers):
            print(f'  L{i}: |I-diff|={res.get(f"L{i}_idiff", 0):.6f}', end='')
            if f'L{i}_W_pred_grad' in res:
                print(f'  W_pred_grad={res[f"L{i}_W_pred_grad"]:.6f}', end='')
            if f'L{i}_alpha_grad' in res:
                print(f'  alpha_grad={res[f"L{i}_alpha_grad"]:.6f}', end='')
            if f'L{i}_pred_error_mean' in res:
                print(f'  |pred_err|={res[f"L{i}_pred_error_mean"]:.4f}', end='')
            print()
        
        del model


# ═══════════════════════════════════════════════════════════════════════
# Training comparison
# ═══════════════════════════════════════════════════════════════════════

def train_model(model_cfg, mirror_cls, loss_cls, n_steps=500, use_structured=True):
    """Train and track |I-diff| convergence."""
    if loss_cls == 'ce+info':
        model = ApproachCStack(model_cfg).to(device)
    else:
        model = WideBindStack(model_cfg).to(device)
    
    # Replace mirrors
    for layer in model.layers:
        old = layer.mirror
        new = mirror_cls(D, G=G, k=k, w_pred_scale_init=model_cfg.w_pred_scale_init).to(device)
        new.load_state_dict(old.state_dict(), strict=False)
        layer.mirror = new
        del old
    
    optimizer = torch.optim.AdamW(model.param_groups(model_cfg.lr), betas=(0.9, 0.95))
    
    def get_lr(step):
        if step < model_cfg.warmup_steps:
            return model_cfg.lr * (step + 1) / model_cfg.warmup_steps
        progress = (step - model_cfg.warmup_steps) / (n_steps - model_cfg.warmup_steps)
        return model_cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    
    state, gs = None, None
    metrics = {'step': [], 'loss': [], 'idiff': [], 'alpha': [], 'gate_var': []}
    
    for step in range(n_steps):
        if use_structured:
            x = structured_data(B, L, model_cfg.vocab).to(device)
            y = structured_data(B, L, model_cfg.vocab).to(device)
        else:
            x = torch.randint(0, model_cfg.vocab, (B, L), device=device)
            y = torch.randint(0, model_cfg.vocab, (B, L), device=device)
        
        model.train()
        optimizer.zero_grad()
        h = model.embed_tokens(x)
        out, state, gs = model(h, state=state, global_state=gs)
        loss = model.compute_loss(out, y, pred_weight=1.0)
        
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), model_cfg.grad_clip)
        optimizer.step()
        
        # Detach states
        state = [(s[0].detach(), s[1].detach(), s[2].detach()) if s is not None else None for s in state]
        gs = gs.detach() if gs is not None else None
        
        # LR update
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        
        if (step + 1) % 50 == 0 or step == 0:
            with torch.no_grad():
                idiff_val = 0.0
                alpha_val = -1.0
                gv_val = 0.0
                for i, layer in enumerate(model.layers):
                    mir = layer.mirror
                    if hasattr(mir, 'W_pred'):
                        eye = torch.eye(k, device=mir.W_pred.device).unsqueeze(0)
                        idiff_val += (mir.W_pred.data - eye).abs().mean().item()
                    if hasattr(mir, 'alpha'):
                        alpha_mean = mir.alpha.data.mean().item()
                        alpha_val = alpha_mean if alpha_val < 0 else alpha_val + alpha_mean
                        idiff_val += abs(alpha_mean - 1.0)
                    gv_val += mir._last_gates.var().item()
                idiff_val /= n_layers
                alpha_val = alpha_val / n_layers if alpha_val > 0 else -1.0
                gv_val /= n_layers
            
            metrics['step'].append(step)
            metrics['loss'].append(loss.item())
            metrics['idiff'].append(idiff_val)
            metrics['alpha'].append(alpha_val)
            metrics['gate_var'].append(gv_val)
            
            alpha_str = f'  alpha={alpha_val:.4f}' if alpha_val > 0 else ''
            print(f'  step {step:4d}  loss={loss.item():.4f}  |I-diff|={idiff_val:.6f}{alpha_str}  gate_var={gv_val:.6f}')
    
    # Verify W_pred learned
    final_idiff = metrics['idiff'][-1]
    initial_idiff = metrics['idiff'][0]
    delta = final_idiff - initial_idiff
    
    del model
    return {'final_idiff': final_idiff, 'initial_idiff': initial_idiff, 'delta': delta, 'metrics': metrics}


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Gradient analysis only')
    parser.add_argument('--approach', type=str, default='',
                        help='Which approach to test (A, B, C, or all)')
    args = parser.parse_args()
    
    approaches_to_test = ['A', 'B', 'C']
    if args.approach:
        approaches_to_test = [args.approach.upper()]
    
    cfg = make_cfg()
    
    # ─── Gradient Analysis ───
    run_gradient_analysis()
    
    if args.quick:
        sys.exit(0)
    
    # ─── Training Comparison ───
    print('\n' + '=' * 70)
    print('TRAINING COMPARISON (500 steps)')
    print('=' * 70)
    
    all_results = {}
    
    if 'A' in approaches_to_test:
        print('\n--- BASELINE (original) ---')
        all_results['baseline'] = train_model(cfg, GroupedCognitiveMirror, 'ce+mse', n_steps=500, use_structured=True)
        
        print('\n--- APPROACH A (no lo/hi split) ---')
        all_results['A'] = train_model(cfg, ApproachAMirror, 'ce+mse', n_steps=500, use_structured=True)
    
    if 'B' in approaches_to_test:
        print('\n--- APPROACH B (scalar alpha) ---')
        all_results['B'] = train_model(cfg, ApproachBMirror, 'ce+mse', n_steps=500, use_structured=True)
    
    if 'C' in approaches_to_test:
        print('\n--- APPROACH C (InfoNCE) ---')
        all_results['C'] = train_model(cfg, GroupedCognitiveMirror, 'ce+info', n_steps=500, use_structured=True)
    
    # ─── Summary ───
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    for name, res in all_results.items():
        print(f'  {name}: |I-diff| {res["initial_idiff"]:.6f} -> {res["final_idiff"]:.6f} (delta={res["delta"]:.6f})')
    
    print('\nDone.')
