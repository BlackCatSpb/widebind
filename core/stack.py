"""WideBind: stack module."""

import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import WideBindConfig
from .block import WideBindBlock
from .embedding import ZeckendorfEmbedding, PartitionedEmbedding, LmHead, PartitionedHead
from .vsa_utils import dct_basis, zeckendorf_codes, sparse_block_codes, vsa_prefix_scan
from .zeckendorf_readout import ZeckendorfReadout

class WideBindStack(nn.Module):
    """Stack of WideBindBlock layers with embedding and lm_head."""
    
    def __init__(self, cfg: WideBindConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = PartitionedEmbedding(cfg)
        if getattr(cfg, 'zeckendorf_readout', False):
            self.lm_head = ZeckendorfReadout(cfg)
        else:
            self.lm_head = PartitionedHead(cfg, embed_basis=self.embed.basis)
        
        self.layers = nn.ModuleList([
            WideBindBlock(cfg, i) for i in range(cfg.n_layers)
        ])
        
        self.register_buffer('final_norm_w', torch.ones(cfg.D))
        # ─── Idea 1: Learnable VSA timescales ───
        self._vsa_log_param = nn.Parameter(torch.tensor([1.7918, 1.2321, 1.1304, 1.1065]))
        # ─── Idea 4: Per-layer τ_l deviation ───
        self._tau_l_dev = nn.Parameter(torch.zeros(cfg.n_layers))
        # c_ema: global state EMA rate = write_rate * tau_mid
        tau_s = self.layers[0]._tau_s
        tau_mid = math.sqrt(tau_s[0].item() * tau_s[-1].item())
        write_rate = 1.0 / math.sqrt(cfg.D)
        self._c_ema_value = write_rate * tau_mid
        self._tau_min_value = tau_s[0].item()
        self._tau_max_value = tau_s[-1].item()
        self._tau_mid_value = tau_mid
        # EMA for exploration (smoothed over ~500 steps)
        self.register_buffer('_expl_ema', torch.zeros(1), persistent=False)
    
    def forward(self, h, state=None, global_state=None, pred_weight=None, adaptive=True,
                context_mem=None, allow_write=None, step=None):
        """h: (B, L, D) — pre-embedded tokens
           state: per-layer memory states from previous forward (or None)
           global_state: cross-layer EMA self-model (or None, created fresh)
           pred_weight: adaptive alpha auxiliary loss weight (or None to compute)
           adaptive: if True, run AdaptiveController (training); if False, skip for speed (inference)
        """
        if state is None:
            state = [None] * len(self.layers)
        B, L, D = h.shape
        
        # ─── Learnable VSA scales (Idea 1) — moved before AdaptiveController loop ───
        vsa_tau = torch.exp(torch.cumsum(F.softplus(self._vsa_log_param), dim=0)) + 1.0
        tau_min = vsa_tau[0]
        tau_max = vsa_tau[-1]
        tau_mid = (tau_min * tau_max).sqrt()
        c_ema = (1.0 / math.sqrt(self.cfg.D)) * tau_mid
        n_layers = len(self.layers)
        
        # ─── Adaptive gate biases from mirror stats (per-layer) ───
        if adaptive:
            with torch.no_grad():
                expl_raw, diff = AdaptiveController.stats(self.layers,
                    expl_thresh=self.cfg.exploration_threshold,
                    diff_thresh=self.cfg.differentiation_threshold)
                self._expl_ema.mul_(0.998).add_(expl_raw * (1.0 - 0.998))
                global_expl = self._expl_ema.clamp(0.0, 1.0).item()
                
                self._pred_weight = (pred_weight if pred_weight is not None
                    else AdaptiveController.pred_weight(self.layers,
                        min_val=0.05, max_val=0.3))
                
                for i, layer in enumerate(self.layers):
                    l_expl, l_diff = AdaptiveController.layer_stats(layer,
                        expl_thresh=self.cfg.exploration_threshold,
                        diff_thresh=self.cfg.differentiation_threshold)
                    lf = i / max(len(self.layers) - 1, 1)
                    dev = torch.tanh(self._tau_l_dev[i])
                    tau_l_val = (tau_min * (tau_max / tau_min) ** (lf * (1.0 + 0.1 * dev))).item()
                    b_i_val = AdaptiveController.layer_b_i(layer, expl=l_expl, tau_l=tau_l_val)
                    b_d_max = getattr(self.cfg, 'vsa_b_d_max', 12.0)
                    b_d_val = AdaptiveController.layer_b_d(layer, expl=l_expl,
                        b_d_max=b_d_max)
                    smooth = getattr(self.cfg, 'vsa_b_d_smooth', 0.99)
                    if smooth >= 1.0:
                        layer.b_i.fill_(b_i_val)
                        layer.b_d.fill_(b_d_val)
                    else:
                        b_d_t = torch.tensor(b_d_val, device=layer.b_d.device, dtype=layer.b_d.dtype)
                        b_i_t = torch.tensor(b_i_val, device=layer.b_i.device, dtype=layer.b_i.dtype)
                        layer.b_d.data.lerp_(b_d_t, 1.0 - smooth)
                        layer.b_i.data.lerp_(b_i_t, 1.0 - smooth)
        
        # Global self-model: running EMA of layer memory centroids
        # Per-layer EMA rates proportional to 1/τ (Proposal V)
        if global_state is None:
            global_state = torch.zeros(n_layers, 1, D, device=h.device, dtype=h.dtype)
        if global_state.dim() == 2:
            global_state = global_state.unsqueeze(0).expand(n_layers, -1, -1).clone()
        elif global_state.shape[0] != n_layers:
            global_state = global_state[0:1].expand(n_layers, -1, -1).clone()
        # ─── Momentum warmup for global_state oscillation (Idea 3) ───
        momentum_beta = 0.0
        if adaptive and step is not None and step >= 5000:
            momentum_beta = 0.8 * min(1.0, (step - 5000) / 5000)
        if momentum_beta > 0:
            if not hasattr(self, '_gs_velocity') or self._gs_velocity.shape != global_state.shape:
                self._gs_velocity = torch.zeros_like(global_state)
            else:
                self._gs_velocity = self._gs_velocity.to(global_state.device)
        new_state = []
        self._pred_cache = []
        for i, (layer, s) in enumerate(zip(self.layers, state)):
            if adaptive:
                l_expl, l_diff = AdaptiveController.layer_stats(layer,
                    expl_thresh=self.cfg.exploration_threshold,
                    diff_thresh=self.cfg.differentiation_threshold)
                mem2v_scale = AdaptiveController.layer_w_mem2v_scale(layer,
                    min_val=self.cfg.w_mem2v_scale_min, max_val=self.cfg.w_mem2v_scale_max,
                    diff=l_diff)
                nscale = AdaptiveController.layer_noise_scale(layer,
                    min_val=self.cfg.noise_scale_min, max_val=self.cfg.noise_scale_max,
                    diff=l_diff)
                tanh_bias_mod = AdaptiveController.tanh_bias_modulation(layer, expl=l_expl)
                spectral_mod = AdaptiveController.spectral_modulation(layer, diff=l_diff)
                pred_scale_mod = AdaptiveController.pred_scale_mod(layer)
            else:
                l_expl = l_diff = 0.5
                mem2v_scale = 1.0
                nscale = 0.0
                tanh_bias_mod = 1.0
                spectral_mod = 1.0
                pred_scale_mod = None
            
            gs_i = global_state[i:i+1].detach().clone()  # (1, 1, D), no grad through global_state (EMA-only)
            if self.cfg.gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint as _cp
                h, s_out = _cp(
                    WideBindStack._checkpointed_block,
                    layer, h, s, gs_i,
                    mem2v_scale, l_diff, nscale,
                    tanh_bias_mod, pred_scale_mod, spectral_mod,
                    context_mem, allow_write, vsa_tau,
                    use_reentrant=False,
                )
            else:
                h, s_out = layer(h, s, global_state=gs_i,
                                 mem2v_scale=mem2v_scale, diff=l_diff, noise_scale=nscale,
                                 tanh_bias_mod=tanh_bias_mod, pred_scale_mod=pred_scale_mod,
                                 spectral_mod=spectral_mod,
                                 context_mem=context_mem, allow_write=allow_write,
                                 tau_s=vsa_tau)
            if s_out is not None:
                mem_state_out = s_out[0]  # (B, S*D) — multi-scale memory state
                B = h.shape[0]
                S_expected = layer._n_scales
                # Guard: checkpoint can flatten state to (B, D); infer actual S from numel
                mem_flat = mem_state_out.reshape(B, -1)
                S = mem_flat.shape[-1] // layer.D
                if S == 0:
                    S = 1
                # Per-layer tau from VSA timescales + per-layer deviation (Idea 4)
                lf = i / max(n_layers - 1, 1)
                dev = torch.tanh(self._tau_l_dev[i])
                tau_l = tau_min * (tau_max / tau_min) ** (lf * (1.0 + 0.1 * dev))
                alpha_l = torch.clamp(1.0 - c_ema / tau_l, min=0.0)
                # Weighted combination of scales для global state
                w = torch.sigmoid(layer.scale_w)  # (S, D), per-channel independent
                if S < S_expected:
                    w = w[:S]  # truncate weights to match available scales
                mem_combined = (mem_flat.reshape(B, S, layer.D) * w.unsqueeze(0)).sum(dim=1)
                mem_avg = mem_combined.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, D)
                if momentum_beta > 0:
                    vel_update = momentum_beta * self._gs_velocity[i:i+1].detach() + (1.0 - momentum_beta) * (mem_avg - gs_i)
                    self._gs_velocity[i:i+1] = vel_update.detach()
                    global_state[i:i+1] = gs_i + (1.0 - alpha_l.detach()) * self._gs_velocity[i:i+1]
                else:
                    global_state[i:i+1] = alpha_l * gs_i + (1.0 - alpha_l) * mem_avg
                s_out = tuple(t.detach() for t in s_out)
            new_state.append(s_out)
            if adaptive:
                mir = layer.mirror
                if mir._cached_pred_k is not None and mir._cached_hp is not None:
                    self._pred_cache.append((mir._cached_pred_k, mir._cached_hp))
        
        return F.rms_norm(h, (self.cfg.D,), self.final_norm_w), new_state, global_state
    
    def embed_tokens(self, tokens):
        """Token indices -> D-space vectors."""
        return self.embed(tokens)
    
    def compute_loss(self, h, targets, pred_weight=None):
        """Returns CE only (aux losses applied via gradient scaling in training step)."""
        ce_loss, _ = self.compute_losses(h, targets, pred_weight=pred_weight)
        return ce_loss
    
    def compute_losses(self, h, targets, pred_weight=None):
        """Compute CE and auxiliary losses separately. Returns raw (unweighted) values.
        
        Returns:
            ce_loss: scalar, cross-entropy loss
            aux_dict: dict of named auxiliary losses (raw, unweighted).
        """
        if isinstance(self.lm_head, ZeckendorfReadout):
            B, L, D = h.shape
            log_probs = self.lm_head.log_probs_for_target(
                h.reshape(-1, D), targets.reshape(-1))
            ce_loss = -log_probs.mean()
        else:
            logits = self.lm_head(h)
            ce = F.cross_entropy(logits.reshape(-1, self.cfg.vocab),
                                 targets.reshape(-1), reduction='none')
            mask = (targets.reshape(-1) != 0) & (targets.reshape(-1) != 2)
            ce = ce * mask.float()
            sw = getattr(self.cfg, 'surprisal_weight', 0.0)
            if self.training and sw > 0:
                with torch.no_grad():
                    ce_ratio = ce / (ce.mean() + 1e-8)
                    w = torch.sigmoid(2.0 * (ce_ratio - 1.0))
                ce_loss = (ce * w).sum() / mask.sum().clamp(min=1)
            else:
                ce_loss = ce.sum() / mask.sum().clamp(min=1)
        pred_loss = 0.0
        n_pred = 0
        cache = getattr(self, '_pred_cache', [])
        for pred_k, hp in cache:
            pred_loss = pred_loss + F.mse_loss(pred_k, hp.detach())
            n_pred = n_pred + 1
        if n_pred > 0:
            pred_loss = pred_loss / n_pred
        
        gate_l1 = 0.0
        n_gates = 0
        for layer in self.layers:
            g = getattr(layer.mirror, '_cached_gate_l1', None)
            if g is not None:
                gate_l1 = gate_l1 + g
                n_gates = n_gates + 1
        if n_gates > 0:
            gate_l1 = gate_l1 / n_gates
        
        reinforce_loss = 0.0
        n_reinf = 0
        for layer in self.layers:
            u = getattr(layer.mirror, '_cached_usefulness', None)
            g = getattr(layer.mirror, '_cached_gate', None)
            if u is not None and g is not None:
                reinforce_loss = reinforce_loss + F.mse_loss(u, g)
                n_reinf = n_reinf + 1
        if n_reinf > 0:
            reinforce_loss = reinforce_loss / n_reinf
        
        balance_loss = 0.0
        n_bal = 0
        for layer in self.layers:
            usage = getattr(layer.mirror, '_cached_gate_usage', None)
            if usage is not None:
                usage_p = usage / (usage.sum() + 1e-10)
                hhi = (usage_p ** 2).sum()
                norm = (hhi - 1.0 / usage.shape[-1]) / (1.0 - 1.0 / usage.shape[-1])
                balance_loss = balance_loss + norm.clamp(min=0)
                n_bal = n_bal + 1
        if n_bal > 0:
            balance_loss = balance_loss / n_bal
        
        diversity_loss = 0.0
        n_div = 0
        for layer in self.layers:
            group_out = getattr(layer.mlp, '_cached_group_out', None)
            if group_out is not None:
                B, L, G, d = group_out.shape
                y = group_out.norm(dim=-1).reshape(-1, G)
                y = y - y.mean(dim=0, keepdim=True)
                cov = y.T @ y / (y.shape[0] - 1 + 1e-10)
                div = F.mse_loss(cov, torch.eye(G, device=cov.device))
                diversity_loss = diversity_loss + div
                n_div = n_div + 1
        if n_div > 0:
            diversity_loss = diversity_loss / n_div
        
        nuc_loss = 0.0
        n_nuc = 0
        for layer in self.layers:
            bind_W = None
            if hasattr(layer, 'bind') and hasattr(layer.bind, 'W_proj'):
                bind_W = layer.bind.W_proj.weight
            if bind_W is not None and bind_W.ndim == 2:
                rank_ub = min(bind_W.shape[0], bind_W.shape[1])
                nuc_iters = max(1, int(math.sqrt(rank_ub)))
                v = torch.randn(bind_W.shape[1], nuc_iters, device=bind_W.device)
                Wv = bind_W @ v
                nuc = Wv.norm(dim=0).mean() * math.sqrt(bind_W.shape[1])
                nuc_loss = nuc_loss + nuc
                n_nuc = n_nuc + 1
        if n_nuc > 0:
            nuc_loss = nuc_loss / n_nuc
        
        orth_loss = 0.0
        n_orth = 0
        for layer in self.layers:
            bind_W = None
            if hasattr(layer, 'bind') and hasattr(layer.bind, 'W_proj'):
                bind_W = layer.bind.W_proj.weight
            if bind_W is not None and bind_W.ndim == 2:
                W_hat = bind_W / bind_W.norm(dim=0, keepdim=True).clamp(min=1e-8)
                gram = W_hat.T @ W_hat
                orth = F.mse_loss(gram, torch.eye(gram.shape[0], device=gram.device))
                orth_loss = orth_loss + orth
                n_orth = n_orth + 1
        if n_orth > 0:
            orth_loss = orth_loss / n_orth
        
        w_m2v_loss = 0.0
        n_m2v = 0
        if getattr(self.cfg, 'w_m2v_hierarchy_weight', 0.0) > 0:
            for i, layer in enumerate(self.layers):
                wm = getattr(layer, 'w_mem2v', None)
                if wm is not None:
                    lf = i / max(len(self.layers) - 1, 1)
                    vsa_tau = torch.exp(torch.cumsum(F.softplus(self._vsa_log_param), dim=0)) + 1.0
                    tau_min_t = vsa_tau[0]
                    tau_max_t = vsa_tau[-1]
                    tau_mid_t = (tau_min_t * tau_max_t).sqrt()
                    dev = torch.tanh(self._tau_l_dev[i])
                    tau_l_t = tau_min_t * (tau_max_t / tau_min_t) ** (lf * (1.0 + 0.1 * dev))
                    target = getattr(self.cfg, 'w_m2v_hierarchy_target', 1.0)
                    target_m2v = target / (1.0 + torch.exp(-(tau_l_t.log() - tau_mid_t.log())))
                    w_m2v_loss = w_m2v_loss + (wm.mean() - target_m2v).pow(2)
                    n_m2v = n_m2v + 1
                    n_m2v = n_m2v + 1
            if n_m2v > 0:
                w_m2v_loss = w_m2v_loss / n_m2v
        branch_loss = 0.0
        n_branch = 0
        if getattr(self.cfg, 'branch_balance_weight', 0.0) > 0:
            for layer in self.layers:
                conv = getattr(layer, '_cache_conv_out', None)
                bnd = getattr(layer, '_cache_bind_out', None)
                mir = getattr(layer, '_cache_mirror_out', None)
                if conv is not None and bnd is not None and mir is not None:
                    vc = conv.norm(dim=-1).var() + 1e-10
                    vb = bnd.norm(dim=-1).var() + 1e-10
                    vm = mir.norm(dim=-1).var() + 1e-10
                    branch_loss = branch_loss + (torch.log(vc) - torch.log(vb)).pow(2)
                    branch_loss = branch_loss + (torch.log(vc) - torch.log(vm)).pow(2)
                    branch_loss = branch_loss + (torch.log(vb) - torch.log(vm)).pow(2)
                    n_branch = n_branch + 3
            if n_branch > 0:
                branch_loss = branch_loss / n_branch
        
        ranking_loss = 0.0
        if getattr(self.cfg, 'ranking_weight', 0.0) > 0:
            for layer in self.layers:
                gu = getattr(layer.mirror, '_cached_gate_usage', None)
                if gu is not None:
                    ls = layer.mirror.log_scale
                    ls_mean = ls.mean(dim=-1)
                    gate_diff = gu.unsqueeze(1) - gu.unsqueeze(0)
                    ls_diff = ls_mean.unsqueeze(1) - ls_mean.unsqueeze(0)
                    ranking_loss = ranking_loss + (F.relu(-ls_diff) * (gate_diff > 0).float()).sum()
        
        signal_entropy = 0.0
        n_sig = 0
        for layer in self.layers:
            w = torch.sigmoid(layer.mirror._signal_log_weights)
            p = w / (w.sum() + 1e-10)  # normalize for entropy
            signal_entropy = signal_entropy - (p * torch.log(p + 1e-10)).sum()
            n_sig = n_sig + 1
        if n_sig > 0:
            signal_entropy = signal_entropy / n_sig
        
        log_scale_reg = 0.0
        n_ls = 0
        for layer in self.layers:
            ls = layer.mirror.log_scale
            excess = (ls - 2.3).clamp(min=0)
            log_scale_reg = log_scale_reg + excess.pow(2).mean()
            n_ls = n_ls + 1
        if n_ls > 0:
            log_scale_reg = log_scale_reg / n_ls
        
        # Diversity: per-layer log_scale variance (inter-expert + intra-expert)
        div_loss_raw = 0.0
        div_w = getattr(self.cfg, 'div_weight', 0.0)
        if div_w > 0:
            for layer in self.layers:
                ls = layer.mirror.log_scale
                d = ls.shape[-1]
                G = ls.shape[0]
                intra_weight = math.sqrt(d / G)
                div_loss_raw = div_loss_raw - (ls.sigmoid().var(dim=0).mean() + intra_weight * ls.sigmoid().var(dim=-1).mean())
            div_loss_raw = div_loss_raw / max(len(self.layers), 1)

        # Gate repulsion: push gate variance up (inverse of balance)
        gate_repulse_loss = 0.0
        gate_rp_w = getattr(self.cfg, 'gate_repulse_weight', 0.0)
        if gate_rp_w > 0:
            n_rp = 0
            for layer in self.layers:
                gate_usage = getattr(layer.mirror, '_cached_gate_usage', None)
                if gate_usage is not None:
                    gate_repulse_loss = gate_repulse_loss - gate_usage.var()
                    n_rp += 1
            if n_rp > 0:
                gate_repulse_loss = gate_repulse_loss / n_rp

        # Alpha novelty: push per-expert alpha apart
        alpha_novelty_loss = 0.0
        alpha_nv_w = getattr(self.cfg, 'alpha_novelty_weight', 0.0)
        if alpha_nv_w > 0:
            n_nv = 0
            for layer in self.layers:
                ad = layer.mirror.alpha_diag
                if ad is not None:
                    alpha_novelty_loss = alpha_novelty_loss - ad.mean(dim=-1).var()
                    n_nv += 1
            if n_nv > 0:
                alpha_novelty_loss = alpha_novelty_loss / n_nv

        decorr_loss = 0.0
        n_decorr = 0
        for layer in self.layers:
            d = getattr(layer.mirror, '_cached_decorr', None)
            if d is not None:
                decorr_loss = decorr_loss + d
                n_decorr = n_decorr + 1
        if n_decorr > 0:
            decorr_loss = decorr_loss / n_decorr
        
        self._cached_losses = {
            'ce': ce_loss.item(),
            'pred': pred_loss.item() if isinstance(pred_loss, torch.Tensor) else pred_loss,
            'gate_l1': gate_l1.item() if isinstance(gate_l1, torch.Tensor) else gate_l1,
            'reinforce': reinforce_loss.item() if isinstance(reinforce_loss, torch.Tensor) else reinforce_loss,
            'balance': balance_loss.item() if isinstance(balance_loss, torch.Tensor) else balance_loss,
            'div': div_loss_raw.item() if isinstance(div_loss_raw, torch.Tensor) else div_loss_raw,
            'gate_repulse': gate_repulse_loss.item() if isinstance(gate_repulse_loss, torch.Tensor) else gate_repulse_loss,
            'alpha_novelty': alpha_novelty_loss.item() if isinstance(alpha_novelty_loss, torch.Tensor) else alpha_novelty_loss,
            'ranking': ranking_loss.item() if isinstance(ranking_loss, torch.Tensor) else ranking_loss,
            'signal_ent': signal_entropy.item() if isinstance(signal_entropy, torch.Tensor) else signal_entropy,
            'ls_reg': log_scale_reg.item() if isinstance(log_scale_reg, torch.Tensor) else log_scale_reg,
            'decorr': decorr_loss.item() if isinstance(decorr_loss, torch.Tensor) else decorr_loss,
        }
        aux_dict = {}
        w_pred = getattr(self.cfg, 'pred_weight', 0.0) or 0.01
        if pred_loss != 0:
            aux_dict['pred'] = pred_loss * w_pred
        w_gate = getattr(self.cfg, 'gate_l1_weight', 0.0001)
        if gate_l1 != 0 and w_gate > 0:
            aux_dict['gate_l1'] = gate_l1 * w_gate
        w_reinf = getattr(self.cfg, 'reinforce_weight', 0.001)
        if reinforce_loss != 0:
            aux_dict['reinforce'] = reinforce_loss * w_reinf
        w_bal = getattr(self.cfg, 'balance_weight', 0.026)
        if balance_loss != 0:
            aux_dict['balance'] = balance_loss * w_bal
        w_div = getattr(self.cfg, 'diversity_weight', 0.001)
        if diversity_loss != 0:
            aux_dict['diversity'] = diversity_loss * w_div
        w_nuc = getattr(self.cfg, 'nuclear_weight', 1e-5)
        if nuc_loss != 0:
            aux_dict['nuc'] = nuc_loss * w_nuc
        w_orth = getattr(self.cfg, 'orth_weight', 1e-4)
        if orth_loss != 0:
            aux_dict['orth'] = orth_loss * w_orth
        if w_m2v_loss != 0:
            aux_dict['w_m2v'] = w_m2v_loss * getattr(self.cfg, 'w_m2v_hierarchy_weight', 0.001)
        if branch_loss != 0:
            aux_dict['branch'] = branch_loss * getattr(self.cfg, 'branch_balance_weight', 0.0)
        w_repulse = getattr(self.cfg, 'gate_repulse_weight', 0.3)
        if div_loss_raw != 0:
            aux_dict['div'] = div_loss_raw * getattr(self.cfg, 'div_weight', 50.0)
        if gate_repulse_loss != 0:
            aux_dict['gate_repulse'] = gate_repulse_loss * w_repulse
        w_novelty = getattr(self.cfg, 'alpha_novelty_weight', 0.05)
        if alpha_novelty_loss != 0:
            aux_dict['alpha_novelty'] = alpha_novelty_loss * w_novelty
        w_rank = getattr(self.cfg, 'ranking_weight', 0.01)
        if ranking_loss != 0:
            aux_dict['ranking'] = ranking_loss * w_rank
        if n_decorr > 0:
            aux_dict['decorr'] = decorr_loss * 0.01
        if n_sig > 0:
            aux_dict['signal_ent'] = signal_entropy * 0.01
        w_ls = getattr(self.cfg, 'log_scale_l2_weight', 0.01)
        if log_scale_reg != 0:
            aux_dict['ls_reg'] = log_scale_reg * w_ls
        return ce_loss, aux_dict
    
    @staticmethod
    def _checkpointed_block(layer, h, state, global_state,
                             mem2v_scale, diff, noise_scale,
                             tanh_bias_mod, pred_scale_mod, spectral_mod,
                             context_mem, allow_write, tau_s):
        """Wrapper for gradient checkpointing — all positional args are passed by checkpoint."""
        return layer(h, state, global_state=global_state,
                     mem2v_scale=mem2v_scale, diff=diff, noise_scale=noise_scale,
                     tanh_bias_mod=tanh_bias_mod, pred_scale_mod=pred_scale_mod,
                     spectral_mod=spectral_mod, context_mem=context_mem,
                     allow_write=allow_write, tau_s=tau_s)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
    
    def param_groups(self, lr=None, weight_decay=None, gate_lr_mult=None):
        """Optimizer parameter groups with λ_d LR hierarchy or legacy flat groups.
        
        When cfg.lambda_lr_hierarchy=True (default), groups follow λ_d^p:
          p=-2: embedding, readout       (0.29×)
          p=-1: MLP cores, bind W_proj   (0.54×)
          p= 0: conv, norm, W_out, head  (1.00×)
          p=+1: mirror projections, α    (1.84×)
          p=+2: gates, w_i, b_i, etc     (3.38×)
          vsa:  b_d, b_i                 (λ^{-4} ≈ 0.087×)
        """
        cfg = self.cfg
        lr = lr or cfg.lr
        wd = weight_decay or cfg.weight_decay
        
        if getattr(cfg, 'lambda_lr_hierarchy', False):
            from .lambda_utils import lambda_d
            lam = lambda_d(cfg.lambda_d)
            mlr = {
                'embed': lam ** (-2),
                'mlp': lam ** (-1),
                'vsa': lam ** (-2),
                'mirror': lam ** (1),
                'gate': lam ** (1),
            }
            groups = {
                'embed':    {'params': [], 'lr': lr * mlr['embed'], 'weight_decay': 0},
                'embed_wd': {'params': [], 'lr': lr * mlr['embed'], 'weight_decay': wd},
                'mlp':      {'params': [], 'lr': lr * mlr['mlp'],   'weight_decay': 0},
                'mlp_wd':   {'params': [], 'lr': lr * mlr['mlp'],   'weight_decay': wd},
                'mirror':   {'params': [], 'lr': lr * mlr['mirror'],'weight_decay': 0},
                'mirror_wd':{'params': [], 'lr': lr * mlr['mirror'],'weight_decay': wd},
                'gate':     {'params': [], 'lr': lr * mlr['gate'],  'weight_decay': 0},
                'gate_wd':  {'params': [], 'lr': lr * mlr['gate'],  'weight_decay': wd},
                'vsa':      {'params': [], 'lr': lr * mlr['vsa'],   'weight_decay': 0},
                'default':  {'params': [], 'lr': lr,                'weight_decay': 0},
                'default_wd':{'params': [], 'lr': lr,               'weight_decay': wd},
            }
            for name, p in self.named_parameters():
                if '.b_d' in name or '.b_i' in name or '.scale_w' in name:
                    groups['vsa']['params'].append(p)
                elif name.startswith('embed.') or name.startswith('lm_head.readout') or name.startswith('lm_head.proj'):
                    k = 'embed_wd' if p.ndim >= 2 else 'embed'
                    groups[k]['params'].append(p)
                elif any(g in name for g in ['.mirror.alpha_diag', '.mirror.w_pred_scale_legacy',
                                              '.log_skip_alpha', '.mirror.W_proj', '.mirror.W_out',
                                              '.mirror.w_temp', '.mirror.w_global',
                                              '.mirror.log_scale', '.mirror.tanh_bias',
                                              '.log_dvar_mod_scale', '.dvar_mod_bias',
                                              '.log_grad_mod_scale', '.grad_mod_bias']):
                    # Mirror projections, alpha, gates -> mirror LR (1.84x)
                    k = 'mirror_wd' if p.ndim >= 2 else 'mirror'
                    groups[k]['params'].append(p)
                elif '.mlp.' in name or '.bind.W_proj.weight' in name or name.endswith('.W_out') or name.endswith('.W_proj'):
                    # Block-level W_proj/W_out (not mirror, caught above) -> mlp speed (0.54x)
                    k = 'mlp_wd' if p.ndim >= 2 else 'mlp'
                    groups[k]['params'].append(p)
                elif any(g in name for g in ['.w_gate', '.b_gate', '.w_delta_gate', '.b_delta_gate',
                                              '.w_i', '.w_d', '.w_q', '.w_q_leaf', '.w_q_ctx', '.w_mem2v',
                                              '.w_k_mu', '.w_q_mu', '.w_mu_mem',
                                              '.w_u', '.w_v']):
                    k = 'gate_wd' if p.ndim >= 2 else 'gate'
                    groups[k]['params'].append(p)
                else:
                    k = 'default_wd' if p.ndim >= 2 else 'default'
                    groups[k]['params'].append(p)
            return [v for v in groups.values() if v['params']]
        
        # ─── Legacy groups (lambda_lr_hierarchy=False) ───
        gate_lr_mult = cfg.gate_lr_mult if gate_lr_mult is None else gate_lr_mult
        decay = []
        no_decay = []
        gate_decay = []
        gate_no_decay = []
        vsa_bias = []
        for name, p in self.named_parameters():
            if '.b_d' in name or '.b_i' in name or '.scale_w' in name:
                vsa_bias.append(p)
                continue
            is_gate = any(g in name for g in ['.w_i', '.w_d', '.w_q', '.w_q_leaf', '.w_q_ctx', '.w_mem2v',
                                               '.w_k_mu', '.w_q_mu', '.w_mu_mem',
                                               '.w_u', '.w_v',
                                               '.tanh_bias', '.log_scale',
                                               '.mirror.W_proj', '.mirror.W_out',
                                               '.mirror.w_temp', '.mirror.w_global',
                                                '.mirror.alpha_diag', '.mirror.w_pred_scale_legacy',
                                               '.mirror.w_gate', '.mirror.b_gate',
                                               '.log_dvar_mod_scale', '.dvar_mod_bias',
                                               '.log_grad_mod_scale', '.grad_mod_bias',
                                               '.log_skip_alpha'])
            if is_gate:
                if p.ndim < 2 or 'w_pred_scale_legacy' in name:
                    gate_no_decay.append(p)
                else:
                    gate_decay.append(p)
            else:
                if p.ndim < 2:
                    no_decay.append(p)
                else:
                    decay.append(p)
        groups = [
            {'params': decay, 'lr': lr, 'weight_decay': wd},
            {'params': no_decay, 'lr': lr, 'weight_decay': 0},
        ]
        if gate_decay:
            groups.append({'params': gate_decay, 'lr': lr * gate_lr_mult, 'weight_decay': wd})
        if gate_no_decay:
            groups.append({'params': gate_no_decay, 'lr': lr * gate_lr_mult, 'weight_decay': 0})
        if vsa_bias:
            vsa_lr_mult = getattr(cfg, 'vsa_b_lr_mult', 0.1)
            groups.append({'params': vsa_bias, 'lr': lr * vsa_lr_mult, 'weight_decay': 0})
        return groups


# ─── Adaptive Controller ──────────────────────────────────────────────


class AdaptiveController:
    """
    Computes ALL adaptive hyperparameters from cognitive mirror state.

    Two fundamental signals drive every parameter:
    ──────────────────────────────────────────────────────────
    exploration = min(1, |mirror| / λ⁻²)
        How much correction is the mirror applying.
        High → model is actively adjusting, needs aggressive config.
        Low → model is stable, needs conservative config.

    differentiation = min(1, var(log_scale) / λ⁻⁴)
        How specialized has the mirror become (per-dim scaling).
        High → mirror has learned which dims to trust/suppress.
        Low → mirror hasn't differentiated, still exploring.

    λ_d hierarchy (d=3): λ₃ ≈ 1.839, λ⁻² ≈ 0.296, λ⁻⁴ ≈ 0.087
    All range defaults below are λ_d d=3 derived.

    Key design: ALL methods work at per-layer AND global resolution.
    ``layer_stats(layer)`` → per-layer (expl, diff)
    ``stats(blocks)`` → global average   (backward compat)

    New intelligent adaptivity:
    ──────────────────────────
    - ``pred_weight(blocks)`` — alpha loss weight scales with diff
      (more temporal learning when mirror has specialized)
    - ``tanh_bias_modulation(layer)`` — tanh_bias amplified by exploration
      (more asymmetric correction when actively exploring)
    - ``spectral_modulation(layer)`` — lambda_k amplified by differentiation
      (more aggressive freq shaping when experts are specialized)
    - ``pred_scale_mod(layer)`` — per-expert w_pred_scale modulation
      from delta_var (experts with volatile dynamics get more
      temporal teaching signal)

    Mathematically derived ranges (λ_d d=3):
    ────────────────────────────────────────
    b_d ∈ [b_d_min, b_d_max] per layer, where b_d_min = 2.0 + 3.0*layer_frac
         expl=1 → b_d = b_d_min (shortest memory)
         expl=0 → b_d = b_d_max (longest memory, configurable vsa_b_d_max)
         L0: τ≈[7, 150] (default b_d_max=5.0), up to τ≈160K (b_d_max=12.0)
         Per-channel via gradient: b_d is (D,) with lerp-slow push to controller target
    b_i  ∈ [-3.0, -1.5] → i_gate ≈ [0.047, 0.18] (write rate via softplus)
    w_mem2v_scale ∈ [0.544, 1.0]  (memory contribution, λ⁻¹ to 1)
    ema_alpha ∈ [0.974, 0.992]  (cross-layer memory, 1-λ⁻⁶ to 1-λ⁻⁸)
    noise_scale ∈ [0.0076, 0.026]  (parameter noise, λ⁻⁸ to λ⁻⁶)
    pred_weight ∈ [0.026, 0.296]  (alpha loss weight, λ⁻⁶ to λ⁻²)
    tanh_bias_mod ∈ [1.0, 1.5]  (exploration amplification)
    spectral_mod ∈ [0.913, 1.087]  (differentiation, 1±λ⁻⁴)
    """
    @staticmethod
    def layer_stats(layer, expl_thresh=0.296, diff_thresh=0.087):
        """Per-layer (exploration, differentiation) from a single block."""
        m = layer.mirror
        ls = m.log_scale.data
        var = ls.var().item()
        mag = m._last_magnitude.item()
        return min(1.0, mag / expl_thresh), min(1.0, var / diff_thresh)

    @staticmethod
    def stats(blocks, expl_thresh=0.296, diff_thresh=0.087):
        """Global average (exploration, differentiation) across all layers."""
        expl_sum = diff_sum = 0.0
        for layer in blocks:
            e, d = AdaptiveController.layer_stats(layer, expl_thresh, diff_thresh)
            expl_sum += e
            diff_sum += d
        n = len(blocks)
        return expl_sum / n, diff_sum / n

    # ─── Per-layer methods ────────────────────────────────────────

    @staticmethod
    def layer_b_d(layer, expl=None, b_d_max=5.0):
        """Per-layer decay bias. Layer uses its own exploration."""
        if expl is None:
            expl, _ = AdaptiveController.layer_stats(layer)
        lf = getattr(layer, 'layer_idx', 0) / max(getattr(layer, 'total_layers', 32) - 1, 1)
        b_d_min = 2.0 + 3.0 * lf
        b_d_val = b_d_max - expl * (b_d_max - b_d_min)
        return max(2.0, min(b_d_max, b_d_val))

    @staticmethod
    def layer_b_i(layer, expl=None, tau_l=None):
        """Per-layer write gate bias. Нормировка: i_gate ∝ 1/τ.
        
        i_gate = softplus(b_i_l). Равновесная норма памяти:
            ‖M_l‖ = i_gate · ‖h‖ · τ_l
        Для ‖M_l‖ = const по слоям: i_gate ∝ 1/τ_l.
        
        Базовое значение: i_gate_ref = 0.182 при τ_ref ≈ 32.
        c = 0.182 · 32 ≈ 5.83.
        i_gate_l = c / τ_l  →  b_i_l = softplus⁻¹(c / τ_l)
        """
        if expl is None:
            expl, _ = AdaptiveController.layer_stats(layer)
        b_i_base = -3.0 + expl * 1.5
        c = 5.83
        if tau_l is not None:
            i_target = min(1.0, c / tau_l)
        else:
            lf = getattr(layer, 'layer_idx', 0) / max(getattr(layer, 'total_layers', 32) - 1, 1)
            tau_l = 8.0 + 141.0 * lf
            i_target = min(1.0, c / tau_l)
        b_i_tau = math.log(max(i_target, 1e-6))
        b_i = b_i_base + b_i_tau
        return max(b_i, -6.0)  # floor: i_gate >= softplus(-6.0) ≈ 0.0025

    @staticmethod
    def layer_w_mem2v_scale(layer, min_val=0.544, max_val=1.0, diff=None):
        """Per-layer memory contribution."""
        if diff is None:
            _, diff = AdaptiveController.layer_stats(layer)
        return max_val - diff * (max_val - min_val)

    @staticmethod
    def layer_noise_scale(layer, min_val=0.0076, max_val=0.026, diff=None):
        """Per-layer parameter noise."""
        if diff is None:
            _, diff = AdaptiveController.layer_stats(layer)
        return max_val - diff * (max_val - min_val)

    @staticmethod
    def layer_ema_alpha(layer, min_val=0.974, max_val=0.992, diff=None):
        """Per-layer EMA rate (for per-layer global_state aggregation)."""
        if diff is None:
            _, diff = AdaptiveController.layer_stats(layer)
        return min_val + diff * (max_val - min_val)

    # ─── New intelligent adaptivity ───────────────────────────────

    @staticmethod
    def pred_weight(blocks, min_val=0.026, max_val=0.296):
        """Adaptive alpha auxiliary loss weight.

        When mirror has differentiated (high diff), temporal prediction
        is more meaningful → increase pred_weight to drive alpha learning.
        When mirror hasn't specialized, pred would be noise → keep low.
        """
        _, diff = AdaptiveController.stats(blocks)
        return min_val + diff * (max_val - min_val)

    @staticmethod
    def tanh_bias_modulation(layer, expl=None):
        """Scale tanh_bias by exploration.

        High exploration → more asymmetric correction needed → amplify.
        Range: [1.0, 1.296] (at most 1+λ⁻² boost).
        """
        if expl is None:
            expl, _ = AdaptiveController.layer_stats(layer)
        return 1.0 + 0.296 * expl

    @staticmethod
    def spectral_modulation(layer, diff=None):
        """Modulate spectral lambda_k by differentiation.

        High diff → mirror has learned structure → amplify spectral
        contrast (more aggressive frequency shaping).
        Low diff → flatten spectral response (conservative).
        Range: [0.913, 1.087] = 1 ± λ⁻⁴.
        """
        if diff is None:
            _, diff = AdaptiveController.layer_stats(layer)
        return 1.0 + 0.087 * (diff - 0.5) * 2.0  # 0.913 at diff=0, 1.087 at diff=1

    @staticmethod
    def pred_scale_mod(layer):
        """Per-expert w_pred_scale modulation from delta_var.
        
        Experts with volatile K-space dynamics (high delta_var relative
        to layer average) get more temporal teaching signal.
        Uses tanh-based soft normalization instead of division to avoid NaN.
        Range: [0.5, 2.0] centered at 1.0.
        """
        dv = layer.mirror._delta_var
        dv_centered = dv - dv.mean()
        return (1.0 + 0.5 * torch.tanh(dv_centered)).clamp(0.1, 3.0)

    # ─── Global (backward-compat) wrappers ────────────────────────

    @staticmethod
    def b_d(blocks, b_d_max=5.0):
        expl, _ = AdaptiveController.stats(blocks)
        return b_d_max - expl * 2.0

    @staticmethod
    def b_i(blocks):
        expl, _ = AdaptiveController.stats(blocks)
        return -3.0 + expl * 1.5

    @staticmethod
    def w_mem2v_scale(blocks, min_val=0.544, max_val=1.0):
        _, diff = AdaptiveController.stats(blocks)
        return max_val - diff * (max_val - min_val)

    @staticmethod
    def ema_alpha(blocks, min_val=0.974, max_val=0.992):
        _, diff = AdaptiveController.stats(blocks)
        return min_val + diff * (max_val - min_val)

    @staticmethod
    def noise_scale(blocks, min_val=0.0076, max_val=0.026):
        _, diff = AdaptiveController.stats(blocks)
        return max_val - diff * (max_val - min_val)



class MirrorLRScheduler:
    """LR scheduler modulated by cognitive mirror state dynamics.

    Growth-ratio multipliers (neutral at growth=1):
      var/alpha/gate growth  →  LR up when specialization grows, down when stalled
    mag_factor (cap): |mirror| above threshold → LR reduced (counter-cyclical)
    Loss damping (persistent): val_loss regression >2% → _loss_lr_factor halved
      (ReduceLROnPlateau semantics; resets to 1.0 on new best).
    """
    def __init__(self, model, optimizer, base_lr=None, warmup=1000,
                 target_var=0.161, mag_threshold=0.296, lr_min_ratio=0.026,
                 max_decay_steps=2584, var_min_for_lr_decay=0.008,
                 cfg=None):
        if cfg is not None:
            base_lr = base_lr or cfg.lr
            warmup = getattr(cfg, 'warmup_steps', warmup)
        self.model = model
        self.optimizer = optimizer
        self.base_lr = base_lr
        self._orig_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.warmup = warmup
        self._step = 0
        self._last_log = 0
        # Adaptive thresholds: EMA of mirror stats
        self._tau_var = None
        self._tau_mag = None
        self._tau_1malpha = None
        self._tau_gate_var = None
        self._tau_ema = 0.99

    def _mirror_stats(self):
        var_sum = 0.0
        mag_sum = 0.0
        alpha_sum = 0.0
        gate_var_sum = 0.0
        n = len(self.model.layers)
        for layer in self.model.layers:
            m = layer.mirror
            ls = m.log_scale.data
            var_sum += ls.var().item()
            mag_sum += m._last_magnitude.item()
            alpha = m.alpha_diag.data
            alpha_sum += (1.0 - alpha).abs().mean().item()
            gate_var_sum += m._last_gates.var().item()
        return var_sum / n, mag_sum / n, alpha_sum / n, gate_var_sum / n

    def report_train_loss(self, train_loss, ce_loss=None):
        """Report training loss for LR damping. Uses CE (not total) to avoid pred_loss false dampings."""
        pass

    def report_val_loss(self, val_loss):
        """Report validation loss — halve _loss_lr_factor when val loss regresses >2%.
        This implements ReduceLROnPlateau semantics on top of the mirror-based schedule."""
        if not hasattr(self, '_best_val_loss'):
            self._best_val_loss = val_loss
            self._loss_lr_factor = 1.0
        if val_loss < self._best_val_loss * 0.998:
            self._best_val_loss = val_loss
            self._loss_lr_factor = min(1.0, self._loss_lr_factor * 1.1)
        elif val_loss > self._best_val_loss * 1.02:
            self._loss_lr_factor = max(0.05, self._loss_lr_factor * 0.5)

    def step(self):
        self._step += 1
        warmup_end = self.warmup
        blend_steps = 50
        if self._step < warmup_end + blend_steps:
            if self._step < warmup_end:
                mult = self._step / max(warmup_end, 1)
                override = max(0.0, 1.0 - mult * 0.7)
            else:
                blend = (self._step - warmup_end) / blend_steps
                mult = 1.0 - blend * 0.3
                override = 0.3 * max(0.0, 1.0 - blend)
            temp_max, temp_min = 2.0, 0.5
            if self._step < warmup_end:
                t = self._step / max(warmup_end, 1)
                temp = temp_max - t * (temp_max - temp_min)
            else:
                blend = min(1.0, (self._step - warmup_end) / blend_steps)
                temp = temp_min + (1.0 - blend) * (temp_max - temp_min) * 0.3
            for layer in self.model.layers:
                layer.mirror._alpha_override.fill_(override)
                layer.mirror._usefulness_temp.fill_(max(temp, 0.1))
        else:
            for layer in self.model.layers:
                layer.mirror._alpha_override.fill_(0.0)
            var, mag, mean_1malpha, gate_var = self._mirror_stats()

            if self._tau_var is None:
                self._tau_var = var + 1e-10
                self._tau_mag = mag + 1e-10
                self._tau_1malpha = mean_1malpha + 1e-10
                self._tau_gate_var = gate_var + 1e-10

            te = self._tau_ema
            self._tau_var = te * self._tau_var + (1 - te) * var
            self._tau_mag = te * self._tau_mag + (1 - te) * mag
            self._tau_1malpha = te * self._tau_1malpha + (1 - te) * mean_1malpha
            self._tau_gate_var = te * self._tau_gate_var + (1 - te) * gate_var

            var_ratio = var / self._tau_var
            var_mult = min(2.0, max(0.5, 1.0 / max(var_ratio, 1e-10)))

            alpha_ratio = mean_1malpha / self._tau_1malpha
            alpha_mult = min(2.0, max(0.5, 1.0 / max(alpha_ratio, 1e-10)))

            gate_ratio = gate_var / self._tau_gate_var
            gate_mult = min(2.0, max(0.5, 1.0 / max(gate_ratio, 1e-10)))

            mag_ratio = mag / self._tau_mag
            mag_factor = min(1.0, max(0.2, 1.0 / max(mag_ratio, 1e-10)))

            mirror_mult = (var_mult * alpha_mult * gate_mult) ** (1/3) * mag_factor
            mult = max(0.05, min(1.0, mirror_mult))
            if hasattr(self, '_loss_lr_factor'):
                mult = mult * self._loss_lr_factor

            if self._step - self._last_log >= 500:
                self._last_log = self._step
                tau_var = self._tau_var.item() if hasattr(self._tau_var, 'item') else self._tau_var
                print(f'  lr_adapt: var(ls)={var:.6f} |1-a|={mean_1malpha:.6f} '
                      f'gate_var={gate_var:.6f} |mirror|={mag:.4f} '
                      f'tau_var={tau_var:.6f} '
                      f'mult={mult:.4f} lr={self.base_lr*mult:.2e}')

        for i, pg in enumerate(self.optimizer.param_groups):
            pg['lr'] = self._orig_lrs[i] * mult

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]

    def state_dict(self):
        sd = {
            'step': self._step,
            'last_log': self._last_log,
            'type': 'MirrorLRScheduler',
            'tau_var': self._tau_var,
            'tau_mag': self._tau_mag,
            'tau_1malpha': self._tau_1malpha,
            'tau_gate_var': self._tau_gate_var,
            'orig_lrs': self._orig_lrs,
        }
        if hasattr(self, '_best_val_loss'):
            sd['best_val_loss'] = self._best_val_loss
            sd['loss_lr_factor'] = self._loss_lr_factor
        return sd

    def load_state_dict(self, sd):
        self._step = sd.get('step', 0)
        self._last_log = sd.get('last_log', 0)
        self._tau_var = sd.get('tau_var')
        self._tau_mag = sd.get('tau_mag')
        self._tau_1malpha = sd.get('tau_1malpha')
        self._tau_gate_var = sd.get('tau_gate_var')
        if 'orig_lrs' in sd:
            self._orig_lrs = sd['orig_lrs']
        if 'best_val_loss' in sd:
            self._best_val_loss = sd['best_val_loss']
            self._loss_lr_factor = sd.get('loss_lr_factor', 1.0)

    def reset_for_new_data(self, reset_warmup_steps=2000):
        self._tau_var = None
        self._tau_mag = None
        self._tau_1malpha = None
        self._tau_gate_var = None


# ─── Verify ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    cfg = WideBindConfig(n_layers=24, D=896, bottleneck=896, bind_K=32, mlp_groups=8)
    model = WideBindStack(cfg).to(device)
    n = model.param_count()
    print(f'  D=896 G=8: params={n:,} ({n/1e6:.2f}M)')
    
    print()
    cfg = WideBindConfig(n_layers=4, D=896, bottleneck=896, bind_K=32)
    model = WideBindStack(cfg).to(device)
    
    x = torch.randint(0, cfg.vocab, (2, 16), device=device)
    h = model.embed_tokens(x)
    out, state, _ = model(h)
    loss = model.compute_loss(out[:, :-1], x[:, 1:])
    loss.backward()
    
    total_grad = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    out_std = out.std().item()
    print(f'Output: {out.shape}  std={out_std:.4f}')
    print(f'Loss: {loss.item():.4f}  Grad: {total_grad:.4f}')
    print('OK' if not math.isnan(loss.item()) and total_grad > 0 else 'FAIL')
