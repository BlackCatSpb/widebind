"""
Generate HTML analysis report from any WideBind checkpoint (.pt or _fcf.pt).
Usage: python scripts/analyze_checkpoint.py <checkpoint.pt>
"""

import os, sys, math, json, textwrap
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn.functional as F
from torch.serialization import add_safe_globals
from core import WideBindConfig, WideBindStack, dct_basis, zeckendorf_codes
from compression import FCF_CPR

add_safe_globals([WideBindConfig])

def fmt(x, dec=4):
    if isinstance(x, float):
        return f'{x:.{dec}f}'
    return str(x)

def svd_eff_rank(w):
    """Effective rank from SVD: (sum s_i)^2 / sum s_i^2."""
    s = torch.linalg.svdvals(w.float())
    return (s.sum() ** 2 / (s ** 2).sum()).item()

def bind_output_eff_rank(layer, x):
    """Effective rank of bind OUTPUT over the batch axis via covariance SVD.
    This is what the Fibonacci twist actually raises (r_p won't move in tied mode)."""
    layer.eval()
    with torch.no_grad():
        bind_out = layer.bind(x)
        flat = bind_out.reshape(-1, bind_out.shape[-1]).float()
        flat = flat - flat.mean(dim=0, keepdim=True)
        cov = flat.t() @ flat / max(flat.shape[0] - 1, 1)
        s = torch.linalg.svdvals(cov)
        s = s[s > 1e-12]
    return (s.sum().pow(2) / s.pow(2).sum()).item() if s.numel() > 0 else 0.0


def top_singular_values(w, n=5):
    s = torch.linalg.svdvals(w.float())
    return [f'{s[i]:.3f}' for i in range(min(n, len(s)))]

def fmt_small(x):
    if abs(x) < 1e-4:
        return f'{x:.2e}'
    return f'{x:.4f}'

def analyze_single_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['cfg']
    step = ckpt.get('step', 0)
    best_val = ckpt.get('best_val_loss', float('inf'))
    if not isinstance(best_val, (int, float)):
        best_val = float('inf')
    has_opt = 'optimizer' in ckpt
    has_sch = 'scheduler' in ckpt
    is_compressed = 'model_compressed' in ckpt
    
    if is_compressed:
        cpr = FCF_CPR()
        ckpt = cpr.load_compressed(ckpt_path, cfg=cfg)
    
    sd = ckpt['model']
    
    model = WideBindStack(cfg)
    missing, _ = model.load_state_dict(sd, strict=False)
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # ─── Weight distribution ───
    all_w = torch.cat([p.data.flatten() for p in model.parameters()])
    
    # ─── Forward pass for activations (adaptive=False: don't mutate b_i/b_d/EMA) ───
    torch.manual_seed(42)
    device = next(model.parameters()).device
    x = torch.randint(0, min(cfg.vocab, 1000), (1, 16)).to(device)
    h = model.embed_tokens(x)
    out, state, _ = model(h, adaptive=False)
    # Sample batch for bind output rank measurement (fresh random, not the same as forward)
    sample_batch = model.embed_tokens(torch.randint(0, min(cfg.vocab, 1000), (1, 32)).to(device))
    
    # ─── Per-layer analysis ───
    layers_data = []
    for i, layer in enumerate(model.layers):
        d = {'idx': i}
        
        # Bind
        bnd = layer.bind
        wp = bnd.W_proj.weight.data   # (D, K)
        wo = bnd.W_out.data           # (K, D) or (S, K, D)
        d['bind_proj_norm'] = wp.norm().item()
        d['bind_out_norm'] = wo.norm().item()
        d['bind_proj_svd'] = top_singular_values(wp)
        d['bind_out_svd'] = top_singular_values(wo if wo.dim() == 2 else wo.reshape(-1, wo.shape[-1]))
        d['bind_proj_rank'] = svd_eff_rank(wp)
        d['bind_mode'] = bnd.mode
        d['bind_S'] = bnd.S
        # bind output effective rank (requires a real batch)
        if 'sample_batch' in locals() and sample_batch is not None and i == 0:
            d['bind_out_rank'] = bind_output_eff_rank(layer, sample_batch)
        else:
            d['bind_out_rank'] = 0.0
        
        # Gates
        for gate in ['w_i', 'w_d', 'w_q', 'w_mem2v', 'w_k_mu', 'w_q_mu', 'w_mu_mem']:
            d[f'{gate}_mean'] = getattr(layer, gate).data.mean().item()
            d[f'{gate}_std'] = getattr(layer, gate).data.std().item()
        
        # Memory gate biases → actual gate values
        # b_d/b_i are (D,) per-channel; show stats across channels
        d['b_i_val'] = layer.b_i.data.mean().item()
        d['b_d_val'] = layer.b_d.data.mean().item()
        d['b_d_std'] = layer.b_d.data.std().item()
        d['b_i_std'] = layer.b_i.data.std().item()
        d['i_gate'] = torch.sigmoid(layer.b_i.data).mean().item()
        d['tau'] = -1.0 / math.log(max(torch.sigmoid(layer.b_d.data).mean().item(), 1e-10))
        
        # Bind bilinear weights
        for vec in ['w_u', 'w_v']:
            d[f'bind_{vec}_mean'] = getattr(bnd, vec).data.mean().item()
            d[f'bind_{vec}_std'] = getattr(bnd, vec).data.std().item()
        
        # Spectral
        d['lambda_k_mean'] = layer.lambda_k.data.mean().item()
        d['lambda_k_std'] = layer.lambda_k.data.std().item()
        
        # Pre-LN
        d['pre_ln_w_mean'] = layer.pre_ln_w.data.mean().item()
        d['pre_ln_w_std'] = layer.pre_ln_w.data.std().item()
        
        # Conv
        cw = layer.conv.weight.data
        d['conv_norm'] = cw.norm().item()
        d['conv_mean'] = cw.mean().item()
        d['conv_std'] = cw.std().item()
        
        # ─── Mirror ───
        m = layer.mirror
        d['mirror_proj_norm'] = m.W_proj.data.norm().item()
        d['mirror_out_norm'] = m.W_out.data.norm().item()
        d['mirror_temp_mean'] = m.w_temp.data.mean().item()
        d['mirror_temp_std'] = m.w_temp.data.std().item()
        d['mirror_global_mean'] = m.w_global.data.mean().item()
        d['mirror_global_std'] = m.w_global.data.std().item()
        d['mirror_sym_u_mean'] = m.w_sym_u.data.mean().item()
        d['mirror_sym_u_std'] = m.w_sym_u.data.std().item()
        d['mirror_sym_v_mean'] = m.w_sym_v.data.mean().item()
        d['mirror_sym_v_std'] = m.w_sym_v.data.std().item()
        d['mirror_log_scale_mean'] = m.log_scale.data.mean().item()
        d['mirror_log_scale_std'] = m.log_scale.data.std().item()
        d['mirror_log_scale_min'] = m.log_scale.data.min().item()
        d['mirror_log_scale_max'] = m.log_scale.data.max().item()
        d['mirror_log_scale_sparsity'] = (m.log_scale.data.abs() < 0.01).float().mean().item() * 100
        
        # ─── Gate / Prediction ───
        d['w_pred_scale_mean'] = m.w_pred_scale.data.mean().item()
        d['w_pred_scale_std'] = m.w_pred_scale.data.std().item()
        d['gate_beta'] = m.b_gate.data.mean().item()  # gate bias (was 'gate_beta' in old arch)
        # Alpha identity check (per-dim diagonal, pred_k = alpha_diag * hp_prev)
        alpha = m.alpha_diag.data  # (G, k)
        d['alpha_mean'] = alpha.mean().item()
        d['alpha_std'] = alpha.std().item()
        d['alpha_min'] = alpha.min().item()
        d['alpha_max'] = alpha.max().item()
        d['alpha_deviation'] = (alpha - 1.0).abs().mean().item()
        
        # Per-expert learned modulation
        d['log_skip_alpha_mean'] = m.log_skip_alpha.data.mean().item()
        d['skip_alpha'] = m.log_skip_alpha.data.exp().mean().item()
        d['log_dvar_mod_scale_mean'] = m.log_dvar_mod_scale.data.mean().item()
        d['dvar_mod_bias_mean'] = m.dvar_mod_bias.data.mean().item()
        d['log_grad_mod_scale_mean'] = m.log_grad_mod_scale.data.mean().item()
        d['grad_mod_bias_mean'] = m.grad_mod_bias.data.mean().item()
        
        # Mirror conv smooth
        d['mirror_conv_norm'] = m.conv_smooth.weight.data.norm().item()
        
        # temp/global/sym vector norms
        for vn in ['w_temp', 'w_global', 'w_sym_u', 'w_sym_v']:
            d[f'mirror_{vn}_std'] = getattr(m, vn).data.std().item()
        
        # ─── MLP ───
        mlp = layer.mlp
        G = mlp.G
        eff_ranks = []
        norms = []
        for g in range(G):
            w = mlp.W_up[g].float()
            s = torch.linalg.svdvals(w)
            eff_ranks.append((s.sum() ** 2 / (s ** 2).sum()).item())
            norms.append(w.norm().item())
        d['mlp_eff_rank'] = sum(eff_ranks) / G
        d['mlp_norm'] = sum(norms) / G
        d['mlp_min_rank'] = min(eff_ranks)
        d['mlp_max_rank'] = max(eff_ranks)
        d['mlp_eff_ranks'] = eff_ranks  # per-group
        d['mlp_group_norms'] = norms  # per-group
        d['mlp_dead_groups'] = sum(1 for r in eff_ranks if r < 0.5 * (sum(eff_ranks) / G))
        d['mlp_up_mean'] = mlp.W_up.data.mean().item()
        d['mlp_up_std'] = mlp.W_up.data.std().item()
        d['mlp_down_mean'] = mlp.W_down.data.mean().item()
        d['mlp_down_std'] = mlp.W_down.data.std().item()
        d['mlp_norm_w_mean'] = mlp.norm_w.data.mean().item()
        
        layers_data.append(d)
    
    # ─── Optimizer analysis ───
    opt_info = {}
    if has_opt:
        opt = ckpt['optimizer']
        if 'state' in opt:
            g_abs_means = []
            g_rms = []
            for st in opt['state'].values():
                if 'exp_avg' in st:
                    g = st['exp_avg']
                    g_abs_means.append(g.abs().mean().item())
                    g_rms.append((g ** 2).mean().item())
            if g_abs_means:
                opt_info['mean_abs_grad'] = sum(g_abs_means) / len(g_abs_means)
                opt_info['rms_grad'] = math.sqrt(sum(g_rms) / len(g_rms))
    
    return model, cfg, step, best_val, total, trainable, all_w, out, layers_data, opt_info, has_opt, has_sch, missing


def append_to_log(ckpt_path, layers_data, model, cfg, step, best_val, all_w, out):
    log_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'TRAINING_LOG.md')
    log_path = os.path.normpath(log_path)
    
    from core import AdaptiveController
    d0 = layers_data[0]
    dL = layers_data[-1]
    ls_stds = [d['mirror_log_scale_std'] for d in layers_data]
    expl, diff = AdaptiveController.stats(model.layers)
    
    val_str = f'val_loss={best_val:.4f}' if best_val != float('inf') else 'no eval'
    
    entry = f'''
### Step {step} — {os.path.basename(ckpt_path)}
| Metric | Value |
|--------|-------|
| Step | {step} |
| best_val_loss | {best_val if best_val != float('inf') else 'N/A'} |
| alpha deviation from 1 | {max(d['alpha_deviation'] for d in layers_data):.4f} |
| alpha mean / std | {sum(d['alpha_mean'] for d in layers_data)/len(layers_data):.3f} / {sum(d['alpha_std'] for d in layers_data)/len(layers_data):.4f} |
| skip_alpha mean | {sum(d['skip_alpha'] for d in layers_data)/len(layers_data):.4f} |
| var(log_scale) mean | {sum(s**2 for s in ls_stds)/len(ls_stds):.6f} |
| log_scale sigma mean | {sum(ls_stds)/len(ls_stds):.4f} |
| MLP eff_rank mean | {sum(d['mlp_eff_rank'] for d in layers_data)/len(layers_data):.1f} |
| Bind proj rank mean (r_p) | {sum(d['bind_proj_rank'] for d in layers_data)/len(layers_data):.1f} |
| Bind out rank mean (r_out) | {sum(d['bind_out_rank'] for d in layers_data)/len(layers_data):.1f} |
| Bind mode | {d0['bind_mode']} S={d0['bind_S']} |
| tau range | [{min(d['tau'] for d in layers_data):.0f}, {max(d['tau'] for d in layers_data):.0f}] |
| i_gate | {d0['i_gate']:.3f} |
| w_pred_scale mu | {d0['w_pred_scale_mean']:.3f} |
| log_dvar_mod_scale | {d0['log_dvar_mod_scale_mean']:.3f} |
| log_grad_mod_scale | {d0['log_grad_mod_scale_mean']:.3f} |
| Exploration | {expl:.4f} |
| Differentiation | {diff:.6f} |
| Output std (fwd) | {out.std():.4f} |
| Weights std | {all_w.std():.4f} |
'''
    
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(entry)
    print(f'  Appended to {os.path.relpath(log_path)}')


def generate_report(ckpt_path, append_log=False):
    model, cfg, step, best_val, total, trainable, all_w, out, layers_data, opt_info, has_opt, has_sch, missing = analyze_single_checkpoint(ckpt_path)
    
    base = os.path.splitext(ckpt_path)[0]
    html_path = base + '_report.html'
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>WideBind — Step {step} Report</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 2em; background: #0d1117; color: #e6edf3; line-height: 1.5; }}
h1 {{ color: #58a6ff; }}
h2 {{ color: #79c0ff; border-bottom: 1px solid #30363d; padding-bottom: 0.2em; }}
h3 {{ color: #58a6ff; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.85em; }}
th, td {{ border: 1px solid #30363d; padding: 3px 8px; text-align: left; }}
th {{ background: #161b22; color: #58a6ff; position: sticky; top: 0; }}
tr:nth-child(even) {{ background: #0d1117; }}
tr:nth-child(odd) {{ background: #161b22; }}
code {{ background: #21262d; padding: 1px 5px; border-radius: 3px; color: #f0883e; }}
.num {{ color: #79c0ff; }}
.warn {{ color: #f85149; }}
.good {{ color: #3fb950; }}
.highlight {{ background: #1f2a3a !important; }}
td:hover {{ background: #21262d; }}
pre {{ background: #161b22; padding: 1em; border-radius: 6px; overflow-x: auto; }}
</style>
</head>
<body>
<h1>WideBind — Step {step}</h1>
<p>checkpoint: {os.path.basename(ckpt_path)} | best_val_loss: {fmt(best_val)}
{" | ⚠ MISSING KEYS: " + str(len(missing)) if missing else ""}</p>

<h2>Architecture</h2>
<table>
<tr><td>Params</td><td class="num">{total:,} ({total/1e6:.2f}M)</td></tr>
<tr><td>Trainable</td><td class="num">{trainable:,}</td></tr>
<tr><td>D / K</td><td class="num">{cfg.D} / {cfg.bind_K}</td></tr>
<tr><td>MLP groups / expand</td><td class="num">{cfg.mlp_groups} / {cfg.mlp_expand}×</td></tr>
<tr><td>Layers</td><td class="num">{cfg.n_layers}</td></tr>
<tr><td>SEQ_LEN / Batch</td><td class="num">{cfg.seq_len} / {cfg.batch_size}</td></tr>
<tr><td>LR / warmup</td><td class="num">{cfg.lr} / {cfg.warmup_steps}</td></tr>
<tr><td>Scheduler</td><td class="num">{cfg.scheduler}</td></tr>
<tr><td>Optimizer</td><td class="num">{"Yes" if has_opt else "No"}</td></tr>
<tr><td>Scheduler state</td><td class="num">{"Yes" if has_sch else "No"}</td></tr>
</table>

<h2>Weight Distribution</h2>
<table>
<tr><td>Mean</td><td class="num">{all_w.mean():.4f}</td></tr>
<tr><td>Std</td><td class="num">{all_w.std():.4f}</td></tr>
<tr><td>Min</td><td class="num">{all_w.min():.4f}</td></tr>
<tr><td>Max</td><td class="num">{all_w.max():.4f}</td></tr>
<tr><td>Output std (fwd)</td><td class="num">{out.std():.4f}</td></tr>
<tr><td>Output mean (fwd)</td><td class="num">{out.mean():.4f}</td></tr>
</table>
'''
    
    # ─── Gate / Prediction Summary ───
    skip_alphas = [d['skip_alpha'] for d in layers_data]
    ls_stds = [d['mirror_log_scale_std'] for d in layers_data]
    html += f'''<h2>Gate & Prediction Summary</h2>
<table>
<tr><td>w_pred_scale μ / σ</td><td class="num">{layers_data[0]["w_pred_scale_mean"]:.3f} / {layers_data[0]["w_pred_scale_std"]:.3f}</td></tr>
<tr><td>alpha μ / σ (L0)</td><td class="num">{layers_data[0]["alpha_mean"]:.4f} / {layers_data[0]["alpha_std"]:.4f}</td></tr>
<tr><td>alpha min / max (L0)</td><td class="num">{layers_data[0]["alpha_min"]:.4f} / {layers_data[0]["alpha_max"]:.4f}</td></tr>
<tr><td>alpha deviation from 1 (avg all layers)</td><td class="num">{sum(d["alpha_deviation"] for d in layers_data)/len(layers_data):.4f}</td></tr>
<tr><td>skip_alpha μ (all layers)</td><td class="num">{sum(skip_alphas)/len(skip_alphas):.4f}</td></tr>
<tr><td>var(log_scale) per-layer mean</td><td class="num">{sum(s**2 for s in ls_stds)/len(ls_stds):.6f}</td></tr>
<tr><td>log_skip_alpha μ (L31)</td><td class="num">{layers_data[-1]["log_skip_alpha_mean"]:.4f}</td></tr>
</table>
'''

    if opt_info:
        html += f'''<h2>Optimizer State (from momentum)</h2>
<table>
<tr><td>Mean |grad| (exp_avg)</td><td class="num">{opt_info["mean_abs_grad"]:.6f}</td></tr>
<tr><td>RMS grad (exp_avg)</td><td class="num">{opt_info["rms_grad"]:.6f}</td></tr>
</table>
'''

    # ─── Per-Layer Table ───
    html += '''<h2>Layer-by-Layer Analysis</h2>
<div style="overflow-x:auto;">
<table>
<tr>
<th>L</th>
<th colspan="2">Bind</th>
<th colspan="4">Gates</th>
<th colspan="2">Spectr.</th>
<th colspan="3">Memory</th>
<th>Conv</th>
<th colspan="3">Mirror (log_scale)</th>
<th colspan="2">MLP</th>
</tr>
<tr>
<th></th>
<th>||W_p||</th><th>r_p</th>
<th>b_i</th><th>i_gate</th><th>b_d</th><th>τ</th>
<th>λ_k μ</th><th>λ_k σ</th>
<th>w_q μ</th><th>w_q σ</th><th>w_m2v μ</th>
<th>||conv||</th>
<th>μ</th><th>σ</th><th>sparse%</th>
<th>r_MLP</th><th>||W_up||</th>
</tr>
'''
    
    for d in layers_data:
        tau_str = f'{d["tau"]:.0f}' if d['tau'] < 9999 else '∞'
        html += f'<tr>'
        html += f'<td>{d["idx"]}</td>'
        # Bind
        html += f'<td class="num">{d["bind_proj_norm"]:.2f}</td>'
        html += f'<td class="num">{d["bind_proj_rank"]:.1f}</td>'
        html += f'<td class="num">{d["bind_out_rank"]:.1f}</td>'
        # Gates
        html += f'<td class="num">{d["b_i_val"]:.2f}</td>'
        html += f'<td class="num">{d["i_gate"]:.3f}</td>'
        html += f'<td class="num">{d["b_d_val"]:.2f}</td>'
        html += f'<td class="num">{tau_str}</td>'
        # Spectral
        html += f'<td class="num">{fmt_small(d["lambda_k_mean"])}</td>'
        html += f'<td class="num">{fmt_small(d["lambda_k_std"])}</td>'
        # Memory vectors
        html += f'<td class="num">{d["w_q_mean"]:.3f}</td>'
        html += f'<td class="num">{d["w_q_std"]:.3f}</td>'
        html += f'<td class="num">{d["w_mem2v_mean"]:.3f}</td>'
        # Conv
        html += f'<td class="num">{d["conv_norm"]:.2f}</td>'
        # Mirror log_scale
        html += f'<td class="num">{fmt_small(d["mirror_log_scale_mean"])}</td>'
        html += f'<td class="num">{fmt_small(d["mirror_log_scale_std"])}</td>'
        html += f'<td class="num">{d["mirror_log_scale_sparsity"]:.0f}%</td>'
        # MLP
        html += f'<td class="num">{d["mlp_eff_rank"]:.1f}</td>'
        html += f'<td class="num">{d["mlp_norm"]:.1f}</td>'
        html += f'</tr>\n'
    
    html += '''</table>
</div>
'''
    
    # ─── Mirror Detail ───
    html += '''<h2>Mirror Detail (per layer)</h2>
<div style="overflow-x:auto;">
<table>
<tr>
<th>L</th>
<th>||W_p||</th><th>||W_o||</th>
<th>β</th><th>α_dev</th>
<th>skip_α</th>
<th>w_temp σ</th><th>w_glob σ</th>
<th>w_sym_u σ</th><th>w_sym_v σ</th>
<th>log_scl [min,max]</th>
<th>dvar_mod</th><th>grad_mod</th>
</tr>
'''
    for d in layers_data:
        html += f'<tr>'
        html += f'<td>{d["idx"]}</td>'
        html += f'<td class="num">{d["mirror_proj_norm"]:.2f}</td>'
        html += f'<td class="num">{d["mirror_out_norm"]:.2f}</td>'
        html += f'<td class="num">{d["gate_beta"]:.4f}</td>'
        html += f'<td class="num">{d["alpha_deviation"]:.4f}</td>'
        html += f'<td class="num">{d["skip_alpha"]:.4f}</td>'
        html += f'<td class="num">{d["mirror_w_temp_std"]:.4f}</td>'
        html += f'<td class="num">{d["mirror_w_global_std"]:.4f}</td>'
        html += f'<td class="num">{d["mirror_w_sym_u_std"]:.4f}</td>'
        html += f'<td class="num">{d["mirror_w_sym_v_std"]:.4f}</td>'
        html += f'<td class="num">[{fmt_small(d["mirror_log_scale_min"])}, {fmt_small(d["mirror_log_scale_max"])}]</td>'
        html += f'<td class="num">{d["log_dvar_mod_scale_mean"]:.3f}</td>'
        html += f'<td class="num">{d["log_grad_mod_scale_mean"]:.3f}</td>'
        html += f'</tr>\n'
    html += '''</table>
'''

    # ─── Adaptive Controller State ───
    from core import AdaptiveController
    expl, diff = AdaptiveController.stats(model.layers)
    html += f'''<h2>Adaptive Controller</h2>
<table>
<tr><td>Exploration</td><td class="num">{expl:.4f}</td><td>|mirror| / 0.3 — how much correction applied</td></tr>
<tr><td>Differentiation</td><td class="num">{diff:.6f}</td><td>var(log_scale) / 0.1 — how specialized per-dim</td></tr>
<tr><td>b_d (τ bias)</td><td class="num">{AdaptiveController.b_d(model.layers):.3f}</td></tr>
<tr><td>b_i (i_gate bias)</td><td class="num">{AdaptiveController.b_i(model.layers):.3f}</td></tr>
<tr><td>w_mem2v_scale</td><td class="num">{AdaptiveController.w_mem2v_scale(model.layers):.4f}</td></tr>
<tr><td>EMA α</td><td class="num">{AdaptiveController.ema_alpha(model.layers):.4f}</td></tr>
<tr><td>Noise scale</td><td class="num">{AdaptiveController.noise_scale(model.layers):.6f}</td></tr>
</table>
'''

    # ─── Layer Summary ───
    html += '''<h2>Layer Summary</h2>
<table>
<tr><th>Metric</th><th>Mean</th><th>Std</th><th>Min (layer)</th><th>Max (layer)</th></tr>
'''
    metrics = [
        ('Bind proj rank', [d['bind_proj_rank'] for d in layers_data], '{:.1f}'),
        ('MLP eff rank', [d['mlp_eff_rank'] for d in layers_data], '{:.1f}'),
        ('β (gate)', [d['gate_beta'] for d in layers_data], '{:.4f}'),
        ('α deviation from 1', [d['alpha_deviation'] for d in layers_data], '{:.4f}'),
        ('skip_alpha', [d['skip_alpha'] for d in layers_data], '{:.4f}'),
        ('w_pred_scale μ', [d['w_pred_scale_mean'] for d in layers_data], '{:.4f}'),
        ('i_gate', [d['i_gate'] for d in layers_data], '{:.4f}'),
        ('τ (decay steps)', [d['tau'] for d in layers_data], '{:.0f}'),
        ('log_scale σ', [d['mirror_log_scale_std'] for d in layers_data], '{:.4f}'),
        ('log_dvar_mod_scale', [d['log_dvar_mod_scale_mean'] for d in layers_data], '{:.3f}'),
        ('log_grad_mod_scale', [d['log_grad_mod_scale_mean'] for d in layers_data], '{:.3f}'),
        ('Conv ||W||', [d['conv_norm'] for d in layers_data], '{:.2f}'),
        ('MLP ||W_up||', [d['mlp_norm'] for d in layers_data], '{:.1f}'),
    ]
    for name, vals, ff in metrics:
        mean_v = sum(vals) / len(vals)
        std_v = (sum((v - mean_v) ** 2 for v in vals) / len(vals)) ** 0.5
        min_v = min(vals)
        max_v = max(vals)
        min_i = vals.index(min_v)
        max_i = vals.index(max_v)
        html += f'<tr><td>{name}</td><td class="num">{ff.format(mean_v)}</td>'
        html += f'<td class="num">{ff.format(std_v)}</td>'
        html += f'<td class="num">{ff.format(min_v)} (L{min_i})</td>'
        html += f'<td class="num">{ff.format(max_v)} (L{max_i})</td></tr>\n'
    
    html += '''</table>
'''

    # ─── GroupedMLP Detail ───
    html += '''<h2>GroupedMLP Analysis (per-group eff_rank)</h2>
<div style="overflow-x:auto;">
<table>
<tr>
<th>L</th><th>mean</th><th>std</th><th>min</th><th>max</th>
<th>dead (r<50%)</th>
<th>groups with rank < 50</th>
</tr>
'''
    for d in layers_data:
        ers = d['mlp_eff_ranks']
        mean_r = d['mlp_eff_rank']
        std_r = (sum((v - mean_r) ** 2 for v in ers) / len(ers)) ** 0.5
        low = [f'{i}({r:.0f})' for i, r in enumerate(ers) if r < 0.5 * mean_r]
        low_str = ', '.join(low[:5])
        if len(low) > 5:
            low_str += f' ... (+{len(low)-5})'
        html += f'<tr>'
        html += f'<td>{d["idx"]}</td>'
        html += f'<td class="num">{d["mlp_eff_rank"]:.1f}</td>'
        html += f'<td class="num">{std_r:.1f}</td>'
        html += f'<td class="num">{d["mlp_min_rank"]:.1f}</td>'
        html += f'<td class="num">{d["mlp_max_rank"]:.1f}</td>'
        html += f'<td class="num">{d["mlp_dead_groups"]}</td>'
        html += f'<td style="font-size:0.8em;">{low_str if low_str else "—"}</td>'
        html += f'</tr>\n'
    html += '''</table>
</div>
'''

    # ─── Missing keys ───
    if missing:
        html += f'''<h2>Missing Keys (not loaded)</h2>
<pre>{"\\n".join(missing)}</pre>
'''

    html += '''
<p style="color: #8b949e; font-size: 0.8em; margin-top: 3em;">Generated by analyze_checkpoint.py</p>
</body>
</html>'''
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'  Report saved to {html_path}')
    
    if append_log:
        append_to_log(ckpt_path, layers_data, model, cfg, step, best_val, all_w, out)
    
    return html_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/analyze_checkpoint.py <checkpoint.pt> [--log]')
        print('  --log : append summary to docs/TRAINING_LOG.md')
        sys.exit(1)
    ckpt_path = sys.argv[1]
    append_log = '--log' in sys.argv
    if not os.path.isfile(ckpt_path):
        print(f'File not found: {ckpt_path}')
        sys.exit(1)
    generate_report(ckpt_path, append_log=append_log)
