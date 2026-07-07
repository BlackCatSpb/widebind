"""
Generate HTML analysis report from a WideBind checkpoint.
Usage: python analyze_checkpoint.py <checkpoint.pt>
"""

import os, sys, math, json
import torch
import torch.nn.functional as F
from torch.serialization import add_safe_globals
from config import WideBindConfig
from core import WideBindStack

add_safe_globals([WideBindConfig])

def fmt(x):
    if isinstance(x, float):
        return f'{x:.4f}'
    return str(x)

def generate_report(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    cfg = ckpt['cfg']
    step = ckpt['step']
    best_val = ckpt.get('best_val_loss', 'N/A')

    model = WideBindStack(cfg)
    model.eval()
    model.load_state_dict(ckpt['model'], strict=False)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Weight stats per module group
    mlp_norms, bind_norms, mem_norms, gate_norms, spec_norms = [], [], [], [], []
    for name, p in model.named_parameters():
        n = p.data.norm().item()
        s = p.data.std().item()
        if 'mlp' in name:
            mlp_norms.append((name, n, s, p.numel()))
        elif 'W_proj' in name or 'W_out' in name:
            bind_norms.append((name, n, s, p.numel()))
        elif 'w_i' in name or 'w_d' in name or 'w_q' in name or 'b_' in name:
            gate_norms.append((name, n, s, p.numel()))
        elif 'lambda_k' in name:
            spec_norms.append((name, n, s, p.numel()))
        elif 'mirror' in name:
            mem_norms.append((name, n, s, p.numel()))

    all_w = torch.cat([p.data.flatten() for p in model.parameters()])

    # Forward pass to get output stats
    torch.manual_seed(42)
    x = torch.randint(0, min(cfg.vocab, 1000), (1, 16))
    h = model.embed_tokens(x)
    out, state = model(h)

    # Layer eff_rank
    layer_stats = []
    for i, layer in enumerate(model.layers):
        if hasattr(layer.mlp, 'W_up'):
            # GroupedMLP: per-group eff_rank, then average
            G = layer.mlp.G
            eff_ranks = []
            norms = []
            for g in range(G):
                w = layer.mlp.W_up[g].float()
                s = torch.linalg.svdvals(w)
                eff_r = (s**2).sum() / s.max()**2
                eff_ranks.append(eff_r.item())
                norms.append(w.norm().item())
            eff_rank_mlp = sum(eff_ranks) / G
            mlp_norm = sum(norms) / G
            mlp_std = layer.mlp.W_up.std().item()
        else:
            w = layer.mlp_up.weight.float()
            s = torch.linalg.svdvals(w)
            eff_rank_mlp = (s**2).sum() / s.max()**2
            mlp_norm = w.norm().item()
            mlp_std = w.std().item()

        layer_stats.append({
            'idx': i,
            'eff_rank_mlp': eff_rank_mlp,
            'mlp_norm': mlp_norm,
            'mlp_std': mlp_std,
            'bind_norm': layer.W_proj.norm().item(),
            'bind_eff_rank': (lambda s=torch.linalg.svdvals(layer.W_proj.float()): (s**2).sum() / s.max()**2)().item(),
            'log_scale_mean': layer.mirror.log_scale.data.mean().item(),
            'log_scale_std': layer.mirror.log_scale.data.std().item(),
        })

    # Gradient stats from optimizer
    grad_info = {}
    if 'optimizer' in ckpt:
        opt = ckpt['optimizer']
        if 'state' in opt:
            g_means, g_vars = [], []
            for pid, st in opt['state'].items():
                if 'exp_avg' in st:
                    g = st['exp_avg']
                    g_means.append(g.abs().mean().item())
                    g_vars.append((g**2).mean().item())
            if g_means:
                grad_info = {
                    'mean_abs_grad': sum(g_means)/len(g_means),
                    'rms_grad': math.sqrt(sum(g_vars)/len(g_vars)),
                }

    # Build HTML
    base = os.path.splitext(ckpt_path)[0]
    html_path = base + '_report.html'

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>WideBind Report — step {step}</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 960px; margin: 0 auto; padding: 2em; background: #0d1117; color: #e6edf3; line-height: 1.5; }}
h1 {{ color: #58a6ff; }}
h2 {{ color: #79c0ff; border-bottom: 1px solid #30363d; padding-bottom: 0.2em; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #30363d; padding: 4px 10px; text-align: left; }}
th {{ background: #161b22; color: #58a6ff; }}
tr:nth-child(even) {{ background: #0d1117; }}
tr:nth-child(odd) {{ background: #161b22; }}
code {{ background: #21262d; padding: 1px 5px; border-radius: 3px; color: #f0883e; }}
.num {{ color: #79c0ff; }}
</style>
</head>
<body>
<h1>WideBind — Step {step} Report</h1>
<p>checkpoint: {os.path.basename(ckpt_path)} | best_val_loss: {fmt(best_val)}</p>

<h2>Overview</h2>
<table>
<tr><td>Params</td><td class="num">{total:,} ({total/1e6:.2f}M)</td></tr>
<tr><td>Trainable</td><td class="num">{trainable:,}</td></tr>
<tr><td>D / K / bottleneck</td><td class="num">{cfg.D} / {cfg.bind_K} / {cfg.bottleneck}</td></tr>
<tr><td>MLP groups / expand</td><td class="num">{getattr(cfg,'mlp_groups','?')} / {getattr(cfg,'mlp_expand','?')}×</td></tr>
<tr><td>Layers</td><td class="num">{cfg.n_layers}</td></tr>
<tr><td>Weight mean / std</td><td class="num">{all_w.mean():.4f} / {all_w.std():.4f}</td></tr>
<tr><td>Weight min / max</td><td class="num">{all_w.min():.4f} / {all_w.max():.4f}</td></tr>
<tr><td>Output std (forward)</td><td class="num">{out.std():.4f}</td></tr>
'''

    if grad_info:
        html += f'''<tr><td>Mean |grad|</td><td class="num">{grad_info["mean_abs_grad"]:.6f}</td></tr>
<tr><td>RMS grad</td><td class="num">{grad_info["rms_grad"]:.6f}</td></tr>
'''

    html += '''</table>

<h2>Weight Norms by Group</h2>
<table>
<tr><th>Group</th><th>Top-3 params</th><th>||W||</th><th>std</th></tr>
'''

    for group_name, group_data in [
        ('MLP', mlp_norms), ('Bind', bind_norms),
        ('Gates', gate_norms), ('Spectral', spec_norms), ('Mirror', mem_norms)]:
        top = group_data[:3]
        for name, n, s, sz in top:
            short = name.replace('layers.', 'L').replace('.weight', '')
            html += f'<tr><td>{group_name}</td><td><code>{short}</code></td><td class="num">{n:.1f}</td><td class="num">{s:.4f}</td></tr>'

    html += '''</table>

<h2>Layer Analysis</h2>
<table>
<tr><th>L</th><th>eff_r(MLP)</th><th>||W_mlp||</th><th>eff_r(bind)</th><th>||W_bind||</th><th>log_scale μ</th><th>log_scale σ</th></tr>
'''
    for ls in layer_stats:
        html += f'<tr><td>{ls["idx"]}</td>'
        html += f'<td class="num">{ls["eff_rank_mlp"]:.1f}</td>'
        html += f'<td class="num">{ls["mlp_norm"]:.1f}</td>'
        html += f'<td class="num">{ls["bind_eff_rank"]:.1f}</td>'
        html += f'<td class="num">{ls["bind_norm"]:.1f}</td>'
        html += f'<td class="num">{ls["log_scale_mean"]:.4f}</td>'
        html += f'<td class="num">{ls["log_scale_std"]:.4f}</td>'
        html += '</tr>'

    html += '''</table>

<h2>Memory Gates (mean across layers)</h2>
<table>
'''
    for gate_name in ['w_i', 'w_d', 'w_q', 'b_i', 'b_d', 'w_mem2v']:
        vals = [getattr(l, gate_name).data.mean().item() for l in model.layers]
        html += f'<tr><td><code>{gate_name}</code></td><td class="num">{sum(vals)/len(vals):.4f}</td></tr>'

    html += '''</table>

</body>
</html>'''

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'  Report saved to {html_path}')
    return html_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python analyze_checkpoint.py <checkpoint.pt>')
        sys.exit(1)
    generate_report(sys.argv[1])
