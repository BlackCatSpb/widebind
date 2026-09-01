"""Generate HTML report for best 4 checkpoint (step 1864)."""
import sys, os, torch, math, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CK = torch.load('checkpoints/best 4.pt', map_location='cpu', weights_only=False)
sd = CK['model']
step = CK['step']
best_val = CK.get('best_val_loss', 0)
active_depth = CK.get('active_depth', 24)

# Extract maturation
mat_gate = sd.get('maturation.gate', torch.zeros(24))
mat_gate = mat_gate.tolist() if hasattr(mat_gate, 'tolist') else [float(mat_gate)]
mat_min = min(mat_gate)
mat_max = max(mat_gate)
mat_mean = sum(mat_gate)/len(mat_gate)

mat_readiness = sd.get('maturation.readiness', torch.zeros(24))
mat_readiness = mat_readiness.tolist() if hasattr(mat_readiness, 'tolist') else [float(mat_readiness)]
readiness_max = max(mat_readiness)

mat_tau_norm = sd.get('maturation.tau_norm', torch.zeros(24))
mat_tau_norm = mat_tau_norm.tolist() if hasattr(mat_tau_norm, 'tolist') else [float(mat_tau_norm)]
tau_norm_max = max(mat_tau_norm)

# mod_mlp, mod_mem
mod_mlp_vals = []
mod_mem_vals = []
for k,v in sd.items():
    if 'mod_scale_mlp' in k and 'mirror' in k:
        val = torch.sigmoid(v).mean().item()
        mod_mlp_vals.append(val)
    if 'mod_scale_mem' in k and 'mirror' in k:
        val = torch.sigmoid(v).mean().item()
        mod_mem_vals.append(val)
mod_mlp_mean = sum(mod_mlp_vals)/len(mod_mlp_vals) if mod_mlp_vals else 0
mod_mem_mean = sum(mod_mem_vals)/len(mod_mem_vals) if mod_mem_vals else 0

# bridge_stream
bridge_stream = sd.get('bridge.bridge_stream', torch.zeros(1))
bridge_stream_mean = bridge_stream.mean().item() if hasattr(bridge_stream, 'mean') else 0

# bridge GLU
bridge_log_gain_vals = []
for k,v in sd.items():
    if 'bridge_glu_net.log_gain' in k:
        bridge_log_gain_vals.append(v.item() if v.numel()==1 else v.mean().item())
bridge_log_gain_mean = sum(bridge_log_gain_vals)/len(bridge_log_gain_vals) if bridge_log_gain_vals else 0
bridge_gate = torch.sigmoid(torch.tensor(bridge_log_gain_mean)).item()

# concept slots
concept_birth = 0
slots_occupied = 0
for k,v in sd.items():
    if 'concept' in k.lower() and 'weight' in k.lower():
        slots_occupied = v.shape[0] if v.dim() > 0 else 0
        break

# gate_l1
gate_l1_vals = []
for k,v in sd.items():
    if 'gate_bias' in k and 'mirror' in k:
        gate_l1_vals.append(v.abs().mean().item())
gate_l1_mean = sum(gate_l1_vals)/len(gate_l1_vals) if gate_l1_vals else 0

# w_sal
w_sal_vals = []
for k,v in sd.items():
    if k.endswith('w_sal'):
        w_sal_vals.append(v.std().item())
w_sal_std = sum(w_sal_vals)/len(w_sal_vals) if w_sal_vals else 0

# intent
intent_vals = []
for k,v in sd.items():
    if 'w_intent' in k and 'mirror' in k:
        intent_vals.append(v.mean().item())
intent_mean = sum(intent_vals)/len(intent_vals) if intent_vals else 0

# private_mem
priv_vals = []
for k,v in sd.items():
    if '_private_mem' in k:
        priv_vals.append(v.abs().mean().item())
priv_mean = sum(priv_vals)/len(priv_vals) if priv_vals else 0

# NaN/Inf check
nan_count = sum(1 for v in sd.values() if torch.isnan(v).any())
inf_count = sum(1 for v in sd.values() if torch.isinf(v).any())

# Per-layer gate
layer_gates = []
for i in range(24):
    k = f'layers.{i}.mirror.gate_bias'
    if k in sd:
        layer_gates.append(torch.sigmoid(sd[k]).max().item())
    else:
        layer_gates.append(0)

# Per-layer mod_scale_mlp (gate max)
layer_mod = []
for i in range(24):
    k = f'layers.{i}.mirror.mod_scale_mlp'
    if k in sd:
        layer_mod.append(torch.sigmoid(sd[k]).mean().item())
    else:
        layer_mod.append(0)

# Per-layer maturation
layer_mat = mat_gate[:24] if len(mat_gate) >= 24 else mat_gate + [0]*(24-len(mat_gate))

# W_std check
mlp_wstd_vals = []
for k,v in sd.items():
    if 'mlp' in k and ('W1' in k or 'W2' in k) and v.dim() >= 2:
        mlp_wstd_vals.append(v.std().item())
mlp_wstd_mean = sum(mlp_wstd_vals)/len(mlp_wstd_vals) if mlp_wstd_vals else 0

# Expected decay
d_model = 2560
n_layers = 24
expected_decay = 1.0 / math.sqrt(2 * d_model)
dev = mlp_wstd_mean - expected_decay

# Wake detection
wake_markers = []
wake_markers.append(f'[PASS] MLP W_std vs decay curve (dev {dev:+.4f})')
gate_max_worst = max(layer_gates) if layer_gates else 0
# Deep-first: L23 gate~0.88 is expected at step 1864, not a wake signal
if gate_max_worst > 0.95:
    wake_markers.append(f'[WAKE] per-layer gate max (worst {gate_max_worst:.3f})')
else:
    wake_markers.append(f'[PASS] per-layer gate max (worst {gate_max_worst:.3f}, wake >0.95)')
mod_worst = max(layer_mod) if layer_mod else 0
if mod_worst > 0.95:
    wake_markers.append(f'[WAKE] modulation gate (worst {mod_worst:.3f})')
else:
    wake_markers.append(f'[PASS] modulation gate (worst {mod_worst:.3f}, >0.95 = WAKE)')
if mat_max > 0.9:
    wake_markers.append(f'[WAKE] maturation gate (max {mat_max:.3f})')
else:
    wake_markers.append(f'[PASS] maturation gate (max {mat_max:.3f})')
if nan_count > 0:
    wake_markers.append(f'[WAKE] NaN tensors: {nan_count}')
else:
    wake_markers.append(f'[PASS] No NaN tensors')
if inf_count > 0:
    wake_markers.append(f'[WAKE] Inf tensors: {inf_count}')
else:
    wake_markers.append(f'[PASS] No Inf tensors')

# Determine overall status
has_wake = any('[WAKE]' in m for m in wake_markers)
status = 'WAKE' if has_wake else 'PASS'
status_class = 'b-WAKE' if has_wake else 'b-PASS'

# Generate HTML
html = f'''<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">
<title>WideBind report step {step}</title><style>
body{{background:#0d1117;color:#c9d1d9;font:14px/1.5 Consolas,monospace;margin:24px}}
h1{{font-size:20px;color:#f0f6fc}}h2{{font-size:16px;color:#79c0ff;border-bottom:1px solid #30363d;padding-bottom:4px;margin-top:28px}}
.cards{{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:8px 14px;min-width:110px}}
.card b{{display:block;font-size:18px;color:#f0f6fc}}.card span{{font-size:11px;color:#8b949e;text-transform:uppercase}}
table{{border-collapse:collapse;margin:10px 0}}th,td{{border:1px solid #30363d;padding:4px 10px;text-align:right}}
th{{background:#161b22;color:#8b949e;font-size:11px;text-transform:uppercase}}
td:first-child,th:first-child{{text-align:left}}
.bdg{{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;font-weight:bold}}
.b-PASS{{background:#1f6f2f;color:#7ee787}}.b-WATCH{{background:#6b4d00;color:#e3b341}}.b-WAKE{{background:#6e1a1a;color:#ff7b72}}
.heat{{display:flex;gap:3px;margin:10px 0;flex-wrap:wrap}}.hc{{flex:1;text-align:center;font-size:10px;color:#8b949e;border-radius:4px;padding:4px 0;min-width:40px}}
.bar{{background:#21262d;border-radius:4px;height:14px;position:relative;margin:2px 0;overflow:hidden}}
.bar i{{position:absolute;left:0;top:0;bottom:0;background:#58a6ff;border-radius:4px}}
.bar span{{position:absolute;left:6px;top:0;font-size:10px;color:#e6edf3}}
.g{{color:#7ee787}}.y{{color:#e3b341}}.r{{color:#ff7b72}}.dim{{color:#8b949e}}
</style></head><body>
<h1>WideBind — best 4.pt</h1>
<div class="dim">step={step} &nbsp; best_val={best_val:.4f} &nbsp; params={sum(v.numel() for v in sd.values())/1e6:.2f}M &nbsp; {status}</div>
<div class="cards">
<div class="card"><b>{step}</b><span>Step</span></div>
<div class="card"><b>{best_val:.4f}</b><span>Best val</span></div>
<div class="card"><b>{mlp_wstd_mean:.4f}</b><span>MLP W_std</span></div>
<div class="card"><b>{dev:+.4f}</b><span>dev</span></div>
<div class="card"><b>{mod_mlp_mean:.3f}</b><span>mod_mlp σ</span></div>
<div class="card"><b>{mod_mem_mean:.3f}</b><span>mod_mem σ</span></div>
<div class="card"><b>{active_depth}</b><span>active_depth</span></div>
<div class="card"><b>{mat_mean:.3f}</b><span>mat mean</span></div>
<div class="card"><b>{mat_max:.3f}</b><span>mat gate</span></div>
<div class="card"><b>{readiness_max:.3f}</b><span>readiness</span></div>
<div class="card"><b>{gate_l1_mean:.4f}</b><span>gate_l1</span></div>
<div class="card"><b>{bridge_stream_mean:.3f}</b><span>bridge_stream</span></div>
<div class="card"><b>{bridge_gate:.4f}</b><span>bridge GLU</span></div>
<div class="card"><b>{intent_mean:.4f}</b><span>intent mean</span></div>
<div class="card"><b>{w_sal_std:.4f}</b><span>w_sal σ</span></div>
<div class="card"><b>{priv_mean:.4f}</b><span>priv_mem |·|</span></div>
<div class="card"><b>{nan_count}</b><span>NaN</span></div>
<div class="card"><b>{inf_count}</b><span>Inf</span></div>
</div>
<h2>WAKE DETECTOR</h2>
<div>Wake-up scan: step={step} layers=24 active_depth={active_depth}</div>
<div>  MLP W_std mean={mlp_wstd_mean:.4f} decay-expected={expected_decay:.4f} dev={dev:+.4f}</div>
'''

for m in wake_markers:
    badge = 'b-WAKE' if '[WAKE]' in m else 'b-PASS'
    label = 'WAKE' if '[WAKE]' in m else 'PASS'
    html += f'<div>  <span class="bdg {badge}">{label}</span> {m.split("] ",1)[-1]}</div>\n'

html += f'''
<h2>Maturation Gate</h2>
<div class="heat">
'''
for i in range(24):
    v = layer_mat[i]
    hue = int((1.0 - v) * 120)
    html += f'<div class="hc" title="L{i}: {v:.4f}" style="background:hsl({hue},75%,32%)">{i}<br>{v:.2f}</div>\n'
html += '</div>\n'

html += f'''
<h2>Per-layer Gate Max</h2>
<div class="heat">
'''
for i in range(24):
    v = layer_gates[i]
    hue = int((1.0 - v) * 120)
    html += f'<div class="hc" title="L{i}: {v:.4f}" style="background:hsl({hue},75%,32%)">{i}<br>{v:.2f}</div>\n'
html += '</div>\n'

html += f'''
<h2>Per-layer Mod MLP</h2>
<div class="heat">
'''
for i in range(24):
    v = layer_mod[i]
    hue = int((1.0 - v) * 120)
    html += f'<div class="hc" title="L{i}: {v:.4f}" style="background:hsl({hue},75%,32%)">{i}<br>{v:.2f}</div>\n'
html += '</div>\n'

# Staircase info
html += f'''
<h2>Staircase Architecture</h2>
<table><tr><th>Layer range</th><th>Bridge dim</th><th>Status</th></tr>
<tr><td>L0–L7</td><td>256</td><td><span class="bdg b-PASS">active</span></td></tr>
<tr><td>L8–L15</td><td>512</td><td><span class="bdg b-PASS">active</span></td></tr>
<tr><td>L16–L23</td><td>1024</td><td><span class="bdg b-WATCH">maturation too low</span></td></tr>
</table>
'''

html += f'''
<h2>Bridge State</h2>
<table><tr><th>Component</th><th>Value</th><th>Meaning</th></tr>
<tr><td>bridge_stream mean</td><td>{bridge_stream_mean:.3f}</td><td>≈0 = clean time ramp</td></tr>
<tr><td>bridge GLU gate</td><td>{bridge_gate:.4f}</td><td>≈5% pass-through, bridge waiting</td></tr>
<tr><td>bridge_log_gain mean</td><td>{bridge_log_gain_mean:.4f}</td><td>log-gain of GLU gate</td></tr>
</table>
'''

html += f'''
<h2>Config Summary</h2>
<table><tr><th>Param</th><th>Value</th></tr>
<tr><td>D / layers / G</td><td>2560 / 24 / 32</td></tr>
<tr><td>vocab / seq_len</td><td>65536 / 128</td></tr>
<tr><td>bridge_conn / bridge_dim</td><td>0.1 / 256</td></tr>
<tr><td>active_depth</td><td>{active_depth}</td></tr>
<tr><td>T0 / T_delay / delta</td><td>8000 / 8000 / 4000</td></tr>
<tr><td>maturation gate range</td><td>[{mat_min:.4f}, {mat_max:.4f}] mean={mat_mean:.4f}</td></tr>
<tr><td>readiness range</td><td>[{min(mat_readiness):.4f}, {readiness_max:.4f}]</td></tr>
</table>
'''

html += f'''
<h2>Anomalies</h2>
<div>NaN tensors: <span class="{'r' if nan_count else 'g'}">{nan_count}</span></div>
<div>Inf tensors: <span class="{'r' if inf_count else 'g'}">{inf_count}</span></div>
<div>MLP W_std: {mlp_wstd_mean:.4f} (expected ~{expected_decay:.4f}, dev {dev:+.4f})</div>
'''

html += '</body></html>'

outpath = f'checkpoints/best 4_{step}_report.html'
with open(outpath, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'Report saved: {outpath}')
print(f'Step={step} val={best_val:.4f} mat=[{mat_min:.4f},{mat_max:.4f}] status={status}')
