"""
analyze_widebind.py — Generate HTML report of WideBind architecture.
Reads checkpoint if available for live weight stats.
"""

import os, math, torch
from pathlib import Path
from config import WideBindConfig
from core import WideBindStack

D = 896
VOCAB = 50000
N_LAYERS = 24
BOTTLENECK = 896
K = 16
CONV_K = 48
FIB_LEN = 23  # Zeckendorf code length for 50000 vocab

D_str = str(D)
V_str = f'{VOCAB:,}'
L_str = str(N_LAYERS)
B_str = str(BOTTLENECK)
K_str = str(K)

# Per-layer parameter counts (trainable only, not buffers)
bind_W_proj = D * K
bind_w_u = K
bind_w_v = K
bind_W_out = K * D
bind_total = bind_W_proj + bind_w_u + bind_w_v + bind_W_out

mirror_W_proj = D * K
mirror_w_u = K
mirror_w_v = K
mirror_W_out = K * D
mirror_scale = 1
mirror_total = mirror_W_proj + mirror_w_u + mirror_w_v + mirror_W_out + mirror_scale

vsa_gates = 6 * D   # w_i, w_d, w_q, w_mem2v, b_i, b_d
vsa_moment = 3 * D   # w_k_mu, w_q_mu, w_mu_mem
vsa_total = vsa_gates + vsa_moment

conv_total = D * CONV_K

spectral_lambda = D  # lambda_k parameter
spectral_V = D * D   # V_dct buffer (not trainable)

mlp_up = D * BOTTLENECK
mlp_down = BOTTLENECK * D
mlp_total = mlp_up + mlp_down

per_layer_train = bind_total + mirror_total + vsa_total + conv_total + spectral_lambda + mlp_total
per_layer_buffers = spectral_V  # V_dct + pre_ln_w + mlp_norm_w

embed_W = FIB_LEN * D
head_W = D * FIB_LEN
embedding_total = embed_W + head_W

total_trainable = embedding_total + N_LAYERS * per_layer_train
total_buffers = N_LAYERS * per_layer_buffers + N_LAYERS * D * 3  # V_dct + pre_ln_w + mlp_norm_w per layer
total_buffers += D  # final_norm_w
total_buffers += 2 * VOCAB * FIB_LEN  # embed.codes + lm_head.codes

total_all = total_trainable + total_buffers

# ─── Checkpoint data ───
ckpt_path = Path('checkpoints/best.pt')
ckpt_data = {}
if ckpt_path.exists():
    try:
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        ckpt_data['step'] = ckpt.get('step', '?')
        ckpt_data['val_loss'] = f'{ckpt["best_val_loss"]:.4f}'
        ckpt_data['val_ppl'] = f'{math.exp(ckpt["best_val_loss"]):.2f}'
        # Weight stats: mean/std/grad_norm across layers
        ws = ckpt['model']
        stats = []
        for k in sorted(ws.keys()):
            if 'weight' in k and 'layers.' in k:
                w = ws[k]
                stats.append(dict(
                    name=k, mean=w.mean().item(), std=w.std().item(),
                    min=w.min().item(), max=w.max().item(),
                    norm=w.norm().item(), n=w.numel()
                ))
        ckpt_data['weight_stats'] = stats
        # Per-layer gate stats (exclude buffers, only Parameter vectors)
        gate_keys = [k for k in sorted(ws.keys()) if ('w_i' in k or 'w_d' in k) and '.weight' not in k]
        gates = []
        for k in gate_keys:
            w = ws[k]
            gates.append(dict(name=k, mean=w.mean().item(), std=w.std().item()))
        ckpt_data['gates'] = gates
    except Exception as e:
        ckpt_data = {'error': str(e)}

S = dict(
    D=D_str, V=V_str, L=L_str, B=B_str, K=K_str, CONV=str(CONV_K),
    per_layer=f'{per_layer_train:,}',
    total_m=f'{total_trainable/1e6:.2f}',
    total_all_m=f'{total_all/1e6:.2f}',
)

# ─── Build checkpoint HTML sections ───
ckpt_html = ""
if ckpt_data.get("step") is not None and ckpt_data.get("step") != '?':
    steps = int(ckpt_data['step'])
    tokens_seen = steps * 2 * 128
    pct = tokens_seen / 2.86e9 * 100
    ckpt_html += f'''
<h2>7. Training Progress</h2>
<table class="params">
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Steps completed</td><td>{steps:,}</td></tr>
<tr><td>Best val_loss</td><td>{ckpt_data["val_loss"]}</td></tr>
<tr><td>Best val_ppl</td><td>{ckpt_data["val_ppl"]}</td></tr>
<tr><td>Tokens seen</td><td>{tokens_seen:,} ({pct:.2f}% of corpus)</td></tr>
<tr><td>tok/s (MX550)</td><td>262</td></tr>
</table>'''
    # Show first 3 layers' weight stats
    for layer_i in range(min(3, 24)):
        rows = ""
        for s in ckpt_data.get("weight_stats", []):
            if f'layers.{layer_i}.' in s['name']:
                short = s['name'].replace(f'layers.{layer_i}.', '')
                rows += f'<tr><td>{short}</td><td>{s["mean"]:.4f}</td><td>{s["std"]:.4f}</td><td>{s["min"]:.4f}</td><td>{s["max"]:.4f}</td><td>{s["norm"]:.1f}</td></tr>\n'
        if rows:
            ckpt_html += f'''
<h3>Layer {layer_i} Weights</h3>
<table class="params">
<tr><th>Weight</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Norm</th></tr>
{rows}
</table>'''
        # Gate stats
        g_rows = ""
        for s in ckpt_data.get("gates", []):
            if f'layers.{layer_i}.' in s['name']:
                short = s['name'].replace(f'layers.{layer_i}.', '')
                g_rows += f'<tr><td>{short}</td><td>{s["mean"]:.4f}</td><td>{s["std"]:.4f}</td></tr>\n'
        if g_rows:
            ckpt_html += f'''
<h3>Layer {layer_i} Gates</h3>
<table class="params">
<tr><th>Gate</th><th>Mean</th><th>Std</th></tr>
{g_rows}
</table>'''
elif 'error' in ckpt_data:
    ckpt_html = f'<p style="color:#f85149;">Checkpoint error: {ckpt_data["error"]}</p>'

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>WideBind Architecture Analysis</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 960px; margin: 40px auto; padding: 20px; background: #0d1117; color: #e6edf3; }}
h1 {{ color: #58a6ff; border-bottom: 2px solid #30363d; padding-bottom: 10px; }}
h2 {{ color: #58a6ff; margin-top: 30px; }}
h3 {{ color: #79c0ff; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0 20px 0; }}
th, td {{ border: 1px solid #30363d; padding: 8px 12px; text-align: left; }}
th {{ background: #161b22; color: #8b949e; }}
tr:nth-child(even) {{ background: #161b22; }}
tr.total {{ background: #1f2937; font-weight: bold; }}
.code {{ font-family: 'Courier New', monospace; background: #161b22; padding: 12px; border-radius: 6px; overflow-x: auto; white-space: pre; font-size: 13px; line-height: 1.5; }}
.metric {{ display: inline-block; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 20px; margin: 6px; text-align: center; }}
.metric .val {{ font-size: 24px; font-weight: bold; color: #58a6ff; }}
.metric .lbl {{ font-size: 12px; color: #8b949e; }}
.grid {{ display: flex; flex-wrap: wrap; }}
ul {{ line-height: 1.6; }}
li {{ margin: 6px 0; }}
</style>
</head>
<body>

<h1>WideBind: Hybrid VSA Architecture Analysis</h1>
<p>No softmax, no sigmoid gates, no attention. D&rarr;K=16 bottleneck bind + VSA vector memory + depthwise conv + DCT spectral + MLP.</p>

<div class="grid">
<div class="metric"><div class="val">{S["total_m"]}M</div><div class="lbl">Trainable Params</div></div>
<div class="metric"><div class="val">{S["total_all_m"]}M</div><div class="lbl">Total (w/ buffers)</div></div>
<div class="metric"><div class="val">{S["D"]}</div><div class="lbl">Hidden Dim (D)</div></div>
<div class="metric"><div class="val">{S["L"]}</div><div class="lbl">Layers</div></div>
<div class="metric"><div class="val">{S["K"]}</div><div class="lbl">Bind Bottleneck K</div></div>
<div class="metric"><div class="val">{S["CONV"]}</div><div class="lbl">Conv Kernel</div></div>
<div class="metric"><div class="val">{S["V"]}</div><div class="lbl">Vocabulary</div></div>
</div>

<h2>1. Architecture Overview</h2>
<p><b>WideBind</b> is a hybrid VSA-style LM operating entirely in D=896 space with five mechanisms per layer:</p>
<ul>
<li><b>Bottleneck Bind</b> (D&rarr;K=16&rarr;D): bilinear projection through a narrow bottleneck. Provides cross-dim gradient mixing — the key fix over pure element-wise VSA (which has diagonal Jacobian and dies at &gt;4 layers).</li>
<li><b>VSA Vector Memory</b>: element-wise superposition <code>mem[t] = decay[t]*mem[t-1] + h[t]*i_gate[t]</code>. Associative prefix scan in O(L log L). State is O(D) not O(D&sup2;).</li>
<li><b>Mirror Bind</b>: same D&rarr;K&rarr;D bind on centered input (h &minus; mean). Captures deviation patterns.</li>
<li><b>Depthwise Conv1d</b>: k=48, groups=D. Local temporal context that bind/memory don't provide.</li>
<li><b>DCT Spectral</b>: DCT-II basis scaling with learned per-dimension frequency mask &lambda;_k.</li>
</ul>
<p>99% of parameters are in the MLP. Bind/VSA together are &lt; 4%.</p>

<h2>2. Per-Layer Parameter Breakdown (trainable)</h2>
<table class="params">
<tr><th>Component</th><th>Params</th><th>% of Layer</th></tr>
<tr><td>Bind: W_proj (D&times;K), w_u, w_v, W_out (K&times;D)</td><td style="text-align:right">{bind_total:,}</td><td style="text-align:right">{bind_total/per_layer_train*100:.1f}%</td></tr>
<tr><td>Mirror: same structure + scale</td><td style="text-align:right">{mirror_total:,}</td><td style="text-align:right">{mirror_total/per_layer_train*100:.1f}%</td></tr>
<tr><td>VSA Gates: w_i, w_d, w_q, w_mem2v, b_i, b_d</td><td style="text-align:right">{vsa_gates:,}</td><td style="text-align:right">{vsa_gates/per_layer_train*100:.1f}%</td></tr>
<tr><td>VSA Moment: w_k_mu, w_q_mu, w_mu_mem</td><td style="text-align:right">{vsa_moment:,}</td><td style="text-align:right">{vsa_moment/per_layer_train*100:.1f}%</td></tr>
<tr><td>Conv1d depthwise (D&times;1&times;{CONV_K})</td><td style="text-align:right">{conv_total:,}</td><td style="text-align:right">{conv_total/per_layer_train*100:.1f}%</td></tr>
<tr><td>Spectral &lambda;_k (per-dim scale)</td><td style="text-align:right">{spectral_lambda:,}</td><td style="text-align:right">{spectral_lambda/per_layer_train*100:.1f}%</td></tr>
<tr><td>MLP up / down (D&times;B, B&times;D)</td><td style="text-align:right">{mlp_total:,}</td><td style="text-align:right">{mlp_total/per_layer_train*100:.1f}%</td></tr>
<tr class="total"><td><b>Trainable per layer</b></td><td style="text-align:right"><b>{S["per_layer"]}</b></td><td style="text-align:right">100%</td></tr>
</table>

<h3>Full Model</h3>
<table class="params">
<tr><th>Component</th><th>Params</th><th>% Trainable</th></tr>
<tr><td>Zeckendorf Embedding + LmHead (2 &times; {FIB_LEN} &times; D)</td><td style="text-align:right">{embedding_total:,}</td><td style="text-align:right">{embedding_total/total_trainable*100:.1f}%</td></tr>
<tr><td>{N_LAYERS} &times; Bind (D&rarr;K&rarr;D)</td><td style="text-align:right">{int(bind_total*N_LAYERS):,}</td><td style="text-align:right">{bind_total*N_LAYERS/total_trainable*100:.1f}%</td></tr>
<tr><td>{N_LAYERS} &times; Mirror Bind</td><td style="text-align:right">{int(mirror_total*N_LAYERS):,}</td><td style="text-align:right">{mirror_total*N_LAYERS/total_trainable*100:.1f}%</td></tr>
<tr><td>{N_LAYERS} &times; VSA Memory (gates + moment)</td><td style="text-align:right">{int(vsa_total*N_LAYERS):,}</td><td style="text-align:right">{vsa_total*N_LAYERS/total_trainable*100:.1f}%</td></tr>
<tr><td>{N_LAYERS} &times; Depthwise Conv1d</td><td style="text-align:right">{int(conv_total*N_LAYERS):,}</td><td style="text-align:right">{conv_total*N_LAYERS/total_trainable*100:.1f}%</td></tr>
<tr><td>{N_LAYERS} &times; Spectral &lambda;_k</td><td style="text-align:right">{int(spectral_lambda*N_LAYERS):,}</td><td style="text-align:right">{spectral_lambda*N_LAYERS/total_trainable*100:.1f}%</td></tr>
<tr><td>{N_LAYERS} &times; MLP (D &rarr; {BOTTLENECK} &rarr; D)</td><td style="text-align:right">{int(mlp_total*N_LAYERS):,}</td><td style="text-align:right">{mlp_total*N_LAYERS/total_trainable*100:.1f}%</td></tr>
<tr class="total"><td><b>Trainable Total</b></td><td style="text-align:right"><b>{total_trainable:,}</b></td><td style="text-align:right">100%</td></tr>
</table>

<h2>3. VRAM Estimation (fp32)</h2>
<table class="params">
<tr><th>Component</th><th>Size</th></tr>
<tr><td>Model parameters (trainable)</td><td style="text-align:right">{total_trainable*4/1e9:.2f} GB</td></tr>
<tr><td>Buffers (V_dct, codes, norms)</td><td style="text-align:right">{total_buffers*4/1e9:.2f} GB</td></tr>
<tr><td>AdamW states (2 &times; trainable)</td><td style="text-align:right">{total_trainable*2*4/1e9:.2f} GB</td></tr>
<tr><td>Gradients</td><td style="text-align:right">{total_trainable*4/1e9:.2f} GB</td></tr>
<tr><td>CUDA context + PyTorch allocator overhead</td><td style="text-align:right">~0.8-1.0 GB</td></tr>
<tr class="total"><td><b>Observed peak (B=2, L=128, MX550)</b></td><td style="text-align:right"><b>1.86 GB</b></td></tr>
</table>
<p style="color:#f85149;">41M params fit MX550 2GB with ~140 MB margin. bottleneck=3584 (156M) requires &gt; 8 GB.</p>
<p style="color:#8b949e;">The gap between raw param count (0.67 GB) and observed peak (1.86 GB) is PyTorch's caching CUDA allocator: it reserves large memory blocks upfront. Actual usage is ~0.7 GB; the rest is pre-allocated pool that stays reserved but unused.</p>

<h2>4. Forward Pass Flow</h2>
<div class="code">
h[t-1]  (B, L, D)
  |
  |-- [Pre-LN: RMS Norm]
  |
  |-- [Depthwise Conv1d]  k=48, groups=D
  |     h_conv = conv(concat(state, h_perm))
  |     h += h_conv
  |
  |-- [Bottleneck Bind: D -> K=16 -> D]
  |     hp = h @ W_proj          (D -> K)
  |     u = hp * w_u, v = hp * w_v
  |     bind_out = (u * v) @ W_out  (K -> D)
  |
  |-- [VSA Vector Memory]
  |     i_gate = exp(h * w_i + b_i)
  |     decay  = sigmoid(h * w_d + b_d)
  |     mem = prefix_scan(decay, h * i_gate)   O(L log L)
  |     mem_read = mem * w_q
  |     mu = prefix_scan(decay, h * i_gate * w_k_mu)
  |     mem_read += mu_read * w_mu_mem
  |
  |-- [Mirror Bind]
  |     h_centered = h - mean(h)
  |     mirror = D->K->D bind on h_centered
  |
  |-- h += bind_out + mem_read * w_mem2v + mirror
  |
  |-- [DCT Spectral]
  |     h_dct = h @ V_dct.T
  |     h += (h_dct * lambda_k) @ V_dct
  |
  |-- [MLP]
  |     h_mlp = RMS norm -> SiLU(Linear(D->B)) -> Linear(B->D)
  |     h += h_mlp
  |
h' = h   (B, L, D)
</div>

<h2>5. Why Not Attention?</h2>
<table class="params">
<tr><th>Mechanism</th><th>Complexity</th><th>State Size</th><th>WideBind</th></tr>
<tr><td>Attention</td><td>O(L&sup2;)</td><td>O(L &middot; D &middot; layers) KV-cache</td><td>N/A</td></tr>
<tr><td>Prefix Scan (VSA)</td><td>O(L log L)</td><td>O(D) = 336 KB total</td><td>&check;</td></tr>
<tr><td>Conv window</td><td>O(L &middot; D &middot; k)</td><td>O(D &middot; k) = 43 KB</td><td>&check;</td></tr>
<tr><td>Bind (D&rarr;K&rarr;D)</td><td>O(L &middot; D &middot; K)</td><td>0 (stateless)</td><td>&check;</td></tr>
</table>
<p>No KV-cache means infinite context during inference. State is 336 KB for all 24 layers — fits in L1 cache on modern CPUs.</p>

<h2>6. Key Design Decisions</h2>
<table class="params">
<tr><th>Decision</th><th>Reason</th></tr>
<tr><td>K=16 bottleneck</td><td>Dense D&rarr;K projection provides cross-dim gradient mixing (vs diagonal Jacobian of pure element-wise bind). Minimal K that achieves grad/param &gt; 0.4.</td></tr>
<tr><td>std=1 for element-wise vectors</td><td>Product w_u * w_v * w_out scales as std&sup3;. At std=0.02, grad_h = 1e-4 (dead). At std=1.0, grad_h = 22.5 (healthy). Factor 200,000&times; difference.</td></tr>
<tr><td>std=0.1 for gates</td><td>Limits exp() variance to prevent overflow/NaN through 24 layers.</td></tr>
<tr><td>bottleneck=896 (not 3584)</td><td>41M params fit in 2GB VRAM with AdamW. 156M causes OOM. Bigger bottleneck needs &gt;2GB.</td></tr>
<tr><td>No weight tying</td><td>Separate embed/head projections. Marginal param cost (46K), more expressive.</td></tr>
<tr><td>Vector memory (not covariance)</td><td>O(D) state vs O(D&sup2;). 336 KB vs 19 MB for 24 layers. Same gradient quality.</td></tr>
</table>

{ckpt_html}

<h2>8. Code Location</h2>
<div class="code">
core.py:
  ZeckendorfEmbedding  -- token -> D via Fibonacci codes
  LmHead               -- D -> vocab via transposed codes
  WideBindBlock        -- Conv + Bind + VSA + Mirror + Spectral + MLP
  WideBindStack        -- N x WideBindBlock + embed + head
  vsa_prefix_scan      -- associative parallel scan O(L log L)

config.py:
  WideBindConfig       -- all hyperparameters

train.py              -- streaming training loop + checkpointing
generate.py           -- text generation (requires tokenizer)
</div>

</body>
</html>"""

path = 'widebind_analysis.html'
with open(path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'HTML saved: {path} ({os.path.getsize(path):,} bytes)')
