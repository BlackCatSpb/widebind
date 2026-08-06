import json

with open('notebooks/colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update markdown cell (index 0)
nb['cells'][0]['source'] = [
    "# WideBind Colab Training\n",
    "\n",
    "**Model**: D=2560, 24 layers, 32 experts, TrajectorySpiralBind, SigmoidCodedHead\n",
    "**Features**: Adaptive maturity collective, adaptive mirror temperature, private memory\n",
    "**VRAM target**: T4 (16GB) — fits comfortably (~3-4GB)\n",
    "**Data**: token_stream_*.bin files in Google Drive\n",
    "\n",
    "---"
]

# Update model build cell (index 4)
nb['cells'][4]['source'] = [
    "# @title 4. Build Model (T4-optimised: D=2560, 24 layers, sigmoid head)\n",
    "gc.collect()\n",
    "torch.cuda.empty_cache()\n",
    "\n",
    "cfg = WideBindConfig(\n",
    "    D=2560,\n",
    "    n_layers=24,\n",
    "    bind_K=32,\n",
    "    vocab=65536,\n",
    "    mlp_groups=32,\n",
    "    mlp_expand=4,\n",
    "    seq_len=128,\n",
    "    lr=3e-4,\n",
    "    max_steps=300000,\n",
    "    warmup_steps=101,\n",
    "    log_interval=55,\n",
    "    eval_interval=233,\n",
    "    save_interval=987,\n",
    "    scheduler='mirror',\n",
    "    private_mem=True,\n",
    "    expert_asymmetry=True,\n",
    "    meta_trust=True,\n",
    "    data_dir=DATA_DIR,\n",
    "    save_dir=SAVE_DIR,\n",
    "    log_dir=LOG_DIR,\n",
    "    grad_clip=0.5,\n",
    "    conv_kernel=48,\n",
    "    gradient_checkpointing=False,\n",
    "    head_mode='sigmoid_coded',\n",
    "    head_normalize=True,\n",
    "    bind_twist_mode='trajectory_spiral',\n",
    "    bind_traj_dims=3,\n",
    "    hybrid_alpha_max=0.7,\n",
    "    hybrid_alpha_min=0.3,\n",
    "    w_pred_scale_init=3.0,\n",
    "    collective_layer=True,\n",
    "    collective_layer_idx=None,\n",
    "    collective_read_out=False,\n",
    "    collective_uncert_theta=0.5,\n",
    "    collective_uncert_kappa=3.0,\n",
    "    collective_contra_thresh=-0.1,\n",
    "    collective_contra_gain=6.0,\n",
    "    collective_maturity_thresh=0.12,\n",
    ")\n",
    "\n",
    "model = WideBindStack(cfg).to(device)\n",
    "n_params = model.param_count()\n",
    "print(f'Model: {n_params:,} params ({n_params/1e6:.2f}M)')\n",
    "print(f'Head: {cfg.head_mode} (normalize={cfg.head_normalize})')\n",
    "print(f'Bind: {cfg.bind_twist_mode} (S={cfg.bind_traj_dims}, K={cfg.bind_K})')\n",
    "\n",
    "gc.collect()\n",
    "torch.cuda.empty_cache()\n"
]

# Update training loop cell - fix codec-specific metrics
for idx, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if 'pw_diag' in src and 'proto_mean' in src:
            new_src = src.replace(
                "pw_diag = model.lm_head.pred_w.diag().mean().item()\n"
                "                alpha = torch.tanh(model.lm_head.proto) * model.lm_head.codes\n"
                "                proto_mean = alpha[model.lm_head.codes.bool()].mean().item()",
                "head_temp = model.lm_head.log_temp.data.mean().item()"
            )
            new_src = new_src.replace(
                "pw={pw_diag:.3f} proto={proto_mean:.4f} ",
                "head_temp={head_temp:.3f} "
            )
            new_src = new_src.replace(
                "idiff = gvar = ls_var = pw_diag = proto_mean = 0.0",
                "idiff = gvar = ls_var = head_temp = 0.0"
            )
            cell['source'] = [line + '\n' for line in new_src.split('\n')]
            cell['source'][-1] = cell['source'][-1].rstrip('\n') + '\n'
            print(f'Updated training loop cell at index {idx}')
            break

with open('notebooks/colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Notebook updated!')
