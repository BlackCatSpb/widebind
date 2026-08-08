import json

with open('notebooks/colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update markdown
nb['cells'][0]['source'] = [
    "# EVA (Единая Вычислительная Архитектура) Colab Training\n",
    "\n",
    "**Model**: D=2560, 24 layers, 32 experts, TrajectorySpiralBind, SigmoidCodedHead\n",
    "**Features**: Variable Precision Memory, Explicit Reasoning, Adaptive Collective\n",
    "**VRAM target**: T4 (16GB)\n",
    "**Data**: token_stream_*.bin files in Google Drive\n",
    "\n",
    "---"
]

# Update model build cell (index 4)
nb['cells'][4]['source'] = [
    "# @title 4. Build Model (T4-optimised: D=2560, 24 layers)\n",
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
    "    bind_twist_gate=True,\n",
    "    collective_layer=True,\n",
    "    collective_layer_idx=None,\n",
    "    collective_read_out=True,\n",
    "    collective_uncert_theta=0.5,\n",
    "    collective_uncert_kappa=3.0,\n",
    "    collective_contra_thresh=-0.1,\n",
    "    collective_contra_gain=6.0,\n",
    "    collective_maturity_thresh=0.12,\n",
    "    surprisal_weight=0.3,\n",
    "    branch_balance_weight=0.1,\n",
    "    variable_precision=True,\n",
    "    precision_threshold=0.3,\n",
    "    explicit_reasoning=True,\n",
    "    reasoning_max_steps=8,\n",
    ")\n",
    "\n",
    "model = WideBindStack(cfg).to(device)\n",
    "n_params = model.param_count()\n",
    "print(f'Model: {n_params:,} params ({n_params/1e6:.2f}M)')\n",
    "print(f'Head: {cfg.head_mode} (normalize={cfg.head_normalize})')\n",
    "print(f'Bind: {cfg.bind_twist_mode} (gate={cfg.bind_twist_gate})')\n",
    "print(f'Collective: maturity={cfg.collective_maturity_thresh}, read_out={cfg.collective_read_out}')\n",
    "print(f'Variable Precision: {cfg.variable_precision} (threshold={cfg.precision_threshold})')\n",
    "print(f'Explicit Reasoning: {cfg.explicit_reasoning} (max_steps={cfg.reasoning_max_steps})')\n",
    "\n",
    "gc.collect()\n",
    "torch.cuda.empty_cache()\n"
]

with open('notebooks/colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Notebook updated for EVA with Variable Precision + Explicit Reasoning!')
