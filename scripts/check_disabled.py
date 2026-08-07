import sys
sys.path.insert(0, '.')
from core import WideBindConfig

# Base config matching best.pt
cfg = WideBindConfig()
cfg.D = 2560
cfg.n_layers = 24
cfg.bind_K = 32
cfg.vocab = 65536
cfg.mlp_groups = 32
cfg.seq_len = 128
cfg.lambda_d_enabled = False
cfg.head_mode = 'sigmoid_coded'
cfg.head_normalize = True
cfg.bind_twist_mode = 'trajectory_spiral'
cfg.bind_traj_dims = 3
cfg.hybrid_alpha_max = 0.7
cfg.hybrid_alpha_min = 0.3
cfg.w_pred_scale_init = 3.0
cfg.private_mem = True
cfg.meta_trust = True
cfg.collective_layer = True
cfg.collective_layer_idx = None
cfg.uncert_theta = 0.5
cfg.uncert_kappa = 3.0
cfg.contra_thresh = -0.1
cfg.contra_gain = 6.0
cfg.maturity_thresh = 0.12

print('=== Currently disabled features ===')
print()

# 1. surprisal_weight - focus on informative tokens
print('1. surprisal_weight = 0.0 -> 0.3')
print('   Focus loss on surprising tokens (high CE).')
print('   Speeds up learning on predictable patterns.')
cfg.surprisal_weight = 0.3

# 2. branch_balance_weight - balance conv/bind/mirror contributions
print('2. branch_balance_weight = 0.0 -> 0.1')
print('   Equalize log-variance of conv/bind/mirror branches.')
print('   Prevents one branch from dominating.')
cfg.branch_balance_weight = 0.1

# 3. aux_mirror_weight - auxiliary world model loss
print('3. aux_mirror_weight = 0.0 -> 0.05')
print('   Auxiliary cosine loss for mirror predictions.')
print('   Improves mirror self-consistency.')
cfg.aux_mirror_weight = 0.05

# 4. bind_twist_gate - adaptive aperture
print('4. bind_twist_gate = False -> True')
print('   Per-token adaptive aperture in bind.')
print('   Allows dynamic cross-mixing strength.')
cfg.bind_twist_gate = True

# 5. collective_read_out - enable concept read
print('5. collective_read_out = False -> True')
print('   Enable reading from collective concept bank.')
print('   Adds concept signal to residual.')
cfg.collective_read_out = True

# 6. gradient_checkpointing - save VRAM
print('6. gradient_checkpointing = False -> True')
print('   Trade compute for VRAM (essential for T4 16GB).')
cfg.gradient_checkpointing = True

print()
print('=== Final enabled features ===')
print(f'surprisal_weight: {cfg.surprisal_weight}')
print(f'branch_balance_weight: {cfg.branch_balance_weight}')
print(f'aux_mirror_weight: {cfg.aux_mirror_weight}')
print(f'bind_twist_gate: {cfg.bind_twist_gate}')
print(f'collective_read_out: {cfg.collective_read_out}')
print(f'gradient_checkpointing: {cfg.gradient_checkpointing}')
