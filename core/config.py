"""WideBind configuration with λ_d hierarchy support."""

from dataclasses import dataclass, field
from .lambda_utils import LambdaConfig

_LAMBDA_OVERRIDE_DOC = (
    "Set to None to use λ_d-derived value (recommended for Experiment 1)."
)


@dataclass
class WideBindConfig:
    D: int = 4096
    n_layers: int = 32
    bind_K: int = 64
    vocab: int = 50000
    seq_len: int = 128
    batch_size: int = 2
    lr: float = 3e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    grad_clip: float = 0.5
    dtype: str = 'float32'

    # ─── λ_d hierarchy ─────────────────────────────────────────
    lambda_d: int = 3            # dimension of generalized golden ratio
    lambda_d_enabled: bool = True  # True = apply λ_d derivation in __post_init__

    # Symmetry constraints
    tie_bind: bool = True  # True = W_out = W_proj^T (autoencoder bind bottleneck)
    tie_mirror_proj: bool = True  # True = mirror W_out = W_proj^T (per-expert K-space AE)

    # Zeckendorf Readout (experimental)
    zeckendorf_readout: bool = False  # True = replace LM head with Zeckendorf tree

    # Embed
    code_dim: int = 32
    code_sparsity: int = 6

    # Mirror
    mirror_k: int = 32
    mirror_k_staircase: bool = True  # True = k_l∈{4,8,16} по третям глубины
    w_pred_scale_init: float = 3.0
    log_scale_init_std: float = 0.05
    mlp_groups: int = 32
    mlp_expand: int = 4
    private_mem: bool = False  # cross-expert private memory bank (meta-cognitive layer)
    signal_entropy_weight: float = 0.001  # entropy regularization on 5 signal weights (0=disabled)
    log_scale_l2_weight: float = 0.01  # L2 on exp(log_scale) > 10 to prevent gradient explosion
    div_weight: float = 0.01  # expert diversity: sum-of-squares push, no /N (0=disabled)
    ranking_weight: float = 0.1  # pairwise order ls_mean by gate_usage (0=disabled)

    # Scheduler (values below will be overridden by λ_d when lambda_d_enabled=True)
    scheduler: str = 'mirror'
    target_var: float = 0.1
    mag_threshold: float = 0.3
    lr_min_ratio: float = 0.05
    max_decay_steps: int = 50000
    var_min_for_lr_decay: float = 0.005

    # AdaptiveController (values below will be overridden by λ_d when lambda_d_enabled=True)
    exploration_threshold: float = 0.25
    differentiation_threshold: float = 0.08
    w_mem2v_scale_min: float = 0.5
    w_mem2v_scale_max: float = 1.0
    ema_alpha_min: float = 0.90
    ema_alpha_max: float = 0.99
    noise_scale_min: float = 0.001
    noise_scale_max: float = 0.05
    delta_var_ema_min: float = 0.80
    delta_var_ema_max: float = 0.99

    # Optimizer
    gate_lr_mult: float = 5.0
    lambda_lr_hierarchy: bool = True  # True = LR mult по степеням λ_d^p

    # w_m2v hierarchy by τ (Proposal IV)
    w_m2v_hierarchy_target: float = 1.0  # m — max target for deep layers
    w_m2v_hierarchy_weight: float = 0.001  # λ_weight for w_m2v regularisation

    # Init stds
    w_d_init_std: float = 0.1
    conv_init_std: float = 0.01

    # Conv
    conv_kernel: int = 48

    # Spectral
    spec_lo: float = 0.5
    spec_hi: float = 1.5
    lambda_sliding: bool = True

    # Memory
    cov_multi_timescale: bool = True
    cov_tau_lo: int = 3
    cov_tau_hi: int = 200

    # Gate sparsity (auxiliary loss weight for expert specialization)
    gate_l1_weight: float = 0.001   # L1 penalty on expert gates (0=disabled)
    # Expert reinforcement: align gate with usefulness prediction
    reinforce_weight: float = 0.01  # MSE(gate, usefulness) aux loss weight
    # Load balancing: encourages uniform expert usage across tokens
    balance_weight: float = 0.01  # CV(gate_usage) aux loss weight (0=disabled)
    # Diversity loss: decorrelate per-group MLP outputs
    diversity_weight: float = 0.001  # ||cov - I||² weight (0=disabled)
    # Nuclear norm regularization for bind W_proj
    nuclear_weight: float = 1e-5  # stochastic ||W||_* weight (0=disabled)
    # Orthogonality regularization for bottleneck bind
    orth_weight: float = 1e-4  # ||Ŵ^TŴ - I||² weight (0=disabled)
    # Surprisal-weighted loss: focus on informative tokens
    surprisal_weight: float = 0.0  # γ, 0=disabled, 0.5=mild, 1.0=aggressive

    # Branch balance: equalize log-variance of conv/bind/mirror (Proposal V-3)
    branch_balance_weight: float = 0.0  # λ_B, 0=disabled

    # VSA long-range memory
    vsa_b_d_max: float = 12.0       # max b_d (τ≈160K at 12.0, was 5.0/τ≈150)
    vsa_b_d_smooth: float = 0.999   # per-step lerp rate towards controller target
                                    # 0.999 = 0.1%/step (τ_lerp≈1000 steps)
                                    # 1.0 = instant overwrite (old behavior)
    vsa_b_lr_mult: float = 0.1      # optimizer LR multiplier for b_d/b_i

    # BottleneckBind twist: inter-channel bilinear mixing via golden-angle shifts
    bind_twist_mode: str = "off"         # "off" | "shift" | "cascade"
    bind_twist_S: int = 4                # number of shifts (1 when mode=off)
    bind_twist_ocular: str = "tied"      # "tied" | "multi" — per-shift W_out
    bind_twist_scheme: str = "golden"    # "golden" | "fibonacci"
    bind_twist_gate: bool = False        # per-token adaptive aperture via hp

    # Gradient accumulation
    accum_steps: int = 1  # effective batch = batch_size * seq_len * accum_steps

    # Training
    max_steps: int = 500000
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    patience: int = 999999
    resume: str = ''

    # Paths
    data_dir: str = ''
    save_dir: str = 'checkpoints'
    log_dir: str = 'logs'

    def __post_init__(self):
        if self.lambda_d_enabled:
            self._apply_lambda_d()

    def _apply_lambda_d(self):
        lc = LambdaConfig(self.lambda_d)
        self.warmup_steps = lc.warmup_steps
        self.target_var = lc.target_var
        self.mag_threshold = lc.mag_threshold
        self.lr_min_ratio = lc.lr_min_ratio
        self.max_decay_steps = lc.max_decay_steps
        self.var_min_for_lr_decay = lc.var_min_for_lr_decay
        self.exploration_threshold = lc.exploration_threshold
        self.differentiation_threshold = lc.differentiation_threshold
        self.w_mem2v_scale_min = lc.mem2v_scale_min
        self.w_mem2v_scale_max = lc.mem2v_scale_max
        self.ema_alpha_min = lc.ema_alpha_min
        self.ema_alpha_max = lc.ema_alpha_max
        self.noise_scale_min = lc.noise_scale_min
        self.noise_scale_max = lc.noise_scale_max
        self.delta_var_ema_min = lc.delta_var_ema_min
        self.delta_var_ema_max = lc.delta_var_ema_max
        self.gate_lr_mult = lc.gate_lr_mult
        self.log_scale_init_std = lc.log_scale_init_std
        self.conv_init_std = lc.conv_init_std
        self.w_d_init_std = lc.w_d_init_std
        self.log_interval = lc.log_interval
        self.eval_interval = lc.eval_interval
        self.save_interval = lc.save_interval
        self.patience = lc.patience
