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
    grad_clip: float = 1.0
    dtype: str = 'float32'

    # ─── λ_d hierarchy ─────────────────────────────────────────
    lambda_d: int = 3            # dimension of generalized golden ratio
    lambda_d_enabled: bool = True  # True = apply λ_d derivation in __post_init__

    # Symmetry constraints
    tie_bind: bool = True  # True = W_out = W_proj^T (autoencoder bind bottleneck)
    tie_mirror_proj: bool = True  # True = mirror W_out = W_proj^T (per-expert K-space AE)

    # Zeckendorf Readout (experimental)
    zeckendorf_readout: bool = False  # True = replace LM head with Zeckendorf tree

    # Temporal Zeckendorf (experimental)
    temporal_zeckendorf: bool = False  # True = use Zeckendorf-based temporal decay

    # Embed
    code_dim: int = 32
    code_sparsity: int = 6

    # Mirror
    mirror_k: int = 32
    w_pred_scale_init: float = 3.0
    log_scale_init_std: float = 0.05
    mlp_groups: int = 32
    mlp_expand: int = 4

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
