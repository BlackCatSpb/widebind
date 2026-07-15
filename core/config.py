"""WideBind configuration."""

from dataclasses import dataclass, field

@dataclass
class WideBindConfig:
    D: int = 4096
    n_layers: int = 32
    bind_K: int = 32             # bottleneck for bind projection (align with 32 segments)
    vocab: int = 50000
    seq_len: int = 128
    batch_size: int = 2
    lr: float = 3e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    dtype: str = 'float32'

    # Embed
    code_dim: int = 32           # K: число сегментов для PartitionedEmbedding
    code_sparsity: int = 6       # S: единиц на токен (C(32,6)=906K>=50000)

    # Mirror
    mirror_k: int = 32           # K-space dim per expert (d/k=128/32=4:1)
    w_pred_scale_init: float = 3.0
    log_scale_init_std: float = 0.05
    # MLP
    mlp_groups: int = 32         # D/mlp_groups=128, mirror_k=32 → d/k=4:1
    mlp_expand: int = 4

    # Scheduler
    scheduler: str = 'mirror'
    target_var: float = 0.1
    mag_threshold: float = 0.3
    lr_min_ratio: float = 0.05
    max_decay_steps: int = 50000
    var_min_for_lr_decay: float = 0.005  # log_scale init std=0.05 → var=0.0025; only decay when var exceeds 2× init noise

    # AdaptiveController
    exploration_threshold: float = 0.25   # normalization: |mirror| / thresh → [0,1]
    differentiation_threshold: float = 0.08  # normalization: var(ls) / thresh → [0,1]
    w_mem2v_scale_min: float = 0.5       # memory contribution when diff=1
    w_mem2v_scale_max: float = 1.0       # memory contribution when diff=0
    ema_alpha_min: float = 0.90          # global EMA rate when diff=0
    ema_alpha_max: float = 0.99          # global EMA rate when diff=1
    noise_scale_min: float = 0.001       # parameter noise when diff=1
    noise_scale_max: float = 0.05        # parameter noise when diff=0
    delta_var_ema_min: float = 0.80      # δ_var EMA rate when diff=0 (fast, ~5-step TC)
    delta_var_ema_max: float = 0.99      # δ_var EMA rate when diff=1 (slow, ~100-step TC)

    # Optimizer
    gate_lr_mult: float = 5.0     # LR boost for gate weight params

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
