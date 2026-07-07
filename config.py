"""WideBind configuration."""

from dataclasses import dataclass, field

@dataclass
class WideBindConfig:
    D: int = 896
    n_layers: int = 24
    bottleneck: int = 3584       # MLP hidden dim
    bind_K: int = 16             # bottleneck for bind projection
    vocab: int = 50000
    seq_len: int = 128
    batch_size: int = 2
    lr: float = 3e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    dtype: str = 'float32'

    # MLP
    mlp_groups: int = 8
    mlp_expand: int = 8

    # Scheduler
    scheduler: str = 'mirror'

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
