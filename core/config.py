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

    # False = обучать EOS-токен (границы предложений), True = маскировать (старое поведение)
    mask_eos: bool = True

    # ─── λ_d hierarchy ─────────────────────────────────────────
    lambda_d: int = 3            # dimension of generalized golden ratio
    lambda_d_enabled: bool = True  # True = apply λ_d derivation in __post_init__

    tie_bind: bool = True  # True = W_out = W_proj^T (autoencoder bind bottleneck)
    tie_mirror_proj: bool = True  # True = mirror W_out = W_proj^T (per-expert K-space AE)

    # Variable Precision Memory
    variable_precision: bool = False  # True = add exact sequence memory on top of VSA
    precision_threshold: float = 0.3  # gate threshold to activate exact memory

    # Explicit Reasoning (chain-of-thought)
    explicit_reasoning: bool = False  # True = enable thinking tokens and reasoning memory
    reasoning_max_steps: int = 8  # max reasoning steps in chain-of-thought
    reasoning_ramp_steps: int = 1000  # exp ramp of block influence: scale = 1 - exp(-t/ramp)
    reasoning_adaptive: bool = False  # True = per-step gates (adaptive depth); False = old single step
    reasoning_gate_stop_threshold: float = 0.5  # loop stops when mean gate < threshold

    # AMP (Automatic Mixed Precision)
    use_amp: bool = False  # True = mixed precision (requires CUDA, ~2x speed)

    head_mode: str = "sigmoid_coded"
    head_normalize: bool = True
    code_dim: int = 32
    code_sparsity: int = 6

    mirror_k: int = 32
    mirror_k_staircase: bool = True  # True = k_l∈{8,16,32} по третям глубины
    w_pred_scale_init: float = 3.0
    log_scale_init_std: float = 0.05
    mlp_groups: int = 32
    mlp_expand: int = 4
    # Force uniform log_skip_alpha=0 on build/resume (SMF L0-depth fix).
    # Default False: only matters when resuming an OLD checkpoint that carries
    # the 17.8x L0 bias — set True to neutralize it without retraining from scratch.
    reset_skip_alpha: bool = False
    private_mem: bool = True  # cross-expert private memory bank (meta-cognitive layer)

    # ─── Spec 1: Asymmetric expert init ───
    expert_asymmetry: bool = True  # break symmetry: different alpha, log_scale, W_proj per expert

    # ─── Spec 3: Recursive meta-trust ───
    meta_trust: bool = True  # track trust dynamics, penalize unstable experts (requires private_mem)

    collective_layer: bool = True
    collective_layer_idx: int = None
    collective_read_out: bool = False
    collective_S: int = 8
    collective_uncert_theta: float = 0.5
    collective_uncert_kappa: float = 3.0
    collective_contra_thresh: float = -0.1
    collective_contra_gain: float = 6.0
    collective_birth_gap: float = 0.55
    collective_maturity_thresh: float = 0.12

    log_scale_l2_weight: float = 0.01  # L2 on exp(log_scale) > 10 to prevent gradient explosion
    orth_weight: float = 0.0  # ortho-gran loss; 0=off (32x D²=4096² gram graphs cost 2GB+ VRAM)
    div_weight: float = 50.0   # sigmoid-bounded log_scale divergence (bypasses spectral alignment)
    ranking_weight: float = 0.01  # pairwise order ls_mean by gate_usage (bypasses spectral alignment)
    gate_repulse_weight: float = 0.3  # push gate variance up (inverse of balance, bypasses spectral)
    alpha_novelty_weight: float = 0.05  # push per-expert alpha apart (heuristic, no spectral)
    gate_bias_scale: float = 2.0  # linspace init for gate bias per expert [-scale, scale]
    gate_bias_scale_per_layer: bool = True  # 0.5 (first layer) -> 2.0 (last layer)

    # Scheduler (values below will be overridden by λ_d when lambda_d_enabled=True)
    scheduler: str = 'mirror'
    target_var: float = 0.1
    mag_threshold: float = 0.3
    lr_min_ratio: float = 0.05
    max_decay_steps: int = 50000
    var_min_for_lr_decay: float = 0.005

    # Per-layer LS-based LR modulation (индивидуальная адаптация по var(log_scale))
    per_layer_ls_lr: bool = False  # True = per-layer mult из fast/slow EMA var(ls)
    ls_ema_fast: float = 0.99
    ls_ema_slow: float = 0.999
    ls_mult_min: float = 0.5
    ls_mult_max: float = 2.0
    ls_mirror_mult_max: float = 2.0  # кламп итога irm*ls_mult для mirror-градиентов

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
    w_m2v_hierarchy_weight: float = 0.01  # λ_weight for w_m2v regularisation (drives _tau_l_dev adaptation)

    # Intent Bridge: own τ-ladder by τ (context integration timescale, separate from memory)
    intent_tau_hierarchy_target: float = 0.3  # desired integration rate alpha for intent_state
    intent_tau_hierarchy_weight: float = 0.01  # λ_weight for intent-τ regularisation (drives _tau_intent_dev adaptation)

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
    gate_l1_weight: float = 0.0001   # L1 penalty on expert gates (0=disabled)
    # Expert reinforcement: align gate with usefulness prediction
    reinforce_weight: float = 0.001  # MSE(gate, usefulness) aux loss weight
    # Load balancing: encourages uniform expert usage across tokens
    balance_weight: float = 0.026  # λ⁻⁶ → HHI-based load balancing (adaptive)
    # Diversity loss: decorrelate per-group MLP outputs
    diversity_weight: float = 0.001  # ||cov - I||² weight (0=disabled)
    # Nuclear norm regularization for bind W_proj
    nuclear_weight: float = 1e-5  # stochastic ||W||_* weight (0=disabled)
    orth_weight: float = 1e-4  # ||Ŵ^TŴ - I||² weight (0=disabled)
    # Surprisal-weighted loss: focus on informative tokens
    surprisal_weight: float = 0.0  # γ, 0=disabled, 0.5=mild, 1.0=aggressive

    # Branch balance: equalize log-variance of conv/bind/mirror (Proposal V-3)
    branch_balance_weight: float = 0.0  # λ_B, 0=disabled

    # Gradient-reactive governance loss (prototype): open MLP gate where the MLP
    # output actually changes the CE loss. Aligns per-expert mlp_mod to
    # g_target = ||∂CE/∂mlp_out|| (detached). 0 = disabled (default).
    gradalign_weight: float = 0.0

    # VSA long-range memory
    vsa_b_d_max: float = 12.0       # max b_d (τ≈160K at 12.0, was 5.0/τ≈150)
    vsa_b_d_smooth: float = 0.999   # per-step lerp rate towards controller target
                                    # 0.999 = 0.1%/step (τ_lerp≈1000 steps)
                                    # 1.0 = instant overwrite (old behavior)
    vsa_b_lr_mult: float = 0.1      # optimizer LR multiplier for b_d/b_i

    # ─── Qwen3-inspired upgrades ───
    bind_qk_norm: bool = True            # RMSNorm on hp before bottleneck cross (≈QK-Norm)
    rope_theta: float = 1000000.0        # RoPE base frequency (Qwen3: 1e6)
    rope_scaling: float = 1.0            # RoPE scaling factor (linear)
    mlp_swiglu: bool = True              # SwiGLU gate_proj parallel to up_proj (Qwen3-style)

    bind_twist_mode: str = "trajectory_spiral"
    bind_twist_S: int = 4
    bind_traj_dims: int = 3
    hybrid_alpha_max: float = 0.7
    hybrid_alpha_min: float = 0.3
    bind_twist_ocular: str = "tied"
    bind_twist_scheme: str = "golden"
    bind_twist_gate: bool = False

    # Trajectory manifold (FCF): beams + Zeckendorf decay on trajectory bind
    traj_manifold: bool = False        # clever wrap: TrajectoryManifoldBind instead of Spiral
    traj_beams: int = 0                # число лучей: 0 = авто ceil(sqrt(buffer))
    traj_buffer_size: int = 1024       # буфер переходов (Mini: 512; 1024→32 луча автоматом)
    traj_cos_threshold: float = 0.5    # cos-порог кластеризации лучей
    traj_rebuild_interval: int = 128   # пересборка лучей каждые N переходов
    traj_gain: float = 0.05            # масштаб вклада манифолда

    # ─── Intent Bridge (нисходяще-восходящая передача «намерения» экспертам) ───
    # Эксперты «подхватывают» восходящий сигнал (то, что идёт наверх и станет
    # логитом). Реализуется как обёртка: IntentProbe + zero-init w_intent/b_intent.
    intent_bridge: bool = False    # True = добавить мост ( checkpoint-совместимо: ноль-эффект при init)
    intent_topdown: bool = True    # зарезервировано: форма нисходящей трансляции intent_state

    # Gradient accumulation
    accum_steps: int = 1  # effective batch = batch_size * seq_len * accum_steps

    compile: bool = False
    gradient_checkpointing: bool = True  # trade compute for memory, essential on T4

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
