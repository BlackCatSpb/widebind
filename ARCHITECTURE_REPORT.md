# EVA / WideBind — Complete Architectural Analysis Report

## Table of Contents
1. [Project Overview](#overview)
2. [File-by-File Module Map](#module-map)
3. [Forward Pass Flow](#forward-flow)
4. [Training Flow](#training-flow)
5. [Bottlenecks & Issues](#bottlenecks)

---

## 1. PROJECT OVERVIEW

EVA (WideBind) is a PyTorch neural network implementing Vector Symbolic Architecture (VSA) with cognitive mirror experts, adaptive memory, and explicit reasoning. The architecture uses:

- **Sparse block codes** for token embedding (not standard embeddings)
- **Bottleneck bind** (Fibonacci-twisted bilinear cross-mixing) in D→K→D space
- **32-group cognitive mirror** experts operating in independent K-space subspaces
- **Multi-scale VSA memory** (4 fixed τ scales: 8, 32, 128, 512)
- **DCT spectral gating** for frequency-domain modulation
- **Streaming memory bank** (hierarchical L1+L2+L3)
- **Maturation controller** (unified per-layer wake-up gate)
- **Semantic bridge** (cross-layer self-supervised probe)
- **Intent bridge** (top-down expert modulation)
- **Adaptive reasoning** (chain-of-thought with per-step gating)
- **Triad verification** (re-circulation on low confidence)

Default config: D=4096, n_layers=32, G=32 experts, k=32/16/8 (staircase), vocab=50000, seq_len=128.

---

## 2. FILE-BY-FILE MODULE MAP

### core/config.py (400 lines)

**Class: `WideBindConfig`** (dataclass, line 12)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__post_init__` | 371 | Applies λ_d hierarchy if enabled | None | None | `_apply_lambda_d` | All construction | No |
| `_apply_lambda_d` | 375 | Overrides ~20 hyperparameters from generalized golden ratio λ_d | None | None | `LambdaConfig` | `__post_init__` | No |

**Key config values:**
- D=4096, n_layers=32, bind_K=64, vocab=50000, seq_len=128
- mlp_groups=32 (→ d=128 per expert), mlp_expand=4
- mirror_k=32 (staircase: 8/16/32 by depth thirds)
- reasoning_max_steps=8, reasoning_adaptive=True
- bridge_dim=256, bridge_conn=0.1
- maturation_enabled=True, matur_T0=8000, matur_T_delay=8000
- private_mem=True, memory_bank=False (default off)
- bridge_glu=True, bridge_glu_beta=0.25
- intent_bridge=True

---

### core/bind.py (651 lines)

**Class: `_ExpRMSNorm`** (line 11)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 14 | Creates RMSNorm weight | K | None | nn.Parameter | — | Yes (weight) |
| `forward` | 18 | RMSNorm via explicit formula | x: (..., K) | (..., K) | torch.rsqrt | BottleneckBind.hp_norm | Yes |

**Function: `migrate_bind_state_dict`** (line 21)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `migrate_bind_state_dict` | 21 | Converts old bind state dict keys to new format | sd, n_layers, mode, S | dict | re | train.py resume | No |

**Function: `_golden_shifts`** (line 49)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `_golden_shifts` | 49 | Generates S golden-angle circular shifts for bind | K: int, S: int | list[int] | math.floor | BottleneckBind.__init__ | No |

**Class: `BottleneckBind`** (line 62)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 78 | Bilinear cross-mixing with Fibonacci/golden-angle shifts | D, K, cfg | None | _ExpRMSNorm, _golden_shifts, nn.Linear | WideBindBlock.bind | Yes (W_proj, w_u, w_v, W_out, mix_logit, log_tau, w_gate_proj, w_bind_bias) |
| `_tie_hook` | 138 | Copies W_proj to W_out for tied mode | module, inp | None | torch.no_grad | register_forward_pre_hook | No |
| `_cross` | 142 | Bilinear cross-mixing: left * roll(right, shift) | left, right, shift | tensor | torch.roll | forward | No |
| `forward` | 145 | Main bind: D→K bottleneck, bilinear mixing, K→D output | h: (B, L, D) | (B, L, D) | hp_norm, W_proj, w_u, w_v, W_out | WideBindBlock | Yes |

**Bind modes:**
- "off": legacy diagonal (u·v), no shifts
- "shift": sum of S shifted bilinear products
- "cascade": Fibonacci-nested cascade, monomials up to order F_S

**Function: `_fibonacci_shifts`** (line 218)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `_fibonacci_shifts` | 218 | Fibonacci sequence-based circular shifts | K, S | list[int] | _golden_shifts (fallback) | BottleneckBind | No |

**Class: `SpiralBind`** (line 235)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 236 | Complex spiral bind with frequency/phase modulation | D, K, cfg | None | _ExpRMSNorm, nn.Linear | WideBindBlock | Yes (w_u_re/im, w_v_re/im, W_freq, W_phase, W_out, W_proj, w_bind_bias) |
| `forward` | 257 | Spiral bilinear binding with complex exponentials | h: (B, L, D) | (B, L, D) | self | WideBindBlock | Yes |

**Class: `TrajectorySpiralBind`** (line 303)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 304 | Trajectory-based spiral bind with n_dims trajectory history | D, K, cfg | None | _ExpRMSNorm, nn.Linear | WideBindBlock | Yes (w_u_re/im, w_v_re/im, W_freq, W_phase, W_out, W_proj, freq_scale, w_bind_bias) |
| `_hybrid_alpha` | 348 | Returns hybrid alpha (HRR vs element-wise blend) | None | float | self.training, _step_count | forward | No |
| `_hrr_bind` | 355 | HRR (holographic reduced representation) bind | a, b | tensor | _circ_conv_idx, torch.einsum | _hybrid_bind | No |
| `_hybrid_bind` | 359 | Alpha-weighted blend of HRR and element-wise bind | a, b | tensor | _hrr_bind | forward | No |
| `forward` | 365 | Full trajectory spiral bind with coherence measure | h: (B,L,D), traj_state | (result, new_traj, coherence) | hp_norm, W_proj, cos/sin of trajectory | WideBindBlock | Yes |

**Class: `TrajectoryManifoldBind`** (line 457)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 473 | Extends TrajectorySpiralBind with manifold beam clustering | D, K, cfg | None | super().__init__ | WideBindBlock | Yes (W_man, logit_gain, log_tau + all parent) |
| `_fib_list` | 508 | Static: Fibonacci sequence up to max_n | max_n | list | — | _zeck_weight | No |
| `_zlen` | 514 | Static: Zeckendorf decomposition length | fibs, n | int | — | _zeck_weight | No |
| `_hrr_unbind` | 526 | HRR unbind (correlation) | a, b | tensor | _circ_corr_idx | _hybrid_unbind | No |
| `_hybrid_unbind` | 530 | Alpha-weighted HRR+element-wise unbind | a, b | tensor | _hrr_unbind | _push_transitions | No |
| `_push_transitions` | 536 | Write unbind(hp_t, hp_{t-1}) transitions to ring buffer | hp | None | _hybrid_unbind, _rebuild_beams | forward | No |
| `_rebuild_beams` | 559 | VSA-cluster transitions into beam centers | None | None | cos_threshold | _push_transitions | No |
| `_zeck_weight` | 605 | Zeckendorf-based age decay for beams | age | float | _zlen, _fib_cache | _manifold_read | No |
| `_manifold_read` | 611 | Hybrid attention read from manifold beams | hp: (B,L,K) | (B,L,K) | beam_centers, logit_gain, log_tau | forward | Yes (logit_gain, log_tau) |
| `forward` | 641 | Trajectory + manifold bind combined | h, traj_state | (result, new_traj, coherence) | super().forward, _push_transitions, _manifold_read | WideBindBlock | Yes |

**Function: `_fib_sequence`** (line 280), **`_zeckendorf_levels`** (line 287) — helpers for trajectory.

---

### core/mirror.py (798 lines)

**Class: `BridgeGLU`** (line 11)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 19 | GLU gating: Wg, Wv linear + log_gain | G, k | None | nn.Linear | GroupedCognitiveMirror | Yes (Wg, Wv, log_gain) |
| `forward` | 28 | delta → sigmoid(Wg·flat) * sigmoid(Wv·flat) * sigmoid(log_gain) | delta: (B,L,G,k) | (B,L,G) | torch.sigmoid | GroupedCognitiveMirror.bridge_glu_net | Yes |

**Class: `GroupedCognitiveMirror`** (line 36)

This is the largest and most complex module — 32 expert mirrors.

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 73 | 32-group mirror expert ensemble | D, G, k, ... | None | AdaptiveGate, BridgeGLU | WideBindBlock.mirror | Yes (W_proj, W_out, w_temp, w_global, conv_smooth, w_sym_u/v, alpha_diag, tanh_bias, log_scale, w_gate, b_gate, w_delta_gate, gate_bias, log_skip_alpha, w_alpha, b_alpha, log_dvar_mod_scale, log_grad_mod_scale, dvar_mod_bias, grad_mod_bias, _signal_log_weights, w_help, w_contra, mod_scale_mlp, mod_scale_mem, w_intent, b_intent, w_sal) |
| `_sync_W_out` | 297 | Sync W_out = W_proj^T for tied mode | None | None | torch.no_grad | forward_pre_hook | No |
| `forward` | 301 | Main mirror: project→K-space→signals→gate→output | h, mem_all, global_state, diff, tanh_bias_mod, pred_scale_mod, context_mem, allow_write, step, intent, salience, maturity | (mirror, mlp_mod, mem_mod, hp, pred_error_norm) | BridgeGLU, AdaptiveGate | WideBindBlock | Yes |
| `cache_grad_norms` | 733 | Store per-subspace gradient norm after backward | grad_h | None | torch.no_grad | training loop | No |
| `debug_mind` | 744 | Return meta-cognitive stats dict | None | dict | torch.no_grad | smart_controller, analyze.py | No |
| `meta_signals` | 786 | GPU-friendly (trust_max, gate_ema_mean) tensors | None | (tensor, tensor) | torch.no_grad | smart_controller.generate | No |

**Mirror forward signal computation (within forward, line 301-731):**

1. **Split h into G subspaces**: h_g = h.reshape(B,L,G,d)
2. **Project to K-space**: hp = einsum(h_g, W_proj) → (B,L,G,k)
3. **Bipolar pos_id binding**: hp *= _pos_id_buf
4. **hp_prev**: shifted by 1 timestep
5. **temp_k**: (hp - mc_k) * w_temp — deviation from memory centroid
6. **pred_error**: hp - hp_prev * alpha_diag — predictive error (per-dim tau)
7. **smooth_k**: hp - conv_smooth(hp) — local coherence
8. **sym_k**: (hp * w_sym_u) * (hp_prev * w_sym_v) — temporal symmetry
9. **help_k**: private memory read via cross-expert attention (when enabled)
10. **Signal EMA normalization**: per-signal RMS normalization
11. **Learnable signal weights**: sigmoid(_signal_log_weights)
12. **Delta merge**: weighted sum of all signals, RMS-normalized
13. **Gate modulation**: grad_mod, dvar_mod, intent_bridge, salience, contradiction
14. **Expert gate**: sigmoid(gate_logits) — per-token, per-expert
15. **MLP mod**: hybrid_gate(usefulness_logits) * BridgeGLU (when enabled)
16. **Mirror output**: tanh(W_out·delta) + skip_alpha * linear, scaled by log_scale
17. **SMF gate**: alpha = sigmoid(cat(h_norm, m_norm) · w_alpha + b_alpha)
18. **Output**: mirror * expert_gate * maturation

---

### core/adaptive_gate.py (125 lines)

**Class: `AdaptiveGate`** (line 32)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 46 | Sigmoid-softmax hybrid gate | n_features, tau_init, learnable_tau | None | nn.Parameter | GroupedCognitiveMirror | Yes (log_tau) |
| `forward` | 60 | gate = sigmoid(logits) * (1 + softmax(logits/tau)) | logits: (*, n_features) | (*, n_features) | torch.sigmoid, F.softmax | GroupedCognitiveMirror.hybrid_gate | Yes |
| `get_diagnostics` | 90 | Return gate metrics | None | dict | — | logging | No |

**Class: `AdaptiveGateBundle`** (line 100)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 107 | Multiple named gates | None | None | ModuleDict | — | Yes (gates) |
| `add` | 111 | Register a new gate | name, n_features, tau_init | None | AdaptiveGate | — | Yes |
| `forward` | 115 | Forward through named gate | name, logits | tensor | self.gates[name] | — | Yes |
| `get_all_diagnostics` | 119 | Return all gate diagnostics | None | dict | — | logging | No |

---

### core/memory_bank.py (627 lines)

**Function: `_memory_attention`** (line 51)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `_memory_attention` | 51 | Hybrid attention: sigmoid(scores) * (1 + softmax(scores/tau)) | q, k, temp, bridge_dim, softmax_free, age_decay | attn: (B,L,n_slots) | torch.sigmoid, F.softmax | L1Buffer.read, L2Bank.read, L3Concepts.read | No |

**Class: `L1Buffer`** (line 96)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 104 | Rolling buffer of K sentence embeddings | D, bridge_dim, n_slots, softmax_free | None | nn.Linear, _memory_attention | StreamingMemoryBank.l1 | Yes (proj, q_proj, out_proj, log_tau) |
| `write` | 127 | Write sentence embedding (ring buffer, overwrite oldest) | embedding: (D,) | None | torch.no_grad | StreamingMemoryBank.forward | No |
| `read` | 145 | Read via hybrid attention | query: (B,L,D) | (B,L,D) | _memory_attention | StreamingMemoryBank.forward | Yes |
| `get_stats` | 164 | Return write statistics | None | dict | — | diagnostics | No |
| `reset` | 170 | Clear buffer | None | None | — | StreamingMemoryBank.reset | No |

**Class: `L2Bank`** (line 177)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 188 | Learned memory bank with N slots | D, bridge_dim, n_slots, softmax_free | None | nn.Linear, nn.Sequential, _memory_attention | StreamingMemoryBank.l2 | Yes (W_k, W_v, W_o, q_proj, val_norm, keys, vals, novelty_gate, log_tau, key_log_scale, val_log_scale) |
| `write` | 229 | Write with novelty gating, prioritizes consumed slots | embedding | slot_idx | torch.no_grad, F.normalize | StreamingMemoryBank.forward | No |
| `mark_consumed` | 277 | Mark slot as consumed by L3 concept birth | slot | None | — | StreamingMemoryBank.forward | No |
| `read` | 283 | Read via hybrid attention | query: (B,L,D) | (B,L,D) | _memory_attention | StreamingMemoryBank.forward | Yes |
| `get_stats` | 302 | Return bank statistics | None | dict | — | diagnostics | No |
| `reset` | 313 | Clear all slots | None | None | — | StreamingMemoryBank.reset | No |

**Class: `L3Concepts`** (line 324)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 345 | Emergent concept clustering from L2 slots | D, bridge_dim, n_concepts, birth_threshold, ... | None | _memory_attention | StreamingMemoryBank.l3 | Yes (concept_keys, concept_vals, q_proj, out_proj, log_tau, val_log_scale) |
| `write` | 378 | Concept birth/update from L2 key+val | l2_key, l2_val, confidence | bool | F.normalize, torch.no_grad | StreamingMemoryBank.forward | No |
| `read` | 439 | Read from concepts via attention | query: (B,L,D) | (B,L,D) | _memory_attention | StreamingMemoryBank.forward | Yes |
| `get_active_concepts` | 458 | Count active concepts | None | int | — | diagnostics | No |
| `get_stats` | 462 | Return concept statistics | None | dict | — | diagnostics | No |
| `reset` | 471 | Clear all concepts | None | None | — | StreamingMemoryBank.reset | No |

**Class: `StreamingMemoryBank`** (line 481)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 494 | Combined L1+L2+L3 memory bank | D, bridge_dim, l1_slots, l2_slots, l3_concepts, ... | None | L1Buffer, L2Bank, L3Concepts, nn.Sequential | WideBindStack.memory_bank | Yes (fusion, log_scale) |
| `forward` | 534 | Read from all levels, write at sentence boundaries | h: (B,L,D), tokens, step, mat_gate | (B,L,D) | L1/L2/L3.read, L1/L2/L3.write | WideBindStack.forward | Yes |
| `reset` | 598 | Clear all memory | None | None | — | — | No |
| `get_diagnostics` | 604 | Return diagnostic info | None | dict | — | logging | No |

---

### core/stack.py (1878+ lines)

**Class: `WideBindStack`** (line 17)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 20 | Full model: embed + layers + head + reasoning + bridge + maturation | cfg | None | PartitionedEmbedding, SigmoidCodedHead, WideBindBlock, ReasoningMemory, ThinkingTokenHead, ReasoningGate, SemanticBridge, LayerBridgeGate, StreamingMemoryBank, MaturationController | train.py, generate.py | Yes |
| `forward` | 144 | Main forward: embed→layers→reasoning→triad→output | h, state, global_state, pred_weight, adaptive, context_mem, allow_write, step, reasoning_buffer, reasoning_count, intent_state, tokens, _triad_depth | (h, new_state, global_state, (reasoning_buffer, reasoning_count)) | All layers, AdaptiveController, MaturationController, SemanticBridge, StreamingMemoryBank | train.py, generate.py | Yes |
| `_knowledge_signal` | 574 | Compute model confidence features (B,8) | h, state | (B,8) | lm_head | _adaptive_reasoning | No |
| `_last_conf` | 621 | Confidence of head on last position | h | (B,) | lm_head | triad_reason, _adaptive_reasoning | No |
| `_adaptive_reasoning` | 631 | Adaptive-depth reasoning loop | h, s, state, reasoning_buffer, reasoning_count | h_acc | ReasoningMemory, ReasoningGate, _knowledge_signal, _last_conf | forward | Yes (reasoning_gate) |
| `reasoning_scale` | 739 | Property: reasoning influence scale | None | float | reasoning_scale_override, reasoning_ramp_steps | forward | No |
| `reset_reasoning` | 747 | Reset reasoning buffer | None | None | — | generate.py | No |
| `embed_tokens` | 752 | Token indices → D-space vectors | tokens | (B,L,D) | embed | generate.py | Yes |
| `projector_signals` | 757 | Property: write_event, concept_id from collective layer | None | (tensor, tensor) | — | projector.py | No |
| `compute_loss` | 770 | Returns CE only | h, targets, pred_weight, h_emb | scalar | compute_losses | training | Yes |
| `_finalize_ce` | 775 | Mask PAD/EOS + surprisal weighting | ce, targets | scalar | — | compute_losses | No |
| `compute_salience` | 792 | Word importance from head output | logits | (B,L,1) | — | observe_output | No |
| `observe_output` | 804 | Store salience of current output for next step | logits | None | compute_salience | training loop | No |
| `compute_losses` | 810 | CE + all auxiliary losses (raw, unweighted) | h, targets, pred_weight, h_emb | (ce_loss, aux_dict) | lm_head, bridge.loss, all layer mirrors | LossBalancer.backward | Yes |
| `_checkpointed_block` | 1203 | Static wrapper for gradient checkpointing | layer, h, state, ... | (h_out, s_out, pen, hp) | WideBindBlock | forward (training) | Yes |
| `param_count` | 1223 | Total parameter count | None | int | — | logging | No |
| `apply_mlp_depth_gradient_boost` | 1226 | Register backward hooks to boost deep MLP gradients | exp | None | register_hook | train.py | No |
| `collective_stats` | 1256 | Per-layer collective concept stats | None | dict or None | — | logging | No |
| `param_groups` | 1275 | Optimizer parameter groups with λ_d LR hierarchy | lr, weight_decay, gate_lr_mult | list[dict] | lambda_d | build_optimizer | No |

**Class: `AdaptiveController`** (line 1426)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `layer_stats` | 1476 | Per-layer (exploration, differentiation) | layer, expl_thresh, diff_thresh | (float, float) | mirror.log_scale, mirror._last_magnitude | stack.forward | No |
| `stats` | 1484 | Global average (expl, diff) across all layers | blocks, expl_thresh, diff_thresh | (float, float) | layer_stats | stack.forward | No |
| `layer_b_d` | 1498 | Per-layer decay bias from exploration | layer, expl, b_d_max | float | layer_stats | stack.forward | No |
| `layer_b_i` | 1507 | Per-layer write gate bias from exploration+τ | layer, expl, tau_l | float | layer_stats | stack.forward | No |
| `layer_w_mem2v_scale` | 1533 | Per-layer memory contribution | layer, min_val, max_val, diff | float | layer_stats | stack.forward | No |
| `layer_noise_scale` | 1540 | Per-layer parameter noise | layer, min_val, max_val, diff | float | layer_stats | stack.forward | No |
| `layer_ema_alpha` | 1547 | Per-layer EMA rate | layer, min_val, max_val, diff | float | layer_stats | stack.forward | No |
| `pred_weight` | 1556 | Adaptive alpha aux loss weight | blocks, min_val, max_val | float | stats | stack.forward | No |
| `tanh_bias_modulation` | 1567 | Scale tanh_bias by exploration | layer, expl | float | layer_stats | stack.forward | No |
| `spectral_modulation` | 1578 | Modulate spectral lambda_k by differentiation | layer, diff | float | layer_stats | stack.forward | No |
| `pred_scale_mod` | 1591 | Per-expert pred error modulation from delta_var | layer | tensor | mirror._delta_var | stack.forward | No |
| `b_d`, `b_i`, `w_mem2v_scale`, `ema_alpha`, `noise_scale` | 1606-1629 | Global wrappers (backward compat) | blocks | float | stats | — | No |

**Class: `MirrorLRScheduler`** (line 1633)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 1642 | LR scheduler modulated by cognitive mirror state | model, optimizer, base_lr, warmup, ... | None | — | LRController | No |
| `_mirror_stats` | 1675 | Aggregate var(log_scale), |mirror|, alpha, gate_var | None | (var, mag, alpha, gate_var) | step | No |
| `_update_ls_mult` | 1691 | Per-layer LR mult from fast/slow EMA of var(ls) | None | list or None | — | step | No |
| `report_train_loss` | 1724 | Report training loss (no-op) | train_loss, ce_loss | None | — | training | No |
| `report_val_loss` | 1728 | Adaptive LR damping on val loss regression | val_loss | None | — | training | No |
| `step` | 1769 | Update LR with warmup + mirror adaptive multiplier | None | None | _mirror_stats, _update_ls_mult | LRController.step | No |
| `get_last_lr` | 1859 | Return current LRs | None | list[float] | — | training | No |
| `state_dict` | 1862 | Serialize scheduler state | None | dict | — | checkpoint | No |
| `load_state_dict` | 1879+ | Restore scheduler state | sd | None | — | resume | No |

---

### core/block.py (504 lines)

**Class: `PrecisionGate`** (line 15)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 16 | Linear→sigmoid gate | D | None | nn.Linear | WideBindBlock | Yes (gate) |
| `forward` | 20 | sigmoid(linear(h)) | h: (B,L,D) | (B,L,1) | torch.sigmoid | WideBindBlock.forward | Yes |

**Class: `ExactSequenceMemory`** (line 24)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 25 | Q/K/V attention memory | D, k, softmax_free | None | nn.Linear | WideBindBlock | Yes (query, key, value, proj) |
| `forward` | 34 | Scaled dot-product attention (sigmoid or softmax) | h: (B,L,D) | (B,L,D) | F.softmax or sigmoid | WideBindBlock.forward | Yes |

**Class: `WideBindBlock`** (line 48)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 62 | Full block: bind + mirror + VSA memory + conv + spectral + MLP + collective | cfg, layer_idx | None | BottleneckBind/SpiralBind/TrajectorySpiralBind/TrajectoryManifoldBind, GroupedCognitiveMirror, GroupedMLP, PrecisionGate, ExactSequenceMemory, CollectiveConceptLayer, Conv1d | WideBindStack.layers | Yes (pre_ln_w, w_i, w_d, w_q, w_q_leaf, w_q_ctx, w_mem2v, w_q_dyn, w_i_dyn, w_d_pen, w_bind_gate, scale_w, b_i, b_d, gamma_surprisal, bind_coh_gate, w_k_mu, w_q_mu, w_mu_mem, conv, lambda_k, + bind/mirror/mlp/collective/precision/exact submodules) |
| `forward` | 201 | Full block forward: preLN→conv→bind→VSA→mirror→output→spectral→MLP | h, state, global_state, ... | (h, state_out) | All submodules | WideBindStack.forward | Yes |
| `base_parameters` | 493 | Property: all params except mirror | None | list | — | — | No |
| `mirror_parameters` | 498 | Property: all mirror params | None | list | — | — | No |

**Block forward flow (line 201-491):**
1. Pre-LN: RMS normalization
2. Conv: depthwise 1D convolution + residual
3. Bind: D→K→D via BottleneckBind/TrajectorySpiralBind + residual
4. VSA Memory: multi-scale prefix scan (S=4 scales) with content-dependent write/decay
5. Mirror: GroupedCognitiveMirror(h, mem_all, ...) → (mirror, mlp_mod, mem_mod, hp, pred_error_norm)
6. Output: enhanced = bind_gated + mem_modulated + mirror + collective
7. Variable Precision: PrecisionGate + ExactSequenceMemory (optional)
8. Spectral: DCT basis * lambda_k modulation + residual
9. MLP: GroupedMLP with mirror_gate + residual

---

### core/mlp.py (80 lines)

**Class: `GroupedMLP`** (line 9)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 21 | Grouped bottleneck MLP with per-group expansion (SwiGLU) | D, expand, groups, swiglu, gate_b_init | None | — | WideBindBlock.mlp | Yes (W_gate/W_up/W_down, norm_w, mlp_gate_a, mlp_gate_b) |
| `forward` | 50 | Grouped SwiGLU: silu(gate) * up → down, with mirror_gate modulation | h: (B,L,D), mirror_gate: (B,L,G) | (B,L,D) | F.silu, torch.matmul | WideBindBlock.forward | Yes |

**MLP flow:**
1. RMSNorm: norm_w * h / sqrt(mean(h²))
2. Reshape to G groups of d dims
3. SwiGLU: silu(W_gate·h) * (a + b·mirror_gate) * W_up·h → W_down
4. Cache _cached_group_out for diversity loss

---

### core/embedding.py (387 lines)

**Class: `RotaryEmbedding`** (line 11)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 12 | Precompute RoPE frequencies | D, theta, scaling, max_len | None | — | PartitionedEmbedding | No |
| `_build_cache` | 22 | Build cos/sin cache for length L | L | None | — | forward | No |
| `forward` | 31 | Apply rotary position embedding | x: (B,L,D) | (B,L,D) | cos, sin | PartitionedEmbedding | No |

**Class: `ZeckendorfEmbedding`** (line 45)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 50 | Token→D via Zeckendorf codes + learned projection | cfg | None | zeckendorf_codes | — (legacy) | Yes (proj) |
| `forward` | 58 | proj(codes[tokens]) | tokens | (B,L,D) | — | — | Yes |

**Class: `PartitionedEmbedding`** (line 63)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 78 | Token→D via sparse block codes + dense mixing + RoPE | cfg | None | sparse_block_codes, RotaryEmbedding | WideBindStack.embed | Yes (embed_mix, basis) |
| `forward` | 100 | codes→sigmoid(M·codes)→outer(basis)→RoPE | tokens | (B,L,D) | — | WideBindStack.embed_tokens | Yes |

**Class: `LmHead`** (line 114)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 116 | D→vocab via Zeckendorf code projection (legacy) | cfg | None | zeckendorf_codes | — (legacy) | Yes (proj) |
| `forward` | 124 | proj(h) @ codes.T | h: (B,L,D) | (B,L,V) | — | — | Yes |

**Class: `PartitionedHead`** (line 129)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 142 | Segment-addressed readout + per-token bias | cfg, embed_basis | None | sparse_block_codes | — (alt head) | Yes (readout, token_bias) |
| `forward` | 159 | Σ_k z_{vk}·⟨h_k, r_k⟩ + b_v | h: (B,L,D) | (B,L,V) | — | — | Yes |

**Class: `SigmoidCodedHead`** (line 166)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 167 | Sigmoid-coded head with hybrid gate | cfg, embed_basis | None | sparse_block_codes | WideBindStack.lm_head | Yes (readout, bit_bias, log_temp, token_bias) |
| `_gates` | 188 | Compute per-bit gate logits | h, temp_factor, bus_bias | (B,L,K) | — | forward, log_probs_for_target | Yes |
| `_su` | 209 | Sigmoid-softmax hybrid: log-odds computation | zt | (u, base) | torch.sigmoid, torch.softmax | forward, log_probs_for_target | Yes |
| `forward` | 221 | Full head: gates→log-odds→token logits | h, bus_bias | (B,L,V) | _gates, _su, codes | WideBindStack.compute_losses | Yes |
| `log_probs_for_target` | 235 | Efficient log-probs for specific target tokens | h, targets, bus_bias | (N,) | _gates, _su, codes | CE computation | Yes |

**Class: `CognitiveCodedHead`** (line 263)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 264 | Cognitive head with prior, social bias, resonance | cfg, embed_basis, k_mirror | None | sparse_block_codes, nn.Embedding | — (alt head) | Yes (readout, log_temp_base, w_res, w_stab, bit_bias, W_q_prior, W_k_prior, alpha_prior, w_prior_scale, beta_social, w_energy, gamma, W_code_mod, token_shift_embed, proj_shift, token_bias) |
| `set_cognitive_state` | 305 | Set external cognitive signals | pred_error, private_mem, trust_matrix, contra_graph, dominance | None | — | WideBindBlock | No |
| `_compute_z` | 313 | Compute z with prior, social bias, resonance, context | h, B, L, device | (z, base) | — | forward, log_probs_for_target | Yes |
| `_shift_all` | 358 | Token shift embedding for all tokens | None | (V,K) | — | forward | Yes |
| `_shift_targets` | 362 | Token shift for specific targets | token_ids | (N,K) | — | log_probs_for_target | Yes |
| `forward` | 367 | Full cognitive head | h | (B,L,V) | _compute_z, _shift_all | — | Yes |
| `log_probs_for_target` | 375 | Efficient log-probs | h, targets | (N,) | _compute_z, _shift_targets | — | Yes |

---

### core/spectrum_gate.py (182 lines)

**Class: `SpectrumGate`** (line 33)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 44 | Sigmoid-softmax spectrum gate with learnable tau | n_features, tau_init, tau_min, tau_max, learnable_tau | None | — | LayerBridgeGate, WideBindStack | Yes (log_tau) |
| `forward` | 83 | gate = sigmoid(logits) * (1 + softmax(logits/tau)) | logits: (*, n_features) | (*, n_features) | torch.sigmoid, F.softmax | LayerBridgeGate.gates | Yes |
| `set_tau` | 119 | Manually set tau | tau | None | — | — | No |
| `get_diagnostics` | 124 | Return diagnostic metrics | None | dict | — | logging | No |

**Class: `SpectrumGateBundle`** (line 136)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 150 | Multiple named gates | None | None | ModuleDict | — | Yes (gates) |
| `add` | 154 | Register a new gate | name, n_features, tau_init, ... | None | SpectrumGate | — | Yes |
| `forward` | 168 | Forward through named gate | name, logits | tensor | self.gates[name] | — | Yes |
| `get_all_diagnostics` | 172 | Return all gate diagnostics | None | dict | — | logging | No |
| `get_modes` | 180 | Return current mode for each gate | None | dict | — | logging | No |

---

### core/layer_bridge_gate.py (249 lines)

**Class: `SpectrumGate`** (line 20) — local variant with external tau support

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 34 | Sigmoid-softmax gate with optional external tau | n_features, tau_init | None | — | LayerBridgeGate.gates | Yes (log_tau) |
| `forward` | 39 | gate = sigmoid * (1 + softmax/tau), tau_external overrides | logits, tau_external | gate | torch.sigmoid, F.softmax | LayerBridgeGate.forward | Yes |

**Class: `LayerBridgeGate`** (line 58)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 76 | Per-layer SpectrumGate for bridge routing | n_layers, health_features, tau_min, tau_max | None | ModuleList(SpectrumGate) | WideBindStack.layer_bridge_gate | Yes (gates) |
| `_effective_tau` | 94 | Compute effective tau from maturation | maturation | tensor | — | forward, stack | No |
| `forward` | 105 | Weighted bridge input from layer diagnostics + maturation | layer_outputs, diagnostics, tau_maturation, global_ready | (bridge_input, gate_weights, gate_info) | SpectrumGate | WideBindStack.forward | Yes |
| `get_diagnostics` | 204 | Extract per-layer diagnostics from model state | layers, bridge_contribution | (n_layers, health_features) | — | forward | No |

---

### core/concept_layer.py (279 lines)

**Class: `CollectiveConceptLayer`** (line 16)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 17 | Adaptive maturity concept layer with S slots | D, k, S, uncert_theta, ... | None | nn.Linear, nn.Parameter | WideBindBlock.collective | Yes (_birth_log_scale, W_o, _read_scale, _temp) |
| `_update_maturity` | 77 | Adaptive maturity from residual variance | resvar | None | — | forward | No |
| `_maybe_write` | 101 | Mature-gated concept write/update/birth | hp, pen, allow_write | (write_event, best) | F.normalize, torch.no_grad | forward | No |
| `forward` | 187 | Main: write concepts + read with gate-weighted expert attention | h, hp, pen, resvar, allow_write, mature_override, gate | out: (B,L,D) | F.normalize, torch.sigmoid/einsum | WideBindBlock.forward | Yes |
| `birth_gate_mean` | 264 | Average birth gate weight | None | float | — | diagnostics | No |
| `get_diagnostics` | 269 | Return concept diagnostics | None | dict | — | logging | No |

---

### core/reasoning.py (178 lines)

**Class: `ReasoningTokens`** (line 11) — static token IDs (THINK=65536, STEP=65537, ANSWER=65538, END=65539)

**Class: `ReasoningMemory`** (line 19)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 24 | Reasoning step encoder + attention + output proj | D, max_steps | None | nn.Linear | WideBindStack.reasoning_memory | Yes (step_encoder, step_query, step_key, step_value, output_proj) |
| `forward` | 40 | Encode current step, attend to buffer, update buffer | h, reasoning_buffer, reasoning_count, record | (output, new_buffer, new_count) | torch.sigmoid attention | _adaptive_reasoning | Yes |

**Class: `ReasoningGate`** (line 106)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 121 | Per-step decision gates for adaptive reasoning depth | D, max_steps, know_dim | None | nn.Linear | WideBindStack.reasoning_gate | Yes (proj, know_proj, r_proj) |
| `forward` | 139 | tanh(proj(h) + know_proj(know) + r_proj(r)) | h, know, r | (B,L,max_steps) | torch.tanh | _adaptive_reasoning | Yes |
| `logits` | 154 | Raw logits for straight-through gradient | h, know, r | (B,L,max_steps) | — | _adaptive_reasoning | Yes |

**Class: `ThinkingTokenHead`** (line 164)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 169 | Head predicting thinking tokens | D, num_reasoning_tokens | None | nn.Linear | WideBindStack.thinking_head | Yes (reasoning_proj) |
| `forward` | 173 | linear(h) → reasoning token logits | h: (B,L,D) | (B,L,4) | — | — | Yes |

---

### core/maturation.py (138 lines)

**Class: `MaturationController`** (line 47)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 48 | Unified per-layer maturation gate (time ramp) | n_layers, tau_min, tau_max, cfg | None | — | WideBindStack.maturation | No (buffers only) |
| `_update_tau_norm` | 77 | Update tau_norm from per-layer deviation | dev | None | — | step_gate | No |
| `step_gate` | 86 | Per-layer sigmoid time ramp: M_l(t) | step, tau_dev | (n_layers,) | sigmoid | WideBindStack.forward | No |
| `global_ready` | 113 | Property: True when ALL layers above threshold | None | bool | — | WideBindStack.forward | No |
| `global_readiness_ratio` | 124 | Property: fraction of mature layers | None | float | — | diagnostics | No |
| `update` | 128 | Update readiness EMA from per-layer pred_err | step, pred_err | None | torch.no_grad | WideBindStack.forward (end) | No |

---

### core/bridge.py (181 lines)

**Class: `SemanticBridge`** (line 31)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 32 | Per-layer semantic bridge with stream injection | D, n_layers, bridge_dim, depth, cfg | None | nn.Sequential, nn.Linear | WideBindStack.bridge | Yes (probe, emb_proj, stream_proj, stream_log_scale, stream_log_weights) |
| `readiness` | 79 | Scalar readiness [0,1] from bridge competence | None | tensor | torch.no_grad | MaturationController | No |
| `start_forward` | 93 | Reset predictions list for new forward | None | None | — | WideBindStack.forward | No |
| `probe_layer` | 96 | Emit semantic vector for a layer | h_l: (B,L,D) | (B,L,bridge_dim) | probe | WideBindStack.forward | Yes |
| `inject_layer` | 100 | Add cross-layer stream signal to hidden state | i, h_l, maturity | h_l: (B,L,D) | stream_proj, sigmoid weights | WideBindStack.forward | Yes |
| `update_stream` | 126 | EMA-update persistent stream | i, s_l | None | torch.no_grad | WideBindStack.forward | No |
| `record` | 132 | Record prediction for loss | s_l | None | — | WideBindStack.forward | No |
| `reset_stream` | 138 | Clear bridge stream | None | None | — | — | No |
| `loss` | 141 | Self-supervised bridge loss: predict next token embedding | y, embed_fn | scalar or None | F.cosine_similarity | compute_losses | Yes |

---

### core/vsa_utils.py (134 lines)

| Function | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|----------|------|-------------|--------|---------|--------------|-------------|-----------|
| `dct_basis` | 9 | DCT-II basis vectors (n,n) | n | tensor | math.cos | WideBindBlock, PartitionedHead, CognitiveCodedHead, compression | No |
| `zeckendorf_codes` | 19 | Fibonacci Zeckendorf binary codes | vocab | (V,K≈23) | — | ZeckendorfEmbedding, LmHead, compression | No |
| `fib_sigmoid_init` | 39 | Fibonacci-based sigmoid bias init | n, fib_vals | tensor | — | WideBindBlock.scale_w, bind, bottleneck | No |
| `sparse_block_codes` | 57 | Sparse block codes: exactly S ones from K per token | vocab, K, S | (V,K) | math.comb | PartitionedEmbedding, SigmoidCodedHead, CognitiveCodedHead, compression | No |
| `vsa_prefix_scan` | 90 | Associative parallel prefix scan for VSA memory | a, b, state | (result, final_state) | — | WideBindBlock (legacy, now inline scan) | No |

---

### core/lambda_utils.py (392 lines)

| Function/Class | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|----------------|------|-------------|--------|---------|--------------|-------------|-----------|
| `lambda_d` | 20 | Positive root of x^d = x^{d-1} + ... + 1 | d | float | — | LambdaConfig, config.py | No |
| `fib` | 41 | Classical Fibonacci | n | int | — | LambdaConfig | No |
| `generalized_fib` | 49 | d-step Fibonacci | n, d | int | — | — | No |
| `LambdaConfig` | 67 | All hyperparameters derived from λ_d | d | — | lambda_d, fib | config.py._apply_lambda_d | No |
| `spectral_radius` | 308 | Power iteration estimate of ρ(J) | model, h, n_steps, n_iters | float | torch.autograd.functional.jvp | diagnostics | No |

**LambdaConfig properties (all float):**
- exploration_threshold: λ⁻² ≈ 0.30
- differentiation_threshold: λ⁻⁴ ≈ 0.087
- mem2v_scale_min/max, ema_alpha_min/max, noise_scale_min/max
- delta_var_ema_min/max, warmup_steps, max_decay_steps
- target_var, mag_threshold, lr_min_ratio
- gate_lr_mult, pred_weight_max/min
- tanh_bias_mod_max, spectral_mod_lo/hi
- log_scale_init_std, conv_init_std, w_d_init_std
- eval_interval, save_interval, log_interval, patience

---

### core/adaptation.py (545 lines)

| Function/Class | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|----------------|------|-------------|--------|---------|--------------|-------------|-----------|
| `set_active_depth` | 74 | Freeze blocks with index >= k | model, k | int | — | DepthController | No |
| `DepthController` | 85 | Progressive unfreezing on val-loss plateau | model, n_layers, init_k, ... | — | set_active_depth | train.py | No |
| `build_optimizer` | 188 | AdamW with Layer-wise LR Decay (LLRD) | model, base_lr, llrd_decay, ... | AdamW | lambda_d | train.py | No |
| `LRController` | 220 | Warmup + mirror-adaptive LR + plateau damping | model, optimizer, cfg, ... | — | MirrorLRScheduler | train.py | No |
| `FailureDetector` | 294 | Statistical divergence detection (3σ rule) | model, lr_controller, ... | — | — | train.py | No |
| `GradientClipper` | 388 | Adaptive Gradient Clipping (AGC, scale-free) | c, eps | — | — | train.py | No |
| `LossBalancer` | 420 | Multi-task aux balancing (spectral alignment) | align, align_cap, eval_interval | — | torch.autograd.grad | train.py | No |

---

### core/projector.py (70 lines)

**Class: `Projector`** (line 24)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 27 | Readout of words from hidden state via concept signals | tokenizer | None | Tokenizer | — | No |
| `segment` | 30 | Static: write_event → list of (start,end) spans | write_event: (B,L) bool | list[list[(int,int)]] | — | read_words, concept_spans | No |
| `read_words` | 50 | Decode token spans to words | ids, write_event | list[list[str]] | segment | — | No |
| `concept_spans` | 61 | Get concept ID at end of each word | concept_id, write_event | list[list[int]] | segment | — | No |

---

### core/compression.py (361 lines)

| Function/Class | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|----------------|------|-------------|--------|---------|--------------|-------------|-----------|
| `is_removable` | 13 | Check if key is a deterministically-computable buffer | k | bool | — | FCF_CPR | No |
| `is_scalar_gate` | 16 | Check if tensor is uniform b_i/b_d (safe to scalar-fold) | k, v | bool | — | FCF_CPR | No |
| `quantize_tensor` | 25 | Uniform 8-bit quantization | t, n_bits | (indices, min, scale) | — | FCF_CPR | No |
| `dequantize_tensor` | 43 | Restore fp32 from uint8 | indices, min, scale, dtype | tensor | — | FCF_CPR | No |
| `quantize_tensor_channel` | 51 | Per-channel uniform 8-bit quantization | t, dim, n_bits | (indices, mins, scales) | — | FCF_CPR | No |
| `dequantize_tensor_channel` | 80 | Restore fp32 from per-channel uint8 | indices, mins, scales, orig_shape, dtype | tensor | — | FCF_CPR | No |
| `analyze_sd` | 94 | Print detailed state dict analysis | sd | groups dict | — | scripts | No |
| `FCF_CPR` | 152 | Checkpoint compressor class | — | — | — | generate.py | No |
| `FCF_CPR.compress_sd` | 156 | Compress model state dict | sd | (compressed, meta) | — | save_compressed | No |
| `FCF_CPR.decompress_sd` | 206 | Restore full state dict from compressed | compressed, meta, cfg | sd | dct_basis, sparse_block_codes | load_compressed | No |
| `FCF_CPR.save_compressed` | 242 | Save compressed checkpoint | ckpt, save_path | size | — | scripts | No |
| `FCF_CPR.load_compressed` | 269 | Load and decompress checkpoint | load_path, cfg | ckpt | — | generate.py | No |

---

### core/training_guard.py (34 lines)

Re-exports from `core.adaptation`: LossBalancer, DepthController, LRController, FailureDetector, GradientClipper, set_active_depth, build_optimizer.

### core/curriculum.py (64 lines)

**Class: `CurriculumTracker`** (line 13)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 25 | Loss-based chunk sampling with temperature decay | n_streams, tau_0, tau_min, decay_steps, momentum | None | — | — | No |
| `tau` | 39 | Property: temperature decay | None | float | math.exp | sample_probs | No |
| `update` | 44 | Update EMA loss for a stream | stream_idx, loss_val | None | — | — | No |
| `sample_probs` | 53 | p_i ∝ exp(L_i / τ) | cur_step | (n_streams,) | torch.softmax | — | No |

### core/live_inference.py (231 lines)

**Class: `MirrorMonitor`** (line 18)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 25 | Rolling tracer of internal model states | model, max_history | None | — | LiveInference | No |
| `clear` | 30 | Reset history | None | None | — | — | No |
| `capture` | 43 | Read internal metrics from all layers after forward | global_state | None | AdaptiveController.stats | LiveInference | No |
| `summary` | 93 | Return mean/std over last window steps | window | dict | torch.stack | — | No |

**Class: `LiveInference`** (line 118)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 141 | Continuous stateful inference wrapper | model, cfg, monitor, max_history | None | MirrorMonitor | scripts | No |
| `think` | 155 | Run internal self-dialogue steps (zero input loopback) | n_steps, h | (B,1,D) | model.forward | — | No |
| `respond` | 186 | Process actual input through live model | h | (B,L,D) | model.forward | — | No |
| `reset_state` | 206 | Reset all internal states | None | None | — | — | No |
| `generate` | 215 | think→prefill→generate tokens | prompt_ids, gen_len, think_steps | list[int] | think, respond, lm_head | — | No |

### core/word_num.py (92 lines)

| Function | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|----------|------|-------------|--------|---------|--------------|-------------|-----------|
| `n_of` | 41 | Product of prime-letters (anagram-invariant) | word | int | PHI | — | No |
| `v_of` | 49 | Polynomial (order-preserving) | word | int | _CODE, _BASE | — | No |
| `factors` | 57 | Factorize N back to letters | n | str | _PRIMES, INV | — | No |
| `gcd_of` | 69 | GCD of two words' compositions | a, b | int | n_of, math.gcd | — | No |
| `lcm_of` | 73 | LCM of two words' compositions | a, b | int | n_of, math.lcm | — | No |
| `morph_sim` | 77 | Morphological similarity in [0,1] | a, b | float | n_of, math.gcd, math.log | — | No |
| `log_size` | 89 | Logarithmic composition size | word | float | n_of | — | No |

### core/migrate.py (36 lines)

| Function | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|----------|------|-------------|--------|---------|--------------|-------------|-----------|
| `migrate_state_dict` | 12 | Migrate old checkpoint for new bind W_out +K, bind_coh_gate, freq_scale | sd, model | (sd, changed) | torch | train.py resume | No |

### core/amp_optim.py (154 lines)

| Function/Class | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|----------------|------|-------------|--------|---------|--------------|-------------|-----------|
| `build_amp_groups` | 46 | Build param_groups for AmpAdam by roles | model, lr, betas, eps, ... | list[dict] | — | — | No |
| `AmpAdam` | 97 | Adam with projection for factorized head | groups, betas, eps | — | torch.optim.Optimizer | — | No |

---

### scripts/train.py (656 lines)

| Function/Class | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|----------------|------|-------------|--------|---------|--------------|-------------|-----------|
| `_detach_state` | 24 | Recursively detach state tensors | st | detached st | — | training loop | No |
| `TokenStream` | 36 | Memory-mapped uint16 token stream | path | — | np.memmap | train | No |
| `TokenStream.get_batch` | 40 | Get batch of (x, y, offset) | seq_len, batch_size, offset, vocab | (x, y, offset) | torch.from_numpy | training | No |
| `create_lr_scheduler` | 54 | Linear warmup + cosine decay | optimizer, warmup, max_steps, lr | LambdaLR | — | (unused, replaced by MirrorLRScheduler) | No |
| `_opt_param_names` | 64 | Map optimizer params to names | model, optimizer | list[str] | — | _restore_optimizer | No |
| `_restore_optimizer` | 69 | Restore AdamW state by parameter name | optimizer, model, ckpt_opt | bool | — | resume | No |
| `train` | 140 | Main training loop | cfg, resume_path | None | WideBindStack, build_optimizer, LRController, DepthController, LossBalancer, FailureDetector, GradientClipper | __main__ | No |

**Training loop flow (train.py:140-656):**
1. Load data streams (token_stream_*_eos.bin)
2. Create WideBindStack, apply MLP depth gradient boost
3. Create optimizer (LLRD AdamW), LR controller, depth controller, loss balancer, watchdog, clipper
4. Resume from checkpoint if available
5. Main loop:
   - Get batch from random stream
   - Forward pass (h, state, global_state) = model(embed, state, ...)
   - Compute CE + aux losses
   - LossBalancer.backward (spectral alignment)
   - GradientClipper (AGC)
   - optimizer.step, scheduler.step
   - depth.update on eval boundaries
   - watchdog.check for divergence
   - Save checkpoints periodically

### scripts/smart_controller.py (275 lines)

| Function/Class | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|----------------|------|-------------|--------|---------|--------------|-------------|-----------|
| `lerp` | 22 | Linear interpolation | a, b, t | float | — | SmartController | No |
| `smooth` | 26 | Smoothstep [0,1] | x | float | — | SmartController | No |
| `SmartController` | 35 | Self-governing inference controller using model's own signals | model, vocab, reasoning_on, no_trunc | — | torch, model.debug_mind/meta_signals | generate.py --smart | No |
| `SmartController._compute_tau` | 63 | Per-layer VSA timescale → temporal profile | model | None | — | __init__ | No |
| `SmartController._entropy` | 79 | Shannon entropy of logits | logits | float | torch.softmax | decide | No |
| `SmartController._repetition` | 84 | N-gram repetition detector | None | bool | self.recent | decide | No |
| `SmartController._rep_pressure` | 93 | Continuous repetition pressure signal | None | float | self.recent, alarm_window | decide | No |
| `SmartController.decide` | 105 | Per-token generation parameter decision | logits, mind, step | (temp, top_p, top_k, rep_pen, alpha) | — | smart_generate | No |
| `SmartController.sample` | 179 | Sample token with repetition penalty + top-p/top-k | logits, temp, top_p, top_k, rep_pen | int | F.softmax, torch.multinomial | smart_generate | No |
| `smart_generate` | 198 | Generate with SmartController | model, prompt, controller, max_new_tokens, ... | (text, decisions) | — | scripts | No |

### scripts/generate.py (445 lines)

| Class | Line | Description |
|-------|------|-------------|
| `AdaptiveSampler` | 16 | Self-governing sampling with entropy-based exploration/exploitation |

### scripts/analyze.py (1616 lines)

Full checkpoint analyzer: static analysis, live forward, head decomposition, gradient diagnostics, anomaly tracking, checkpoint comparison.

---

## 3. FORWARD PASS FLOW

### 3.1 Complete Data Flow (Training)

```
Token IDs (B, L)
    │
    ▼
PartitionedEmbedding.forward()
    ├─ codes[tokens] → (B,L,K) sparse binary
    ├─ sigmoid(embed_mix · codes) → (B,L,K) dense mixing
    ├─ outer(codes, basis) → (B,L,D) embedding
    └─ RoPE → (B,L,D) positional
    │
    ▼
[Optional: StreamingMemoryBank.forward(h, tokens)]
    ├─ L1.read(h) → mem_l1 (B,L,D)
    ├─ L2.read(h) → mem_l2 (B,L,D)
    ├─ L3.read(h) → mem_l3 (B,L,D)
    └─ h = h + scale * fusion(cat(h, mem_l1, mem_l2, mem_l3))
    │
    ▼
For each layer i (0..n_layers-1):
    │
    ├─ [Intent Bridge] intent_probe(h) → intent_i (1,1,G,Kmax)
    │   └─ truncate to layer k, scale by maturation
    │
    ├─ [Semantic Bridge] bridge.inject_layer(i, h)
    │   └─ h += scale * stream_proj(combined_neighbors)
    │
    ├─ [Memory Bank] memory_bank(h, tokens) per layer (if enabled)
    │
    ├─ WideBindBlock.forward(h, state, global_state, ...)
    │   │
    │   ├─ Pre-LN: h = pre_ln_w * h / sqrt(mean(h²))
    │   │
    │   ├─ Conv: h += conv1d(h)  (depthwise, causal)
    │   │
    │   ├─ Bind: bind_out = BottleneckBind(h) or TrajectorySpiralBind(h)
    │   │   ├─ hp = W_proj(h) + bias  → (B,L,K)
    │   │   ├─ bilinear mixing with shifts/trajectory
    │   │   └─ bind_out = mixed @ W_out  → (B,L,D)
    │   │
    │   ├─ VSA Memory (multi-scale, S=4):
    │   │   ├─ i_gate = softplus(h*w_i + b_i + γ*pred_error)
    │   │   ├─ d_mod = sigmoid(h*w_d + b_d)
    │   │   ├─ decay = tau_scale * d_mod  (per-scale per-channel)
    │   │   ├─ mem_input = h * i_gate * dynamic_write_mod
    │   │   ├─ Parallel prefix scan (4 scales, chunked)
    │   │   ├─ Weighted combination: mem_all = Σ sigmoid(scale_w) * mem_per_scale
    │   │   ├─ mem_read = mem_all * w_q + mem_leaf * w_q_leaf + mem_all * w_q_ctx
    │   │   └─ mem_read += mu_read * w_mu_mem
    │   │
    │   ├─ Mirror: mirror, mlp_mod, mem_mod, hp, pred_error_norm
    │   │   = GroupedCognitiveMirror(h, mem_all, global_state, ...)
    │   │   ├─ Split h → h_g (B,L,G,d)
    │   │   ├─ Project: hp = einsum(h_g, W_proj) → (B,L,G,k)
    │   │   ├─ pos_id binding: hp *= pos_id_buf
    │   │   ├─ Signals:
    │   │   │   ├─ temp_k = (hp - mc_k) * w_temp
    │   │   │   ├─ pred_error = (hp - hp_prev*alpha) / hp_norm
    │   │   │   ├─ smooth_k = hp - conv_smooth(hp)
    │   │   │   ├─ sym_k = (hp*w_sym_u) * (hp_prev*w_sym_v)
    │   │   │   └─ [help_k from private memory]
    │   │   ├─ EMA-normalize each signal
    │   │   ├─ Weighted merge: delta = Σ sigmoid(w_sig) * signal_normed
    │   │   ├─ RMS-normalize delta
    │   │   ├─ Usefulness predictor → gate logits
    │   │   ├─ Expert gate: sigmoid(gate_logits + grad_mod + dvar_mod + intent)
    │   │   ├─ Mirror out: tanh(W_out·delta) + skip*linear, scaled by log_scale
    │   │   ├─ SMF gate: alpha = sigmoid(cat(h,m)·w_alpha + b_alpha)
    │   │   ├─ BridgeGLU: glu = sigmoid(Wg·delta) * sigmoid(Wv·delta)
    │   │   └─ MLP mod: base = hybrid_gate(usefulness) * (1 + β*(2*glu-1))
    │   │
    │   ├─ Output:
    │   │   ├─ bind_gated = bind_out * mem_mod * bind_gate
    │   │   ├─ mem_modulated = mem_read * read_mod * mem_mod
    │   │   ├─ enhanced = bind_gated + mem_modulated*w_mem2v + mirror
    │   │   ├─ [CollectiveConceptLayer output added]
    │   │   └─ h += enhanced
    │   │
    │   ├─ Variable Precision Memory (optional):
    │   │   ├─ precision = sigmoid(linear(h))
    │   │   └─ h += precision * exact_memory(h) * gate
    │   │
    │   ├─ Spectral:
    │   │   ├─ h_dct = h @ V_dct.T * lambda_k * spectral_mod
    │   │   └─ h += h_dct @ V_dct
    │   │
    │   └─ MLP:
    │       ├─ h_mlp = GroupedMLP(h, mirror_gate=mlp_mod)
    │       ├─ SwiGLU: silu(W_gate·h) * (a + b*mlp_mod) * W_up·h → W_down
    │       └─ h += h_mlp
    │
    ├─ [Bridge probe] bridge.record(probe(h))
    ├─ [Bridge update] bridge.update_stream(i, probe(h))
    ├─ [Maturation update] pred_errs.append(mirror._cached_pred_error_norm)
    └─ [Global state update] global_state[i] = α * gs + (1-α) * mem_avg
    │
    ▼
After all layers:
    ├─ h = final_norm_w * h / sqrt(mean(h²))
    │
    ├─ [Maturation update] maturation.update(step, pred_errs)
    │
    ├─ [Explicit Reasoning] (if enabled and scale > 0):
    │   └─ _adaptive_reasoning(h, s, state, ...):
    │       ├─ know = _knowledge_signal(h)  (B,8)
    │       ├─ For i in 0..K-1:
    │       │   ├─ r_i = ReasoningMemory(h, buf, count)
    │       │   ├─ l_i = ReasoningGate.logits(h_acc, know, r_i)[..., i]
    │       │   ├─ a_i = tanh(l_i)  (gate ∈ (-1,1))
    │       │   ├─ contrib = a_i * normalize(r_i) * w_soft * run_mask
    │       │   └─ h_acc += s * contrib / denom_accum
    │       └─ return h_acc
    │
    └─ [Triad re-circulation] (inference only, if confidence < threshold):
        ├─ _conf = _last_conf(h).mean()
        └─ if _conf < triad_conf_thr:
            h2 = self.forward(h, ..., _triad_depth+1)  (recursive)
            h = 0.5*h + 0.5*h2
    │
    ▼
Output: (B, L, D)
    │
    ▼
SigmoidCodedHead.forward(h, bus_bias):
    ├─ z = (h_g * readout).sum(-1) / T + bit_bias + bus_bias
    ├─ gate = sigmoid(z) * (1 + softmax(z/tau))
    ├─ log_odds = log(gate/(1-gate))
    ├─ logits = log_odds @ codes.T + log(1-gate).sum(-1) + token_bias
    └─ logits = logits - logsumexp(logits)  (normalized)
    │
    ▼
Output: (B, L, vocab) logits
```

### 3.2 Memory Bank Read/Write at Each Stage

**Write** (at sentence boundaries, token == 2):
- L1: overwrite oldest slot with sentence mean
- L2: overwrite oldest/consumed slot with projected key+val
- L3: if L2 key matches existing concept → update; else → birth (mark L2 consumed)

**Read** (every token position):
- L1: hybrid attention over rolling buffer
- L2: hybrid attention over learned slots
- L3: hybrid attention over emergent concepts
- Fusion: cat(h, mem_l1, mem_l2, mem_l3) → MLP → scale * fused → inject into h

### 3.3 Mirror Expert Processing

Each of 32 experts operates in its own d=128 subspace:
1. **Project** to K-space (k=8/16/32)
2. **Compute 4+ signals**: temporal, predictive, smooth, symmetry, [help from private memory]
3. **EMA-normalize** each signal independently
4. **Merge** with learnable Fibonacci-initialized weights
5. **Gate** using usefulness predictor + gradient/delta modulation
6. **Output**: tanh(linear) + skip connection, scaled by log_scale

### 3.4 Reasoning Cycle

1. `_knowledge_signal(h)` → 8-dim vector (confidence, entropy, mem agreement, etc.)
2. For each step i (0..K-1):
   - `r_i = ReasoningMemory(h, buffer)` → encode current, attend to past
   - `l_i = ReasoningGate.logits(h_acc, know, r_i)` → step gate logit
   - `a_i = tanh(l_i)` → gate value (-1, 1)
   - If previous gate closed → step doesn't run (run_mask)
   - `contrib = a_i * normalize(r_i)` → gated, normalized contribution
   - Validate: check if confidence improves → w_soft scales contribution
   - `h_acc += s * accum / Σ|w_i|` → accumulated reasoning
3. Buffer updated only for committed steps

### 3.5 Maturation Gates

`M_l(t) = sigmoid((t - (T0 + α*(1-τ_norm)*T_delay)) / δt)`

- **Deep layers** (τ_norm≈1): open at t ≈ T0 = 8000 steps
- **Shallow layers** (τ_norm≈0): open at t ≈ T0 + T_delay = 16000 steps
- Gates: BridgeGLU, private-memory write, bridge injection, intent bus

---

## 4. TRAINING FLOW

### 4.1 Loss Computation

**Primary loss:**
- Cross-entropy (CE) on logits vs targets, with PAD/EOS masking and optional surprisal weighting

**Auxiliary losses (all raw, unweighted):**

| Loss | Weight Config | Description |
|------|--------------|-------------|
| `pred` | 0 | MSE(pred_k, hp.detach()) — temporal prediction |
| `gate_l1` | gate_l1_weight=0.0001 | Mean expert gate (sparsity) |
| `reinforce` | reinforce_weight=0.001 | MSE(usefulness, gate) — align gate with self-assessment |
| `balance` | balance_weight=0.026 | HHI-based load balancing (uniform expert usage) |
| `diversity` | diversity_weight=0.001 | ||cov(group_out) - I||² — decorrelate groups |
| `nuc` | nuclear_weight=1e-5 | Stochastic nuclear norm of bind W_proj |
| `orth` | orth_weight=1e-4 ||Ŵ^TŴ - I||² — orthogonality of bind W_proj |
| `div` | div_weight=50.0 | sigmoid-bounded log_scale divergence |
| `ranking` | ranking_weight=0.01 | Pairwise order loss: ls_mean vs gate_usage |
| `gate_repulse` | gate_repulse_weight=0.3 | Push gate variance up |
| `alpha_novelty` | alpha_novelty_weight=0.05 | Push per-expert alpha apart |
| `decorr` | 0 | Decorrelation of weighted mirror signals |
| `signal_ent` | 0 | Entropy of signal weights (diversity) |
| `ls_reg` | 0 | L2 on log_scale excess above 2.3 |
| `branch` | branch_balance_weight=0.0 | Equalize log-variance of conv/bind/mirror |
| `w_m2v` | w_m2v_hierarchy_weight=0.01 | Drive w_mem2v toward τ-hierarchy target |
| `intent_tau` | intent_tau_hierarchy_weight=0.01 | Drive intent τ-deviation toward target |
| `bridge_conn` | bridge_conn=0.1 | Bridge cosine loss: predict next-token embedding |
| `pred_w` | 0 | MSE(pred_w, I) — identity prediction weights |

### 4.2 Gradient Flow

1. **CE loss** → through head → through all layers → parameters
2. **Aux losses** → LossBalancer.backward:
   - Compute `g_CE = ∇ CE`
   - Compute `g_aux = ∇ (Σ aux_i)`
   - `cos = ⟨g_CE, g_aux⟩ / (||g_CE||·||g_aux||)`
   - `scale = clamp(cos, 0, cap) * ||g_CE|| / (||g_aux|| + ε)`
   - `p.grad = g_CE + scale * g_aux` (only when cos > 0)
3. **GradientClipper** (AGC): clip iff `||g|| > c·||θ||`
4. **MLP depth boost**: `p.grad *= exp(mlp_depth_lr_exp * layer_idx)` for MLP params

### 4.3 Optimizer Groups (λ_d hierarchy, default)

| Group | LR multiplier | Parameters |
|-------|--------------|------------|
| embed | λ⁻² ≈ 0.296 | embed.embed_mix, embed.basis |
| embed_wd | λ⁻² | (same with weight_decay) |
| mlp | λ⁻¹ ≈ 0.544 | mlp.W_*, mlp.norm_w, mlp.mlp_gate_*, bind.W_proj |
| mirror | λ¹ ≈ 1.839 | mirror.W_proj, mirror.W_out, mirror.alpha_diag, mirror.log_scale, mirror.tanh_bias, mirror.w_temp, mirror.w_global |
| gate | λ¹ | mirror.w_gate, mirror.b_gate, mirror.w_delta_gate, mirror.gate_bias, w_i, w_d, w_q |
| vsa | λ⁻² | b_d, b_i, scale_w |
| bridge | 1.0 (base) | bridge.*, bridge_glu_net.*, intent_probe, bus_head_proj, layer_bridge_gate.* |
| default | 1.0 | everything else |

**Layer-wise LR decay (LLRD):** `lr_layer_i = lr_group * llrd^i` (llrd=0.9)

### 4.4 What Gets Updated vs Frozen

**Always trainable:**
- Embedding: embed_mix, basis
- All bind params: W_proj, w_u, w_v, W_out, mix_logit, log_tau
- Mirror: W_proj, W_out, alpha_diag, log_scale, tanh_bias, w_temp, w_global, w_gate, b_gate, all signal weights
- MLP: W_gate, W_up, W_down, norm_w, mlp_gate_a, mlp_gate_b
- VSA: w_i, w_d, w_q, w_q_leaf, w_q_ctx, w_mem2v, b_i, b_d, scale_w, w_k_mu, w_q_mu, w_mu_mem
- Spectral: lambda_k
- Head: readout, bit_bias, log_temp, token_bias
- Bridge: probe, emb_proj, stream_proj, stream_log_scale, stream_log_weights
- Intent: intent_probe, bus_head_proj
- Maturation: None (all buffers)
- Reasoning: step_encoder, step_query/key/value, output_proj, reasoning_gate

**Frozen (by DepthController):**
- Blocks with index >= k are frozen (requires_grad=False)
- Default: first 8 blocks active, rest frozen until plateau triggers unfreezing

**Maturation-gated (effectively frozen at init):**
- BridgeGLU gates (live but scaled by maturity → ~0 at start)
- Private memory writes (gated by maturity > threshold)
- Bridge injection (scaled by maturity)
- Intent bus (scaled by maturity)

---

## 5. BOTTLENECKS & ISSUES

### 5.1 Dead Code Paths

1. **`ZeckendorfEmbedding`** (embedding.py:45): Legacy, replaced by `PartitionedEmbedding`. Not used in default config.

2. **`LmHead`** (embedding.py:114): Legacy Zeckendorf head. Replaced by `SigmoidCodedHead`. Only used if `head_mode='zeckendorf'`.

3. **`PartitionedHead`** (embedding.py:129): Alternative head, not used by default (`head_mode='sigmoid_coded'`).

4. **`vsa_prefix_scan`** (vsa_utils.py:90): Standalone function, now inlined in `WideBindBlock.forward` with chunked parallel scan. Dead code.

5. **`create_lr_scheduler`** (train.py:54): Returns a cosine LambdaLR scheduler. Replaced by `MirrorLRScheduler` / `LRController`. Dead code.

6. **`SpiralBind`** (bind.py:235): Not used in default config (bind_twist_mode='trajectory_spiral'). Only instantiated if `bind_mode == "spiral"`.

7. **`AdaptiveGateBundle`** (adaptive_gate.py:100): Generic bundle, only used as `self.hybrid_gate = AdaptiveGate(...)` inside mirror. The Bundle class is unused.

8. **`SpectrumGateBundle`** (spectrum_gate.py:136): Generic bundle, not used anywhere in the codebase.

9. **`CurriculumTracker`** (curriculum.py:13): Not imported or used in train.py or any other module.

10. **`AmpAdam`** (amp_optim.py:97): Custom optimizer for AMP codec head. Not used with `SigmoidCodedHead` (the default head).

### 5.2 Unused Parameters

1. **`lm_head.proj`** in `LmHead` — legacy head, not used.

2. **`collective_layer_idx = None`** — when None, collective is added to EVERY layer (not just one). This is expensive (32 CollectiveConceptLayers).

3. **`orth_weight: float = 0.0`** (line 149) is defined TWICE in WideBindConfig (also line 238). The second definition (`1e-4`) overrides the first (`0.0`). This is confusing.

4. **`_damp_tau`** in mirror (line 295): Only used during inference (not self.training), and the dampening effect on alpha_eff is minimal.

5. **`_pos_id_buf`** in mirror (line 220): Deterministic position code, registered as buffer. Used in forward but never learned. Potential redundancy with RoPE in embedding.

### 5.3 Gradient Flow Problems

1. **`pred_loss`** (stack.py:852-856): `F.mse_loss(pred_k, hp.detach())` — hp is DETACHED, so gradient only flows through pred_k (alpha_diag). The main loss path through hp is severed. This is intentional but means pred_loss doesn't help the trunk learn better representations.

2. **`bridge.loss`** (bridge.py:141): Uses `h.detach()` for the probe — gradient from bridge loss doesn't flow into the trunk. Only the probe parameters learn. This is documented as intentional (prevents CE divergence).

3. **`SemanticBridge.inject_layer`** (bridge.py:100): Stream injection has gradient through `stream_proj`, but the stream itself is EMA-updated (detached). No cross-step BPTT through the stream.

4. **`Private memory write`** (mirror.py:494-512): All writes under `torch.no_grad()`. No gradient through memory updates. Memory is a pure buffer.

5. **`MaturationController.update`** (maturation.py:128): All updates under `torch.no_grad()`. Maturity is a pure buffer, not learnable.

6. **VSA prefix scan** (block.py:329-366): Uses `torch.no_grad()` for the log-space scan? No — it's under regular autograd. But the fp32 cast (`decay.float()`) breaks gradient flow in fp16 training (needs explicit `.to(dtype)` for grad).

### 5.4 Redundant Calculations

1. **`AdaptiveController.stats`** is called in the outer loop (stack.py:201) AND again inside the per-layer loop (stack.py:320). The outer call computes global averages; the inner call recomputes per-layer stats. This means each layer's (expl, diff) is computed twice — once for global, once for local. Could cache per-layer results.

2. **`_knowledge_signal`** (stack.py:574): Called in `_adaptive_reasoning` at each reasoning step. It runs `self.lm_head(h)` which is a full forward through the head — expensive at 8 reasoning steps × 32 layers.

3. **`debug_mind`** (mirror.py:744): Syncs ~10 tensors to CPU per call. Only used in SmartController, but if called in the hot loop, it would stall the GPU. `meta_signals()` is the optimized alternative.

4. **Memory bank `forward`** (memory_bank.py:534): Runs L1/L2/L3 reads at EVERY token position, but writes only at sentence boundaries. The read is O(n_slots) per position — expensive for long sequences.

5. **`SigmoidCodedHead.forward`** (embedding.py:221): Computes full logits for ALL vocab tokens, then normalizes. `log_probs_for_target` (line 235) is the efficient path for training (only computes target log-probs). Ensure training uses `log_probs_for_target`, not `forward`.

6. **`Spectral transform`** (block.py:476-479): `h @ V_dct.T` then `result @ V_dct` — two matrix multiplications of (B,L,D) × (D,D). For D=4096, this is 4096² = 16M multiplications per layer per direction. The DCT basis is orthogonal, so this could be done with FFT in O(D log D).

### 5.5 Methods That Do Similar Things

1. **`BridgeGLU`** (mirror.py:11) vs **`AdaptiveGate`** (adaptive_gate.py:32) vs **`SpectrumGate`** (spectrum_gate.py:33, layer_bridge_gate.py:20): All three implement sigmoid-softmax hybrid gating with minor variations:
   - BridgeGLU: GLU variant (Wg·delta, Wv·delta, log_gain)
   - AdaptiveGate: direct sigmoid(softmax) blend with learnable tau
   - SpectrumGate: same formula as AdaptiveGate with tau_external support

2. **`L1Buffer.read`**, **`L2Bank.read`**, **`L3Concepts.read`**: All use the same `_memory_attention` function with identical code patterns. Could be unified into a base class.

3. **`SigmoidCodedHead._su`** and **`AdaptiveGate.forward`**: Both compute `sigmoid(x) * (1 + softmax(x/tau))`. The head's version returns log-space; the gate returns raw. Same mathematical operation, different APIs.

4. **`compute_salience`** (stack.py:792) and **`CognitiveCodedHead._compute_z`** (embedding.py:313): Both extract per-position importance from the head output, but with different formulas and purposes.

### 5.6 Architecture Concerns

1. **No feed-forward residual in bind**: The bind module (BottleneckBind) goes D→K→D, but the residual is only `h = h + enhanced` AFTER combining bind + memory + mirror. There's no dedicated residual around bind itself. If bind output is large, it could dominate.

2. **VSA memory write/read asymmetry**: Writes use content-dependent gating (w_i, b_i) but reads use a simpler linear combination (w_q, w_q_leaf, w_q_ctx). The read path may not be expressive enough to selectively retrieve from the multi-scale memory.

3. **32 CollectiveConceptLayers**: When `collective_layer=True` and `collective_layer_idx=None`, a CollectiveConceptLayer is created for EVERY block. Each has S=8 concept slots with attention over K-space. This is 32 × 8 × K parameters just for concept management.

4. **Private memory per expert**: Each of 32 layers has G=32 experts, each with a (k,) private memory vector. Total: 32 × 32 × 32 = 32K private memory entries — small but spread thin.

5. **Reasoning buffer**: Fixed-size (B, max_steps=8, D) tensor. At 8 steps of D=4096, that's 128KB per batch element. For batch_size=2, that's 256KB. Negligible.

6. **Intent bridge bus**: `_K_max=32` (max mirror k). Stream is (1,1,G=32,K_max=32) per layer. 32 layers × 32 × 32 = 32K values. Negligible.

### 5.7 Performance Bottlenecks

1. **Prefix scan** (block.py:329-374): The VSA memory uses a 2-level chunked prefix scan. For L=128 and CHUNK=32, that's 4 chunks × 2 levels = 8 scan operations per layer per scale. With S=4 scales, that's 32 scan operations per layer. For 32 layers: 1024 prefix scans. Each involves log-space cumsum + exponentiation. This is the most expensive part of the forward pass.

2. **DCT spectral transform** (block.py:476-479): Two D×D matrix multiplications per layer. For D=4096 and 32 layers: 32 × 2 × 4096² ≈ 1B FLOPs.

3. **Private memory cross-attention** (mirror.py:409-428): G×G attention per token (32×32=1024 attention scores). For B=2, L=128: 2 × 128 × 1024 = 262K sigmoid operations.

4. **Reasoning loop** (stack.py:631-737): Up to K=8 iterations, each involving ReasoningMemory (linear + attention) + ReasoningGate (3 linear projections) + _last_conf (full head forward). The head forward is the most expensive part.

5. **Triad re-circulation** (stack.py:552-570): Recursively calls the entire forward pass up to 3 times when confidence is low. Each re-circulation is a full forward pass. Bounded by `triad_max_passes=3`.

---

*Report generated from analysis of all 54 Python files in the WideBind/EVA codebase.*
*Total estimated lines of code: ~10,000+ across core/ and scripts/.*
