"""Streaming Memory Bank for EVA — hierarchical L1 + L2 + L3.

Architecture:
  L1 (immediate): rolling buffer of last K sentence embeddings
    - No learning, just storage + attention read
    - Always active, no maturation gating

  L2 (learned): VSA-mediated memory bank with N slots
    - Write: at sentence boundaries (SEP token = 2)
    - Read: attention over slots at each token
    - Selective: novelty-based write gating
    - Differentiable: gradients flow through write/read

  L3 (concepts): emergent from L2 slot clustering
    - Cluster L2 keys by cosine similarity
    - Concept birth: when cluster confidence > threshold
    - Concept update: running mean of cluster members
    - Read: attention over concepts (higher-level abstractions)
    - Long-range memory (tau ~ 500+)

Integration:
  - Sits AFTER embedding, BEFORE first layer
  - Write at sentence boundaries (token == 2)
  - Read at every token position
  - NOT gated by maturation (always active)
  - Maturation controls depth of processing, not memory access

Design principles from EVA:
  - Softmax-free (sigmoid attention) — regime B
  - Bridge dim matches bridge_dim config
  - Compatible with gradient checkpointing
  - Persistent buffers for inference
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Buffer(nn.Module):
    """Rolling buffer of last K sentence embeddings.

    Simple storage: at each sentence boundary, append the mean embedding.
    Read: attention over buffer contents.
    No learning needed (just projections for compatibility).
    """
    def __init__(self, D: int, bridge_dim: int, n_slots: int = 3):
        super().__init__()
        self.D = D
        self.bridge_dim = bridge_dim
        self.n_slots = n_slots
        # Project stored embeddings to bridge space (for keys)
        self.proj = nn.Linear(D, bridge_dim)
        # Query projection for attention read
        self.q_proj = nn.Linear(D, bridge_dim)
        # Output projection: buf is (n_slots, D), read output is (B, L, D)
        self.out_proj = nn.Linear(D, D)
        # Learnable temperature for attention
        self.log_temp = nn.Parameter(torch.tensor(1.0))

        # Persistent buffer: (n_slots, D)
        self.register_buffer('buf', torch.zeros(n_slots, D), persistent=True)
        self.register_buffer('buf_age', torch.zeros(n_slots), persistent=True)
        self._write_idx = 0

    def write(self, embedding: torch.Tensor) -> None:
        """Write sentence embedding to buffer (round-robin)."""
        slot = self._write_idx % self.n_slots
        self.buf.data[slot] = embedding.detach().float()
        self.buf_age.data[slot] = 0.0
        self.buf_age.data[slot != torch.arange(self.n_slots, device=self.buf_age.device)] += 1.0
        self._write_idx += 1

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Read from buffer using attention.

        query: (B, L, D) — current hidden state
        returns: (B, L, D) — memory read output
        """
        B, L, _ = query.shape
        q = self.q_proj(query)  # (B, L, bridge_dim)
        k = self.proj(self.buf)  # (n_slots, bridge_dim)
        v = self.out_proj(self.buf)  # (n_slots, D)

        temp = torch.exp(self.log_temp).clamp(min=0.1, max=10.0)
        attn = torch.sigmoid((q @ k.T) / math.sqrt(self.bridge_dim) * temp)  # (B, L, n_slots)

        read = (attn @ v)  # (B, L, D)
        return read

    def reset(self) -> None:
        self.buf.zero_()
        self.buf_age.zero_()
        self._write_idx = 0


class L2Bank(nn.Module):
    """Learned memory bank with N slots.

    Write: VSA-mediated (bind summary with context).
    Read: attention over slots.
    Selective: novelty gate (write only if different from existing).
    """
    def __init__(self, D: int, bridge_dim: int, n_slots: int = 16):
        super().__init__()
        self.D = D
        self.bridge_dim = bridge_dim
        self.n_slots = n_slots

        self.W_k = nn.Linear(D, bridge_dim)
        self.W_v = nn.Linear(D, bridge_dim)
        self.W_o = nn.Linear(bridge_dim, D)

        self.q_proj = nn.Linear(D, bridge_dim)

        self.keys = nn.Parameter(torch.randn(n_slots, bridge_dim) * 0.02)
        self.vals = nn.Parameter(torch.randn(n_slots, bridge_dim) * 0.02)

        self.novelty_gate = nn.Sequential(
            nn.Linear(D, bridge_dim),
            nn.GELU(),
            nn.Linear(bridge_dim, 1),
        )

        self.log_temp = nn.Parameter(torch.tensor(1.0))

        self.register_buffer('slot_age', torch.zeros(n_slots), persistent=True)
        self.register_buffer('slot_novelty', torch.ones(n_slots), persistent=True)
        self._write_idx = 0

    def write(self, embedding: torch.Tensor) -> bool:
        """Write to bank if novelty gate allows.

        embedding: (D,) — sentence summary
        returns: True if wrote
        """
        novelty_score = torch.sigmoid(self.novelty_gate(embedding))

        if novelty_score.item() < 0.3 and self._write_idx >= self.n_slots:
            return False

        slot = self._write_idx % self.n_slots
        with torch.no_grad():
            new_key = self.W_k(embedding.detach())
            new_val = self.W_v(embedding.detach())

        self.keys.data[slot] = new_key
        self.vals.data[slot] = new_val
        self.slot_age.data[slot] = 0.0
        self.slot_novelty.data[slot] = novelty_score.item()
        mask = torch.arange(self.n_slots, device=self.slot_age.device) != slot
        self.slot_age.data[mask] += 1.0
        self._write_idx += 1
        return True

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Read from bank using attention.

        query: (B, L, D)
        returns: (B, L, D)
        """
        B, L, _ = query.shape
        q = self.q_proj(query)  # (B, L, bridge_dim)
        k = self.keys  # (n_slots, bridge_dim)
        v = self.vals  # (n_slots, bridge_dim)

        temp = torch.exp(self.log_temp).clamp(min=0.1, max=10.0)
        attn = torch.sigmoid((q @ k.T) / math.sqrt(self.bridge_dim) * temp)

        age_decay = torch.exp(-0.01 * self.slot_age)
        attn = attn * age_decay.unsqueeze(0).unsqueeze(0)

        attn_sum = attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        attn = attn / attn_sum

        read = attn @ v  # (B, L, bridge_dim)
        return self.W_o(read)  # (B, L, D)

    def reset(self) -> None:
        self.keys.data.zero_()
        self.vals.data.zero_()
        self.slot_age.zero_()
        self.slot_novelty.zero_()
        self._write_idx = 0


class L3Concepts(nn.Module):
    """Emergent concept layer — clusters L2 slots into higher-level abstractions.

    Mechanism:
      - At each boundary write, check L2 key against existing concepts
      - If cosine similarity > threshold: update concept (running mean)
      - If no match and confidence > threshold: birth new concept
      - Read: attention over concepts (higher-level than L2)

    Concept birth is triggered by:
      1. L2 key doesn't match any existing concept
      2. The new key is confident enough (from L2 novelty gate)
      3. An empty slot is available (or least-used concept is evicted)

    Concept death: unused concepts decay via age and get evicted.
    """
    def __init__(self, D: int, bridge_dim: int, n_concepts: int = 8,
                 birth_threshold: float = 0.7, update_momentum: float = 0.1):
        super().__init__()
        self.D = D
        self.bridge_dim = bridge_dim
        self.n_concepts = n_concepts
        self._birth_threshold = birth_threshold
        self._update_momentum = update_momentum

        # Concept keys/values (learnable)
        self.concept_keys = nn.Parameter(torch.randn(n_concepts, bridge_dim) * 0.02)
        self.concept_vals = nn.Parameter(torch.randn(n_concepts, bridge_dim) * 0.02)

        # Query projection for read
        self.q_proj = nn.Linear(D, bridge_dim)
        self.out_proj = nn.Linear(bridge_dim, D)

        # Temperature
        self.log_temp = nn.Parameter(torch.tensor(1.0))

        # Tracking
        self.register_buffer('concept_age', torch.zeros(n_concepts), persistent=True)
        self.register_buffer('concept_count', torch.zeros(n_concepts), persistent=True)
        self.register_buffer('concept_confidence', torch.zeros(n_concepts), persistent=True)
        self._n_births = 0

    @torch.no_grad()
    def write(self, l2_key: torch.Tensor, confidence: float = 0.5) -> bool:
        """Try to write L2 key into a concept slot.

        l2_key: (bridge_dim,) — the L2 key that was written
        confidence: float — novelty confidence from L2 gate
        returns: True if wrote (updated or birthed)
        """
        # Normalize for cosine similarity
        key_n = F.normalize(l2_key.unsqueeze(0), dim=-1)  # (1, bridge_dim)
        concept_n = F.normalize(self.concept_keys.data, dim=-1)  # (n_concepts, bridge_dim)

        # Cosine similarity to existing concepts
        sims = (key_n @ concept_n.T).squeeze(0)  # (n_concepts,)

        # Find best match
        best_sim, best_idx = sims.max(0)
        best_idx = best_idx.item()
        best_sim = best_sim.item()

        # Update existing concept if similarity > threshold
        if best_sim > self._birth_threshold:
            alpha = self._update_momentum
            self.concept_keys.data[best_idx] = F.normalize(
                self.concept_keys.data[best_idx] * (1 - alpha) + l2_key * alpha, dim=-1)
            self.concept_vals.data[best_idx] = (
                self.concept_vals.data[best_idx] * (1 - alpha) + l2_key * alpha)
            self.concept_count.data[best_idx] += 1
            self.concept_confidence.data[best_idx] = (
                self.concept_confidence.data[best_idx] * 0.9 + best_sim * 0.1)
            self.concept_age.data[best_idx] = 0.0
            return True

        # Birth: find empty slot or evict least-used concept
        empty = torch.nonzero(self.concept_count == 0)
        if empty.numel() > 0:
            idx = empty[0].item()
        elif confidence > self._birth_threshold:
            # Evict concept with lowest confidence * count (least established)
            utility = self.concept_confidence * torch.clamp(self.concept_count, min=1)
            idx = utility.argmin().item()
        else:
            return False

        # Birth or overwrite
        self.concept_keys.data[idx] = F.normalize(l2_key.unsqueeze(0), dim=-1).squeeze(0)
        self.concept_vals.data[idx] = l2_key.clone()
        self.concept_age.data[idx] = 0.0
        self.concept_count.data[idx] = 1
        self.concept_confidence.data[idx] = confidence
        self._n_births += 1
        return True

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Read from concepts using attention.

        query: (B, L, D)
        returns: (B, L, D)
        """
        B, L, _ = query.shape
        q = self.q_proj(query)  # (B, L, bridge_dim)
        k = self.concept_keys  # (n_concepts, bridge_dim)
        v = self.concept_vals  # (n_concepts, bridge_dim)

        temp = torch.exp(self.log_temp).clamp(min=0.1, max=10.0)
        attn = torch.sigmoid((q @ k.T) / math.sqrt(self.bridge_dim) * temp)  # (B, L, n_concepts)

        # Age-based decay
        age_decay = torch.exp(-0.005 * self.concept_age)  # slower decay than L2 (concepts are long-range)
        attn = attn * age_decay.unsqueeze(0).unsqueeze(0)

        # Normalize
        attn_sum = attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        attn = attn / attn_sum

        read = attn @ v  # (B, L, bridge_dim)
        return self.out_proj(read)  # (B, L, D)

    def get_active_concepts(self) -> int:
        """Number of concepts with count > 0."""
        return int((self.concept_count > 0).sum().item())

    def reset(self) -> None:
        self.concept_keys.data.zero_()
        self.concept_vals.data.zero_()
        self.concept_age.zero_()
        self.concept_count.zero_()
        self.concept_confidence.zero_()
        self._n_births = 0


class StreamingMemoryBank(nn.Module):
    """Combined L1 + L2 + L3 memory bank for EVA.

    Integration points:
    - forward(h, tokens, step, mat_gate): read from L1+L2+L3 at each position
    - write_boundary(embedding): called at sentence boundaries
    - reset(): clear all memory (for new sequence)

    Writes are gated by maturation (like private_mem).
    Reads are always active.
    """
    def __init__(self, D: int, bridge_dim: int,
                 l1_slots: int = 3, l2_slots: int = 16,
                 l3_concepts: int = 8, l3_birth_threshold: float = 0.7,
                 min_write_maturation: float = 0.3,
                 cfg=None):
        super().__init__()
        self.D = D
        self.bridge_dim = bridge_dim
        self.cfg = cfg
        self._min_write_maturation = min_write_maturation

        # L1: rolling buffer (immediate, ~last 3 sentences)
        self.l1 = L1Buffer(D, bridge_dim, n_slots=l1_slots)

        # L2: learned bank (short-term, ~16 slots)
        self.l2 = L2Bank(D, bridge_dim, n_slots=l2_slots)

        # L3: emergent concepts (long-range, ~8 concepts from L2 clustering)
        self.l3 = L3Concepts(D, bridge_dim, n_concepts=l3_concepts,
                             birth_threshold=l3_birth_threshold)

        # Fusion gate: combine L1 + L2 + L3 + current state
        self.fusion = nn.Sequential(
            nn.Linear(D * 4, D),
            nn.GELU(),
            nn.Linear(D, D),
        )
        # Gate init: start as no-op
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

        # Injection scale (starts small, grows if helpful)
        self.log_scale = nn.Parameter(torch.tensor(-2.0))

        # Track sentence boundaries
        self._in_sentence = True
        self._sent_start = 0

    def forward(self, h: torch.Tensor, tokens: torch.Tensor,
                step: int = None, mat_gate: float = None) -> torch.Tensor:
        """Read from memory at each position.

        h: (B, L, D) — current hidden state (after embedding)
        tokens: (B, L) — token ids (for boundary detection)
        step: current training step (for logging)
        mat_gate: float — maturation gate value (0-1), gates writes
        returns: (B, L, D) — memory-augmented hidden state
        """
        B, L, D = h.shape
        is_sep = (tokens == 2)  # SEP token = sentence boundary

        # Determine if writes are allowed
        _can_write = (mat_gate is None) or (mat_gate >= self._min_write_maturation)

        # Detect boundaries and write to all levels
        with torch.no_grad():
            for b in range(B):
                sent_start = 0
                for t in range(L):
                    if is_sep[b, t]:
                        summary = h[b, sent_start:t+1].mean(0)  # (D,)

                        # Write to L1 (always when allowed)
                        if _can_write:
                            self.l1.write(summary)

                        # Write to L2 (novelty-gated, only when allowed)
                        wrote_l2 = False
                        if _can_write:
                            wrote_l2 = self.l2.write(summary)

                        # Write to L3 (concept clustering from L2 key, only when allowed)
                        if _can_write and wrote_l2:
                            last_slot = (self.l2._write_idx - 1) % self.l2.n_slots
                            l2_key = self.l2.keys[last_slot]  # (bridge_dim,)
                            conf = self.l2.slot_novelty[last_slot].item()
                            self.l3.write(l2_key, confidence=conf)

                        sent_start = t + 1

        # Read from all levels
        mem_l1 = self.l1.read(h)  # (B, L, D)
        mem_l2 = self.l2.read(h)  # (B, L, D)
        mem_l3 = self.l3.read(h)  # (B, L, D)

        # Fusion: combine current + L1 + L2 + L3
        combined = torch.cat([h, mem_l1, mem_l2, mem_l3], dim=-1)  # (B, L, 4D)
        fused = self.fusion(combined)  # (B, L, D)

        # Injection with bounded scale
        scale = torch.tanh(self.log_scale)  # in (-1, 1)

        # When maturation too low, bypass memory bank entirely (no-op)
        if not _can_write:
            return h

        return h + scale * fused

    def reset(self) -> None:
        """Clear all memory (for new sequence)."""
        self.l1.reset()
        self.l2.reset()
        self.l3.reset()

    def get_diagnostics(self) -> dict:
        """Return diagnostic info for logging."""
        return {
            'l1_write_idx': self.l1._write_idx,
            'l2_write_idx': self.l2._write_idx,
            'l2_novelty_mean': self.l2.slot_novelty.mean().item(),
            'l2_age_mean': self.l2.slot_age.mean().item(),
            'l3_n_concepts': self.l3.get_active_concepts(),
            'l3_n_births': self.l3._n_births,
            'l3_confidence_mean': self.l3.concept_confidence.mean().item(),
            'mem_scale': torch.tanh(self.log_scale).item(),
        }
