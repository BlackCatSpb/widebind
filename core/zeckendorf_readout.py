"""Zeckendorf tree readout with learnable centroids.

Port from fcp/ld_model/readout.py with WideBind adaptations.
Replaces DxV LM head with tree-structured decoder (K levels, ~86K params).
"""

import torch
import torch.nn.functional as F


def fibonacci_bases(vocab_size: int) -> list[int]:
    """Fibonacci numbers up to vocab_size (Zeckendorf bases).

    F[0]=1, F[1]=2, F[k]=F[k-1]+F[k-2].
    Covers tokens 0..V-1 iff F_K - 1 >= V-1.
    """
    fibs = [1, 2]
    while fibs[-1] < vocab_size:
        fibs.append(fibs[-1] + fibs[-2])
    return fibs


def zeckendorf_code(token_id: int, fibs: list[int]) -> list[int]:
    """Zeckendorf representation: binary, no consecutive 1s, MSB first."""
    bits = []
    remaining = token_id
    prev = False
    for f in reversed(fibs):
        if remaining >= f and not prev:
            bits.append(1)
            remaining -= f
            prev = True
        else:
            bits.append(0)
            prev = False
    return bits


class ZeckendorfReadout(torch.nn.Module):
    """Zeckendorf tree readout replacing D->V LM head.

    P(token|h) = prod_k P(bit_k | h, state_k)

    Centroids: c[k, state, bit] in R^D.
    K ~ log_phi(V) levels instead of full V-way softmax.
    """

    def __init__(self, cfg):
        super().__init__()
        self.vocab = cfg.vocab
        self.D = cfg.D

        fibs = fibonacci_bases(cfg.vocab)
        self.K = len(fibs)
        self.register_buffer('fibs', torch.tensor(fibs, dtype=torch.long))

        self.max_representable = fibs[-1] + (fibs[-2] if len(fibs) > 1 else 1) - 1
        valid_vocab = min(self.vocab, self.max_representable + 1)

        codes = torch.zeros(valid_vocab, self.K, dtype=torch.long)
        for i in range(valid_vocab):
            bits = zeckendorf_code(i, fibs)
            for k, b in enumerate(bits):
                codes[i, k] = b
        self.register_buffer('codes', codes)

        self.centroids = torch.nn.Parameter(
            torch.randn(self.K, 2, 2, self.D) * 0.1
        )
        with torch.no_grad():
            self.centroids[:, 1, 1, :] = 0.0

    @staticmethod
    def _mask_logits(logit: torch.Tensor) -> torch.Tensor:
        """Mask illegal transitions in Zeckendorf tree.

        logit: (..., K, 2, 2) — (batch, level, state, digit)
        c[:,1,1,:] is invalid (no consecutive 1s). Set logit -> -inf
        so softmax assigns P(digit=1 | state=1) = 0.
        """
        mask = torch.full_like(logit, float('-inf'))
        mask[..., 0, :] = logit[..., 0, :]     # state=0: both digits allowed
        mask[..., 1, 0] = logit[..., 1, 0]     # state=1: only digit=0 allowed
        return mask

    def log_probs_for_target(self, h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """O(B * K) log P(target|h) without full V' computation.

        Args:
            h: (B, D)
            target: (B,)
        Returns:
            log_probs: (B,)
        """
        B, D = h.shape
        K = self.K

        c = self.centroids.float()
        h_exp = h.view(B, 1, 1, 1, D)
        c_exp = c.view(1, K, 2, 2, D)
        logit = (h_exp * c_exp).sum(dim=-1)
        logit = self._mask_logits(logit)
        log_probs_k_s = F.log_softmax(logit, dim=-1)
        log_p_flat = log_probs_k_s.reshape(B, K, 4)

        codes = self.codes.to(h.device)
        prev_bits = torch.zeros_like(codes)
        if K > 1:
            prev_bits[:, 1:] = codes[:, :-1]
        combined_idx = prev_bits * 2 + codes

        Vp = codes.shape[0]
        target_clamped = target.clamp(0, Vp - 1)
        idx = combined_idx[target_clamped]
        log_p = log_p_flat[torch.arange(B, device=h.device)[:, None],
                           torch.arange(K, device=h.device), idx]
        return log_p.sum(dim=-1)

    def forward_log_probs(self, h: torch.Tensor) -> torch.Tensor:
        """Full log P(token|h) for all valid Zeckendorf tokens.

        Args:
            h: (B, D)
        Returns:
            log_probs: (B, V')
        """
        B, D = h.shape
        K = self.K

        c = self.centroids.float()
        h_exp = h.view(B, 1, 1, 1, D)
        c_exp = c.view(1, K, 2, 2, D)
        logit = (h_exp * c_exp).sum(dim=-1)
        logit = self._mask_logits(logit)
        log_probs_k_s = F.log_softmax(logit, dim=-1)
        log_p_flat = log_probs_k_s.reshape(B, K, 4)

        Vp, _ = self.codes.shape
        codes = self.codes
        prev_bits = torch.zeros_like(codes)
        if K > 1:
            prev_bits[:, 1:] = codes[:, :-1]
        combined_idx = prev_bits * 2 + codes

        log_probs = torch.zeros(B, Vp, device=h.device)
        for k in range(K):
            log_probs += log_p_flat[:, k, combined_idx[:, k]]
        return log_probs

    def predict(self, h: torch.Tensor, greedy: bool = True,
                temperature: float = 1.0) -> torch.Tensor:
        """Generate token via Zeckendorf tree traversal.

        Args:
            h: (B, D)
            greedy: argmax or sample
            temperature: sampling temperature
        Returns:
            tokens: (B,)
        """
        B, D = h.shape
        device = h.device
        K = self.K
        c = self.centroids.float()

        h_exp = h.view(B, 1, 1, 1, D)
        c_exp = c.view(1, K, 2, 2, D)
        logit = (h_exp * c_exp).sum(dim=-1) / temperature
        logit = self._mask_logits(logit)
        probs = F.softmax(logit, dim=-1)

        tokens = torch.zeros(B, dtype=torch.long, device=device)
        fibs = self.fibs

        for b in range(B):
            state = 0
            token_id = 0
            for k in range(K):
                p1 = probs[b, k, state, 1]
                if state == 1:
                    bit = 0
                elif greedy:
                    bit = 1 if p1 > 0.5 else 0
                else:
                    bit = 1 if torch.rand(1, device=device).item() < p1.item() else 0
                if bit:
                    token_id += fibs[k].item()
                state = bit
            tokens[b] = min(token_id, self.vocab - 1)

        return tokens

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
