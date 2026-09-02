"""
Collective Concept Layer — adaptive maturity version.

Maturity is detected automatically from mirror residual variance:
when CV = std/mean drops below 1/lambda_d for ceil(lambda_d) consecutive steps,
the layer becomes "mined" and starts writing concepts.
No hardcoded write_delay — adapts to training dynamics.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CollectiveConceptLayer(nn.Module):
    def __init__(
        self,
        D: int,
        k: int,
        S: int = 8,
        uncert_theta: float = 0.5,
        uncert_kappa: float = 3.0,
        contra_thresh: float = -0.1,
        contra_gain: float = 6.0,
        birth_gap: float = 0.55,
        maturity_thresh: float = 0.12,
        seed: int = 0,
        cfg=None,
        softmax_free=None,
        novelty_threshold: float = 0.15,
    ):
        super().__init__()
        self.cfg = cfg
        self.softmax_free = (softmax_free if softmax_free is not None
                             else (getattr(cfg, 'softmax_free', True) if cfg is not None else True))
        self.D = D
        self.k = k
        self.S = int(S)
        self._uncert_theta = float(uncert_theta)
        self._uncert_kappa = float(uncert_kappa)
        self._contra_thresh = float(contra_thresh)
        self._contra_gain = float(contra_gain)
        self._birth_gap = float(birth_gap)
        self._maturity_thresh = float(maturity_thresh)
        self._novelty_threshold = float(novelty_threshold)
        self._births_skipped_novelty = 0
        self._births_allowed = 0
        # Рождение концепта (горизонт событий, режим Б): sigmoid(-1)~0.27,
        # растёт если проекция потенциала помогает CE.
        self._birth_log_scale = nn.Parameter(torch.tensor(-1.0))

        g = torch.Generator().manual_seed(seed)
        m_init = torch.randn(S, k, generator=g)
        self.register_buffer('M', F.normalize(m_init, dim=-1))
        self.register_buffer('U_s', torch.zeros(S))
        self.register_buffer('N_s', torch.zeros(S, dtype=torch.long))
        self.register_buffer('_step', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_mature', torch.zeros(1))
        self.register_buffer('_gate_u', torch.zeros(1))
        self.register_buffer('_gate_c', torch.zeros(1))
        self.register_buffer('_cached_birth_gate', torch.tensor(0.0), persistent=False)
        self._last_write_step = -1

        # Сигналы для прожектора (считывание слов из скрытого состояния):
        # _write_event — на каких позициях записан новый/обновлённый концепт (≈ граница слова)
        # _concept_id   — в какой слот концепта отображается позиция (best match)
        self.register_buffer('_write_event', torch.zeros(1, 1, dtype=torch.bool), persistent=False)
        self.register_buffer('_concept_id', torch.zeros(1, 1, dtype=torch.long), persistent=False)

        self.W_o = nn.Linear(S * k, D, bias=False)
        nn.init.orthogonal_(self.W_o.weight)
        self._read_scale = nn.Parameter(torch.tensor(0.0))
        self._temp = nn.Parameter(torch.tensor(2.0))

    @torch.no_grad()
    def _update_maturity(self, resvar):
        """Adaptive maturity from residual variance stabilization."""
        if resvar is None:
            return
        if isinstance(resvar, torch.Tensor):
            resvar = resvar.detach().item()
        lam = getattr(self.cfg, 'lambda_d', 3) if self.cfg else 3
        lam_inv = 1.0 / lam
        if not hasattr(self, '_resvar_ema'):
            self.register_buffer('_resvar_ema', torch.tensor(resvar))
            self.register_buffer('_resvar_var', torch.tensor(1.0))
            self.register_buffer('_mature_count', torch.zeros(1, dtype=torch.long))
        ema_rate = lam_inv
        delta = resvar - self._resvar_ema.item()
        self._resvar_ema.fill_(self._resvar_ema.item() + ema_rate * delta)
        self._resvar_var.mul_(1 - ema_rate).add_(delta * delta * ema_rate)
        cv = (self._resvar_var.item() ** 0.5) / (abs(self._resvar_ema.item()) + 1e-8)
        stable = cv < lam_inv
        if stable:
            self._mature_count += 1
        else:
            self._mature_count.zero_()
        self._mature.fill_(1.0 if self._mature_count.item() >= math.ceil(lam) else 0.0)

    @torch.no_grad()
    def _maybe_write(self, hp, pen, allow_write):
        """Mature-gated, confident+novel slot refinement and birth.

        Возвращает (write_event, best):
          write_event: (B, L) bool — на позициях, где записан концепт (граница разметки)
          best:        (B, L) long — слот концепта с наилучшим совпадением
        """
        self._step += 1
        B = hp.shape[0]
        L = hp.shape[1]
        zeros_ev = torch.zeros(B, L, dtype=torch.bool, device=hp.device)
        if not allow_write:
            return zeros_ev, zeros_ev.long()

        # Разделяем зрелость: для РОЖДЕНИЯ достаточно начальной стадии (0.1),
        # для ОБНОВЛЕНИЯ существующих слотов нужна полная зрелость (0.5).
        mat = self._mature.item()
        can_refine = mat >= 0.5
        can_birth = mat >= 0.1
        if not can_refine and not can_birth:
            return zeros_ev, zeros_ev.long()

        B, L, G, k = hp.shape
        shared = hp.mean(dim=-2)
        shared_n = F.normalize(shared, dim=-1)
        M_n = F.normalize(self.M, dim=-1)
        sim = shared_n @ M_n.T
        best = sim.argmax(dim=-1)
        best_sim = sim.max(dim=-1).values
        d_min = 1.0 - best_sim
        conf = torch.sigmoid(-pen)
        write_event = torch.zeros(B, L, dtype=torch.bool, device=hp.device)
        did_write = False

        # Обновление существующих слотов — только при полной зрелости
        if can_refine:
            refine_thresh = conf.median().clamp(min=0.01)
            for s in range(self.S):
                mask = (best == s) & (conf >= refine_thresh)
                if mask.any():
                    write_event |= mask
                    upd = F.normalize(shared[mask].mean(dim=0), dim=-1)
                    if self.N_s[s].item() < 10:
                        self.M.data[s] = upd
                    else:
                        alpha = 0.01
                        self.M.data[s] = F.normalize(
                            self.M[s] * (1 - alpha) + upd * alpha, dim=-1)
                    self.N_s[s] += mask.sum().item()
                    did_write = True

        # Рождение новых концептов — при начальной зрелости (can_birth)
        if can_birth:
            birth_thresh = conf.quantile(0.25).clamp(min=0.01)
            empty = torch.nonzero(self.N_s == 0)
            # Novelty gate: birth only if max dist to existing concepts > threshold
            # (i.e., best cosine similarity < 1 - threshold)
            novel = (d_min > self._novelty_threshold) & (conf >= birth_thresh)
            # Track skipped births for diagnostics
            birth_candidates = (conf >= birth_thresh)
            if birth_candidates.any() and not novel.any():
                self._births_skipped_novelty += birth_candidates.sum().item()
            if novel.any():
                self._births_allowed += novel.sum().item()
                write_event |= novel
                if empty.numel() > 0:
                    idx = empty[0].item()
                    self.M.data[idx] = F.normalize(shared[novel].mean(dim=0), dim=-1)
                    self.N_s[idx] += 1
                else:
                    evict = int(torch.argmin(self.U_s).item())
                    self.M.data[evict] = F.normalize(shared[novel].mean(dim=0), dim=-1)
                    self.N_s[evict] = 1
                    self.U_s[evict] = 0.0
                did_write = True

        if did_write:
            self._last_write_step = int(self._step.item())

        occ = torch.zeros(self.S)
        for s in range(self.S):
            occ[s] = (best == s).float().mean().item()
        self.U_s.mul_(0.99).add_(occ.to(self.U_s), alpha=0.01)
        return write_event, best

    def forward(self, h, hp, pen, resvar=None, allow_write=None, mature_override=None, gate=None):
        _write = allow_write is None or allow_write
        if mature_override is not None:
            self._mature.fill_(float(mature_override))
        elif self.training:
            self._update_maturity(resvar)
        write_event, best = self._maybe_write(hp, pen, _write)
        # Прожектор: сохраняем сигналы границ слов и id концептов
        self._write_event = write_event
        self._concept_id = best

        B, L, G, k = hp.shape
        M_n = F.normalize(self.M, dim=-1)
        # Подсознание = множество экспертов (G каналов K-пространства) и их
        # гейтов. Образ мышления статьи: понимание — не усреднение, а параллельное
        # множество взглядов. Сравнение с концептами ведётся ПЕР-ЭКСПЕРТНО
        # (sim_g: каждый эксперт несёт своё понимание), и лишь затем множество
        # переводится gate-взвешенно в читаемый для Сознания уровень. Тем самым
        # множественность подсознания не стягивается в точку ДО интерпретации
        # (проводимость, а не коллапс).
        gate_w = gate.float() if (gate is not None and gate.shape[-1] == G) else torch.ones(B, L, G, device=hp.device)
        gsum = gate_w.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        hpg = F.normalize((hp * gate_w.unsqueeze(-1)).sum(dim=-2) / gsum, dim=-1)  # (B,L,k) ядро подсознания
        sim_g = torch.einsum('blgk,sk->blgs', hp, M_n)            # (B,L,G,S) множество пониманий
        temp = self._temp.clamp(min=0.5)
        if self.softmax_free:
            # Режим Б: нормированное сигмоид-среднее ПО ЭКСПЕРТАМ (каждый эксперт
            # сам мягко взвешивает концепт-слоты, без конкуренции), затем
            # перевод множества в читаемый уровень. Сохраняет лакуну.
            a_g = torch.sigmoid(sim_g * temp)
            a_g = a_g / a_g.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        else:
            a_g = torch.softmax(sim_g * temp, dim=-1)
        a = (a_g * gate_w.unsqueeze(-1)).sum(dim=-2) / gsum        # (B,L,S) перевод множества
        occ_w = (self.U_s / (self.U_s.max() + 1e-8)).clamp(0, 1)
        blend = (a.unsqueeze(-1) * occ_w.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                 * M_n.unsqueeze(0).unsqueeze(0))
        read = self.W_o(blend.reshape(B, L, -1))

        with torch.no_grad():
            u_gate = torch.sigmoid(self._uncert_kappa * (pen.unsqueeze(-1) - self._uncert_theta))
            out_n = F.normalize(read, dim=-1)
            h_n = F.normalize(h.detach(), dim=-1)
            cos_c = (out_n * h_n).sum(dim=-1, keepdim=True)
            c_gate = torch.sigmoid(self._contra_gain * (cos_c - self._contra_thresh))
            if self.training:
                self._gate_u.fill_(u_gate.mean().item())
                self._gate_c.fill_(c_gate.mean().item())

        scale = torch.sigmoid(self._read_scale)
        out = read * u_gate * c_gate * scale

        if self.softmax_free and self.S >= 2:
            # ─── Проекция горизонта событий (рождение концепта) ───
            # best_sim низок => вход — Феномен (лакуна между известными).
            # Порождаем потенциал: выпуклая сигмоид-смесь ближайших концептов
            # (сохраняет лакуну) + остаток новизны (то, что вне известного).
            # Это и есть «лиса» из «собака+кошка»: новый концепт в промежутке.
            best_sim = a.max(dim=-1, keepdim=True).values            # (B,L,1)
            birth_open = torch.sigmoid(self._contra_gain * (self._birth_gap - best_sim))
            birth_gate = torch.sigmoid(self._birth_log_scale) * birth_open
            self._cached_birth_gate = birth_gate.detach().mean()
            if birth_gate.mean().item() > 1e-4:
                topk = torch.topk(a, k=2, dim=-1)
                w12 = torch.sigmoid(topk.values * temp)               # (B,L,2)
                w12 = w12 / w12.sum(dim=-1, keepdim=True)             # нормировка
                M12 = M_n[topk.indices]                               # (B,L,2,k)
                neigh = (w12.unsqueeze(-1) * M12).sum(dim=-2)         # выпуклая смесь ближайших
                # residual = новизна вне известного, по ядру подсознания (множество
                # экспертов уже переведено gate-взвешенно в hpg)
                residual = hpg - (a.unsqueeze(-1) * M_n).sum(dim=-2)
                horizon = F.normalize(neigh + residual, dim=-1)       # проекция горизонта
                birth = self.W_o(horizon.unsqueeze(-2).expand(-1, -1, self.S, -1).reshape(B, L, -1))  # -> то же пространство D
                out = out + birth_gate * birth * u_gate * c_gate

        return out

    @torch.no_grad()
    def birth_gate_mean(self):
        """Средний вес рождения концепта (проекция горизонта событий). 0 = спит."""
        return self._cached_birth_gate

    @torch.no_grad()
    def get_diagnostics(self):
        """Return concept layer diagnostics."""
        return {
            'births_allowed': self._births_allowed,
            'births_skipped_novelty': self._births_skipped_novelty,
            'novelty_threshold': self._novelty_threshold,
            'mature': self._mature.item(),
            'n_slots_used': int((self.N_s > 0).sum().item()),
            'n_slots_total': self.S,
        }
