"""Streaming memory test: parallel forward, memory bank writes at boundaries.

Key test: can L2 memory help predict sent2 from sent1's summary?
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, time

D = 96
N_LAYERS = 2
N_HEADS = 3
SEQ_LEN = 40
VOCAB = 64
BATCH = 32
STEPS = 600
LR = 8e-4


class MemoryBank(nn.Module):
    def __init__(self, n_slots=64):
        super().__init__()
        self.n_slots = n_slots
        self.K = nn.Linear(D, D)
        self.V = nn.Linear(D, D)
        self.O = nn.Linear(D, D)
        self.keys = nn.Parameter(torch.randn(n_slots, D) * 0.02)
        self.vals = nn.Parameter(torch.randn(n_slots, D) * 0.02)
        self.age = nn.Parameter(torch.zeros(n_slots), requires_grad=False)
        self.writes = 0

    def read(self, q):
        k = self.K(self.keys)
        v = self.V(self.vals)
        a = torch.softmax(q @ k.T / math.sqrt(D), dim=-1)
        return self.O(a @ v)

    def write(self, embedding):
        s = self.age.argmin().item()
        e = embedding.detach()
        self.keys.data[s] = self.K(e)
        self.vals.data[s] = self.V(e)
        self.age.data[s] = self.age.max() + 1
        self.writes += 1


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.n = nn.LayerNorm(D)
        self.a = nn.MultiheadAttention(D, N_HEADS, batch_first=True)
        self.m = nn.Sequential(nn.Linear(D, D*2), nn.GELU(), nn.Linear(D*2, D))
    def forward(self, x):
        h = self.n(x)
        return x + self.a(h,h,h)[0] + self.m(self.n(x + self.a(h,h,h)[0]))


class Model(nn.Module):
    def __init__(self, use_mem=False, slots=64):
        super().__init__()
        self.use_mem = use_mem
        self.emb = nn.Embedding(VOCAB, D)
        self.layers = nn.ModuleList([Layer() for _ in range(N_LAYERS)])
        self.head = nn.Linear(D, VOCAB)
        if use_mem:
            self.mem = MemoryBank(slots)
            self.mix = nn.Linear(D*2, D)

    def forward(self, tok, tgt=None):
        B, S = tok.shape
        x = self.emb(tok)

        if self.use_mem:
            sent_start = 0
            mem_reads = torch.zeros(B, S, D, device=tok.device)
            for b in range(B):
                boundary = -1
                for t in range(S):
                    if tok[b, t].item() == 2:
                        summary = x[b, sent_start:t+1].mean(0)
                        self.mem.write(summary)
                        sent_start = t + 1
                        boundary = t
                if self.mem.writes > 0:
                    for b2 in range(B):
                        reads = []
                        for t in range(S):
                            q = x[b2, t]
                            reads.append(self.mem.read(q))
                        mem_reads[b2] = torch.stack(reads)

            x = x + self.mix(torch.cat([x, mem_reads], dim=-1))

        h = x
        for layer in self.layers:
            h = layer(h)

        logits = self.head(h)
        if tgt is not None:
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1))
            acc = (logits.argmax(-1) == tgt).float().mean()
            return loss, acc
        return logits


def gen_repeat_mutate(n=300):
    """sent2 = sent1 with 1 token mutated."""
    data = []
    for _ in range(n):
        L = torch.randint(3, 6, (1,)).item()
        s1 = torch.randint(3, VOCAB, (L,))
        s2 = s1.clone()
        s2[-1] = torch.randint(3, VOCAB, (1,))
        full = torch.cat([s1, torch.tensor([2]), s2, torch.tensor([2])])
        full = F.pad(full, (0, SEQ_LEN - full.shape[0]))
        tgt = torch.cat([full[1:], torch.tensor([0])])
        data.append((full, tgt))
    return data


def gen_random(n=300):
    data = []
    for _ in range(n):
        full = torch.randint(0, VOCAB, (SEQ_LEN,))
        full[torch.randint(5, 15, (1,)).item()] = 2
        full[torch.randint(20, 35, (1,)).item()] = 2
        tgt = torch.cat([full[1:], torch.tensor([0])])
        data.append((full, tgt))
    return data


def run_test(data_fn, name, n=300):
    print(f"\n{'='*50}")
    print(f"TEST: {name}")
    print(f"{'='*50}")
    data = data_fn(n)
    split = int(n * 0.75)
    train_d, test_d = data[:split], data[split:]

    results = {}
    for label, use_mem in [("no-mem", False), ("with-mem", True)]:
        model = Model(use_mem=use_mem, slots=64)
        np = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        t0 = time.time()

        model.train()
        for step in range(STEPS):
            idx = torch.randint(0, split, (BATCH,))
            tok = torch.stack([train_d[i][0] for i in idx])
            tgt = torch.stack([train_d[i][1] for i in idx])
            loss, acc = model(tok, tgt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if (step+1) % 200 == 0:
                print(f"  [{label}] {step+1}: loss={loss.item():.3f} acc={acc.item():.4f}")

        elapsed = time.time() - t0

        model.eval()
        tl, ta, ntok = 0, 0, 0
        with torch.no_grad():
            for tok, tgt in test_d:
                loss, acc = model(tok.unsqueeze(0), tgt.unsqueeze(0))
                tl += loss.item()
                ta += acc.item() * SEQ_LEN
                ntok += SEQ_LEN

        results[label] = {
            'ppl': math.exp(tl/len(test_d)),
            'acc': ta/ntok,
            'params': np,
            'time': elapsed,
            'writes': model.mem.writes if use_mem else 0,
        }

    no = results['no-mem']
    wi = results['with-mem']
    print(f"\n  No mem:   ppl={no['ppl']:.2f}  acc={no['acc']:.4f}  ({no['params']:,} params, {no['time']:.1f}s)")
    print(f"  With mem: ppl={wi['ppl']:.2f}  acc={wi['acc']:.4f}  ({wi['params']:,} params, {wi['time']:.1f}s)")
    da = wi['acc'] - no['acc']
    dp = no['ppl'] - wi['ppl']
    oh = (wi['params'] - no['params']) / no['params'] * 100
    print(f"  delta acc: {da:+.4f}  delta ppl: {dp:+.2f}  overhead: {oh:.1f}%  writes: {wi['writes']}")
    return da, dp, oh


if __name__ == '__main__':
    print("=" * 50)
    print("STREAMING MEMORY BANK - MINI TEST")
    print("=" * 50)

    d1, p1, o1 = run_test(gen_random, "RANDOM (no cross-sentence dependency)")
    d2, p2, o2 = run_test(gen_repeat_mutate, "REPEAT+MUTATE (sent2 = sent1 modified)")

    print(f"\n{'='*50}")
    print("SUMMARY:")
    print(f"  Random:       delta_acc={d1:+.4f}  delta_ppl={p1:+.2f}")
    print(f"  Repeat+mutate: delta_acc={d2:+.4f}  delta_ppl={p2:+.2f}")
    if d2 > d1 + 0.005:
        print("  >> Memory bank helps MORE on correlated data = concept validated!")
    elif d2 > 0:
        print("  >> Marginal improvement on correlated data")
    else:
        print("  >> No clear benefit yet, needs tuning")
    print("=" * 50)
