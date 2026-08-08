"""Долгий синтетический прогон SignedAmpCodec с AmpAdam (без сравнения с softmax).

Цель — стабильный градиент факторизованной кодечной головы на большом числе
шагов: маржинальная цель (margin_loss, scale-invariant) + AmpAdam с проекциями.

Мониторы каждые 100 шагов:
  margin  — маржинальная потеря (должна падать и быть ОГРАНИЧЕННОЙ);
  top1    — точность на свежих парах (должна расти на структурированных задачах);
  gnorms  — нормы градиентов по ролям head/embed/backbone;
  diag    — σ_on/σ_off, gain, |bias| max, разброс α, сатурация |a|>0.9.

В конце — сводка стабильности: мин/макс margin, NaN/Inf-флаги, доли шагов
с параметрами на границах коробок.
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from core import WideBindConfig, WideBindStack
from core.amp_optim import build_amp_groups, AmpAdam

parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=2000)
parser.add_argument('--vocab', type=int, default=1024)
parser.add_argument('--task', choices=['copy', 'shift', 'rand', 'counter'], default='shift')
parser.add_argument('--recipe', choices=['carry', 'fresh'], default='carry')
parser.add_argument('--seeds', type=int, default=1)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--head-scale', type=float, default=1.0)
parser.add_argument('--embed-scale', type=float, default=0.5)
parser.add_argument('--amp-scale', type=float, default=1.0,
                    help='масштаб записи кода в эмбеддинг (cfg.amp_scale)')
parser.add_argument('--hinge-weight', type=float, default=1.0,
                    help='вес шарнира против истинного argmax-конкурента')
parser.add_argument('--pred-w', action='store_true',
                    help='механизм A: оператор перехода W_pred в кодовом пространстве')
parser.add_argument('--phasor', action='store_true',
                    help='«корни из единицы»: плотные cos/sin-коды (переходы = вращения)')
parser.add_argument('--hybrid', action='store_true',
                    help='hybrid: sparse-позиции (разделение) + cos/sin (вращение)')
parser.add_argument('--reg-w', type=float, default=0.0,
                    help='механизм C: вес регрессии чтения на код цели')
parser.add_argument('--sigma-min', type=float, default=None,
                    help='нижняя граница σ (cfg.amp_sigma_min, по умолчанию 0.2)')
parser.add_argument('--freeze-backbone', action='store_true',
                    help='train только lm_head + embed (проверка: читаемо ли код из h_L)')
parser.add_argument('--obj', choices=['mh', 'ce'], default='mh',
                    help='mh = margin+hinge(+reg);  ce = одна CE-цель (S1 упрощение)')
parser.add_argument('--no-echo', action='store_true',
                    help='отключить код-канал (эхо записи) в счёте (S2 абляция)')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}  steps={args.steps} vocab={args.vocab} task={args.task} '
      f'recipe={args.recipe} lr={args.lr} seeds={args.seeds}')
print(f'head_scale={args.head_scale} embed_scale={args.embed_scale} '
      f'amp_scale={args.amp_scale} hinge_w={args.hinge_weight}')

B, L = 1, 64


def sample(task, cfg, prev=None):
    if task == 'counter':
        # Детерминированный ход по словарю: x_{t+1} = x_t + 1. Цель — НАСТОЯЩИЙ
        # next-token: y_t = x_{t+1}. У перехода есть структура, выучить её можно
        # и небольшим числом параметров (в отличие от shift-random, где y=x+1
        # при случайном x — это словарь на V записей).
        x0 = prev if prev is not None else torch.randint(0, cfg.vocab, (B, 1), device=device)
        x = (x0 + torch.arange(L, device=device).unsqueeze(0)) % cfg.vocab
        y = (x + 1) % cfg.vocab
        return x, y
    x = torch.randint(0, cfg.vocab, (B, L), device=device)
    if task == 'copy':
        y = x.clone()
    elif task == 'shift':
        y = (x + 1) % cfg.vocab
    else:
        y = torch.randint(0, cfg.vocab, (B, L), device=device)
    return x, y


def top1(model, cfg, state=None):
    with torch.no_grad():
        model.eval()
        x, y = sample(args.task, cfg)
        h_emb = model.embed_tokens(x)
        out, st, gl = model(h_emb, state=state, global_state=None)
        logits = model.lm_head(out, None if args.no_echo else h_emb)
        acc = (logits.argmax(-1).reshape(-1) == y.reshape(-1)).float().mean().item()
    model.train()
    if args.recipe == 'carry' and st is not None:
        state = [(t[0].detach(), t[1].detach(), t[2].detach()) if t is not None else None for t in st]
    else:
        state = None
    return acc, state


def code_fidelity(model, h, out, y):
    """Насколько код выжил в стеке: corr проекции (embed и final) с α цели.

    Плоская корреляция по активным записям (N·S). Если fin ~ 0 при emb ~ 0.8 —
    стек уничтожает записанный код, маржине нечего усиливать.
    """
    with torch.no_grad():
        m = model.lm_head
        N = h.shape[0] * h.shape[1]
        alpha = (torch.tanh(m.proto[y]) * m.codes[y]).reshape(N, m.K).float()
        mask = m.codes[y].float().reshape(N, m.K).bool()
        co = lambda a: torch.corrcoef(
            torch.stack([a[mask].flatten(), alpha[mask].flatten()]))[0, 1].item()
        c_emb = co(m._proj(h.reshape(N, m.K, -1)))
        c_fin = co(m._proj(out.reshape(N, m.K, -1)))
        return c_emb, c_fin


def head_diag(model):
    m = model.lm_head
    son, soff, semb = m._sigmas()
    bias = (m.token_bias - m.token_bias.mean()).clamp(-4, 4)
    return {
        'son': son.mean().item(),
        'soff': soff.mean().item(),
        'gain_max': torch.exp(m.log_gain).item() if m.log_gain.numel() == 1 else
                    torch.exp(m.log_gain).max().item(),
        'gain_min': torch.exp(m.log_gain).min().item(),
        'bias_max': bias.abs().max().item(),
        'alpha_std': torch.tanh(m.proto).std().item(),
        'o_std': m.o.std().item(),
        's_emb': m._sigmas()[2].mean().item(),
        'gain_emb': torch.exp(m.log_gain_emb).clamp(0.1, 4.0).mean().item(),
    }


def run(seed):
    torch.manual_seed(seed)
    cfg = WideBindConfig(D=896, n_layers=4, mlp_groups=8, seq_len=L, batch_size=B,
                         lr=args.lr, amp_codec=True, vocab=args.vocab,
                         amp_scale=args.amp_scale, amp_pred=args.pred_w,
                         amp_phasor=args.phasor, amp_hybrid=args.hybrid,
                         amp_obj=args.obj,
                         amp_sigma_min=args.sigma_min if args.sigma_min is not None else 0.2)
    model = WideBindStack(cfg).to(device)
    if args.freeze_backbone:
        for name, p in model.named_parameters():
            if not (name.startswith('lm_head.') or name.startswith('embed.')):
                p.requires_grad_(False)
    groups = build_amp_groups(model, lr=args.lr, head_scale=args.head_scale,
                              embed_scale=args.embed_scale)
    opt = AmpAdam(groups)
    print(f'\n=== seed {seed}: {model.param_count()/1e6:.2f}M, '
          f'roles: head={len([g for g in groups if g["role"]=="head"])} '
          f'embed={len([g for g in groups if g["role"]=="embed"])} '
          f'backbone={len([g for g in groups if g["role"]=="backbone"])} ===')

    state, gs = None, None
    est = None
    cx = None
    t0 = time.time()
    margin_min, margin_max = 1e9, -1e9
    nan_flag = False
    at_bound = {'sigma': 0, 'gain': 0, 'bias': 0}
    checks = 0
    for step in range(args.steps):
        x, y = sample(args.task, cfg, cx)
        if args.task == 'counter':
            cx = x[0, -1:]  # продолжение счётчика между батчами
        opt.zero_grad()
        h = model.embed_tokens(x)
        out, st, gl = model(h, state=state, global_state=gs)
        mh = model.lm_head
        hf = out.reshape(-1, cfg.D)
        t = y.reshape(-1)
        he = h.reshape(-1, cfg.D)
        hem = None if args.no_echo else he
        if args.obj == 'ce':
            ce = mh.ce_loss(hf, t, hem)
            loss = ce.mean()
            margin = hinge = reg = None
        else:
            margin = mh.margin_loss(hf, t, hem)
            hinge = mh.argmax_hinge(hf, t, hem)
            loss = margin.mean() + args.hinge_weight * hinge.mean()
            if args.reg_w > 0:
                reg = mh.code_reg(hf, t, hem)
                loss = loss + args.reg_w * reg.mean()
            else:
                reg = None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if not torch.isfinite(loss):
            nan_flag = True
        if checks % 100 == 0 or step < 3:
            gn = {}
            for name, p in model.named_parameters():
                if p.grad is not None:
                    r = 'head' if name.startswith('lm_head.') else ('embed' if name.startswith('embed.') else 'backbone')
                    gn[r] = gn.get(r, torch.zeros((), device=device)) + p.grad.detach().norm().square()
            gn = {k: v.sqrt().item() for k, v in gn.items()}
        opt.step()
        if args.recipe == 'carry' and st is not None:
            state = [(t[0].detach(), t[1].detach(), t[2].detach()) if t is not None else None for t in st]
            gs = gl.detach()
        else:
            state, gs = None, None
        mv = ce.mean().item() if args.obj == 'ce' else margin.mean().item()
        metric = 'ce' if args.obj == 'ce' else 'margin'
        margin_min, margin_max = min(margin_min, mv), max(margin_max, mv)
        checks += 1
        if checks % 100 == 0 or step < 3:
            acc, est = top1(model, cfg, est)
            d = head_diag(model)
            cf = code_fidelity(model, h, out, y)
            rv = reg.mean().item() if reg is not None else float('nan')
            hv = hinge.mean().item() if hinge is not None else float('nan')
            print(f'  step {step+1:5d}  {metric}={mv:8.3f} hinge={hv:6.3f} reg={rv:6.3f}  top1={acc*100:6.2f}%  '
                  f'gnorm h/e/b={gn.get("head",0):.2f}/{gn.get("embed",0):.2f}/{gn.get("backbone",0):.2f}  '
                  f'son={d["son"]:.3f} soff={d["soff"]:.3f} semb={d["s_emb"]:.3f} '
                  f'gain=[{d["gain_min"]:.2f},{d["gain_max"]:.2f}] '
                  f'bias={d["bias_max"]:.2f} alpha_std={d["alpha_std"]:.3f} '
                  f'corr emb/fin={cf[0]:.2f}/{cf[1]:.2f}')
    dt = time.time() - t0
    acc, _ = top1(model, cfg, est)
    d = head_diag(model)
    print(f'  time {dt:.1f}s ({args.steps/dt:.1f} steps/s)')
    print(f'  FINAL top1={acc*100:.2f}%  {metric} range=[{margin_min:.3f}, {margin_max:.3f}]  nan={nan_flag}')
    print(f'  diag: son={d["son"]:.3f} soff={d["soff"]:.3f} semb={d["s_emb"]:.3f} '
          f'gain=[{d["gain_min"]:.2f},{d["gain_max"]:.2f}] gain_emb={d["gain_emb"]:.2f} '
          f'bias_max={d["bias_max"]:.2f} alpha_std={d["alpha_std"]:.3f} o_std={d["o_std"]:.3f}')
    return {'margin_min': margin_min, 'margin_max': margin_max, 'acc': acc, 'nan': nan_flag}


for seed in range(args.seeds):
    run(seed)
print('\nDONE_OK')
