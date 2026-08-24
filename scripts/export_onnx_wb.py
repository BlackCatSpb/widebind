import sys, os, time, inspect, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch
import onnxruntime as ort
from core import WideBindConfig, WideBindStack
from core.migrate import migrate_state_dict
from scripts.generate import AdaptiveSampler, load_russian_tokenizer
from compression import FCF_CPR


def build_model(ckpt):
    cfg = ckpt['cfg']
    model = WideBindStack(cfg).eval()
    sd = dict(ckpt['model'])
    sd, n_mig = migrate_state_dict(sd, model)
    model.load_state_dict(sd, strict=False)
    if n_mig:
        print('migrated', n_mig, 'keys')
    if model.explicit_reasoning:
        model.reasoning_enabled_step = int(ckpt.get('reasoning_enabled_step', 0))
    return model, cfg


def make_wrapper(n_layers, n_fields, used_fields):
    # used_fields[li] = list of field indices that are real tensors
    field = lambda li, fi: f's_{li}_{fi}'
    params = (['self', 'h', 'global_state', 'step', 'reasoning_buffer', 'reasoning_count']
              + [field(li, fi) for li in range(n_layers) for fi in used_fields[li]])
    lines = []
    lines.append('        state = []')
    for li in range(n_layers):
        parts = []
        for fi in range(n_fields):
            if fi in used_fields[li]:
                parts.append(field(li, fi))
            else:
                parts.append('None')
        lines.append('        state.append((' + ', '.join(parts) + '))')
    lines.append('        out, ns, gs2, (rb2, rc2) = self.model(h, state, global_state=global_state,'
                 ' adaptive=False, context_mem=None, allow_write=None, step=step,'
                 ' reasoning_buffer=reasoning_buffer, reasoning_count=reasoning_count)')
    # outputs: out, then per layer per used field, then gs2, rb2, rc2
    ret = ['out']
    out_order = []
    for li in range(n_layers):
        for fi in used_fields[li]:
            ret.append(f'ns[{li}][{fi}]')
            out_order.append((li, fi))
    ret += ['gs2', 'rb2', 'rc2']
    lines.append('        return (' + ', '.join(ret) + ')')
    src = 'class ExportWrapper(torch.nn.Module):\n'
    src += '    def __init__(self, model):\n        super().__init__()\n        self.model = model\n'
    src += '    def forward(' + ', '.join(params) + '):\n' + '\n'.join(lines) + '\n'
    g = {'torch': torch}
    exec(src, g)
    return g['ExportWrapper'], out_order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('onnx')
    ap.add_argument('--tokens', type=int, default=30)
    ap.add_argument('--prompt', default='кто ты?')
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--top-k', type=int, default=0)
    ap.add_argument('--rep-penalty', type=float, default=2.0)
    ap.add_argument('--reuse', action='store_true')
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'model' not in ckpt and 'model_compressed' in ckpt:
        print('compressed checkpoint -> decompress')
        ckpt = FCF_CPR().load_compressed(args.checkpoint)
    model, cfg = build_model(ckpt)
    n_layers = len(model.layers)
    D = cfg.D
    L = cfg.seq_len
    B = 1
    max_steps = getattr(cfg, 'reasoning_max_steps', 8)

    # dry run to capture state structure
    with torch.no_grad():
        h0 = torch.zeros(B, L, D)
        gs0 = torch.zeros(n_layers, 1, D)
        step0 = torch.tensor(0, dtype=torch.long)
        rb0 = torch.zeros(B, max_steps, D)
        rc0 = torch.tensor(0, dtype=torch.long)
        out_ref, ns_ref, gs_ref, (rb_ref, rc_ref) = model(
            h0, None, global_state=gs0, adaptive=False, step=step0,
            reasoning_buffer=rb0, reasoning_count=rc0)
    n_fields = max(len(s) for s in ns_ref)
    used_fields = [[fi for fi in range(len(ns_ref[li])) if ns_ref[li][fi] is not None]
                   for li in range(n_layers)]
    state_shapes = {}
    for li in range(n_layers):
        for fi in used_fields[li]:
            state_shapes[(li, fi)] = tuple(ns_ref[li][fi].shape)
    print(f'captured: n_layers={n_layers} D={D} L={L} max_steps={max_steps} '
          f'n_fields={n_fields} used_per_layer={[len(u) for u in used_fields]}')
    for li in range(n_layers):
        print(f'  layer {li}: used fields {used_fields[li]} shapes {[state_shapes[(li,fi)] for fi in used_fields[li]]}')

    Wrapper, out_order = make_wrapper(n_layers, n_fields, used_fields)
    wrapper = Wrapper(model).eval()
    state_dummies = [torch.zeros(*state_shapes[(li, fi)])
                     for li in range(n_layers) for fi in used_fields[li]]
    ex_args = (h0, gs0, step0, rb0, rc0) + tuple(state_dummies)

    # export
    if not (args.reuse and os.path.isfile(args.onnx)):
        print('Exporting ONNX (torch.export + onnx.dynamo)...')
        t0 = time.time()
        with torch.no_grad():
            ep = torch.export.export(wrapper, ex_args, strict=False)
            torch.onnx.export(ep, ex_args, args.onnx, dynamo=True, opset_version=17)
        print(f'  export time {time.time()-t0:.1f}s -> {args.onnx}')

    # numeric check ORT vs python
    opt = ort.SessionOptions()
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort.set_default_logger_severity(3)
    sess = ort.InferenceSession(args.onnx,
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                                sess_options=opt)
    inames = [i.name for i in sess.get_inputs()]
    onames = [o.name for o in sess.get_outputs()]
    feeds = {'h': h0.numpy(), 'global_state': gs0.numpy(),
             'step': step0.numpy(), 'reasoning_buffer': rb0.numpy(),
             'reasoning_count': rc0.numpy()}
    didx = 0
    for li in range(n_layers):
        for fi in used_fields[li]:
            feeds[f's_{li}_{fi}'] = state_dummies[didx].numpy()
            didx += 1
    res = sess.run(onames, feeds)
    out_ort = torch.from_numpy(res[0])
    diff = (out_ort - out_ref).abs().max().item()
    print(f'ORT vs python out: max abs diff = {diff:.4e}  (provider={sess.get_providers()})')
    if diff > 1e-1:
        print('WARNING: ONNX mismatch large; generation may diverge.')

    # generation loop via ORT (CUDA)
    tok = load_russian_tokenizer()
    if tok is None:
        raise FileNotFoundError('russian_tokenizer/tokenizer.json not found')
    prompt_tokens = tok.encode(args.prompt).ids
    tokens = torch.tensor(prompt_tokens, dtype=torch.long)
    head = model.lm_head
    tb = head.token_bias.data
    h_emb_ok = 'h_emb' in inspect.signature(head.forward).parameters
    state_flat = { (li, fi): torch.zeros(*state_shapes[(li, fi)])
                  for li in range(n_layers) for fi in used_fields[li] }
    gs = torch.zeros(n_layers, 1, D)
    rb = torch.zeros(B, max_steps, D)
    rc = torch.tensor(0, dtype=torch.long)
    recent = set()

    def gen_step(step, first=False):
        nonlocal state_flat, rb, rc, gs, tokens
        ctx = tokens[-L:]
        if len(ctx) < L:
            ctx = torch.cat([torch.zeros(L - len(ctx), dtype=torch.long), ctx])
        ctx = ctx.unsqueeze(0)
        with torch.no_grad():
            h = model.embed_tokens(ctx)
        feeds = {'h': h.detach().numpy(), 'global_state': gs.numpy(),
                 'step': torch.tensor(step, dtype=torch.long).numpy(),
                 'reasoning_buffer': rb.numpy(), 'reasoning_count': rc.numpy()}
        didx = 0
        for li in range(n_layers):
            for fi in used_fields[li]:
                feeds[f's_{li}_{fi}'] = state_flat[(li, fi)].numpy()
                didx += 1
        res = sess.run(onames, feeds)
        out = torch.from_numpy(res[0])
        for oi, (li, fi) in enumerate(out_order):
            state_flat[(li, fi)] = torch.from_numpy(res[1 + oi]).clone()
        rb = torch.from_numpy(res[1 + len(out_order) + 1]).clone()
        rc = torch.from_numpy(res[1 + len(out_order) + 2]).clone()
        gs = torch.zeros(n_layers, 1, D)
        with torch.no_grad():
            if h_emb_ok:
                logits = head(out[:, -1:, :], h[:, -1:, :])[0, 0]
            else:
                logits = head(out[:, -1:, :])[0, 0]
            logits = (logits - tb) + 1.0 * tb
            logits = logits / args.temperature
            for rid in list(recent)[-5:]:
                logits[rid] -= args.rep_penalty
            if args.top_k > 0:
                vals, _ = torch.topk(logits, args.top_k)
                logits[logits < vals[-1:]] = -float('inf')
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
        recent.add(int(nxt))
        tokens = torch.cat([tokens, nxt], dim=0)

    # warmup
    for s in range(3):
        gen_step(s)
    tokens = torch.tensor(prompt_tokens, dtype=torch.long)
    recent.clear()
    state_flat = { (li, fi): torch.zeros(*state_shapes[(li, fi)])
                  for li in range(n_layers) for fi in used_fields[li] }
    rb = torch.zeros(B, max_steps, D); rc = torch.tensor(0, dtype=torch.long)
    gs = torch.zeros(n_layers, 1, D)
    t0 = time.time()
    for s in range(args.tokens):
        gen_step(s)
    dt = time.time() - t0
    text = tok.decode(tokens.tolist(), skip_special_tokens=True)
    print(f'\n[ORT CUDA] {args.tokens} tok / {dt:.2f}s = {args.tokens/dt:.2f} tok/s')
    print(f'[ORT CUDA] TEXT: {text}')


if __name__ == '__main__':
    main()
