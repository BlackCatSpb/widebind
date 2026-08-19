"""Экспорт мини-модели в ONNX (проверка совместимости когерентности спиралей с torch.export)."""
import sys, os, math, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from core import WideBindConfig, WideBindStack

cfg = WideBindConfig(
    D=256, n_layers=2, bind_K=16, mlp_groups=4, mlp_expand=2,
    seq_len=32, batch_size=1, private_mem=True,
    traj_manifold=False, gradient_checkpointing=False,
    explicit_reasoning=False,
)
model = WideBindStack(cfg).eval()

B, L, D = 1, 32, cfg.D
h = torch.randn(B, L, D)
state = [None] * len(model.layers)
gs = torch.zeros(len(model.layers), 1, D)
step = torch.tensor(0, dtype=torch.long)
rb = torch.zeros(0)
rc = torch.tensor(0, dtype=torch.long)

def run(model, h, state, gs, step, rb, rc):
    out, new_state, gs2, (rb2, rc2) = model(
        h, state, global_state=gs,
        pred_weight=None, adaptive=False, context_mem=None, allow_write=None,
        step=step, reasoning_buffer=rb, reasoning_count=rc)
    return out, new_state, gs2, rb2, rc2

with torch.no_grad():
    ref_out, _, _, _, _ = run(model, h, state, gs, step, rb, rc)
    print('python forward OK, out', tuple(ref_out.shape))

    kwargs = {'pred_weight': None, 'adaptive': False, 'context_mem': None,
              'allow_write': None, 'step': step,
              'reasoning_buffer': rb, 'reasoning_count': rc}
    ep = torch.export.export(model, (h, state, gs), kwargs=kwargs, strict=False)
    print('torch.export OK')

    onnx_path = os.path.join(os.path.dirname(__file__), '..', 'spiral_coherence_test.onnx')
    torch.onnx.export(ep, (h, state, gs), kwargs=kwargs, dynamo=True, opset_version=17)
    print('torch.onnx.export OK ->', onnx_path)

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    inames = [i.name for i in sess.get_inputs()]
    feeds = {}
    for i in sess.get_inputs():
        name = i.name
        if name == 'h': feeds[name] = h.numpy()
        elif name == 'global_state': feeds[name] = gs.numpy()
        elif name == 'step': feeds[name] = step.numpy()
        elif name == 'reasoning_buffer': feeds[name] = rb.numpy()
        elif name == 'reasoning_count': feeds[name] = rc.numpy()
        else: feeds[name] = torch.zeros(tuple(i.shape)).numpy()
    res = sess.run(None, feeds)
    ort_out = torch.from_numpy(res[0])
    diff = (ort_out - ref_out).abs().max().item()
    print(f'ORT vs python: max abs diff = {diff:.6e}')
    assert diff < 1e-3, f'ONNX mismatch: {diff}'
    print('ONNX compatibility PASS')
    os.remove(onnx_path)
    print('cleaned up')