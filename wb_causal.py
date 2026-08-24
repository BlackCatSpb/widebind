"""WideBind step_10857 — перенос методики rUadapter.
Тест A: steering направления animal на финальном hidden (flip/КЛ).
Тест B: линейная композициональность по слоям (corr(dAB, dA+dB)).
Тест C: поворот базиса cos(d_L, d_final) по слоям для 4 концептов.
Модель маленькая (D=2560, 24 слоя, ~0.56GB), CPU-форварды дешёвые."""
import sys, os, time, json, re, inspect
sys.path.insert(0, r"C:\Users\black\OneDrive\Desktop\WideBind")
import numpy as np
import torch
from torch.serialization import add_safe_globals
from core import WideBindConfig, WideBindStack
from core.lambda_utils import LambdaConfig
from tokenizers import Tokenizer

CKPT = r"C:\Users\black\OneDrive\Desktop\WideBind\checkpoints\step_10857.pt"
TOK  = r"C:\Users\black\OneDrive\Desktop\WideBind\wb\russian_tokenizer\tokenizer.json"
T0=time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)

add_safe_globals([WideBindConfig, LambdaConfig])
state = torch.load(CKPT, map_location='cpu', mmap=True, weights_only=False)
cfg = state['cfg']
step = state.get('step'); bvl = state.get('best_val_loss')
log(f"checkpoint step={step} best_val_loss={bvl}  cfg D={cfg.D} nL={cfg.n_layers} vocab={cfg.vocab}")
model = WideBindStack(cfg)
model.load_state_dict(state['model'], strict=False)
model.eval(); del state
N = len(model.layers)
log("model ready params(MB)=", round(model.param_count()*4/1e6,1))

tok = Tokenizer.from_file(TOK); tok.enable_padding(pad_id=0, pad_token='<|pad|>')
maxid = max(tok.encode("кошка мяукает и собака лает текст случилось хорошо плохо маленький большой женщина мужчина").ids)
log("tokenizer ok, max id =", maxid, "(vocab", cfg.vocab, ")")

h_emb_ok = 'h_emb' in inspect.signature(model.lm_head.forward).parameters
cap = [None]*N
def make_hook(i):
    def hook(m, inp, outp):
        cap[i] = outp[0][0,-1,:].detach().clone()
    return hook
for i in range(N):
    model.layers[i].register_forward_hook(make_hook(i))

@torch.no_grad()
def run(phrase):
    ids = tok.encode(phrase).ids
    assert max(ids) < cfg.vocab, f"id {max(ids)} >= vocab"
    t = torch.tensor([ids])
    h = model.embed_tokens(t)
    out,_,_,_ = model(h, adaptive=False, step=None)
    return [cap[i].clone() for i in range(N)], out[0,-1,:].clone(), h[0,-1,:].clone()

@torch.no_grad()
def logits_of(out_vec, h_vec):
    o = out_vec.unsqueeze(0).unsqueeze(0); he = h_vec.unsqueeze(0).unsqueeze(0)
    if h_emb_ok: return model.lm_head(o, he)[0,0]
    return model.lm_head(o)[0,0]

CON = {
 "animal": (["кошка мяукает","кошка ловит мышей","у кошки мягкая шерсть","кошка спит на окне","домашняя кошка игрива"],
           ["собака лает","собака охраняет дом","у собаки густая шерсть","собака гуляет во дворе","домашняя собака преданна"]),
 "sentiment": (["день был хорошим","отзыв хороший","настроение хорошее","результат хороший","вечер хороший"],
               ["день был плохим","отзыв плохой","настроение плохое","результат плохой","вечер плохой"]),
 "size": (["дом маленький","город маленький","зал маленький","стол маленький","шкаф маленький"],
          ["дом большой","город большой","зал большой","стол большой","шкаф большой"]),
 "gender": (["женщина высокая","женщина вошла","эта женщина умна","старая женщина","сильная женщина"],
            ["мужчина высокий","мужчина вошёл","этот мужчина умён","старый мужчина","сильный мужчина"]),
}
# предвычислить per-layer h_L [5,24,D], out_last [5,D], embed_last [5,D] для A/B каждого концепта
HL, OL, HLLAST = {}, {}, {}
for c,(Ap,Bp) in CON.items():
    ha = np.array([run(p)[0] for p in Ap]); hb = np.array([run(p)[0] for p in Bp])  # after-layer [5,24,D]
    oa = np.array([run(p)[1].numpy() for p in Ap]); ob = np.array([run(p)[1].numpy() for p in Bp])
    hla = np.array([run(p)[2].numpy() for p in Ap]); hlb = np.array([run(p)[2].numpy() for p in Bp])
    HL[c] = (ha, hb); OL[c] = (oa, ob); HLLAST[c] = (hla, hlb)
    log(f"  {c}: HL/A shape {ha.shape}")

def normrows(x): return x/np.linalg.norm(x,axis=1,keepdims=True)

# ===== Тест C: поворот базиса cos(d_enter_L, d_final) =====
# d_enter_L = направление на residual, ВХОДЯЩЕМ в слой L (как rUadapter q_proj input):
#   L=0 -> эмбеддинг; L>=1 -> выход слоя L-1. d_final = направление на финальном hidden.
testC={}
for c in CON:
    ha,hb=HL[c]; oa,ob=OL[c]; hla,hlb=HLLAST[c]
    dF = normrows((oa.mean(0)-ob.mean(0))[None])[0]  # final direction [D]
    cosL=[]
    for L in range(N):
        if L==0: vA, vB = hla, hlb
        else:    vA, vB = ha[:,L-1,:], hb[:,L-1,:]
        dL = normrows((vA.mean(0)-vB.mean(0))[None])[0]
        cosL.append(float(np.dot(dL,dF)))
    testC[c]=dict(cos_to_final=cosL)
    Ls=[0,4,8,12,16,20,23]
    log(f"  {c} cos(d_enter_L,d_final): "+" ".join(f"L{l}:{cosL[l]:.2f}" for l in Ls))

# ===== Тест A: steering animal на финали =====
ap, bp = CON["animal"]
oa,ob = OL["animal"]
d = (ob.mean(0)-oa.mean(0)); d /= np.linalg.norm(d)
rng=np.random.default_rng(0); r=rng.standard_normal(2560); r/=np.linalg.norm(r)
probe = ap[0]  # "кошка мяукает"
_, out_clean, h_clean = run(probe)
lg_clean = logits_of(out_clean, h_clean).softmax(-1).numpy()
KL = lambda p,q: float(np.sum(p*np.log((p+1e-12)/(q+1e-12))))
dog_ids = list(set(tok.encode("собака").ids) & set(range(cfg.vocab)))
cat_ids = list(set(tok.encode("кошка").ids) & set(range(cfg.vocab)))
testA={"dog_mass_clean":float(lg_clean[dog_ids].sum()), "cat_mass_clean":float(lg_clean[cat_ids].sum())}
for alpha in [5,20,50,100]:
    res={}
    for tag,vec in [("d",d),("rnd",r)]:
        out_s = out_clean + alpha*torch.tensor(vec, dtype=out_clean.dtype)
        lg = logits_of(out_s, h_clean).softmax(-1).numpy()
        kl = KL(lg, lg_clean); top=int(np.argmax(lg))
        res[tag]=dict(kl=kl, top=tok.decode([top]), top_p=float(lg[top]))
        testA[f"a{alpha}_{tag}"]=res[tag]
    log(f"  steer a={alpha} d: KL={res['d']['kl']:.3f} top={res['d']['top']!r}({res['d']['top_p']:.4f}) "
        f"| rnd KL={res['rnd']['kl']:.3f} | dog_mass {testA['dog_mass_clean']:.4f}->{lg[dog_ids].sum():.4f}")

# ===== Тест B: композициональность по слоям (cos(dAB_agg, dA+dB)) =====
R = "и это случилось"
hR = np.array(run(R)[0])  # [24,D]
pairs=[("animal","sentiment"),("animal","size"),("animal","gender"),("sentiment","size")]
testB={}
for (c1,c2) in pairs:
    ha1,_=HL[c1]; ha2,_=HL[c2]
    dA = normrows(ha1.mean(0)-hR)        # d_L концепта c1  [24,D]
    dB = normrows(ha2.mean(0)-hR)        # d_L концепта c2  [24,D]
    dAB_agg=np.zeros((N,2560))
    for i in range(5):
        ab=f"{CON[c1][0][i]} {CON[c2][0][i]}"
        dAB_agg += np.array([run(ab)[0][L].numpy() for L in range(N)]) - hR
    dAB_agg /= 5
    dAB_n=normrows(dAB_agg); target=normrows(dA+dB)
    corrL=[float(np.dot(dAB_n[L],target[L])) for L in range(N)]
    testB[f"{c1}+{c2}"]=corrL
    L=[0,4,8,12,16,20,23]
    log(f"  comp {c1}+{c2}: corr="+ " ".join(f"L{l}:{corrL[l]:.2f}" for l in L))

OUT=os.path.join(r"C:\Users\black\OneDrive\Desktop\WideBind","wb_region_causal_report.json")
json.dump(dict(step=step, best_val_loss=bvl, testA=testA, testB=testB, testC=testC),
          open(OUT,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
log("saved", OUT)
