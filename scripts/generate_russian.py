"""
EVA Russian Generation Monitor - Full version with BPE tokenizer
"""
import torch
import sys
import os
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from core import WideBindConfig, WideBindStack

def main():
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/best.pt'

    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = ckpt['cfg']
    model = WideBindStack(cfg)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()

    step = ckpt.get('step', '?')
    val_loss = ckpt.get('best_val_loss', 0)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"=== EVA Generation Monitor ===")
    print(f"Checkpoint: step={step}, val_loss={val_loss:.4f}")
    print(f"Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    print()

    # Import generate function
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from generate import generate

    prompts = [
        "Привет, как дела?",
        "Москва — столица России",
        "Искусственный интеллект — это",
        "В начале было Слово",
    ]

    for prompt in prompts:
        try:
            text = generate(model, prompt, max_new_tokens=40, temperature=0.7, top_k=40)
            print(f"[{prompt}]")
            print(f"  -> {text[:200]}")
            print()
        except Exception as e:
            print(f"[{prompt}]")
            print(f"  ERROR: {e}")
            print()

    # Quality assessment
    print("=== Quality Assessment ===")
    if val_loss > 10.5:
        stage = "Random letters"
    elif val_loss > 10.0:
        stage = "Letter fragments"
    elif val_loss > 9.5:
        stage = "Word fragments"
    elif val_loss > 9.0:
        stage = "Simple words"
    elif val_loss > 8.5:
        stage = "Simple phrases"
    elif val_loss > 8.0:
        stage = "Connected text"
    elif val_loss > 7.0:
        stage = "Coherent text"
    else:
        stage = "Advanced reasoning"

    print(f"Current stage: {stage} (val_loss={val_loss:.4f})")
    print()

    # Milestones
    print("Milestones:")
    targets = [(10.0, "Fragments -> Words"),
               (9.5, "Words -> Phrases"),
               (9.0, "Phrases -> Text"),
               (8.0, "Text -> Coherence"),
               (7.0, "Coherence -> Reasoning")]
    for target_loss, desc in targets:
        if val_loss > target_loss:
            remaining = val_loss - target_loss
            est_steps = int(remaining / 0.05 * 250)
            print(f"  val_loss={target_loss}: {desc} (~{est_steps} steps)")
            break
    else:
        print("  All milestones reached!")

if __name__ == '__main__':
    main()
