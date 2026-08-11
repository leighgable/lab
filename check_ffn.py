import torch
from config import load_gemma_config
from chat import Gemma3ChatModel
import os

model_path = "/run/media/leigh/MEDIA/gemma_3n/"
config_path = os.path.join(model_path, "config.json")
config = load_gemma_config(config_path)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
model = Gemma3ChatModel.from_hf_pretrained(model_path, config, device=device)
model.eval()

print("\nLayer 0 FFN Stats:")
ffn = model.transformer.h[0].ffn
print(f"Gate-Up weight: mean={ffn.gate_up_proj.w.mean().item():.4f}, std={ffn.gate_up_proj.w.std().item():.4f}")
print(f"Down weight: mean={ffn.down_proj.w.mean().item():.4f}, std={ffn.down_proj.w.std().item():.4f}")

print("\nLayer 0 Norm Stats:")
print(f"Input Norm: mean={model.transformer.h[0].input_layernorm.weight.mean().item():.4f}, std={model.transformer.h[0].input_layernorm.weight.std().item():.4f}")
print(f"Pre-FFN Norm: mean={model.transformer.h[0].pre_feedforward_layernorm.weight.mean().item():.4f}, std={model.transformer.h[0].pre_feedforward_layernorm.weight.std().item():.4f}")

# Check logits for some words
prompt = "The capital of France is"
tokens = model.tokenizer.encode(prompt, add_bos=False)
token_tensor = torch.tensor([tokens], device=device)

with torch.no_grad():
    outputs = model(tokens=token_tensor)
    logits = outputs.logits[:, -1, :]
    
    # Check top 5 pieces
    probs = torch.softmax(logits.float(), dim=-1)
    top_p, top_i = torch.topk(probs, k=5)
    
    print(f"\nTop 5 pieces for prompt '{prompt}':")
    for p, i in zip(top_p[0], top_i[0]):
        print(f"'{model.tokenizer.IdToPiece(i.item())}': {p.item():.4f}")
