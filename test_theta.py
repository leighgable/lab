import torch
from config import load_gemma_config
from chat import Gemma3ChatModel
import os

model_path = "/run/media/leigh/MEDIA/gemma_3n/"
config_path = os.path.join(model_path, "config.json")
config = load_gemma_config(config_path)

# OVERRIDE RoPE
# config.text.rope_theta = 10000.0

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
model = Gemma3ChatModel.from_hf_pretrained(model_path, config, device=device)
model.eval()

# Test prompt: "The capital of France is"
prompt = "The capital of France is"
tokens = model.tokenizer.encode(prompt, add_bos=True)
token_tensor = torch.tensor([tokens], device=device)

print(f"Input tokens: {tokens}")

with torch.no_grad():
    outputs = model(tokens=token_tensor)
    logits = outputs.logits[:, -1, :]
    probs = torch.softmax(logits.float(), dim=-1)
    top_probs, top_ids = torch.topk(probs, k=5)
    
    print("\nTop 5 predictions:")
    for p, i in zip(top_probs[0], top_ids[0]):
        token_str = model.tokenizer.decode([i.item()])
        print(f"ID {i.item():6d}: {p.item():.4f} | '{token_str}'")
