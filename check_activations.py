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

# Test prompt: "Hello!"
prompt = "Hello!"
tokens = model.tokenizer.encode(prompt, add_bos=True)
token_tensor = torch.tensor([tokens], device=device)

print(f"Input tokens: {tokens}")

with torch.no_grad():
    # Capture hidden states
    outputs = model.transformer(input_ids=token_tensor, segment_pos=torch.arange(len(tokens), device=device).unsqueeze(0))
    h = outputs.hidden_states # [batch, seq, d_model]
    
    print(f"\nFinal Hidden States (before head): mean={h.mean().item():.4f}, std={h.std().item():.4f}, max={h.max().item():.4f}, min={h.min().item():.4f}")
    
    # Check layer 0 output
    x_base, _ = model.transformer.wte(input_ids=token_tensor, d_model=2048)
    print(f"Embeddings: mean={x_base.mean().item():.4f}, std={x_base.std().item():.4f}")
    
    x_stack = x_base.unsqueeze(0).repeat(4, 1, 1, 1)
    block0 = model.transformer.h[0]
    # Simulate block forward (no cache)
    # segment_pos is same
    x_out, _, _, _ = block0(x_stack=x_stack, segment_pos=torch.arange(len(tokens), device=device).unsqueeze(0))
    h0 = x_out[0]
    print(f"Layer 0 output: mean={h0.mean().item():.4f}, std={h0.std().item():.4f}")

