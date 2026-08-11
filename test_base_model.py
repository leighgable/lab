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

# Test prompt: "The capital of France is"
prompt = "The capital of France is"
tokens = model.tokenizer.encode(prompt, add_bos=True)
token_tensor = torch.tensor([tokens], device=device)

print(f"Input tokens: {tokens}")

with torch.no_grad():
    # Greedily generate 10 tokens
    current_tokens = token_tensor
    past_key_values = None
    
    generated = []
    for _ in range(10):
        # We need to handle segment_pos and cache manually here or use model.generate
        # But generate handles templates internally. Let's use model.generate but 
        # modify it to NOT use templates if we pass a flag? 
        # Actually, let's just use the transformer directly.
        
        batch_size, seq_len = current_tokens.shape
        segment_pos = torch.arange(seq_len, device=device).unsqueeze(0)
        
        outputs = model.transformer(input_ids=current_tokens, segment_pos=segment_pos)
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        generated.append(next_token.item())
        current_tokens = torch.cat([current_tokens, next_token], dim=-1)

    response = model.tokenizer.decode(generated)
    print(f"\nResponse: {response}")
