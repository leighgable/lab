import torch
import sys
import os

# Mock configuration for testing
from config import Gemma3nConfig, TextConfig, VisionConfig, AudioConfig
from chat import Gemma3ChatModel

def test_multimodal_forward():
    print("Testing Multimodal Forward Pass...")
    
    # 1. Setup Mock Config
    text_cfg = TextConfig(
        hidden_size=512, 
        num_hidden_layers=2, 
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        intermediate_size=[1024, 1024]
    )
    vision_cfg = VisionConfig(hidden_size=2048)
    audio_cfg = AudioConfig(input_feat_size=128, hidden_size=512)
    
    config = Gemma3nConfig(
        architectures=["Gemma3nForCausalLM"],
        text=text_cfg,
        vision=vision_cfg,
        audio=audio_cfg,
        vision_soft_tokens_per_image=256,
        audio_soft_tokens_per_image=50
    )

    # 2. Instantiate Model
    model = Gemma3ChatModel(config)
    model.eval()

    # 3. Prepare Mock Inputs
    # Tokens with placeholders (-2 for image, -4 for audio)
    tokens = torch.tensor([[100, -2, 101, -4, 102]]) 
    
    # Image patches [b, n, num_patches, patch_dim]
    # num_patches=256 (16x16), patch_dim=588 (assuming 14x14x3)
    images = torch.randn(1, 1, 256, 588) 
    
    # Audio MEL [batch, frames, bins]
    audio = torch.randn(1, 800, 128) 
    audio_mask = torch.zeros(1, 800, dtype=torch.bool)

    # 4. Run Forward Pass
    print("Running forward pass...")
    with torch.no_grad():
        logits, cache = model(
            tokens=tokens,
            images=images,
            audio=audio,
            audio_mask=audio_mask
        )

    print(f"Forward pass successful!")
    print(f"Logits shape: {logits.shape}")
    
    # Expected seq length:
    # 100 (1) + boi + 256 + eoi + 101 (1) + boa + 188 + eoa + 102 (1) + some \n\n (4?)
    # Input.prepare_multimodal_stream adds \n\n around BOI/EOI and BOA/EOA
    print(f"Interleaved sequence length: {logits.shape[1]}")

if __name__ == "__main__":
    try:
        test_multimodal_forward()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
