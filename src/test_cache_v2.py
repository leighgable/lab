import torch
from config import Gemma3nConfig, TextConfig
from blocks import Gemma3nTransformer

def test_init_cache():
    text_config = TextConfig(
        num_hidden_layers=10,
        num_kv_shared_layers=3,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=128,
        hidden_size=1024,
        max_position_embeddings=1024,
        sliding_window=256,
        layer_types=["sliding_attention"] * 10
    )
    config = Gemma3nConfig(
        architectures=["Gemma3nForCausalLM"],
        text=text_config,
        vision=None,
        audio=None
    )
    
    model = Gemma3nTransformer(config)
    batch_size = 2
    caches = model.init_cache(batch_size=batch_size)
    
    assert len(caches) == 10
    
    # 0, 1, 2 should share caches[0]
    # 3, 4, 5 should share caches[3]
    # 6, 7, 8 should share caches[6]
    # 9 should share caches[9]
    
    assert caches[0] is caches[1]
    assert caches[0] is caches[2]
    assert caches[0] is not caches[3]
    
    assert caches[3] is caches[4]
    assert caches[3] is caches[5]
    assert caches[3] is not caches[6]
    
    assert caches[6] is caches[7]
    assert caches[6] is caches[8]
    assert caches[6] is not caches[9]
    
    print("Cache sharing logic verified!")
    
    # Check cache sizes
    # Sliding window is 256, max_pos is 1024.
    # Should be 256.
    assert caches[0]['k'].shape[1] == 256
    print("Cache size logic verified!")

    # Test forward with dynamic cache init
    input_ids = torch.zeros((2, 5), dtype=torch.long)
    segment_pos = torch.arange(5).unsqueeze(0).repeat(2, 1)
    
    # This should trigger init_cache internally
    logits, new_caches = model(input_ids, segment_pos, use_cache=True)
    assert len(new_caches) == 10
    assert new_caches[0] is not None
    print("Forward dynamic cache init verified!")

if __name__ == "__main__":
    test_init_cache()
