from __future__ import annotations
import torch
from pydantic import (
  BaseModel,
  Field,
  ConfigDict,
  ValidationError
  )
import json


def load_gemma_config(path: str) -> Gemma3nConfig:
    """
    Loads a Gemma 3N configuration from a JSON file.
    Supports both HF-style nested configs and custom flat configs.
    """
    with open(path, 'r') as f:
        config_dict = json.load(f)

    # 1. Map HF-style nested keys to our Pydantic names
    # text_config -> text, vision_config -> vision, etc.
    mappings = {
        "text_config": "text",
        "vision_config": "vision",
        "audio_config": "audio"
    }
    
    for hf_key, my_key in mappings.items():
        if hf_key in config_dict and my_key not in config_dict:
            config_dict[my_key] = config_dict.pop(hf_key)

    # 2. If sub-configs are still missing (flat config), 
    # we use the root dict to populate them
    if "text" not in config_dict:
        # Many flat configs are essentially TextConfigs
        config_dict["text"] = config_dict.copy()
        
    # 3. Validate and return
    try:
        return Gemma3nConfig.model_validate(config_dict)
    except ValidationError as e:
        print(f"Configuration validation failed: {e}")
        raise

class VisionConfig(BaseModel):
  model_config = ConfigDict(extra='ignore',
                            arbitrary_types_allowed=True,)
  do_pooling: bool = Field(default=False) # false
  hidden_size: int = Field(default=2048) # 2048
  initializer_range: float = Field(default=0.02) # 0.02
  label_names: list[str] = Field(default=["LABEL_0", "LABEL_1"]) 
  num_classes: int = Field(default=2) # 2
  rms_norm_eps: float = Field(default=1e-06) # 1e-06
  vocab_offset: int = Field(default=262144) # 262144
  vocab_size: int = Field(default=128) # 128

class AudioConfig(BaseModel):
  model_config = ConfigDict(extra='ignore',
                            arbitrary_types_allowed=True,
                          )
  conf_attention_chunk_size: int = Field(default=12)  # 12
  conf_attention_context_left: int = Field(default=13) #  13,
  conf_attention_context_right: int = Field(default=0)# 0,
  conf_attention_logit_cap: float = Field(default=50.0) # 50.0,
  conf_conv_kernel_size: int = Field(default=5)  #  5,
  conf_num_attention_heads: int = Field(default=8) # 8,
  conf_num_hidden_layers: int = Field(default=12) # 12,
  conf_reduction_factor: int = Field(default=4) # 4,
  conf_residual_weight: float = Field(default=0.5) # 0.5,
  gradient_clipping: float = Field(default=10_000_000_000.0) # 10000000000.0,
  hidden_size: int = Field(default=1536) # 1536,
  input_feat_size: int = Field(default=128) # 128,
  rms_norm_eps: float = Field(default=1e-06) # 1e-06,
  sscp_conv_channel_size: list[int] = Field(default=[128, 32]) # [128, 32]
  sscp_conv_group_norm_eps: float = Field(default=0.001) # 0.001,
  sscp_conv_kernel_size: list[list[int]] = Field(default=[[3,3],[3,3]])
  # [[3,3], [3,3]],
  sscp_conv_stride_size: list[list[int]] = Field(default=[[2,2],[2,2]]) # [[2,2], [2,2]],
  torch_dtype: str = Field(default="torch.bfloat16") # bfloat16
  param_dtype: str = Field(default="torch.float32")
  vocab_offset: int = Field(default=262272) # 262272,
  vocab_size: int = Field(default=128) # 128

class TextConfig(BaseModel):
  model_config = ConfigDict(extra='ignore',
                            arbitrary_types_allowed=True,
                          )
  activation_sparsity_pattern: list[float] = Field(default=[
      0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
  ])             # [ 0.95 10x, 0.0 20x ]
  altup_active_idx: int = Field(default=0)             # 0,
  altup_coef_clip: float = Field(default=120.0)        # 120.0,
  altup_correct_scale: bool = Field(default=True)      # true,
  altup_num_inputs: int = Field(default=4)             # 4,
  attention_bias: bool = Field(default=False)          # false,
  attention_dropout: float = Field(default=0.0)        # 0.0,
  final_logit_softcapping: float = Field(default=30.0) # 30.0,
  head_dim: int = Field(default=256)                   # 256,
  hidden_activation: str = Field(default="gelu_pytorch_tanh") # gelu_pytorch_tanh
  hidden_size: int = Field(default=2048)                # 2048,
  hidden_size_per_layer_input: int = Field(default=256) # 256,
  initializer_range: float = Field(default=0.02)        # 0.02,
  intermediate_size: list[int] = Field(default=[
    8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192
  ]) # [ 8192 x32]
  laurel_rank: int = Field(default=64)                  # 64,
  layer_types: list[str] = Field(default=[
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
  ])
  max_position_embeddings: int = Field(default=32768)   # 32768,
  num_attention_heads: int = Field(default=8)           # 8,
  num_hidden_layers: int = Field(default=30)            # 30,
  num_key_value_heads: int = Field(default=2)           # 2,
  num_kv_shared_layers: int = Field(default=10)         # 10,
  rms_norm_eps: float = Field(default=1e-06)            # 1e-06,
  rope_local_base_freq: float = Field(default=10000.0)  # 10000.0,
  rope_scaling: float | None = Field(default=None) # null, ?
  rope_theta: float = Field(default=1_000_000.0)        # 1_000_000.0,
  sliding_window: int = Field(default=512)              # 512,
  use_cache: bool = Field(default=True)                 # true,
  vocab_size: int = Field(default=262400)               # 262400,
  vocab_size_per_layer_input: int = Field(default=262144)  # 262144
    
class Gemma3nConfig(BaseModel):
  model_config = ConfigDict(extra='ignore',
                            arbitrary_types_allowed=True,
                            ignored_types=(TextConfig, VisionConfig, AudioConfig)
                          )
  architectures: list[str]
  vision_soft_tokens_per_image: int = Field(default=256) # 256  
  audio_soft_tokens_per_image: int = Field(default=188) # 188,
  audio_token_id: int = Field(default=262273) # 262273,
  boa_token_id: int = Field(default=256000) # 256000,
  boi_token_id: int = Field(default=255999) # 255999,
  eoa_token_id: int = Field(default=262272) # 262272,
  eoi_token_id: int = Field(default=262144) # 262144,
  image_token_id: int = Field(default=262145) # 262145,
  initializer_range: float = Field(default=0.02) # 0.02,

  text: TextConfig
  vision: VisionConfig | None
  audio: AudioConfig | None
