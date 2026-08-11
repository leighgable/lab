from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn
import einops
from config import Gemma3nConfig, AudioConfig
from matryoshka_modules import (
    MatryoshkaAttention,
    MatryoshkaNorm,
    LayerCache,
    Embedder,
    create_layer_cache,
    AdaptiveSurprisal,
)
from matryoshka_layers import (
    MatryoshkaFFN,
    PerLayerEmbedding,
    AlternatingUpdates,
    LaurelBlock,
    Einsum,
    precompute_rope_cache,
)
from audio_modules import (
        Gemma3nAudioConformerAttention,
        Gemma3nAudioConformerFeedForward,
        Gemma3nAudioSubSampleConvProjection,
        Gemma3nAudioConformerLightConv1d,
    )

DOUBLE_NEWLINE_TOKEN = 108
IMAGE_SOFT_TOKEN_PLACEHOLDER = -2
AUDIO_SOFT_TOKEN_PLACEHOLDER = -4

@dataclass
class Output:
  """Output of the Gemma model.

  Attributes:
    logits: Predicted logits of the model.
    cache: Updated cache if the input cache is not None, None elsewhere.
    hidden_states: The hidden states of the model.
  """

  # When `return_last_only`, `logits` is `*B V`
  logits: torch.Tensor # [b, l, v] | [b, v]
  cache: LayerCache | None
  hidden_states: torch.Tensor | None # [b, l, d] | [b, d]

@dataclass
class Gemma3nAudioEncoderModelOutput:
    last_hidden_state: torch.Tensor
    audio_mel_mask: torch.Tensor

@dataclass    
class Input:
    """
    Container for raw multimodal inputs.
    Used to prepare the interleaved token stream before embedding.
    """
    tokens: torch.Tensor              # [b, l] raw text tokens
    images: torch.Tensor | None = None # [b, n, h, w, c] raw image patches/pixels
    audio: torch.Tensor | None = None  # [b, a, samples] raw audio
    config: Gemma3nConfig | None = None

    def prepare_multimodal_stream(self, 
                                 image_placeholder: int = IMAGE_SOFT_TOKEN_PLACEHOLDER,
                                 audio_placeholder: int = AUDIO_SOFT_TOKEN_PLACEHOLDER,
                                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Interleaves text, image, and audio placeholders based on templates.
        Returns:
            new_tokens: [batch, total_len]
            modality_mask: [batch, total_len] (0=Text, 1=Image, 2=Audio)
        """
        tokens = self.tokens
        # Define templates
        image_seq = [
            DOUBLE_NEWLINE_TOKEN,
            self.config.boi_token_id,
            *[image_placeholder] * self.config.vision_soft_tokens_per_image,
            self.config.eoi_token_id,
            DOUBLE_NEWLINE_TOKEN,
        ]
        
        audio_seq = [
            DOUBLE_NEWLINE_TOKEN,
            self.config.boa_token_id,
            *[audio_placeholder] * self.config.audio_soft_tokens_per_image,
            self.config.eoa_token_id,
            DOUBLE_NEWLINE_TOKEN,
        ]

        batch_tokens = []
        batch_masks = []
        max_len = 0

        for b in range(tokens.shape[0]):
            row_tokens = []
            row_mask = []
            
            for token in tokens[b]:
                t_val = token.item()
                if t_val == image_placeholder:
                    row_tokens.extend(image_seq)
                    # \n\n, boi, [soft tokens], eoi, \n\n
                    # Only the soft tokens part (index 2 to -2) should be modality 1
                    mask_seq = [0, 0] + [1] * self.config.vision_soft_tokens_per_image + [0, 0]
                    row_mask.extend(mask_seq)
                elif t_val == audio_placeholder:
                    row_tokens.extend(audio_seq)
                    # \n\n, boa, [soft tokens], eoa, \n\n
                    mask_seq = [0, 0] + [2] * self.config.audio_soft_tokens_per_image + [0, 0]
                    row_mask.extend(mask_seq)
                else:
                    row_tokens.append(t_val)
                    row_mask.append(0)
            
            max_len = max(max_len, len(row_tokens))
            batch_tokens.append(torch.tensor(row_tokens, dtype=torch.long))
            batch_masks.append(torch.tensor(row_mask, dtype=torch.long))

        # Pad sequences to match max_len
        padded_tokens = torch.full((tokens.shape[0], max_len), 0, dtype=torch.long, device=tokens.device)
        padded_masks = torch.full((tokens.shape[0], max_len), 0, dtype=torch.long, device=tokens.device)

        for i, (t, m) in enumerate(zip(batch_tokens, batch_masks)):
            l = len(t)
            padded_tokens[i, :l] = t
            padded_masks[i, :l] = m

        return padded_tokens, padded_masks

    
class Gemma3nBlock(nn.Module):
    def __init__(self,
                 config: Gemma3nConfig,
                 block_idx: int,
    ):
        super().__init__()
        self.block_idx = block_idx
        self.config = config
        
        # 1. Augmentation Layers
        self.altup = AlternatingUpdates(config=config.text)
        self.laurel = LaurelBlock(config=config.text)
        self.per_layer_embedding = PerLayerEmbedding(config=config.text)
        
        # 2. Attention Branch
        self.input_layernorm = MatryoshkaNorm(
            weight_shape=(config.text.hidden_size,),
            pattern="h",
        )
        self.attn = MatryoshkaAttention(
            config=config.text,
            block_idx=block_idx,
        )
        self.post_attention_layernorm = MatryoshkaNorm(
            weight_shape=(config.text.hidden_size,),
            pattern="h"
        )
        
        # 3. FFN Branch
        self.pre_feedforward_layernorm = MatryoshkaNorm(
            weight_shape=(config.text.hidden_size,),
            pattern="h"
        )
        self.ffn = MatryoshkaFFN(
            config=config.text,
            block_idx=block_idx,
        )
        self.post_feedforward_layernorm = MatryoshkaNorm(
            weight_shape=(config.text.hidden_size,),
            pattern="h",
        )

        # 4. Surprisal Monitor (only on global attention blocks for efficiency)
        self.surprisal_monitor = None
        if self.attn.attn_type == "full_attention":
            self.surprisal_monitor = AdaptiveSurprisal(config)

    def forward(
        self,
        x_stack: torch.Tensor, # [num_inputs, batch, seq, d_model]
        segment_pos: torch.Tensor,
        per_layer_latent: torch.Tensor | None = None, # latent for PLE
        past_key_values: LayerCache | None = None,
        shared_kv_states: LayerCache | None = None,
        attention_mask: torch.Tensor | None = None,
        d_model: int | None = None,
        d_ff: int | None = None,
        prefix_length: int = 0,
        rope_pos: torch.Tensor | None = None,
        cos_cache: torch.Tensor | None = None,
        sin_cache: torch.Tensor | None = None,
        modality_mask: torch.Tensor | None = None,
    ):
        d_model = d_model or x_stack.shape[-1]
        d_ff = d_ff or self.config.text.intermediate_size[self.block_idx]
        active_idx = self.config.text.altup_active_idx
        
        x_stack = self.altup.predict(x_stack, d_model)
        
        # get active residual stream
        h = x_stack[active_idx]
        
        # ple update
        if per_layer_latent is not None:
            h = h + self.per_layer_embedding(h, per_layer_latent, self.block_idx, d_model)
            x_stack[active_idx] = h
            
        # laurel update 
        x_stack = self.laurel(x_stack, d_model)
        # refresh 
        h = x_stack[active_idx]
        
        # attn branch
        residual = h
        h = self.input_layernorm(h, h=d_model)
        
        # Attn returns (new_kv_cache, new_shared_kv, attn_output)
        past_key_values, shared_kv_states, h = self.attn(
            x=h,
            d_model=d_model,
            segment_pos=segment_pos,
            kv_cache=past_key_values,
            attn_mask=attention_mask,
            kv_shared_cache=shared_kv_states,
            prefix_length=prefix_length,
            rope_pos=rope_pos,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            modality_mask=modality_mask,
        )
        
        h = self.post_attention_layernorm(h, h=d_model)
        h = residual + h
        
        # ffn branch
        residual = h
        h = self.pre_feedforward_layernorm(h, h=d_model)
        h = self.ffn(h, d_model, d_ff)
        h = self.post_feedforward_layernorm(h, h=d_model)
        h = residual + h
        
        # insert updated active stream and sync stack
        x_stack = self.altup.correct(x_stack, h, d_model)

        # 5. Adaptive Slicing: Monitor surprisal and update d_model for the next layer
        if self.surprisal_monitor is not None:
            d_model = self.surprisal_monitor(
                h=h,
                d_model=d_model,
                max_width=self.config.text.hidden_size
            )
        
        return x_stack, past_key_values, shared_kv_states, d_model


class Gemma3nTransformer(nn.Module):
    def __init__(self, config: Gemma3nConfig):
        super().__init__()
        self.config = config
        
        self.wte = Embedder(config)
        
        self.h = nn.ModuleList([
            Gemma3nBlock(config, i) for i in range(config.text.num_hidden_layers)
        ])
        
        self.final_norm = MatryoshkaNorm(
            weight_shape=(config.text.hidden_size,),
            pattern="h"
        )
        self.lm_head = Einsum(
            weight_shape=(config.text.vocab_size, config.text.hidden_size),
            pattern="vh", # v=vocab, h=hidden
        )
        
        # tie weights
        self.lm_head.w = self.wte.input_embedding_table
        for block in self.h:
            if block.surprisal_monitor is not None:
                block.surprisal_monitor.head.w = self.wte.input_embedding_table
        
        self.register_buffer("global_cos", None, persistent=False)
        self.register_buffer("global_sin", None, persistent=False)
        self.register_buffer("local_cos", None, persistent=False)
        self.register_buffer("local_sin", None, persistent=False)
        self._init_rope()

    def _init_rope(self):
        max_seq = self.config.text.max_position_embeddings
        head_dim = self.config.text.head_dim
        
        # Determine target dtype
        dtype_str = getattr(self.config.text, "torch_dtype", "float32")
        dtype_map = {
            "torch.bfloat16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "torch.float16": torch.float16,
            "float16": torch.float16,
        }
        target_dtype = dtype_map.get(dtype_str, torch.float32)

        g_cos, g_sin = precompute_rope_cache(max_seq, head_dim, self.config.text.rope_theta)
        l_cos, l_sin = precompute_rope_cache(max_seq, head_dim, self.config.text.rope_local_base_freq)
        
        self.global_cos, self.global_sin = g_cos.to(target_dtype), g_sin.to(target_dtype)
        self.local_cos, self.local_sin = l_cos.to(target_dtype), l_sin.to(target_dtype)

    def init_cache(self, batch_size: int, dtype: torch.dtype = torch.bfloat16) -> list[LayerCache]:
        """
        Initializes the KV cache for all layers.
        Layers that share KV heads will share the same LayerCache object to save memory.
        
        Note: We always allocate for the MAXIMUM number of KV heads 
        (defined in config.num_key_value_heads) to support dynamic Matryoshka slicing.
        """
        device = next(self.parameters()).device
        num_layers = self.config.text.num_hidden_layers
        shared_period = self.config.text.num_kv_shared_layers
        
        caches = []
        current_shared_cache = None
        
        for i in range(num_layers):
            if i % shared_period == 0:
                # producer layer
                cache_size = self.config.text.max_position_embeddings
                # Sliding window layers only need to store the window size
                if self.config.text.layer_types[i] == "sliding_attention":
                    cache_size = min(cache_size, self.config.text.sliding_window)
                
                current_shared_cache = create_layer_cache(
                    batch_size=batch_size,
                    cache_size=cache_size,
                    num_key_value_heads=self.config.text.num_key_value_heads,
                    head_dim=self.config.text.head_dim,
                    dtype=dtype,
                    device=device
                )
            
            # producer, consumer share same cache
            caches.append(current_shared_cache)
            
        return caches

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_pos: torch.Tensor,
        d_model: int | None = None,
        d_ff: int | None = None,
        past_key_values: list[LayerCache] | None = None,
        vision_tokens: torch.Tensor | None = None,
        audio_tokens: torch.Tensor | None = None,
        prefix_length: int = 0,
        rope_pos: torch.Tensor | None = None,
        use_cache: bool = True,
        modality_mask: torch.Tensor | None = None,
    ):
        d_model = d_model or self.config.text.hidden_size
        d_ff = d_ff or self.config.text.intermediate_size[0] # Simplified
        
        # Initialize cache if requested but not provided (e.g. prefill/first token)
        if use_cache and past_key_values is None:
            batch_size = input_ids.shape[0]
            past_key_values = self.init_cache(batch_size=batch_size)
            
        # x_base: [b, s, d_model]
        # per_layer_inputs: [b, s, num_layers, latent_dim]
        x_base, per_layer_inputs = self.wte(
            input_ids=input_ids,
            d_model=d_model,
            vision_tokens=vision_tokens,
            audio_tokens=audio_tokens,
            modality_mask=modality_mask,
        )
        
        # altup stack [num_inputs, batch, seq, d_model]
        x_stack = x_base.unsqueeze(0).repeat(self.config.text.altup_num_inputs, 1, 1, 1)
        
        new_past_key_values = [] if use_cache else None
        shared_kv = None 
        
        for i, block in enumerate(self.h):
            # per_layer_inputs [b, s, n_layers, latent]
            block_per_layer = None
            if per_layer_inputs is not None:
                block_per_layer = per_layer_inputs[:, :, i, :]
            
            cos = self.global_cos if block.attn.attn_type == "full_attention" else self.local_cos
            sin = self.global_sin if block.attn.attn_type == "full_attention" else self.local_sin
            
            x_stack, kv, _, d_model = block(
                x_stack=x_stack,
                segment_pos=segment_pos,
                per_layer_latent=block_per_layer,
                past_key_values=past_key_values[i] if past_key_values else None,
                shared_kv_states=None,
                d_model=d_model,
                d_ff=self.config.text.intermediate_size[i],
                prefix_length=prefix_length,
                rope_pos=rope_pos,
                cos_cache=cos,
                sin_cache=sin,
                modality_mask=modality_mask,
            )
            if use_cache:
                new_past_key_values.append(kv)
            
        # get active stream and apply final scale/norm
        # altup final scale: [b, s, d_model]
        # x = self.h[-1].altup.scale_corrected_output(x_stack[self.config.text.altup_active_idx], d_model)
        x = x_stack[self.config.text.altup_active_idx]
        x = self.final_norm(x, h=d_model)
        
        logits = self.lm_head("bsh, vh -> bsv", x, h=d_model)

        if self.config.text.final_logit_softcapping is not None:
             logits = torch.tanh(logits / self.config.text.final_logit_softcapping) * self.config.text.final_logit_softcapping
        
        return Output(
            logits=logits,
            cache=new_past_key_values,
            hidden_states=x
        )

    def _encode_and_get_inputs(self,
                               tokens: torch.Tensor,
                               images: torch.Tensor | None = None,
                               audio: torch.Tensor | None = None,
                           ) -> Input:
        """
        Prepares multimodal inputs and returns an Input container.
        """
        if images is not None:
            # Ensure shape is [batch, num_images, height, width, channels]
            if len(images.shape) == 4:
                images = einops.rearrange(images, 'b h w c -> b 1 h w c')
        
        return Input(
            tokens=tokens,
            images=images,
            audio=audio,
            config=self.config
        )

    @classmethod
    def from_hf_pretrained(cls,
                           repo_id: str | None, # or path to local folder
                           config: Gemma3nConfig,  # set with local flag
                           device: str = "cpu",
                           local: bool = True,
                           low_cpu_mem_usage: bool = True,
                           load_tokenizer: bool = True,
                       ) -> tuple[Any, Gemma3nTransformer]:
        """loads weights from a Hugging Face repository (safetensors) with memory efficiency."""
        if not local:
            from huggingface_hub import snapshot_download
        from safetensors import safe_open
        import os
        import gc

        # 0. Determine target dtype from config
        dtype_str = getattr(config.text, "torch_dtype", "float32")
        dtype_map = {
            "torch.bfloat16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "torch.float16": torch.float16,
            "float16": torch.float16,
            "torch.float32": torch.float32,
            "float32": torch.float32,
        }
        target_dtype = dtype_map.get(dtype_str, torch.float32)

        # 1. Initialize model with target dtype to save initial RAM
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(target_dtype)
        try:
            model = cls(config)
        finally:
            torch.set_default_dtype(original_dtype)
            
        if not local:
            try:
                model_path = snapshot_download(repo_id, allow_patterns=["*.safetensors", "tokenizer.model"])
            except Exception as e:
                raise RuntimeError(f"Failed to download from {repo_id}: {e}")
        else:
            model_path = repo_id

        if low_cpu_mem_usage:
            model.to(device)

        # Load weights file by file using safe_open to avoid loading entire files into RAM
        files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
        layers_loaded = [False] * config.text.num_hidden_layers
        
        with torch.no_grad():
            for f_name in files:
                with safe_open(os.path.join(model_path, f_name), framework="pt", device="cpu") as f:
                    state_dict_keys = f.keys()
                    
                    # Try both prefixes
                    for prefix in ["model.", "model.language_model."]:
                        embed_key = f"{prefix}embed_tokens.weight"
                        if embed_key in state_dict_keys:
                            model.wte.input_embedding_table.copy_(f.get_tensor(embed_key))
                            # print(f"Loaded {embed_key}")
                        
                        norm_key = f"{prefix}norm.weight"
                        if norm_key in state_dict_keys:
                            model.final_norm.weight.copy_(f.get_tensor(norm_key))
                            # print(f"Loaded {norm_key}")
                        
                        # per-layer embed table
                        ple_table_key = f"{prefix}embed_tokens_per_layer.weight"
                        if ple_table_key in state_dict_keys:
                             w = f.get_tensor(ple_table_key)
                             # Reshape to [vocab, layers, latent]
                             # Checkpoint: [262144, 7680] -> [262144, 30, 256]
                             w = w.view(w.shape[0], config.text.num_hidden_layers, config.text.hidden_size_per_layer_input)
                             # Copy into the slice of the table
                             model.wte.per_layer_embedding_table[:w.shape[0], :, :].copy_(w)

                    # per-layer weights
                    for i in range(config.text.num_hidden_layers):
                        block = model.h[i]
                        
                        # Find the actual prefix for this layer in the current file
                        actual_hf_prefix = None
                        for p in [f"model.layers.{i}", f"model.language_model.layers.{i}"]:
                            if any(k.startswith(p) for k in state_dict_keys):
                                actual_hf_prefix = p
                                break
                        
                        if actual_hf_prefix is None:
                            continue
                        
                        # print(f"Loading layer {i} from {actual_hf_prefix}")
                        layers_loaded[i] = True

                        norm_map = {
                            "input_layernorm": block.input_layernorm,
                            "post_attention_layernorm": block.post_attention_layernorm,
                            "pre_feedforward_layernorm": block.pre_feedforward_layernorm,
                            "post_feedforward_layernorm": block.post_feedforward_layernorm
                        }
                        for hf_n, my_n in norm_map.items():
                            key = f"{actual_hf_prefix}.{hf_n}.weight"
                            if key in state_dict_keys:
                                my_n.weight.copy_(f.get_tensor(key))

                        # qkv
                        q_key = f"{actual_hf_prefix}.self_attn.q_proj.weight"
                        k_key = f"{actual_hf_prefix}.self_attn.k_proj.weight"
                        v_key = f"{actual_hf_prefix}.self_attn.v_proj.weight"
                        
                        if q_key in state_dict_keys or k_key in state_dict_keys or v_key in state_dict_keys:
                            target_w = block.attn.qkv_einsum.w
                            n_q = config.text.num_attention_heads
                            n_kv = config.text.num_key_value_heads
                            
                            if q_key in state_dict_keys:
                                q_w = f.get_tensor(q_key).view(n_q, config.text.head_dim, -1).permute(0, 2, 1)
                                target_w[:n_q].copy_(q_w)
                            if k_key in state_dict_keys:
                                k_w = f.get_tensor(k_key).view(n_kv, config.text.head_dim, -1).permute(0, 2, 1)
                                target_w[n_q : n_q + n_kv].copy_(k_w)
                            if v_key in state_dict_keys:
                                v_w = f.get_tensor(v_key).view(n_kv, config.text.head_dim, -1).permute(0, 2, 1)
                                target_w[n_q + n_kv : n_q + 2 * n_kv].copy_(v_w)

                        # qkv norm
                        q_n_key = f"{actual_hf_prefix}.self_attn.q_norm.weight"
                        k_n_key = f"{actual_hf_prefix}.self_attn.k_norm.weight"
                        if q_n_key in state_dict_keys or k_n_key in state_dict_keys:
                            target_n = block.attn.qkv_norm.weight
                            if q_n_key in state_dict_keys:
                                q_n = f.get_tensor(q_n_key)
                                if q_n.ndim == 1:
                                     target_n[:config.text.num_attention_heads].copy_(q_n)
                                else:
                                     target_n[:config.text.num_attention_heads].copy_(q_n.view(config.text.num_attention_heads, config.text.head_dim))
                            if k_n_key in state_dict_keys:
                                k_n = f.get_tensor(k_n_key)
                                if k_n.ndim == 1:
                                     target_n[config.text.num_attention_heads : config.text.num_attention_heads + config.text.num_key_value_heads].copy_(k_n)
                                else:
                                     target_n[config.text.num_attention_heads : config.text.num_attention_heads + config.text.num_key_value_heads].copy_(k_n.view(config.text.num_key_value_heads, config.text.head_dim))

                        # output 
                        o_key = f"{actual_hf_prefix}.self_attn.o_proj.weight"
                        if o_key in state_dict_keys:
                            o_w = f.get_tensor(o_key).view(config.text.hidden_size, config.text.num_attention_heads, config.text.head_dim).permute(1, 2, 0)
                            block.attn.attn_vec_einsum.w.copy_(o_w)

                        # mlp
                        gate_key = f"{actual_hf_prefix}.mlp.gate_proj.weight"
                        up_key = f"{actual_hf_prefix}.mlp.up_proj.weight"
                        if gate_key in state_dict_keys or up_key in state_dict_keys:
                            target_ffn = block.ffn.gate_up_proj.w
                            d_ff = config.text.intermediate_size[i]
                            if gate_key in state_dict_keys:
                                target_ffn[:, :d_ff].copy_(f.get_tensor(gate_key).T)
                            if up_key in state_dict_keys:
                                target_ffn[:, d_ff : 2*d_ff].copy_(f.get_tensor(up_key).T)
                        
                        down_key = f"{actual_hf_prefix}.mlp.down_proj.weight"
                        if down_key in state_dict_keys:
                            block.ffn.down_proj.w.copy_(f.get_tensor(down_key).T)

                        # altup, laurel, ple
                        # Handle potential naming differences (alt_up vs altup)
                        altup_hf = None
                        for s in [".altup", ".alt_up"]:
                             if any(k.startswith(actual_hf_prefix + s) for k in state_dict_keys):
                                 altup_hf = actual_hf_prefix + s
                                 break
                        
                        if altup_hf:
                            comp_map = {
                                "modality_router.weight": (block.altup, "modality_router"),
                                "router_norm.weight": (block.altup, "router_norm"),
                                "prediction_coefs.weight": (block.altup, "prediction_proj"),
                                "correction_coefs.weight": (block.altup, "correction_proj"),
                                "correct_output_scale": (block.altup, "correct_output_scale"),
                            }
                            for hf_k, (my_comp, my_attr) in comp_map.items():
                                key = f"{altup_hf}.{hf_k}"
                                if key in state_dict_keys:
                                    w = f.get_tensor(key)
                                    target = getattr(my_comp, my_attr)
                                    target_param = target if isinstance(target, nn.Parameter) else (target.weight if isinstance(target, MatryoshkaNorm) else target.w)
                                    
                                    if target_param.shape == w.shape:
                                        target_param.copy_(w)
                                    elif w.ndim == 2 and target_param.shape == w.T.shape:
                                        target_param.copy_(w.T)
                                    elif target_param.numel() == w.numel():
                                        # Reshape fallback (try to detect if T is needed by comparing dims)
                                        # If target is [H, L] and w is [L, H], we should T then reshape
                                        target_param.copy_(w.reshape(target_param.shape))
                                    else:
                                         print(f"Warning: Shape mismatch for {key}: {w.shape} vs {target_param.shape}")

                        # laurel
                        laurel_prefix = f"{actual_hf_prefix}.laurel"
                        laurel_map = {
                            "linear_left.weight": (block.laurel, "linear_left"),
                            "linear_right.weight": (block.laurel, "linear_right"),
                            "post_laurel_norm.weight": (block.laurel, "post_laurel_norm"),
                        }
                        for hf_k, (my_comp, my_attr) in laurel_map.items():
                            key = f"{laurel_prefix}.{hf_k}"
                            if key in state_dict_keys:
                                w = f.get_tensor(key)
                                target = getattr(my_comp, my_attr)
                                target_param = target if isinstance(target, nn.Parameter) else (target.weight if isinstance(target, MatryoshkaNorm) else target.w)
                                
                                if target_param.shape == w.shape:
                                    target_param.copy_(w)
                                elif w.ndim == 2 and target_param.shape == w.T.shape:
                                    target_param.copy_(w.T)
                                elif target_param.numel() == w.numel():
                                    target_param.copy_(w.reshape(target_param.shape))
                                else:
                                     print(f"Warning: Shape mismatch for {key}: {w.shape} vs {target_param.shape}")

                        # ple
                        ple_map = {
                            "per_layer_input_gate.weight": (block.per_layer_embedding.down_proj, "w"),
                            "per_layer_projection.weight": (block.per_layer_embedding.up_proj, "w"),
                            "post_per_layer_input_norm.weight": (block.per_layer_embedding.post_per_layer_input_norm, "weight") if hasattr(block.per_layer_embedding, "post_per_layer_input_norm") else None,
                        }
                        for hf_k, target_info in ple_map.items():
                            if target_info is None: continue
                            key = f"{actual_hf_prefix}.{hf_k}"
                            if key in state_dict_keys:
                                w = f.get_tensor(key)
                                target_module, target_attr = target_info
                                target_param = getattr(target_module, target_attr) if isinstance(target_module, (nn.Module, nn.Parameter)) else target_module
                                
                                if target_param.shape == w.shape:
                                    target_param.copy_(w)
                                elif w.ndim == 2 and target_param.shape == w.T.shape:
                                    target_param.copy_(w.T)
                                elif target_param.numel() == w.numel():
                                    target_param.copy_(w.reshape(target_param.shape))
                                else:
                                    print(f"Warning: Shape mismatch for {key}: {w.shape} vs {target_param.shape}")



                gc.collect()

        print(f"Loaded {sum(layers_loaded)}/{len(layers_loaded)} layers")
        model.lm_head.w = model.wte.input_embedding_table
        if not low_cpu_mem_usage:
            model.to(device)

        tokenizer = None
        if load_tokenizer:
            import sentencepiece as spm
            tokenizer = spm.SentencePieceProcessor()
            tokenizer.Load(os.path.join(model_path, "tokenizer.model"))
        
        return tokenizer, model

class Gemma3nAudioConformerBlock(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config

        self.ffw_layer_start = Gemma3nAudioConformerFeedForward(self.config)
        self.attention = Gemma3nAudioConformerAttention(self.config)
        self.lconv1d = Gemma3nAudioConformerLightConv1d(self.config)
        self.ffw_layer_end = Gemma3nAudioConformerFeedForward(self.config)
        self.register_buffer("gradient_clipping", torch.tensor(self.config.gradient_clipping), persistent=False)
        self.norm = MatryoshkaNorm(
                                   weight_shape=(
                                       self.config.hidden_size
                                   ),
                                   pattern="h",
                                )

    def forward(self,
                audio_encodings: torch.Tensor,
                audio_mel_mask: torch.BoolTensor
            ) -> torch.Tensor:
        audio_encodings = self.ffw_layer_start(audio_encodings)
        audio_encodings = self.attention(audio_encodings, audio_mel_mask)
        validity_mask_for_lconv = ~audio_mel_mask  # True for valid
        audio_encodings_for_lconv_input = audio_encodings * validity_mask_for_lconv.unsqueeze(-1).to(
            audio_encodings.dtype
        )
        audio_encodings = self.lconv1d(audio_encodings_for_lconv_input)

        audio_encodings = self.ffw_layer_end(audio_encodings)
        audio_encodings = torch.clamp(audio_encodings, -self.gradient_clipping, self.gradient_clipping)
        output = self.norm(audio_encodings, h=self.config.hidden_size)
        return output
    
class Gemma3nAudioEncoder(nn.Module):
    """
    An audio encoder based on the [Universal Speech Model](https://huggingface.co/papers/2303.01037) architecture.
    """

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config

        self.subsample_conv_projection = Gemma3nAudioSubSampleConvProjection(config)
        self.conformer = nn.ModuleList(
            [Gemma3nAudioConformerBlock(config) for _ in range(config.conf_num_hidden_layers)]
        )

    def forward(
        self, audio_mel: torch.Tensor, audio_mel_mask: torch.BoolTensor
    ) -> Gemma3nAudioEncoderModelOutput:
        """Encodes a batch of MELs.

        Args:
            audio_mel: a torch.Tensor of shape [batch, num_frames, num_channels,
              mel_bins].
            audio_mel_mask: a torch.BoolTensor of shape [batch, num_frames].

        Returns:
            Gemma3nAudioEncoderModelOutput containing:
                last_hidden_state: [batch_size, seq_len, hidden_size]
                audio_mel_mask: [batch, seq_len]
        """
        audio_encodings = self.subsample_conv_projection(audio_mel)  # audio_encodings: [B, T_sub, D]

        # Subsample the input audio_mel_mask to match the time dimension of audio_encodings (T_sub)
        t_sub = audio_encodings.shape[1]

        time_stride_product = 1
        for stride_pair_idx in range(len(self.config.sscp_conv_stride_size)):
            time_stride_product *= self.config.sscp_conv_stride_size[stride_pair_idx][0]

        # Create indices for gathering from the original mask.
        # These indices map to original time steps corresponding to the start of each
        # receptive field in the subsampled output.
        indices = torch.arange(t_sub, device=audio_mel_mask.device) * time_stride_product
        indices = torch.clamp(indices, max=audio_mel_mask.shape[1] - 1)  # Ensure indices are valid

        # Expand indices for batch compatibility if B > 1 and indices is 1D.
        if audio_mel_mask.ndim > 1 and indices.ndim == 1:
            indices = indices.unsqueeze(0).expand(audio_mel_mask.shape[0], -1)  # [B, T_sub]
        elif (
            audio_mel_mask.ndim == indices.ndim
            and audio_mel_mask.shape[0] == 1
            and indices.shape[0] != 1
            and t_sub == indices.shape[0]
        ):
            # Handle case where B=1 but indices became [T_sub] instead of [1, T_sub]
            indices = indices.unsqueeze(0)

        current_mask = torch.gather(audio_mel_mask, 1, indices)  # [B, T_sub]

        for block in self.conformer:
            audio_encodings = block(audio_encodings, current_mask)  # Pass the processed mask

        if self.config.conf_reduction_factor > 1:
            audio_encodings = audio_encodings[:, :: self.config.conf_reduction_factor]
            # Reduce the mask as well
            current_mask = current_mask[:, :: self.config.conf_reduction_factor]

        audio_encodings = audio_encodings.masked_fill(current_mask.unsqueeze(-1), 0.0)
        return Gemma3nAudioEncoderModelOutput(
            last_hidden_state=audio_encodings,
            audio_mel_mask=current_mask,
        )
    @classmethod
    def from_hf_pretrained(cls,
                           model_path: str,
                           config: AudioConfig,
                           device: str = "cpu",
                           low_cpu_mem_usage: bool = True
                       ) -> "Gemma3nAudioEncoder":
        from safetensors import safe_open
        import os

        # 1. Determine target dtype from config
        dtype_str = getattr(config, "torch_dtype", "bfloat16").replace("torch.", "")
        target_dtype = getattr(torch, dtype_str, torch.bfloat16)

        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(target_dtype)

        try:
            with torch.device("meta" if low_cpu_mem_usage else "cpu"):
                model = cls(config)

            if low_cpu_mem_usage:
                # Materialize parameters on the target device
                model.to_empty(device=device)
        finally:
            torch.set_default_dtype(original_dtype)

        # Pre-calculate params_dict for efficiency
        params_dict = dict(model.named_parameters())
        prefix = "model.audio_tower."

        files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
        with torch.no_grad():
            for f_name in files:
                # Load tensors to CPU first to avoid VRAM spikes during safe_open
                with safe_open(os.path.join(model_path, f_name), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if not key.startswith(prefix):
                            continue
                        
                        local_key = key[len(prefix):]
                        
                        # Handle the naming mismatch: Checkpoint uses '.weight', Einsum layers use '.w'
                        target_key = local_key
                        if local_key.endswith(".weight"):
                            possible_w_key = local_key[:-6] + "w"
                            if possible_w_key in params_dict:
                                target_key = possible_w_key
                        
                        if target_key in params_dict:
                            param = params_dict[target_key]
                            tensor = f.get_tensor(key)
                            
                            if tensor.shape != param.shape:
                                tensor = tensor.view(param.shape)
                            
                            param.copy_(tensor)
                        else:
                            # Only warn if it's not a known skip or if truly missing
                            print(f"Warning: Key {local_key} (mapped to {target_key}) not found in Gemma3nAudioEncoder")
        
        return model
