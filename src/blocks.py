import torch
import torch.nn as nn
from config import Gemma3nConfig
from matryoshka_modules import (
    MatryoshkaAttention,
    MatryoshkaNorm,
    LayerCache,
    Embedder,
    create_layer_cache,
)
from matryoshka_layers import (
    MatryoshkaFFN,
    PerLayerEmbedding,
    AlternatingUpdates,
    LaurelBlock,
    Einsum,
    precompute_rope_cache,
)

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
    ):
        d_model = d_model or x_stack.shape[-1]
        active_idx = self.config.text.altup_active_idx
        
        x_stack = self.altup.predict(x_stack, d_model)
        
        # get active residual stream
        h = x_stack[active_idx]
        
        # ple update
        if per_layer_latent is not None:
            h = h + self.per_layer_embedding(h, per_layer_latent, self.block_idx, d_model)
            
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
            sin_cache=sin_cache
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
        
        return x_stack, past_key_values, shared_kv_states


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
        
        self.register_buffer("global_cos", None, persistent=False)
        self.register_buffer("global_sin", None, persistent=False)
        self.register_buffer("local_cos", None, persistent=False)
        self.register_buffer("local_sin", None, persistent=False)
        self._init_rope()

    def _init_rope(self):
        max_seq = self.config.text.max_position_embeddings
        head_dim = self.config.text.head_dim
        
        g_cos, g_sin = precompute_rope_cache(max_seq, head_dim, self.config.text.rope_theta)
        l_cos, l_sin = precompute_rope_cache(max_seq, head_dim, self.config.text.rope_local_base_freq)
        
        self.global_cos, self.global_sin = g_cos, g_sin
        self.local_cos, self.local_sin = l_cos, l_sin

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
        soft_tokens: torch.Tensor | None = None,
        soft_token_mask: torch.Tensor | None = None,
        prefix_length: int = 0,
        rope_pos: torch.Tensor | None = None,
        use_cache: bool = True,
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
            input_ids, d_model, soft_tokens, soft_token_mask
        )
        
        # altup stack [num_inputs, batch, seq, d_model]
        x_stack = x_base.unsqueeze(0).repeat(self.config.text.altup_num_inputs, 1, 1, 1)
        
        new_past_key_values = [] if use_cache else None
        shared_kv = None # vertical cache resets every block
        
        for i, block in enumerate(self.h):
            # per_layer_inputs [b, s, n_layers, latent]
            block_per_layer = None
            if per_layer_inputs is not None:
                block_per_layer = per_layer_inputs[:, :, i, :]
            
            cos = self.global_cos if block.attn.attn_type == "full_attention" else self.local_cos
            sin = self.global_sin if block.attn.attn_type == "full_attention" else self.local_sin
            
            x_stack, kv, shared_kv = block(
                x_stack=x_stack,
                segment_pos=segment_pos,
                per_layer_latent=block_per_layer,
                past_key_values=past_key_values[i] if past_key_values else None,
                shared_kv_states=shared_kv,
                d_model=d_model,
                d_ff=self.config.text.intermediate_size[i],
                prefix_length=prefix_length,
                rope_pos=rope_pos,
                cos_cache=cos,
                sin_cache=sin
            )
            if use_cache:
                new_past_key_values.append(kv)
            
        # get active stream and apply final scale/norm
        # altup final scale: [b, s, d_model]
        x = self.h[0].altup.scale_corrected_output(x_stack[self.config.text.altup_active_idx], d_model)
        x = self.final_norm(x, h=d_model)
        
        logits = self.lm_head("bsh, vh -> bsv", x, h=d_model)
        
        return logits, new_past_key_values

    @classmethod
    def from_hf_pretrained(cls,
                           repo_id: str | None, # or path to local folder
                           config: Gemma3nConfig,  # set with local flag
                           device: str = "cpu",
                           local: bool = True,
                       ) -> tuple[spm.SentencePieceProcessor, Gemma3nTransformer]:
        """loads weights from a Hugging Face repository (safetensors) with memory efficiency."""
        if not local:
            from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        import sentencepiece as spm
        import os
        import gc

        # Initialize model on CPU first to save VRAM/RAM during loading
        model = cls(config)
        tokenizer = spm.SentencePieceProcessor()
        
        if not local:
            try:
                model_path = snapshot_download(repo_id, allow_patterns=["*.safetensors", "tokenizer.model"])
            except Exception as e:
                raise RuntimeError(f"Failed to download from {repo_id}: {e}")
        else:
            model_path = repo_id

        tokenizer.Load(os.path.join(model_path, "tokenizer.model"))
        
        # Load weights file by file
        files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
        
        with torch.no_grad():
            for f in files:
                state_dict = load_file(os.path.join(model_path, f))
                
                if "model.embed_tokens.weight" in state_dict:
                    model.wte.input_embedding_table.copy_(state_dict["model.embed_tokens.weight"])
                    
                if "model.norm.weight" in state_dict:
                    model.final_norm.weight.copy_(state_dict["model.norm.weight"])

                # per-layer weights
                for i in range(config.text.num_hidden_layers):
                    hf_prefix = f"model.layers.{i}"
                    block = model.h[i]
                    
                    norm_map = {
                        "input_layernorm": block.input_layernorm,
                        "post_attention_layernorm": block.post_attention_layernorm,
                        "pre_feedforward_layernorm": block.pre_feedforward_layernorm,
                        "post_feedforward_layernorm": block.post_feedforward_layernorm
                    }
                    for hf_n, my_n in norm_map.items():
                        key = f"{hf_prefix}.{hf_n}.weight"
                        if key in state_dict:
                            my_n.weight.copy_(state_dict[key])

                    # qkv
                    q_w = state_dict.get(f"{hf_prefix}.self_attn.q_proj.weight")
                    k_w = state_dict.get(f"{hf_prefix}.self_attn.k_proj.weight")
                    v_w = state_dict.get(f"{hf_prefix}.self_attn.v_proj.weight")
                    
                    if q_w is not None and k_w is not None and v_w is not None:
                        # hf: [num_heads * head_dim, hidden] -> [num_heads, head_dim, hidden] -> [num_heads, hidden, head_dim]
                        q_w = q_w.view(config.text.num_attention_heads, config.text.head_dim, -1).permute(0, 2, 1)
                        k_w = k_w.view(config.text.num_key_value_heads, config.text.head_dim, -1).permute(0, 2, 1)
                        v_w = v_w.view(config.text.num_key_value_heads, config.text.head_dim, -1).permute(0, 2, 1)
                        
                        qkv_fused = torch.cat([q_w, k_w, v_w], dim=0)
                        block.attn.qkv_einsum.w.copy_(qkv_fused)

                    # fused norm 
                    q_n = state_dict.get(f"{hf_prefix}.self_attn.q_norm.weight")
                    k_n = state_dict.get(f"{hf_prefix}.self_attn.k_norm.weight")
                    if q_n is not None and k_n is not None:
                        q_n = q_n.view(config.text.num_attention_heads, config.text.head_dim)
                        k_n = k_n.view(config.text.num_key_value_heads, config.text.head_dim)
                        v_n = torch.ones_like(k_n) # value norm identity if missing
                        qkv_norm_fused = torch.cat([q_n, k_n, v_n], dim=0)
                        block.attn.qkv_norm.weight.copy_(qkv_norm_fused)

                    # output 
                    o_w = state_dict.get(f"{hf_prefix}.self_attn.o_proj.weight")
                    if o_w is not None:
                        o_w = o_w.view(config.text.hidden_size, config.text.num_attention_heads, config.text.head_dim).permute(1, 2, 0)
                        block.attn.attn_vec_einsum.w.copy_(o_w)

                    # ffn fusion
                    gate_w = state_dict.get(f"{hf_prefix}.mlp.gate_proj.weight")
                    up_w = state_dict.get(f"{hf_prefix}.mlp.up_proj.weight")
                    if gate_w is not None and up_w is not None:
                        gate_up_fused = torch.cat([gate_w.T, up_w.T], dim=1)
                        block.ffn.gate_up_proj.w.copy_(gate_up_fused)
                    
                    down_w = state_dict.get(f"{hf_prefix}.mlp.down_proj.weight")
                    if down_w is not None:
                        block.ffn.down_proj.w.copy_(down_w.T)

                    # altup, laurel, ple
                    comp_map = {
                        "alt_up.router.weight": (block.altup, "modality_router"),
                        "alt_up.prediction_coefs.weight": (block.altup, "prediction_proj"),
                        "laurel.linear_left.weight": (block.laurel, "linear_left"),
                        "laurel.linear_right.weight": (block.laurel, "linear_right"),
                    }
                    for hf_k, (my_comp, my_attr) in comp_map.items():
                        key = f"{hf_prefix}.{hf_k}"
                        if key in state_dict:
                            w = state_dict[key]
                            target = getattr(my_comp, my_attr)
                            target.w.copy_(w.T if w.ndim == 2 else w)

                del state_dict
                gc.collect()

        model.lm_head.w = model.wte.input_embedding_table
        model.to(device)
        
        return tokenizer, model


    
