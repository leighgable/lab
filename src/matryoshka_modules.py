import torch
import torch.nn.functional as F
import torch.nn as nn
from config import Gemma3nConfig
from matryoshka_layers import (
    Linear,
    Einsum,
    MatryoshkaNorm,
    apply_rotary_pos_emb,
)

K_MASK = -2.3819763e38
LayerCache = dict[str, torch.Tensor]
# [batch, seq_len, num_attention_heads, head_dim]

def create_sliding_mask(
    q_pos: torch.Tensor,
    k_pos: torch.Tensor,
    window_size: int | None = None,
    prefix_length: int = 0,
    is_causal: bool = True
):
    """
    Creates an attention mask supporting:
    - Causal masking (if is_causal=True)
    - Sliding window (if window_size > 0)
    - Bidirectional prefix (if prefix_length > 0)
    """
    # q_pos: [batch, q_seq], k_pos: [batch, k_seq]
    # dist: [batch, q_seq, k_seq]
    dist = q_pos.unsqueeze(-1) - k_pos.unsqueeze(-2)
    
    # 1. Base Mask (Causal or Bidirectional)
    if is_causal:
        mask = (dist >= 0)
    else:
        mask = torch.ones_like(dist, dtype=torch.bool)
        
    # 2. Sliding Window Constraint
    if window_size is not None and window_size > 0:
        is_in_window = (dist < window_size)
        # KV tokens in the prefix are exempt from sliding window (always visible)
        is_kv_prefix = (k_pos.unsqueeze(-2) < prefix_length)
        mask &= (is_in_window | is_kv_prefix)
        
    # 3. Bidirectional Prefix Exception
    # Allow query tokens in the prefix to see all other prefix tokens (bidirectional)
    if is_causal and prefix_length > 0:
        is_q_prefix = (q_pos.unsqueeze(-1) < prefix_length)
        is_kv_prefix = (k_pos.unsqueeze(-2) < prefix_length)
        mask |= (is_q_prefix & is_kv_prefix)
        
    return mask

class MatryoshkaAttention(nn.Module):
    def __init__(self, config: Gemma3nConfig, block_idx: int):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.embedding_dim = config.hidden_size
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_key_value_heads
        self.attn_type = config.layer_types[block_idx]
        self.rope_base_frequency = config.rope_local_base_freq if self.attn_type == "sliding_attention" else config.rope_theta
        self.attn_logits_softcap = config.final_logit_softcapping
        self.sliding_window_size = config.sliding_window
        self.is_producer: bool = (block_idx % config.num_kv_shared_layers == 0)
      
        # Fused QKV projection: [Max_Heads, Hidden, Head_Dim]
        # Layout: [Q_Heads, K_Heads, V_Heads]
        self.qkv_einsum = Einsum(
            weight_shape=(
                self.num_heads + 2 * self.num_kv_heads,
                self.embedding_dim,
                self.head_dim
            ),
            pattern="ndh",
        )
           
        # Fused QKV normalization: [Max_Heads, Head_Dim]
        self.qkv_norm = MatryoshkaNorm(
            weight_shape=(
                self.num_heads + 2 * self.num_kv_heads,
                self.head_dim,
            ),
            pattern="nh",
        )

        # Output projection: [Heads, Head_Dim, Hidden]
        self.attn_vec_einsum = Einsum(
            weight_shape=(self.num_heads, self.head_dim, self.embedding_dim),
            pattern="nhd",
        )

    def forward(self,
                x: torch.Tensor,
                d_model: int,
                segment_pos: torch.Tensor,
                kv_cache: LayerCache | None = None,
                attn_mask: torch.Tensor | None = None,
                kv_shared_cache: LayerCache | None = None,
                prefix_length: int = 0,
                rope_pos: torch.Tensor | None = None,
                cos_cache: torch.Tensor | None = None,
                sin_cache: torch.Tensor | None = None,
                **side_inputs,
    ) -> tuple[LayerCache | None, LayerCache | None, torch.Tensor]:

        n_heads = (self.num_heads * d_model) // self.embedding_dim
        group_size = self.num_heads // self.num_kv_heads
        n_heads = max(n_heads, group_size)
        n_kv_heads = n_heads // group_size
        total_heads = n_heads + 2 * n_kv_heads
        
        needs_kv = (kv_shared_cache is None)             

        if not needs_kv:
            q = self.qkv_einsum("btd, ndh -> btnh", x, n=n_heads, d=d_model)
            q = self.qkv_norm(q, n=n_heads)
            k, v = kv_shared_cache['k'], kv_shared_cache['v']
        else:
            qkv = self.qkv_einsum("btd, ndh -> btnh", x, n=total_heads, d=d_model)
            qkv = self.qkv_norm(qkv, n=total_heads)
            
            q, k, v = torch.split(qkv, [n_heads, n_kv_heads, n_kv_heads], dim=2)
        
        # Force consistent dtype
        q, k, v = q.to(x.dtype), k.to(x.dtype), v.to(x.dtype)

        # Apply RoPE to Q (and K if it's the current sequence)
        if cos_cache is not None and sin_cache is not None:
            r_pos = rope_pos if rope_pos is not None else segment_pos
            cos = cos_cache[r_pos].unsqueeze(2).to(x.dtype)
            sin = sin_cache[r_pos].unsqueeze(2).to(x.dtype)
            q = apply_rotary_pos_emb(q, cos, sin)
            if needs_kv:
                k = apply_rotary_pos_emb(k, cos, sin)

        if kv_cache is not None:
            batch_size, seq_len = x.shape[:2] 
            curr_idx = kv_cache['end_index'][0].item()

            if needs_kv:
                # Producer: Insert current RoPE-applied tokens into the allocated heads
                kv_cache['k'][:, curr_idx:curr_idx + seq_len, :n_kv_heads, :] = k
                kv_cache['v'][:, curr_idx:curr_idx + seq_len, :n_kv_heads, :] = v
                kv_cache['positions'][:, curr_idx:curr_idx + seq_len] = segment_pos
                kv_cache['end_index'] += seq_len
                actual_end_idx = curr_idx + seq_len
            else:
                actual_end_idx = curr_idx

            # Use full history from the cache
            k = kv_cache['k'][:, :actual_end_idx, :n_kv_heads, :]
            v = kv_cache['v'][:, :actual_end_idx, :n_kv_heads, :]
            k_pos = kv_cache['positions'][:, :actual_end_idx]
        else:
            k_pos = segment_pos

        # Attention Dynamic Generation
        if attn_mask is None:
            window = self.sliding_window_size if self.attn_type == "sliding_attention" else None
            mask_bool = create_sliding_mask(
                q_pos=segment_pos,
                k_pos=k_pos,
                window_size=window,
                prefix_length=prefix_length,
                is_causal=True
            )
            attn_mask = torch.where(mask_bool, 0.0, K_MASK).to(x.dtype)
            # Unsqueeze for broadcasting across heads: [batch, 1, q_seq, k_seq]
            attn_mask = attn_mask.unsqueeze(1)

        # GQA Attention using broadcasting to save memory
        heads_per_group = n_heads // n_kv_heads
        scale = self.head_dim**-0.5
        q_seq_len = x.shape[1]
        k_seq_len = k.shape[1]
        
        if heads_per_group > 1:
            # q: [batch, q_seq, n_heads, head_dim] -> [batch, n_heads, q_seq, head_dim]
            q_reshaped = q.transpose(1, 2)
            # -> [batch, n_kv_heads, heads_per_group, q_seq, head_dim]
            q_reshaped = q_reshaped.view(batch_size, n_kv_heads, heads_per_group, q_seq_len, self.head_dim)
            
            # k: [batch, k_seq, n_kv_heads, head_dim] -> [batch, n_kv_heads, k_seq, head_dim]
            k_reshaped = k.to(x.dtype).transpose(1, 2)
            # [batch, n_kv_heads, 1, k_seq, head_dim]
            k_reshaped = k_reshaped.unsqueeze(2)
            
            # [batch, n_kv_heads, heads_per_group, q_seq, k_seq]
            logits = torch.matmul(q_reshaped, k_reshaped.transpose(-1, -2)) * scale
            # [batch, n_heads, q_seq, k_seq]
            logits = logits.reshape(batch_size, n_heads, q_seq_len, k_seq_len)
        else:
            # [batch, n_heads, q_seq, k_seq]
            logits = torch.einsum("btnh, bsnh -> bnts", q, k.to(x.dtype)) * scale
        
        if self.attn_logits_softcap is not None:
            logits = torch.tanh(logits / self.attn_logits_softcap) * self.attn_logits_softcap
            
        logits = logits + attn_mask
        probs = F.softmax(logits, dim=-1).to(x.dtype)
        
        # Mix values using broadcasting
        if heads_per_group > 1:
            # probs: [batch, n_heads, q_seq, k_seq] -> [batch, n_kv_heads, heads_per_group, q_seq, k_seq]
            probs_reshaped = probs.view(batch_size, n_kv_heads, heads_per_group, q_seq_len, k_seq_len)
            # v: [batch, k_seq, n_kv_heads, head_dim] -> [batch, n_kv_heads, k_seq, head_dim]
            v_reshaped = v.to(x.dtype).transpose(1, 2)
            # [batch, n_kv_heads, 1, k_seq, head_dim]
            v_reshaped = v_reshaped.unsqueeze(2)
            
            # [batch, n_kv_heads, heads_per_group, q_seq, head_dim]
            encoded = torch.matmul(probs_reshaped, v_reshaped)
            # [batch, q_seq, n_heads, head_dim]
            encoded = encoded.reshape(batch_size, n_heads, q_seq_len, self.head_dim).transpose(1, 2)
        else:
            # [batch, q_seq, n_heads, head_dim]
            encoded = torch.einsum("bnts, bsnh -> btnh", probs, v.to(x.dtype))
        
        # Output projection
        attn_output = self.attn_vec_einsum(eqn="btnh, nhd -> btd",
                                           x=encoded,
                                           n=n_heads, 
                                           d=d_model,
                                       )
        
        # vertical cache (shared KV) for subsequent layers in this block
        new_kv_shared_cache = kv_shared_cache
        if self.is_producer:
            batch_size, seq_len = x.shape[:2]
            new_kv_shared_cache = {
                'k': k[:, -seq_len:, :, :],
                'v': v[:, -seq_len:, :, :]
            }
        
        return kv_cache, new_kv_shared_cache, attn_output

def create_layer_cache(
    batch_size: int,
    cache_size: int,
    num_key_value_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
) -> LayerCache:
    """Initializes a single layer's KV cache."""
    return {
        'v': torch.zeros(
            (batch_size, cache_size, num_key_value_heads, head_dim), 
            dtype=dtype, 
            device=device
        ),
        'k': torch.zeros(
            (batch_size, cache_size, num_key_value_heads, head_dim), 
            dtype=dtype, 
            device=device
        ),
        'end_index': torch.zeros((batch_size,), dtype=torch.int32, device=device),
        'positions': torch.zeros((batch_size, cache_size), dtype=torch.int32, device=device),
    }

class HadamardAttention(nn.Module):
    def __init__(self, config: Gemma3nConfig):
        super().__init__()
        self.qkv_proj = Linear(config.hidden_size, (config.num_attention_heads + 2 * config.num_key_value_heads) * config.head_dim, bias=False)

        self.hadamard_scale = nn.Parameter(torch.ones(config.hidden_size))
        self.hadamard_bias = nn.Parameter(torch.zeros(config.hidden_size))

        self.register_buffer("H", self._generate_hadamard(config.hidden_size))

    def _generate_hadamard(self, n):
        import scipy.linalg
        return torch.from_numpy(scipy.linalg.hadamard(n)).float()

    def forward(self, x: torch.Tensor, mask=None):
        q, k, v = self.project_qkv(x)
        
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False
        )

        y = attn_out.transpose(1, 2).reshape(x.shape) # hadamard mixing

        mixed = torch.matmul(y, self.H)
        return self.hadamard_scale * mixed + self.hadamard_bias

class Embedder(nn.Module):
    """
    text, soft tokens ie: vision/audio, per-layer alt/up inputs.
    """
    def __init__(self, config: Gemma3nConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.text.vocab_size
        self.hidden_size = config.text.hidden_size
        
        # [Vocab, Hidden]
        self.input_embedding_table = nn.Parameter(
            torch.empty(self.vocab_size, self.hidden_size)
        )
        nn.init.normal_(self.input_embedding_table, std=config.text.initializer_range)

        # soft embeddings 
        if config.audio:
            self.mm_input_projection = Einsum(
                weight_shape=(config.audio.input_feat_size, config.audio.hidden_size),
                pattern="eh", # e=input_feat_size, h=hidden
            )
            self.mm_soft_embedding_norm = MatryoshkaNorm(
                weight_shape=(config.audio.hidden_size,),
                pattern="h",
            )

        # per_layer_inputs and alt/up
        if config.text.hidden_size_per_layer_input:
            # per-layer offsets: [vocab, layers, latent_dim]
            self.per_layer_embedding_table = nn.Parameter(
                torch.empty(
                    self.vocab_size, 
                    config.text.num_hidden_layers, 
                    config.text.hidden_size_per_layer_input
                )
            )
            nn.init.normal_(self.per_layer_embedding_table, std=config.text.initializer_range)
            
            # residual width: [latent, hidden]
            self.per_layer_projection = Einsum(
                weight_shape=(config.text.hidden_size_per_layer_input, config.text.hidden_size),
                pattern="lh", # l=latent, h=hidden
            )
            self.per_layer_projection_norm = MatryoshkaNorm(
                weight_shape=(config.text.hidden_size,),
                pattern="h",
            )

    def forward(
        self, 
        input_ids: torch.Tensor, 
        d_model: int,
        soft_tokens: torch.Tensor | None = None,
        soft_token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            input_ids: [batch, seq] token IDs
            d_model: Current Matryoshka width
            soft_tokens: [batch, seq, feat_size] 
            soft_token_mask: [batch, seq] true = use soft_tokens
            
        Returns:
            x: [batch, seq, d_model] initial residual stream
            per_layer_inputs: [batch, seq, num_layers, latent_dim] AltUp side-inputs
        """
        x = F.embedding(input_ids, self.input_embedding_table)
        
        if soft_tokens is not None and soft_token_mask is not None:
            # [b, s, e] -> [b, s, h]
            soft_x = self.mm_input_projection("bse, eh -> bsh", soft_tokens, h=self.hidden_size)
            soft_x = self.mm_soft_embedding_norm(soft_x, h=self.hidden_size)
            
            # soft_token_mask: [batch, seq, 1] for broadcasting
            mask = soft_token_mask.unsqueeze(-1)
            x = torch.where(mask, soft_x, x)

        x = x[:, :, :d_model]
        
        # scale by sqrt(max_hidden_size))
        x = x * (self.hidden_size ** 0.5)

        per_layer_inputs = None
        if hasattr(self, 'per_layer_embedding_table'):
            # [batch, seq, num_layers, latent_dim]
            # F.embedding doesn't support 3D weights, use direct indexing
            # input_ids: [b, s], table: [v, l, d] -> [b, s, l, d]
            per_layer_inputs = self.per_layer_embedding_table[input_ids]
            
        return x, per_layer_inputs

    def project_per_layer_input(self, latent: torch.Tensor, d_model: int) -> torch.Tensor:
        """Projects per-layer latent to the current residual width."""
        # latent: [batch, seq, latent_dim]
        x = self.per_layer_projection("bsl, lh -> bsh", latent, h=d_model)
        return self.per_layer_projection_norm(x, h=d_model)
