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
    q_modality: torch.Tensor | None = None,
    k_modality: torch.Tensor | None = None,
    window_size: int | None = None,
    prefix_length: int = 0,
    is_causal: bool = True
):
    """
    Creates an attention mask supporting:
    - Causal masking
    - Sliding window
    - Bidirectional prefix
    - Multimodal "Islands" (Bidirectional for images/audio)
    """
    # dist: [batch, q_seq, k_seq]
    dist = q_pos.unsqueeze(-1) - k_pos.unsqueeze(-2)
    
    # 1. Base Mask (Causal or Bidirectional)
    if is_causal:
        mask = (dist >= 0)
    else:
        mask = torch.ones_like(dist, dtype=torch.bool)
        
    # 2. Multimodal Bidirectional Exception
    # Allow non-text tokens (Image=1, Audio=2) to see each other bidirectionally
    if is_causal and q_modality is not None and k_modality is not None:
        qm = q_modality.unsqueeze(-1)
        km = k_modality.unsqueeze(-2)
        # Same non-text modality can see each other
        # Note: We assume same-modality = same object/block for now.
        is_mm_island = (qm == km) & (qm != 0)
        mask = mask | is_mm_island

    # 3. Sliding Window Constraint
    if window_size is not None and window_size > 0:
        is_in_window = (dist < window_size)
        # KV tokens in the prefix are exempt from sliding window (always visible)
        is_kv_prefix = (k_pos.unsqueeze(-2) < prefix_length)
        mask &= (is_in_window | is_kv_prefix)
        
    # 4. Bidirectional Prefix Exception
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
        self.attn_logits_softcap = 50.0 # config.final_logit_softcapping
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
                modality_mask: torch.Tensor | None = None,
                **side_inputs,
    ) -> tuple[LayerCache | None, LayerCache | None, torch.Tensor]:

        batch_size, seq_len = x.shape[:2] 
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
            # Normalization ONLY for Q and K (indices 0 to n_heads + n_kv_heads)
            q, k, v = torch.split(qkv, [n_heads, n_kv_heads, n_kv_heads], dim=2)
            
            # Use pattern-based norm for q and k
            q = self.qkv_norm(q, n=n_heads)
            # Offset for k in the fused norm weight table
            k = self.qkv_norm(k, n=n_kv_heads, offset=n_heads)
            # V remains un-normalized
        
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
            curr_idx = kv_cache['end_index'][0].item()

            if needs_kv:
                # Producer: Insert current RoPE-applied tokens into the allocated heads
                kv_cache['k'][:, curr_idx:curr_idx + seq_len, :n_kv_heads, :] = k
                kv_cache['v'][:, curr_idx:curr_idx + seq_len, :n_kv_heads, :] = v
                kv_cache['positions'][:, curr_idx:curr_idx + seq_len] = segment_pos
                if modality_mask is not None:
                    kv_cache['modalities'][:, curr_idx:curr_idx + seq_len] = modality_mask
                kv_cache['end_index'] += seq_len
                actual_end_idx = curr_idx + seq_len
            else:
                actual_end_idx = curr_idx

            # Use full history from the cache
            k = kv_cache['k'][:, :actual_end_idx, :n_kv_heads, :]
            v = kv_cache['v'][:, :actual_end_idx, :n_kv_heads, :]
            k_pos = kv_cache['positions'][:, :actual_end_idx]
            k_modality = kv_cache['modalities'][:, :actual_end_idx]
        else:
            k_pos = segment_pos
            k_modality = modality_mask

        # Attention Dynamic Generation
        if attn_mask is None:
            window = self.sliding_window_size if self.attn_type == "sliding_attention" else None
            mask_bool = create_sliding_mask(
                q_pos=segment_pos,
                k_pos=k_pos,
                q_modality=modality_mask,
                k_modality=k_modality,
                window_size=window,
                prefix_length=prefix_length,
                is_causal=True
            )
            attn_mask = torch.where(mask_bool, 0.0, K_MASK).to(x.dtype)
            # Unsqueeze for broadcasting across heads: [batch, 1, q_seq, k_seq]
            attn_mask = attn_mask.unsqueeze(1)

        # GQA Attention using einsum for readability and efficiency
        heads_per_group = n_heads // n_kv_heads
        scale = self.head_dim**-0.5
        
        if heads_per_group > 1:
            # Reshape q to expose the GQA groups: [b, t, n_kv, g, h]
            q_gqa = q.view(batch_size, -1, n_kv_heads, heads_per_group, self.head_dim)
            # Logits: [b, t, n_kv, g, h] * [b, s, n_kv, h] -> [b, n_kv, g, t, s]
            logits = torch.einsum("btngh, bsnh -> bngts", q_gqa, k.to(x.dtype)) * scale
            # Flatten heads: [b, n_heads, t, s]
            logits = logits.reshape(batch_size, n_heads, -1, k.shape[1])
        else:
            # Standard Multi-Head or Multi-Query: [b, t, n, h] * [b, s, n, h] -> [b, n, t, s]
            logits = torch.einsum("btnh, bsnh -> bnts", q, k.to(x.dtype)) * scale
        
        if self.attn_logits_softcap is not None:
            logits = torch.tanh(logits / self.attn_logits_softcap) * self.attn_logits_softcap
            
        logits = logits + attn_mask
        probs = F.softmax(logits.float(), dim=-1).to(x.dtype)
        
        # Mix values using GQA-aware einsum
        if heads_per_group > 1:
            # Reshape probs back to groups: [b, n_kv, g, t, s]
            probs_gqa = probs.view(batch_size, n_kv_heads, heads_per_group, -1, k.shape[1])
            # Mix: [b, n_kv, g, t, s] * [b, s, n_kv, h] -> [b, t, n_kv, g, h]
            encoded = torch.einsum("bngts, bsnh -> btngh", probs_gqa, v.to(x.dtype))
            # Flatten heads: [b, t, n_heads, h]
            encoded = encoded.reshape(batch_size, -1, n_heads, self.head_dim)
        else:
            # Standard mix: [b, n, t, s] * [b, s, n, h] -> [b, t, n, h]
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
        'modalities': torch.zeros((batch_size, cache_size), dtype=torch.int32, device=device),
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
        nn.init.normal_(
                        self.input_embedding_table,
                        std=config.text.initializer_range
                    )

        # soft embeddings 
        if config.vision:
            self.vision_input_projection = Einsum(
                weight_shape=(
                    config.vision.hidden_size, # 2048 for MobileNetV5
                    self.hidden_size
                ),
                pattern="eh",
            )
        if config.audio:
            self.audio_input_projection = Einsum(
                weight_shape=(
                    config.audio.hidden_size, # 512
                    self.hidden_size
                ),
                pattern="eh",
            )
        
        if config.vision or config.audio:
            self.mm_soft_embedding_norm = MatryoshkaNorm(
                weight_shape=(self.hidden_size,),
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
        vision_tokens: torch.Tensor | None = None,
        audio_tokens: torch.Tensor | None = None,
        modality_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # ...
        # Avoid negative indices for placeholders
        safe_input_ids = torch.clamp(input_ids, min=0)
        x = F.embedding(safe_input_ids, self.input_embedding_table)
        
        # scale by sqrt(max_hidden_size))
        x = x * (self.hidden_size ** 0.5)

        if modality_mask is not None:
            # Handle Vision (Modality 1)
            if vision_tokens is not None and hasattr(self, 'vision_input_projection'):
                v_x = self.vision_input_projection("bte, eh -> bth", vision_tokens, h=self.hidden_size)
                v_x = self.mm_soft_embedding_norm(v_x, h=self.hidden_size)
                
                # Scatter into x at modality_mask == 1
                v_mask = (modality_mask == 1)
                # Ensure x and v_x have same hidden dim for this part
                x[v_mask] = v_x.view(-1, self.hidden_size).to(x.dtype)

            # Handle Audio (Modality 2)
            if audio_tokens is not None and hasattr(self, 'audio_input_projection'):
                a_x = self.audio_input_projection("bte, eh -> bth", audio_tokens, h=self.hidden_size)
                a_x = self.mm_soft_embedding_norm(a_x, h=self.hidden_size)
                a_mask = (modality_mask == 2)
                x[a_mask] = a_x.view(-1, self.hidden_size).to(x.dtype)

        x = x[:, :, :d_model]
        
        per_layer_inputs = None
        if hasattr(self, 'per_layer_embedding_table'):
            # [batch, seq, num_layers, latent_dim]
            # F.embedding doesn't support 3D weights, use direct indexing
            # input_ids: [b, s], table: [v, l, d] -> [b, s, l, d]
            per_layer_inputs = self.per_layer_embedding_table[safe_input_ids]
            
            # Mask out non-text positions if modality_mask is provided
            if modality_mask is not None:
                # modality_mask: [b, s], 0 is text
                text_mask = (modality_mask == 0)
                per_layer_inputs = per_layer_inputs * text_mask.unsqueeze(-1).unsqueeze(-1).to(per_layer_inputs.dtype)
            
        return x, per_layer_inputs

    def project_per_layer_input(self, latent: torch.Tensor, d_model: int) -> torch.Tensor:
        """Projects per-layer latent to the current residual width."""
        # latent: [batch, seq, latent_dim]
        x = self.per_layer_projection("bsl, lh -> bsh", latent, h=d_model)
        return self.per_layer_projection_norm(x, h=d_model)

    def encode_vision(self, x: torch.Tensor) -> torch.Tensor:
        """Projects siglip embeddings to the embedding space of the text encoder."""
        x = self.mm_soft_embedding_norm(x)
        x = self.mm_input_projection('...tm,md->...td', x)
        return x


class AdaptiveSurprisal(nn.Module):
    """
    Monitors prediction confidence (surprisal) and suggests a Matryoshka width.
    High surprisal (high entropy) -> Increase d_model
    Low surprisal (low entropy) -> Decrease d_model
    """
    def __init__(self, config: Gemma3nConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.text.hidden_size
        self.vocab_size = config.text.vocab_size
        
        # Mini-norm for the surprisal head
        self.norm = MatryoshkaNorm(
            weight_shape=(self.hidden_size,),
            pattern="h",
        )
        
        # The head itself is just an Einsum, but we'll tie its weight to the main table
        # We don't initialize its own weight here; it will be tied in the Transformer.
        self.head = Einsum(
            weight_shape=(self.vocab_size, self.hidden_size),
            pattern="vh",
        )

    def forward(self, 
                h: torch.Tensor, 
                d_model: int,
                max_width: int,
                min_width: int = 128,
                confidence_threshold: float = 0.8, # If top token < 80% prob, increase width
        ) -> int:
        """
        Args:
            h: [batch, seq, d_model] current hidden state
            d_model: current width
            max_width: maximum allowed width
            min_width: minimum allowed width
            confidence_threshold: prob threshold to trigger width increase
            
        Returns:
            suggested_d_model: power of two width
        """
        # 1. Use a very small slice for the surprisal check (e.g. min_width)
        # to keep the check itself fast.
        check_dim = min(d_model, min_width)
        
        # 2. Project to vocab space (only for the last token in sequence for efficiency)
        # h_last: [batch, 1, check_dim]
        h_last = h[:, -1:, :check_dim]
        h_last = self.norm(h_last, h=check_dim)
        
        # logits: [batch, 1, vocab]
        logits = self.head("bsh, vh -> bsv", h_last, h=check_dim)
        
        # 3. Use Top-K to filter noise and find max confidence
        # k=512 captures the viable candidates in a large vocab
        top_logits, _ = torch.topk(logits.float(), k=512, dim=-1)
        probs = F.softmax(top_logits, dim=-1)
        
        # max_prob: average confidence of the top prediction across the batch
        max_prob = probs[:, :, 0].mean()
        
        # 4. Logic: Move d_model up or down based on confidence
        if max_prob < confidence_threshold:
            # Low confidence -> Increase width to think harder
            return min(d_model * 2, max_width)
        elif max_prob > 0.98:
            # Extremely high confidence -> Decrease width to save compute
            return max(d_model // 2, min_width)
        
        return d_model
