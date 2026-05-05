import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Gemma3nConfig, TextConfig

class Linear(nn.Linear):
    """ Cast to match input dtype in forward. """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(dtype=x.dtype))

class Einsum(nn.Module):
    def __init__(self,
                 weight_shape: tuple[int, ...],
                 pattern: str | None = None,
                 bias_shape: tuple[int, ...] | None = None,
                 scale: float | None = None,
                 init: str = "xavier",
                 bias_init: str = "zeros",
            ):
        super().__init__()
        self.w = nn.Parameter(torch.empty(*weight_shape))
        self.pattern = pattern
        self.bias = nn.Parameter(torch.zeros(*bias_shape)) if bias_shape else None
        self.scale = scale

        if init == "xavier":
            nn.init.xavier_uniform_(self.w)
        elif init == "normal":
            nn.init.normal_(self.w, std=1e-6)
        elif init == "zeros":
            nn.init.zeros_(self.w)
            
        if self.bias is not None:
            if bias_init == "zeros":
                nn.init.zeros_(self.bias)
            elif bias_init == "ones":
                nn.init.ones_(self.bias)

    def forward(self, eqn: str, x: torch.Tensor, **slices) -> torch.Tensor:
        """
        Args:
            eqn: The einsum equation (e.g., 'btd, ndh -> btnh')
            x: Input tensor
            **slices: Dim names and slice sizes (e.g., n=8, d=512)
        """
        # dynamic slicer based on pattern
        slicer = []
        if self.pattern:
            for char in self.pattern:
                slicer.append(slice(0, slices.get(char)))
        else:
            slicer.append(slice(None))
            
        # slice and cast weight
        w = self.w[tuple(slicer)].to(x.dtype)
        y = torch.einsum(eqn, x, w)
        
        # apply optional bias and scale
        if self.bias is not None:
            y = y + self.bias[tuple(slicer)].to(x.dtype)
        if self.scale is not None:
            y = y * self.scale
            
        return y


def precompute_rope_cache(max_seq_len: int, d_head: int, base: float):
    """
    Pre-computes cos and sin tables for RoPE.
    Returns:
        cos, sin: Tensors of shape [max_seq_len, d_head]
    """
    # 1. Calculate the frequencies (theta)
    indices = torch.arange(0, d_head, 2).float()
    inv_freq = 1.0 / (base ** (indices / d_head))
    
    # 2. Calculate the positions (t)
    t = torch.arange(max_seq_len).float()
    
    # 3. Outer product [Seq, d_head // 2]
    freqs = torch.einsum('i, j -> ij', t, inv_freq)
    
    # 4. Create [Seq, d_head] by repeating (standard RoPE layout)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Applies precomputed RoPE to the input tensor.
    x: [batch, seq, heads, head_dim]
    cos, sin: [batch, seq, 1, head_dim] (already indexed/broadcasted)
    """
    return (x * cos) + (rotate_half(x) * sin)

class PerLayerEmbedding(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        # PLE bank stores token-specific latents in the bottleneck dimension
        self.ple_bank = nn.ModuleDict({
            str(i): nn.Embedding(
                config.vocab_size,
                config.head_dim,  # Latent dimension (256)
            ) for i in range(config.num_hidden_layers)
        })

        # down [Hidden, Latent] -> pattern "hl"
        self.down_proj = Einsum(
            weight_shape=(config.hidden_size, config.head_dim),
            pattern="hl",
        )

        # up [Latent, Hidden] -> pattern "lh"
        self.up_proj = Einsum(
            weight_shape=(config.head_dim, config.hidden_size),
            pattern="lh",
        )

    def forward(self,
                x: torch.Tensor,
                latent: torch.Tensor,
                block_idx: int,
                d_model: int,
            ):
        """
        x: current residual stream (batch, seq, d_model)
        latent: per-layer latent from Embedder (batch, seq, latent_dim)
        """
        # residual stream to latent space: [b, s, d_model] -> [b, s, l]
        # h=d_model slices the hidden dimension
        latent_state = self.down_proj('bsi, il -> bsl', x, h=d_model)

        # Match latent dimension if necessary (e.g. if config.head_dim > latent_state dimension)
        # latent_state is based on config.head_dim. latent is from Embedder (config.text.hidden_size_per_layer_input)
        if latent.shape[-1] != latent_state.shape[-1]:
            latent = latent[:, :, :latent_state.shape[-1]]

        # token-specific gating
        gated_latent = latent_state * torch.sigmoid(latent)

        # back up to residual width: [b, s, l] -> [b, s, d_model]
        return self.up_proj('bsl, li -> bsi', gated_latent, h=d_model)        
class MatryoshkaLinear(nn.Module):
    def __init__(self,
                 config: Gemma3nConfig,
                 bias: bool = True,
             ):
        super().__init__()
        self.weight = nn.Parameter(
                                   torch.randn(
                                               config.intermediate_size,
                                               config.hidden_size
                                           )
                               )
        self.bias = nn.Parameter(
                                 torch.zeros(
                                             config.intermediate_size
                                         )
                             ) if bias else None

    def forward(self,
                x: torch.Tensor,
                d_in: int = None,
                d_out: int = None
            ) -> torch.Tensor:
        """ x: (batch, seq, d_in_current) """
        d_i = d_in if d_in is not None else x.shape[-1]
        d_o = d_out if d_out is not None else self.weight.shape[0]

        w = self.weight[:d_o, :d_i]

        out = torch.einsum('bsi, oi -> bso', x, w)

        if self.bias is not None:
            out = out + self.bias[:d_o]

        return out

class MatryoshkaFFN(nn.Module):
    def __init__(self, config: Gemma3nConfig, block_idx: int):
        super().__init__()
        self.activation_sparsity = config.activation_sparsity_pattern[block_idx]
        max_intermediate = config.intermediate_size[block_idx]
        
        # Fuse gate and up into one projection: [Hidden, 2 * Intermediate]
        self.gate_up_proj = Einsum(
            weight_shape=(
                config.hidden_size,
                2 * max_intermediate,
            ),
            pattern="hi", # h=hidden, i=intermediate
        )
        
        self.down_proj = Einsum(
            weight_shape=(
                max_intermediate,
                config.hidden_size,
            ),
            pattern="ih", # i=intermediate, h=hidden
        )

    def forward(self,
                x: torch.Tensor,
                d_model: int,
                d_ff: int,
            ) -> torch.Tensor:
        # fused gate and up projection: [b, s, d_model] -> [b, s, 2 * d_ff]
        gate_up = self.gate_up_proj('bsh, hi -> bsi', x, h=d_model, i=2*d_ff)
        
        # fused output into gate and up branches
        gate, up = torch.split(gate_up, [d_ff, d_ff], dim=-1)

        # activation and sparsity
        if self.activation_sparsity > 0.0:
            gate = self._gaussian_topk(gate)
        
        # GLU logic as per Jax impl.
        junction = gate * F.gelu(up)

        # down projection: [b, s, d_ff] -> [b, s, d_model]
        return self.down_proj('bsi, ih -> bsh', junction, i=d_ff, h=d_model)

    def _gaussian_topk(self, inputs: torch.Tensor) -> torch.Tensor:
        target_sparsity_tensor = torch.tensor(self.activation_sparsity, dtype=torch.float32, device=inputs.device)
        normal_dist = torch.distributions.normal.Normal(0, 1)
        std_multiplier: torch.Tensor = normal_dist.icdf(target_sparsity_tensor)
        std_multiplier = std_multiplier.type(inputs.dtype)
        inputs_mean = torch.mean(inputs, dim=-1, keepdim=True)
        inputs_std = torch.std(inputs, dim=-1, keepdim=True, unbiased=False)
        cutoff_x = inputs_mean + inputs_std * std_multiplier
        return F.relu(inputs - cutoff_x)

class MatryoshkaNorm(nn.Module):
    def __init__(self,
                 weight_shape: tuple[int, ...],
                 pattern: str | None = None,
                 eps: float = 1e-6,
                 with_scale: bool = True,
             ):
        super().__init__()
        self.eps = eps
        self.pattern = pattern
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(weight_shape))

    def forward(self, x: torch.Tensor, **slices):
        # standard RMSNorm on the last dimension
        # x shape: [..., d_curr]
        # calculate in float32 for numerical stability
        x_float = x.float()
        mean_squared = x_float.pow(2).mean(-1, keepdim=True)
        normed = x_float * torch.rsqrt(mean_squared + self.eps)
        
        # learnable weight if enabled
        if self.with_scale:
            slicer = []
            if self.pattern:
                for char in self.pattern:
                    slicer.append(slice(0, slices.get(char)))
            else:
                # simple slicing 
                slicer.append(slice(0, x.shape[-1]))
            
            # slice and cast
            w = self.weight[tuple(slicer)].float()
            
            # reshape scale to (1, ..., 1, D) to match input rank (avoids implicit rank-promotion)
            # follows jax implementation standard
            w = w.view(*([1] * (x.ndim - w.ndim)), *w.shape)
            
            normed = normed * w
            
        return normed.to(x.dtype)
            

class AlternatingUpdates(nn.Module):
    """Alternating Updates (AltUp)
    Optimized implementation using fused Einsum transitions and Matryoshka slicing.
    Ref: https://arxiv.org/abs/2301.11943
    """
    def __init__(self, config: Gemma3nConfig):
        super().__init__()
        self.config = config
        self.num_inputs = config.altup_num_inputs
        
        # 1. Matryoshka-aware Router
        self.modality_router = Einsum(
            weight_shape=(config.hidden_size, self.num_inputs),
            pattern="hi", # h=hidden (to be sliced as d_model), i=altup_inputs
        )
        self.router_norm = MatryoshkaNorm(
            weight_shape=(config.hidden_size,),
            pattern="h",
        )
        self.register_buffer("router_input_scale", torch.tensor(config.hidden_size**-1.0), persistent=False)

        # transition & correction coefs
        # prediction_proj: [num_inputs, num_inputs, num_inputs]
        self.prediction_proj = Einsum(
            weight_shape=(self.num_inputs, self.num_inputs, self.num_inputs),
            pattern="ijk",
        )
        self.correction_proj = Einsum(
            weight_shape=(self.num_inputs, self.num_inputs),
            pattern="ij",
        )
        
        # output scaling
        self.correct_output_scale = nn.Parameter(torch.zeros(config.hidden_size))

    def compute_router_modalities(self, x_active: torch.Tensor, d_model: int) -> torch.Tensor:
        """Maps residual stream to a small modality space."""
        # x_active: [batch, seq, d_model]
        normed = self.router_norm(x_active, h=d_model) * self.router_input_scale
        # (batch, seq, d_model) @ (d_model, num_inputs) -> (batch, seq, num_inputs)
        routed = self.modality_router("bsh, hi -> bsi", normed, h=d_model)
        return torch.tanh(routed.float()).to(x_active.dtype)

    def predict(self, x: torch.Tensor, d_model: int) -> torch.Tensor:
        """
        Predicts the next state for all AltUp inputs.
        x: [num_inputs, batch, seq, d_model]
        """
        # based on the currently active input
        modalities = self.compute_router_modalities(x[self.config.altup_active_idx], d_model)
        
        # compute transition coefficients: [batch, seq, num_inputs, num_inputs]
        # (batch, seq, i) @ (i, j, k) -> (batch, seq, j, k)
        all_coefs = self.prediction_proj("bsi, ijk -> bsjk", modalities)
        
        if self.config.altup_coef_clip is not None:
             all_coefs = all_coefs.clamp(-self.config.altup_coef_clip, self.config.altup_coef_clip)

        # apply transition
        # (batch, seq, j, k) @ (k, batch, seq, d_model) -> (batch, seq, j, d_model)
        predictions = torch.einsum("bsjk, kbsd -> jbsd", all_coefs, x)
        
        return predictions + x

    def correct(self, predictions: torch.Tensor, activated: torch.Tensor, d_model: int) -> torch.Tensor:
        """
        Corrects predictions relative to the actual layer output.
        predictions: [num_inputs, batch, seq, d_model]
        activated: [batch, seq, d_model]
        """
        modalities = self.compute_router_modalities(activated, d_model)
        
        # innovation (error signal): [batch, seq, d_model]
        innovation = activated - predictions[self.config.altup_active_idx]
        
        # correction coefficients: [batch, seq, num_inputs]
        all_coefs = self.correction_proj("bsi, ij -> bsj", modalities) + 1.0
        
        if self.config.altup_coef_clip is not None:
            all_coefs = all_coefs.clamp(1.0 - self.config.altup_coef_clip, 1.0 + self.config.altup_coef_clip)

        # apply correction: [batch, seq, d_model] * [batch, seq, num_inputs] -> [num_inputs, batch, seq, d_model]
        corrected = torch.einsum("bsd, bsj -> jbsd", innovation, all_coefs)
        return corrected + predictions

    def forward(self, x: torch.Tensor, d_model: int) -> torch.Tensor:
        """Scales the provided 3D tensor [batch, seq, d_model]."""
        scale = self.correct_output_scale[:d_model]
        return (x.to(scale.dtype) * scale).to(x.dtype)

    def scale_corrected_output(self, x: torch.Tensor, d_model: int) -> torch.Tensor:
        return self.forward(x, d_model)

class LaurelBlock(nn.Module):
    """Learned Augmented Residual Layer"""

    def __init__(self, config: Gemma3nConfig):
        super().__init__()
        self.config = config

        self.linear_left = Einsum(
                                  weight_shape=(
                                      self.config.hidden_size,
                                      self.config.laurel_rank,),
                              pattern="hr",
                          )
        self.linear_right = Einsum(
                                   weight_shape=(
                                       self.config.laurel_rank,
                                       self.config.hidden_size,
                                   ),
                                   pattern="rh",
                               )
        
        self.post_laurel_norm = MatryoshkaNorm(
                                               weight_shape=(self.config.hidden_size,),
                                               pattern="h",
                                           )

    def forward(self, x: torch.Tensor, d_model: int) -> torch.Tensor:
        # x: [num_inputs, batch, seq, d_model]
        laurel_x: torch.Tensor = self.linear_left(
                                                  'jbsh, hr -> jbsr',
                                                  x,
                                                  h=d_model
                                              )
        laurel_x: torch.Tensor = self.linear_right(
                                                   'jbsr, rh -> jbsh',
                                                   laurel_x,
                                                   h=d_model,
                                               )
        normed_laurel_x = self.post_laurel_norm(laurel_x, h=d_model)  # x.shape[-1]
        return x + normed_laurel_x

