import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence
from config import AudioConfig
from matryoshka_layers import Einsum, MatryoshkaNorm

class Gemma3nAudioRelativePositionEmbedding(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config

        self.num_heads = self.config.conf_num_attention_heads
        self.channels = self.config.hidden_size
        self.head_dim = self.channels // self.num_heads
        self.max_backward = max(0, self.config.conf_attention_context_left - 1)
        self.max_forward = self.config.conf_attention_context_right

        self.pos_proj = Einsum(
            weight_shape=(self.num_heads, self.head_dim, self.channels),
            pattern="nhd",
            bias_shape=None,
        )

        min_timescale = 1.0
        max_timescale = 1.0e4
        num_timescales = self.channels // 2
        log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / max(num_timescales - 1, 1)
        inv_timescales = min_timescale * torch.exp(torch.arange(num_timescales) * -log_timescale_increment)
        self.register_buffer(
            "inv_timescales",
            inv_timescales.float().unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

    def _get_timing_signal_1d_pos(self, position: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        position = position.float().unsqueeze(-1)
        scaled_time = position * self.inv_timescales.to(device=position.device, dtype=torch.float32)
        timing_signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        return timing_signal.type(dtype)

    def _relative_shift(
        self,
        term_bd_before_shift: torch.Tensor,
        batch_size: int,
        num_heads: int,
        num_query_blocks: int,
        query_block_size: int,
        key_context_size: int,
        max_span_plus_1: int,
    ) -> torch.Tensor:
        """Performs the relative shift.

        Args:
          term_bd_before_shift: Tensor of shape [B, N, U, W, F_span]. batch_size
            (B), num_heads (N), num_query_blocks (U), query_block_size (W),
            key_context_size (C = W+L+R), max_span_plus_1 (F_span = L+R+1).

        Returns:
          Tensor of shape [B, N, U, W, C].
        """
        # term_bd_before_shift shape: [B, N, U, W, F_span]
        # Target shape after shift:  [B, N, U, W, C]

        # Padding amount for the last dimension (F_span) to become (C + 1)
        # C = key_context_size
        # F_span = max_span_plus_1
        pad_amount_last_dim = (key_context_size + 1) - max_span_plus_1

        # PyTorch F.pad expects (pad_left, pad_right, pad_top, pad_bottom ...)
        # We only pad the last dimension on the right.
        padding_tuple = (0, pad_amount_last_dim)

        term_bd_padded = nn.functional.pad(term_bd_before_shift, padding_tuple)
        # Shape after pad: [B, N, U, W, C+1]

        # Reshape for slicing (emulating JAX's behavior)
        # [B, N, U, W * (C+1)]
        term_bd_reshaped = term_bd_padded.reshape(
            (
                batch_size,
                num_heads,
                num_query_blocks,
                query_block_size * (key_context_size + 1),
            )
        )

        # Slice to effective [B, N, U, W * C]
        term_bd_sliced = term_bd_reshaped[:, :, :, : query_block_size * key_context_size]

        # Reshape back to [B, N, U, W, C]
        term_bd_shifted = term_bd_sliced.reshape(
            (
                batch_size,
                num_heads,
                num_query_blocks,
                query_block_size,
                key_context_size,
            )
        )
        return term_bd_shifted

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        # queries: [B, U, W, N, H] (batch, num_query_blocks, query_block_size, num_heads, head_dim)
        # keys:    [B, U, C, N, H] (batch, num_query_blocks, key_context_size, num_heads, head_dim)
        # C = W + L + R (key_context_size)
        # F_span = L + R + 1 (max_span + 1)

        batch_size, num_query_blocks, query_block_size, num_heads, head_dim = queries.shape
        _, _, key_context_size, _, _ = keys.shape

        # Relative positions for sinusoidal embeddings: [L, L-1, ..., -R]
        # Length is L+R+1 = self.max_span + 1
        pos_indices = torch.arange(self.max_backward, -self.max_forward - 1, -1, device=queries.device).unsqueeze(
            0
        )  # Shape [1, F_span]

        max_span_plus_1 = pos_indices.shape[1]  # F_span

        sin_emb_timing_signal = self._get_timing_signal_1d_pos(
            pos_indices, dtype=queries.dtype
        )  # Shape [1, F_span, self.channels]

        # Project sinusoidal embeddings: [1, F_span, self.channels] -> [1, F_span, N, H]
        sin_emb = self.pos_proj(
            "b f d, n h d -> b f n h",
            sin_emb_timing_signal,
            n=self.num_heads,
            h=self.head_dim,
            d=self.channels,
        ).squeeze(0)  # Shape [F, N, H]

        # term_ac: Query-Key content interaction
        # queries: [B, U, W, N, H] -> permute to [B, N, U, W, H] for matmul
        # keys:    [B, U, C, N, H] -> permute to [B, N, U, H, C] for matmul
        queries_p = queries.permute(0, 3, 1, 2, 4)  # [B, N, U, W, H]
        keys_p_t = keys.permute(0, 3, 1, 4, 2)  # [B, N, U, H, C]
        term_ac = torch.matmul(queries_p, keys_p_t)  # [B, N, U, W, C]

        # term_bd: Query-Position interaction
        # Original einsum: term_bd_unshifed = torch.einsum('buwnh,fnh->bnuwf', queries, sin_emb)
        # queries shape: [B, U, W, N, H]
        # sin_emb shape: [F, N, H]
        # Target output shape: [B, N, U, W, F]

        # Permute queries to [B, N, U, W, H] for easier broadcasting with sin_emb
        q_permuted = queries.permute(0, 3, 1, 2, 4)

        # Permute sin_emb to [N, H, F] to prepare for matmul
        # sin_emb original is [F, N, H]
        s_permuted = sin_emb.permute(1, 2, 0)  # Shape: [N, H, F]

        # Reshape queries for matmul: [B, N, U*W, H]
        q_reshaped = q_permuted.reshape(batch_size, num_heads, num_query_blocks * query_block_size, head_dim)

        # Perform matmul: [B, N, U*W, H] @ [N, H, F]
        # s_permuted ([N, H, F]) will be broadcast to [B, N, H, F]
        # Result: [B, N, U*W, F]
        term_bd_unshifed_matmul = torch.matmul(q_reshaped, s_permuted)

        # Reshape to target [B, N, U, W, F]
        term_bd_unshifed = term_bd_unshifed_matmul.reshape(
            batch_size,
            num_heads,
            num_query_blocks,
            query_block_size,
            max_span_plus_1,
        )

        # Apply relative shift to term_bd_unshifed
        term_bd_shifted = self._relative_shift(
            term_bd_unshifed,
            batch_size,
            num_heads,
            num_query_blocks,
            query_block_size,
            key_context_size,
            max_span_plus_1,
        )  # Shape [B, N, U, W, C]

        return term_ac + term_bd_shifted


class Gemma3nAudioAttention(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config

        self.num_heads = self.config.conf_num_attention_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

        self.chunk_size = self.config.conf_attention_chunk_size
        self.max_future_horizon = self.config.conf_attention_context_right
        self.max_past_horizon = max(0, self.config.conf_attention_context_left - 1)
        self.attention_logits_soft_cap = self.config.conf_attention_logit_cap
        self.context_size = self.chunk_size + self.max_past_horizon + self.max_future_horizon

        self.relative_position_embedding = Gemma3nAudioRelativePositionEmbedding(config)
        self.per_dim_scale = nn.Parameter(torch.zeros((self.head_dim,)))

        self.q_proj = Einsum(
            weight_shape=(self.num_heads, self.head_dim, self.hidden_size),
            pattern="nhd",
            bias_shape=None,
        )
        self.k_proj = Einsum(
            weight_shape=(self.num_heads, self.head_dim, self.hidden_size),
            pattern="nhd",
            bias_shape=None,
        )
        self.v_proj = Einsum(
            weight_shape=(self.num_heads, self.head_dim, self.hidden_size),
            pattern="nhd",
            bias_shape=None,
        )

        q_scale = self.head_dim**-0.5
        r_softplus_0 = 1.0 / torch.nn.functional.softplus(torch.tensor(0.0))
        self.register_buffer("q_scale", (q_scale * r_softplus_0).clone().detach(), persistent=False)

        local_causal_valid_mask = self.create_local_causal_valid_mask()
        self.register_buffer("local_causal_valid_mask", local_causal_valid_mask, persistent=False)

        self.register_buffer(
            "softcap",
            torch.tensor(self.attention_logits_soft_cap).float(),
            persistent=False,
        )

    def create_local_causal_valid_mask(self):
        lower_causal_mask = torch.tril(
            torch.ones((self.context_size, self.chunk_size), dtype=torch.bool),
            diagonal=0,
        ).T
        upper_causal_mask = torch.tril(
            torch.ones((self.chunk_size, self.context_size), dtype=torch.bool),
            diagonal=self.max_past_horizon + self.max_future_horizon,
        )
        local_causal_valid_mask = torch.ones((self.chunk_size, self.context_size), dtype=torch.bool)
        local_causal_valid_mask = local_causal_valid_mask * lower_causal_mask * upper_causal_mask
        return local_causal_valid_mask

    def _pad_dim1(self, x: torch.Tensor, pad_left: int, pad_right: int) -> torch.Tensor:
        batch, _, *tail_shape = x.shape
        left = x.new_zeros((batch, pad_left, *tail_shape))
        right = x.new_zeros((batch, pad_right, *tail_shape))
        x = torch.cat([left, x, right], dim=1)
        return x

    def _convert_to_block(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Turns a sequence to non overlapping blocks.

        Args:
            hidden_states: a tensor of [batch, time, ...].

        Returns:
            A tensor of [batch, num_blocks, block_size, ...], with necessary
            paddings,
            where output[:, i, ...] are x[:, i*block_size:(i+1)*block_size, ...].
        """
        shape = hidden_states.shape
        b, t = shape[:2]
        num_blocks = (t + self.chunk_size - 1) // self.chunk_size

        if (padding_len := num_blocks * self.chunk_size - t) > 0:
            hidden_states = self._pad_dim1(hidden_states, 0, padding_len)

        permute_dims = (b, num_blocks, self.chunk_size) + shape[2:]
        hidden_states = hidden_states.reshape(permute_dims).contiguous()
        return hidden_states

    def _extract_block_context(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Extracts temporal context for every block.

        Args:
            hidden_states: a tensor of [batch, time, ...].

        Returns:
            A tensor of [batch, num_blocks, context_size, ...], with necessary
            paddings,
            where context_size = block_size + left_context + right_context,
            and output[:, i, ...] are x[:, start-left_context:end+right_context,
            ...],
            start = i * block_size, end = (i + 1) * block_size.
        """
        pad_left = self.max_past_horizon
        # The JAX equivalent padding for signal.frame with pad_mode='valid' is
        # (left_context, right_context + block_size - 1) on the time dimension.
        # PyTorch's _pad_dim1 applies padding symmetrically if only one value is given,
        # or (pad_dim_start, pad_dim_end) if two are given.
        # Our _pad_dim1(x, pad_left, pad_right) pads dim -2 (time for [B,T,N,H])
        # or dim 1 (time for [B,T]).
        # The current pad_right calculation matches the JAX effective padding.
        pad_right = self.max_future_horizon + self.chunk_size - 1
        hidden_states = self._pad_dim1(hidden_states, pad_left, pad_right)

        frame_len = self.context_size
        frame_step = self.chunk_size

        # Directly use unfold without the subframe_factor logic
        # x.unfold(dimension, size, step)
        # dimension=1 (time dimension, assuming x is [B, T_padded, ...])
        # size=frame_len (context_size)
        # step=frame_step (chunk_size)
        x_unfolded = hidden_states.unfold(dimension=1, size=frame_len, step=frame_step)

        # If x was [B, T_padded], x_unfolded is [B, num_blocks, frame_len]
        # If x was [B, T_padded, N, H], x_unfolded is [B, num_blocks, N, H, frame_len]
        # We want to match JAX's typical output for such operations which might be
        # [B, num_blocks, frame_len, N, H] if N, H are present.
        # The relative_position_embedding expects keys as [B, U, C, N, H].
        # If x_unfolded is [B, U, N, H, C(frame_len)], we need to move C.
        if hidden_states.ndim > 2 and x_unfolded.ndim > 3:  # Check if inner dimensions (like N, H) exist
            # Current shape after unfold for [B, T_pad, N, H] is [B, U, N, H, C]
            # Target shape for keys in RPE: [B, U, C, N, H]
            x_unfolded = torch.movedim(x_unfolded, source=-1, destination=2)

        return x_unfolded.contiguous()

    def forward(self, hidden_states: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        # sl.Dense uses jax.numpy.einsum("...a,abcd->...bcd") and jax.numpy.select()
        query_states = self.q_proj(
            "...d, nhd -> ...nh",
            hidden_states,
            n=self.num_heads,
            h=self.head_dim,
            d=self.hidden_size,
        ).contiguous()
        key_states = self.k_proj(
            "...d, nhd -> ...nh",
            hidden_states,
            n=self.num_heads,
            h=self.head_dim,
            d=self.hidden_size,
        ).contiguous()
        value_states = self.v_proj(
            "...d, nhd -> ...nh",
            hidden_states,
            n=self.num_heads,
            h=self.head_dim,
            d=self.hidden_size,
        ).contiguous()

        per_dim_scale_sp = torch.nn.functional.softplus(self.per_dim_scale)

        broadcast_shape = (1, 1, 1, self.head_dim)
        per_dim_scale_sp_broadcast = per_dim_scale_sp.view(broadcast_shape)
        query_states = query_states * self.q_scale * per_dim_scale_sp_broadcast

        batch_size, q_time = query_states.shape[:2]

        query_blocks = self._convert_to_block(query_states)
        key_blocks = self._extract_block_context(key_states)
        value_blocks = self._extract_block_context(value_states)
        num_query_blocks = query_blocks.shape[1]

        # 1. Create a mask indicating originally valid positions.
        original_valid_mask = ~mask  # True for valid, False for padded

        # 2. Extract blocks from this validity mask.
        extracted_valid_mask_blocks = self._extract_block_context(original_valid_mask)

        # If subframe_factor was used in _extract_block_context for a [B, T] input mask,
        # the shape might be [B, U, C/SF, SF]. Reshape to [B, U, C].
        # batch_size and num_query_blocks are known from query_blocks.
        # self.context_size is C.
        if (
            extracted_valid_mask_blocks.ndim == 4
            and extracted_valid_mask_blocks.shape[2] * extracted_valid_mask_blocks.shape[3] == self.context_size
        ):
            extracted_valid_mask_blocks = extracted_valid_mask_blocks.reshape(
                batch_size, num_query_blocks, self.context_size
            )
        # After potential reshape, ensure it's [B, U, C] if it was from a [B,T] mask.
        # This assertion might be too strict if _extract_block_context handles higher-rank inputs differently,
        # but for the mask case, this should hold.
        if extracted_valid_mask_blocks.shape != (
            batch_size,
            num_query_blocks,
            self.context_size,
        ):
            raise ValueError(
                "Shape of extracted_valid_mask_blocks"
                f" {extracted_valid_mask_blocks.shape} is not ({batch_size},"
                f" {num_query_blocks}, {self.context_size}) after potential reshape."
            )

        # 3. Expand dimensions for broadcasting with logits and causal mask.
        # Target shape for broadcasting with logits [B,N,U,W,C]
        # extracted_valid_mask_blocks to [B, 1, U, 1, C]
        condition_from_input_validity = extracted_valid_mask_blocks.unsqueeze(1).unsqueeze(-2)

        # self.local_causal_valid_mask is [W, C], True where allowed by local window.
        # Expand to [1, 1, 1, W, C]
        condition_from_causality = self.local_causal_valid_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # 4. Combine the two conditions.
        # final_condition will be True where a key is *both* originally valid *and* causally accessible.
        # Broadcasts to [B, 1, U, W, C]
        final_condition_for_where = torch.logical_and(
            condition_from_input_validity,
            condition_from_causality.to(condition_from_input_validity.device),  # Ensure same device
        )

        # Embed queries and keys
        logits = self.relative_position_embedding(query_blocks, key_blocks)

        # Apply attention logit softcap
        # Ensure softcap is on the same device as logits
        softcap_val = self.softcap.to(logits.device)
        logits = logits / softcap_val
        logits = torch.tanh(logits)
        logits = logits * softcap_val

        # Apply the combined mask.
        # final_condition_for_where will broadcast with logits [B,N,U,W,C]
        logits = torch.where(final_condition_for_where, logits, torch.finfo(logits.dtype).min)
        probabilities = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32).to(dtype=value_blocks.dtype)

        # context_vectors is adapted from jax.numpy.einsum("BNuwc,BucNH->BuwNH", ...)
        b_dim, n_dim, u_dim, w_dim, c_dim = probabilities.shape
        h_dim = value_blocks.shape[-1]
        prob_bun = probabilities.permute(0, 2, 1, 3, 4).reshape(-1, w_dim, c_dim)
        v_bun = value_blocks.permute(0, 1, 3, 2, 4).reshape(-1, c_dim, h_dim)
        result_bmm = torch.bmm(prob_bun, v_bun)
        context_vectors = result_bmm.reshape(b_dim, u_dim, n_dim, w_dim, h_dim).permute(0, 1, 3, 2, 4)
        context_vectors = context_vectors.reshape(
            (
                batch_size,
                num_query_blocks * self.chunk_size,
                self.num_heads,
                self.head_dim,
            )
        )
        context_vectors = context_vectors[:, :q_time]

        return context_vectors


class Gemma3nAudioCumulativeGroupNorm(nn.Module):
    """Applies Group Normalization cumulatively over the time dimension.

    This layer normalizes the input by calculating the mean and variance
    cumulatively over the time dimension (dim 1). The statistics are computed
    over all feature dimensions (specified by `feature_dims` and `num_channels`)
    for elements marked as valid by the optional `mask`.

    If a `mask` is provided (True for valid, False for invalid/padded),
    invalid time steps do not contribute to the statistics calculation, and
    their corresponding output values are zeroed out.

    Scale and bias, if enabled, are applied per-channel (last dimension).
    This behavior is similar to JAX's `GroupNormalization` with `num_groups=1`
    and `cumulative=True`.
    """

    def __init__(
        self,
        num_channels: int,  # Number of channels (size of the last dimension)
        feature_dims: Sequence[int],  # Sizes of non-channel feature dimensions, e.g., (H, W) for input [B,T,H,W,C]
        eps: float = 1e-3,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.feature_dims = tuple(feature_dims)
        self.eps = eps

        # Scale parameter depends only on the channel dimension
        self.weight = nn.Parameter(torch.ones(num_channels))

        # Axes for normalization: all dimensions except Batch (0) and Time (1).
        # For input [B, T, *feature_dims, C], these are dims from 2 onwards.
        self.reduction_axes = tuple(range(2, 2 + len(self.feature_dims) + 1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Applies cumulative group norm, optionally using a mask.

        Args:
          hidden_states: Input tensor, shape [B, T, *feature_dims, C].

        Returns:
          Normalized tensor with the same shape as x.
        """
        expected_input_suffix = self.feature_dims + (self.num_channels,)
        if hidden_states.shape[2:] != expected_input_suffix:
            raise ValueError(
                f"Input tensor shape suffix {hidden_states.shape[2:]} does not match expected"
                f" suffix (feature_dims + num_channels) {expected_input_suffix}"
            )

        input_dtype = hidden_states.dtype
        # Calculations are performed in float32 for numerical stability.
        calc_dtype = torch.float32
        x_calc = hidden_states.to(calc_dtype)

        # Prepare a broadcastable mask (`mask_calc`).
        # If no mask is provided, treat all elements as valid
        # (mask_calc is all ones).
        # Otherwise, expand the [B, T] mask to [B, T, 1, ..., 1] for broadcasting.
        mask_calc = torch.ones_like(x_calc, dtype=calc_dtype)

        # Cumulative Statistics Calculation
        # 1. Sum of values over reduction axes at each time step.
        sum_values_at_t = torch.sum(x_calc, dim=self.reduction_axes, keepdim=True)
        # 2. Cumulative sum of values over time.
        cum_sum_values = torch.cumsum(sum_values_at_t, dim=1)

        # 3. Count of valid elements in the normalization group at each time step.
        #    (A "group" here consists of all features at a given Batch, Time).
        elements_in_group_at_t = torch.sum(mask_calc, dim=self.reduction_axes, keepdim=True)
        # 4. Cumulative count of valid elements over time.
        cum_count_elements = torch.cumsum(elements_in_group_at_t, dim=1)
        # Avoid division by zero if all preceding elements were masked.
        safe_cum_count_elements = torch.clamp(cum_count_elements, min=1.0)

        # 5. Cumulative mean.
        cum_mean = cum_sum_values / safe_cum_count_elements

        # 6. Sum of squared differences from the cumulative mean.
        #    Only sum for valid elements: (x_calc - cum_mean)^2 * mask_calc.
        #    Using x_calc here for the difference, as cum_mean already accounts for masking.
        squared_diff_from_mean = (x_calc - cum_mean).pow(2)
        sum_sq_diff_at_t = torch.sum(squared_diff_from_mean, dim=self.reduction_axes, keepdim=True)

        # 7. Cumulative sum of squared differences over time.
        cum_sum_sq_diff = torch.cumsum(sum_sq_diff_at_t, dim=1)

        # 8. Cumulative variance.
        cum_variance = cum_sum_sq_diff / safe_cum_count_elements

        # Normalize the input using the calculated cumulative statistics:
        # (x - E[x]) / sqrt(Var[x] + eps)
        normalized_x = (x_calc - cum_mean) * torch.rsqrt(cum_variance + self.eps)

        # Apply affine transformation (scale and bias) if enabled.
        # Scale and bias are applied per-channel (last dimension).
        scale = self.weight.to(calc_dtype)
        # Reshape for broadcasting: [C] -> [1, ..., 1, C]
        scale_view_shape = [1] * (hidden_states.dim() - 1) + [self.num_channels]
        normalized_x = normalized_x * scale.view(scale_view_shape)

        # Zero out outputs for time steps that were originally masked (where mask_calc is 0).
        # This ensures padded/invalid positions in the input result in zero output.
        final_output = normalized_x * mask_calc

        return final_output.to(input_dtype)


class Gemma3nAudioSSCPConvBlock(nn.Module):
    """A single convolution block for the SubSampleConvProjection.

    This block consists of a 2D convolution, followed by CumulativeGroupNorm,
    and a ReLU activation. It handles manual padding for the convolution.
    """

    def __init__(
        self,
        config: AudioConfig,
        idx: int,
        input_freq_dim: int,  # Changed from input_spatial_dim
        manual_padding: tuple[int, int, int, int] = (0, 0, 0, 0),
    ):
        super().__init__()
        self.config = config
        self.manual_padding = manual_padding

        # in_channels is 1 for the first block, or C_out from previous block's conv
        in_channels = 1 if idx == 0 else self.config.sscp_conv_channel_size[idx - 1]
        out_channels = self.config.sscp_conv_channel_size[idx]
        kernel_h, kernel_w = self.config.sscp_conv_kernel_size[idx]
        stride_h, stride_w = self.config.sscp_conv_stride_size[idx]

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(
                kernel_h,
                kernel_w,
            ),  # Kernel (kH, kW) operates on (Time, Freq_dim)
            stride=(stride_h, stride_w),
            padding=(0, 0),  # Manual padding is used
            bias=False,
        )

        # Calculate output frequency dimension (f_out_conv) after this convolution.
        # input_freq_dim is the unpadded width (feature dimension).
        # self.manual_padding is (pad_F_left, pad_F_right, pad_T_top, pad_T_bottom)
        f_in_padded = input_freq_dim + self.manual_padding[0] + self.manual_padding[1]
        f_out_conv = (f_in_padded - kernel_w) // stride_w + 1

        self.norm = Gemma3nAudioCumulativeGroupNorm(
            num_channels=out_channels,  # Channels of the conv output
            feature_dims=(f_out_conv,),  # The frequency dimension size after conv
            eps=self.config.sscp_conv_group_norm_eps,
        )

        self.activation = nn.ReLU()

    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:
        # Input audio_encodings is [B, C_in, T_in, F_in] (e.g., C_in=1)
        # manual_padding is (pad_F_left, pad_F_right, pad_T_top, pad_T_bottom)
        # F.pad applies to last two dims: F_in then T_in
        audio_encodings_padded = F.pad(audio_encodings, self.manual_padding, mode="constant", value=0.0).to(
            self.conv.weight.dtype
        )
        # Expected padded shape for F_in, k_w=3, pad_F=(1,1) -> F_padded = F_in+2
        # Expected padded shape for T_in, k_h=3, pad_T=(0,2) -> T_padded = T_in+2
        audio_encodings_conv = self.conv(audio_encodings_padded)
        # Expected conv output shape: [B, C_out, T_out, F_out]
        # Input to norm is [B, T_out, F_out, C_out]
        x_for_norm = audio_encodings_conv.permute(0, 2, 3, 1).contiguous()
        x_normed = self.norm(x_for_norm)
        # Output of norm is [B, T_out, F_out, C_out], permute back to [B, C_out, T_out, F_out]
        audio_encodings_normed = x_normed.permute(0, 3, 1, 2).contiguous()
        return self.activation(audio_encodings_normed)


class Gemma3nAudioSubSampleConvProjection(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config

        current_f_for_block_input = self.config.input_feat_size  # Start with original feature dim
        calculated_block_padding = []
        calculated_f_out_dims = []  # Tracking frequency dimension output sizes

        for i in range(2):  # Assuming 2 conv layers as per sscp_conv_... arrays
            kernel_h, kernel_w = self.config.sscp_conv_kernel_size[i]
            stride_h, stride_w = self.config.sscp_conv_stride_size[i]

            # Padding for Time (Height for Conv2d) - REVERSE_CAUSAL like
            # JAX 'reverse_causal' padding is (0, kernel_size - 1)
            pad_t_top = 0
            pad_t_bottom = kernel_h - 1

            # Frequency Padding (Width for Conv2d)
            # Based on JAX effective padding (1,1) for F_in=10, K_w=3, S_w=2
            # and the successful test configuration.
            # If kernel/stride/input_freq for frequency changes, this might need re-evaluation
            # to match generic JAX 'SAME' behavior if it differs.
            pad_f_left = 1
            pad_f_right = 1

            manual_padding_tuple = (
                pad_f_left,
                pad_f_right,
                pad_t_top,
                pad_t_bottom,
            )
            calculated_block_padding.append(manual_padding_tuple)

            # Calculate output frequency dimension after this convolution
            # This uses the actual padding applied and kernel/stride.
            f_in_padded = current_f_for_block_input + pad_f_left + pad_f_right
            f_out_after_conv = (f_in_padded - kernel_w) // stride_w + 1  # Assuming dilation_w = 1
            calculated_f_out_dims.append(f_out_after_conv)
            current_f_for_block_input = f_out_after_conv

        self.conv_0 = Gemma3nAudioSSCPConvBlock(
            idx=0,
            input_freq_dim=self.config.input_feat_size,  # Pass original feature dim
            config=self.config,
            manual_padding=calculated_block_padding[0],
        )
        self.conv_1 = Gemma3nAudioSSCPConvBlock(
            idx=1,
            input_freq_dim=calculated_f_out_dims[0],  # Output freq dim from conv_0
            config=self.config,
            manual_padding=calculated_block_padding[1],
        )
        final_c_out = self.config.sscp_conv_channel_size[-1]
        final_f_out = calculated_f_out_dims[-1]  # Final frequency dimension
        self.input_proj_in_features = final_c_out * final_f_out
        self.input_proj_linear = Einsum(
            weight_shape=(self.config.hidden_size, self.input_proj_in_features),
            pattern="hi",
            bias_shape=None,
        )

    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:
        # audio_encodings is [B, T, F_in]
        # Reshape to [B, 1, T, F_in] (Batch, Channels=1, Height=Time, Width=F_in)
        audio_encodings_reshaped = audio_encodings.unsqueeze(1)
        x = self.conv_0(audio_encodings_reshaped)
        x = self.conv_1(x)
        # x from conv_1 is [B, C_out_1, T_out_1, F_out_1]
        b, c_out, t_out, f_out = x.shape
        # Permute to [B, T_out_1, F_out_1, C_out_1] then flatten F_out_1 and C_out_1
        x_permuted = x.permute(0, 2, 3, 1).contiguous()
        output_flattened = x_permuted.view(b, t_out, f_out * c_out)
        output = self.input_proj_linear(
            "...i, hi -> ...h",
            output_flattened,
            h=self.config.hidden_size,
            i=self.input_proj_in_features,
        )
        return output


class Gemma3nAudioConformerAttention(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        self.post_in_features = self.config.hidden_size
        self.register_buffer("gradient_clipping", torch.tensor(self.config.gradient_clipping), persistent=False)
        self.pre_attn_norm = MatryoshkaNorm(
            weight_shape=(self.config.hidden_size,),
            pattern="h",
            eps=self.config.rms_norm_eps,
        )
        self.attn = Gemma3nAudioAttention(self.config)
        self.post = Einsum(
            weight_shape=(self.config.hidden_size, self.post_in_features),
            pattern="hi",
            bias_shape=None,
        )
        self.post_norm = MatryoshkaNorm(
            weight_shape=(self.config.hidden_size,),
            pattern="h",
            eps=self.config.rms_norm_eps,
        )

    def forward(self, audio_encodings: torch.Tensor, audio_mel_mask: torch.BoolTensor) -> torch.Tensor:
        audio_encodings_input_to_attn = audio_encodings
        audio_encodings = torch.clamp(audio_encodings, -self.gradient_clipping, self.gradient_clipping)
        audio_encodings_norm = self.pre_attn_norm(audio_encodings, h=self.config.hidden_size)
        # Output of self.attn is [B, T, NumHeads, HeadDim]
        audio_encodings_attn_out = self.attn(audio_encodings_norm, audio_mel_mask)

        # Reshape from [B, T, NumHeads, HeadDim] to [B, T, NumHeads * HeadDim]
        # NumHeads * HeadDim = hidden_size
        b, t, num_heads, head_dim = audio_encodings_attn_out.shape
        audio_encodings_reshaped = audio_encodings_attn_out.reshape(b, t, num_heads * head_dim)

        audio_encodings = self.post(
            "...i, hi -> ...h",
            audio_encodings_reshaped,
            h=self.config.hidden_size,
            i=self.post_in_features,
        )
        audio_encodings = torch.clamp(audio_encodings, -self.gradient_clipping, self.gradient_clipping)
        return audio_encodings_input_to_attn + self.post_norm(
            audio_encodings, h=self.config.hidden_size
        )


class Gemma3nAudioConformerFeedForward(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config

        self.register_buffer("gradient_clipping", torch.tensor(self.config.gradient_clipping), persistent=False)

        self.pre_layer_norm = MatryoshkaNorm(
            weight_shape=(self.config.hidden_size,),
            pattern="h",
            eps=self.config.rms_norm_eps,
        )
        self.ffw_layer_1 = Einsum(
            weight_shape=(self.config.hidden_size * 4, self.config.hidden_size),
            pattern="ih",
            bias_shape=None,
        )
        self.ffw_layer_2 = Einsum(
            weight_shape=(self.config.hidden_size, self.config.hidden_size * 4),
            pattern="hi",
            bias_shape=None,
        )
        self.post_layer_norm = MatryoshkaNorm(
            weight_shape=(self.config.hidden_size,),
            pattern="h",
            eps=self.config.rms_norm_eps,
        )
        self.post_layer_scale = self.config.conf_residual_weight

    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:
        residual = audio_encodings
        audio_encodings = torch.clamp(audio_encodings, -self.gradient_clipping, self.gradient_clipping)
        audio_encodings = self.pre_layer_norm(audio_encodings, h=self.config.hidden_size)
        audio_encodings: torch.Tensor = self.ffw_layer_1(
            "...h, ih -> ...i",
            audio_encodings,
            i=self.config.hidden_size * 4,
            h=self.config.hidden_size,
        )
        audio_encodings = nn.functional.silu(audio_encodings)
        audio_encodings: torch.Tensor = self.ffw_layer_2(
            "...i, hi -> ...h",
            audio_encodings,
            h=self.config.hidden_size,
            i=self.config.hidden_size * 4,
        )
        audio_encodings = torch.clamp(audio_encodings, -self.gradient_clipping, self.gradient_clipping)
        audio_encodings = self.post_layer_norm(audio_encodings, h=self.config.hidden_size)
        return residual + (audio_encodings * self.post_layer_scale)


class Gemma3nAudioConformerLightConv1d(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config

        self.pre_layer_norm = MatryoshkaNorm(
                                             weight_shape=(
                                                 self.config.hidden_size
                                             ),
                                             pattern="h",
                                             eps=self.config.rms_norm_eps
                                         )
        self.linear_start = Einsum(
                                    weight_shape=(
                                        self.config.hidden_size * 2,
                                        self.config.hidden_size
                                  ),
                                    pattern="oh"
                                  )
        self.depthwise_conv1d = nn.Conv1d(
            in_channels=self.config.hidden_size,
            out_channels=self.config.hidden_size,
            kernel_size=self.config.conf_conv_kernel_size,
            stride=1,
            padding=0,  # Manual causal padding
            groups=self.config.hidden_size,  # Depthwise
            bias=False,
        )
        self.register_buffer("gradient_clipping", torch.tensor(self.config.gradient_clipping), persistent=False)
        self.conv_norm = MatryoshkaNorm(
                                        weight_shape=(
                                            self.config.hidden_size
                                        ),
                                        pattern="h",
                                        eps=self.config.rms_norm_eps,
                                )
        self.linear_end = Einsum(
                                 weight_shape=(
                                     self.config.hidden_size,
                                     self.config.hidden_size
                                 ),
                                 pattern="oh",
                            )

        self.causal_padding = self.config.conf_conv_kernel_size - 1

    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:
        audio_encodings_residual = audio_encodings  # Save for residual connection

        audio_encodings = self.pre_layer_norm(audio_encodings, h=self.config.hidden_size)
        audio_encodings = self.linear_start(
            "...h, oh -> ...o",
            audio_encodings,
            h=self.config.hidden_size,
            o=self.config.hidden_size * 2,
        )
        audio_encodings = torch.nn.functional.glu(audio_encodings, dim=-1)
        # Permute for Conv1d: [B, T, D] -> [B, D, T]
        audio_encodings_permuted = audio_encodings.permute(0, 2, 1)
        # Apply manual causal padding
        audio_encodings_permuted_padded = F.pad(audio_encodings_permuted, (self.causal_padding, 0))
        audio_encodings = self.depthwise_conv1d(audio_encodings_permuted_padded)
        # Permute back: [B, D, T_out] -> [B, T_out, D]
        audio_encodings = audio_encodings.permute(0, 2, 1)
        audio_encodings = torch.clamp(audio_encodings, -self.gradient_clipping, self.gradient_clipping)
        audio_encodings = self.conv_norm(audio_encodings, h=self.config.hidden_size)
        audio_encodings = nn.functional.silu(audio_encodings)
        audio_encodings = self.linear_end(
            "...h, oh -> ...o",
            audio_encodings,
            h=self.config.hidden_size,
            o=self.config.hidden_size,
        )
        output = audio_encodings + audio_encodings_residual
        return output

