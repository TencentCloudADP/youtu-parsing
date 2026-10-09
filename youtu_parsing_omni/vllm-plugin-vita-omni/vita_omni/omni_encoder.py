"""VITA OmniEncoder: shared vision+audio Transformer encoder.

Architecture:
- VisionEmbeddings: Linear patch projection
- AudioEmbeddings: 3×Conv2d stride-2 (8× time downsample) + linear
- SharedTransformerEncoder: Qwen3-style layers (qk-norm + GQA + SwiGLU)
  with bidirectional packed varlen FlashAttention2
- VisionPatchMerger: 2×2 spatial merge + 2-layer MLP
- AudioPatchMerger: 2× temporal merge + 2-layer MLP

Weight keys (HF checkpoint):
  model.omni_model.vision_embeddings.patch_embedding.{weight,bias}
  model.omni_model.audio_embeddings.conv2d{1,2,3}.{weight,bias}
  model.omni_model.audio_embeddings.linear_proj.weight
  model.omni_model.encoder.layers.{i}.{input_layernorm,post_attention_layernorm}.weight
  model.omni_model.encoder.layers.{i}.self_attn.{q_proj,k_proj,v_proj,o_proj}.weight
  model.omni_model.encoder.layers.{i}.self_attn.{q_norm,k_norm}.weight
  model.omni_model.encoder.layers.{i}.mlp.{gate_proj,up_proj,down_proj}.weight
  model.omni_model.vision_merger.{norm,linear_fc1,linear_fc2}.weight
  model.omni_model.audio_merger.{norm,linear_fc1,linear_fc2}.weight
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn.layers.rotary import apply_rotary_emb as flash_apply_rotary_emb
except ImportError:
    flash_attn_varlen_func = None
    flash_apply_rotary_emb = None

from .omni_rope_4d import (
    M_AUDIO,
    M_IMAGE,
    M_VIDEO_AUDIO,
    M_VIDEO_FRAME,
    build_audio_thw_pos_ids,
    build_frame_thw_pos_ids,
    chunked_mthw_rotary,
    split_4d_dims,
)


# ===========================================================================
# RoPE utilities (mirrors YoutuVITAOmniRotaryEmbedding + vision/audio pos emb)
# ===========================================================================


def _rotary_pos_emb_table(dim: int, max_len: int, theta: float = 10000.0,
                          device: torch.device = None) -> torch.Tensor:
    """Build the rotary frequency table: outer(arange, inv_freq).

    Returns: [max_len, dim//2] — raw freq angles (not cos/sin yet).
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    seq = torch.arange(max_len, device=device, dtype=inv_freq.dtype)
    return torch.outer(seq, inv_freq)  # [max_len, dim//2]


def _get_2d_vision_rotary(grid_thw: torch.Tensor, rotary_dim: int,
                          spatial_merge_size: int,
                          theta: float = 10000.0) -> torch.Tensor:
    """Compute 2D vision RoPE (matches HF YoutuVITAOmniEncoder.vision_rot_pos_emb).

    Returns: [total_patches, rotary_dim] — freq angles for flash_attn apply_rotary_emb.
    The output is [N, rotary_dim] = [N, dim//2 * 2] where each half corresponds
    to h-positions and w-positions respectively.
    """
    device = grid_thw.device
    merge = spatial_merge_size

    pos_ids_list = []
    for t, h, w in grid_thw:
        t, h, w = int(t), int(h), int(w)
        # Height positions reshaped by merge pattern
        hpos_ids = torch.arange(h, device=device).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(h // merge, merge, w // merge, merge)
        hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

        # Width positions reshaped by merge pattern
        wpos_ids = torch.arange(w, device=device).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(h // merge, merge, w // merge, merge)
        wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()

        pos_ids_list.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

    pos_ids = torch.cat(pos_ids_list, dim=0)  # [total_patches, 2]
    max_grid_size = int(grid_thw[:, 1:].max().item())

    # Build frequency table
    freq_table = _rotary_pos_emb_table(rotary_dim, max_grid_size, theta, device)
    # freq_table: [max_grid_size, rotary_dim//2]

    # Index: pos_ids [N, 2] → freq_table[pos_ids] → [N, 2, rotary_dim//2] → flatten → [N, rotary_dim]
    rotary_pos_emb = freq_table[pos_ids].flatten(1)  # [N, rotary_dim]
    return rotary_pos_emb


def _get_1d_audio_rotary(lengths: torch.Tensor, rotary_dim: int,
                         theta: float = 10000.0) -> torch.Tensor:
    """Compute 1D audio RoPE (matches HF YoutuVITAOmniEncoder.audio_rot_pos_emb).

    HF logic:
      1. freq_table = rotary_pos_emb(max_len) -> [max_len, dim//2] (dim=64 -> [max_len, 32])
      2. For each audio, take first `length` rows, cat together
      3. Duplicate: cat([rotary, rotary], dim=-1) -> [total, 64]

    Returns: [total_tokens, rotary_dim] — freq angles (same shape as vision RoPE).
    """
    device = lengths.device
    max_len = int(lengths.max().item())

    # Build frequency table: [max_len, rotary_dim//2]
    freq_table = _rotary_pos_emb_table(rotary_dim, max_len, theta, device)

    # Build position ids for packed sequence
    out = []
    for length in lengths.tolist():
        out.append(freq_table[:int(length)])
    rotary_pos_emb = torch.cat(out, dim=0)  # [total_len, rotary_dim//2]

    # Duplicate along feature axis (same as HF)
    rotary_pos_emb = torch.cat([rotary_pos_emb, rotary_pos_emb], dim=-1)
    # [total_len, rotary_dim]
    return rotary_pos_emb


def _apply_rotary_emb_flashattn(q: torch.Tensor, k: torch.Tensor,
                                rotary_pos_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings via flash_attn's apply_rotary_emb.

    Mirrors ref ``vision_apply_rotary_pos_emb_flashatt`` + the encoder
    forward's cos/sin prep (modeling_youtu_vita.py:2690-2697 + 1474-1481):
    ``emb = cat(r, r)``; ``cos/sin = emb.cos()/sin()``; chunk to ``[N, D]``;
    ``apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q)``.

    Args:
        q: [N, num_heads, head_dim]
        k: [N, num_kv_heads, head_dim]
        rotary_pos_emb: [N, D] — raw freq angles (already h/w-concatenated for
            vision, duplicated for audio), identical to ref ``rotary_pos_emb``.
    """
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)   # [N, 2D]
    cos = emb.cos().chunk(2, dim=-1)[0].contiguous()             # [N, D]
    sin = emb.sin().chunk(2, dim=-1)[0].contiguous()
    q_out = flash_apply_rotary_emb(
        q.unsqueeze(0).float(), cos.float(), sin.float()
    ).squeeze(0).type_as(q)
    k_out = flash_apply_rotary_emb(
        k.unsqueeze(0).float(), cos.float(), sin.float()
    ).squeeze(0).type_as(k)
    return q_out, k_out


# ===========================================================================
# Encoder Layers
# ===========================================================================


class OmniRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.variance_epsilon = eps  # alias for compatibility

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        # Match HF: convert back to input dtype BEFORE multiplying weight
        return self.weight * x.to(input_dtype)


class OmniEncoderAttention(nn.Module):
    """Bidirectional packed varlen attention with qk-norm and GQA.

    Tensor-parallel: q/k/v use ``ColumnParallelLinear`` (head-dim column split),
    o uses ``RowParallelLinear`` (row split + all-reduce). ``tp_size`` must
    divide both ``num_heads`` and ``num_kv_heads``. Each rank runs flash attn
    over its LOCAL heads only; the o_proj all-reduce fuses partial outputs.
    HF checkpoint stores q/k/v/o separately, so the per-layer ColumnParallel /
    RowParallel ``weight_loader`` shards them directly (no fusion mapping).
    """

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, rms_norm_eps: float = 1e-6,
                 quant_config=None, prefix: str = ""):
        super().__init__()
        from vllm.distributed import get_tensor_model_parallel_world_size
        from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                                       RowParallelLinear)

        tp_size = get_tensor_model_parallel_world_size()
        if num_heads % tp_size != 0 or num_kv_heads % tp_size != 0:
            raise ValueError(
                f"OmniEncoder TP requires tp_size ({tp_size}) to divide both "
                f"num_heads ({num_heads}) and num_kv_heads ({num_kv_heads}).")

        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        # Per-rank (local) head counts used at runtime for reshaping.
        self.num_heads = num_heads // tp_size
        self.num_kv_heads = num_kv_heads // tp_size
        self.head_dim = head_dim

        self.q_proj = ColumnParallelLinear(
            hidden_size, num_heads * head_dim, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.q_proj")
        self.k_proj = ColumnParallelLinear(
            hidden_size, num_kv_heads * head_dim, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.k_proj")
        self.v_proj = ColumnParallelLinear(
            hidden_size, num_kv_heads * head_dim, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.v_proj")
        self.o_proj = RowParallelLinear(
            num_heads * head_dim, hidden_size, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.o_proj")

        self.q_norm = OmniRMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = OmniRMSNorm(head_dim, eps=rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor,
                cu_seqlens: torch.Tensor, max_seqlen: int,
                rotary_freqs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rotary_freqs: [N, rotary_dim] or [N, rotary_dim//2] frequency angles.
                For vision: [N, rotary_dim] (h and w interleaved).
                For audio: [N, rotary_dim//2].
        """
        N, _ = hidden_states.shape

        # ColumnParallel/RowParallel return (output, bias); bias is None here.
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        q = q.view(N, self.num_heads, self.head_dim)
        k = k.view(N, self.num_kv_heads, self.head_dim)
        v = v.view(N, self.num_kv_heads, self.head_dim)

        # qk-norm (per-head RMSNorm)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply partial rotary (flash_attn style)
        # For vision: freqs [N, 64], chunk into cos[N,32]/sin[N,32], rotate first 64 dims
        # For audio: freqs [N, 32], rotate first 32 dims
        q, k = _apply_rotary_emb_flashattn(q, k, rotary_freqs)

        # FA2 varlen bidirectional
        # flash_attn_varlen_func supports GQA natively when num_heads_q != num_heads_k
        assert flash_attn_varlen_func is not None, "flash_attn required"
        output = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=False,
        )  # [N, local_num_heads_q, head_dim]

        # Reshape LOCAL heads only; RowParallel o_proj all-reduces partials.
        output = output.reshape(N, self.num_heads * self.head_dim)
        output, _ = self.o_proj(output)
        return output


class OmniEncoderMLP(nn.Module):
    """SwiGLU MLP.

    Tensor-parallel: gate/up use ``ColumnParallelLinear`` (intermediate column
    split), down uses ``RowParallelLinear`` (row split + all-reduce). HF stores
    gate/up separately, so each layer's ColumnParallel ``weight_loader`` shards
    them directly (no fusion mapping needed).
    """

    def __init__(self, hidden_size: int, intermediate_size: int,
                 quant_config=None, prefix: str = ""):
        super().__init__()
        from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                                       RowParallelLinear)
        self.gate_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.gate_proj")
        self.up_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.up_proj")
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.down_proj")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, _ = self.gate_proj(x)
        up, _ = self.up_proj(x)
        out, _ = self.down_proj(F.silu(gate) * up)
        return out


class OmniEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, intermediate_size: int, rms_norm_eps: float = 1e-6,
                 quant_config=None, prefix: str = ""):
        super().__init__()
        self.self_attn = OmniEncoderAttention(
            hidden_size, num_heads, num_kv_heads, head_dim, rms_norm_eps,
            quant_config=quant_config, prefix=f"{prefix}.self_attn")
        self.mlp = OmniEncoderMLP(
            hidden_size, intermediate_size,
            quant_config=quant_config, prefix=f"{prefix}.mlp")
        self.input_layernorm = OmniRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = OmniRMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor,
                cu_seqlens: torch.Tensor, max_seqlen: int,
                rotary_freqs: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cu_seqlens, max_seqlen, rotary_freqs)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class SharedTransformerEncoder(nn.Module):
    """Shared bidirectional Transformer body (no final norm - applied in merger).

    Supports per-layer dispatch of ``cu_seqlens`` and ``rotary_freqs`` for the
    video omni-fusion path (``video_fusion_layer_freq`` + 4D RoPE). Fusion
    layers and non-fusion layers may see different attention windows and
    different time encodings for the same physical token. Mirrors HF
    ``YoutuVITAOmniEncoder.forward``:

    - ``cu_seqlens``  : a single ``Tensor`` shared by every layer, OR a
      ``list``/``tuple`` of length ``num_layers`` (one tensor per layer).
    - ``rotary_freqs``: a ``[S, dim_rot]`` tensor shared by every layer, OR a
      stacked ``[num_layers, S, dim_rot]`` tensor (one rotary per layer).
    """

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, intermediate_size: int, num_layers: int,
                 rms_norm_eps: float = 1e-6, quant_config=None, prefix: str = ""):
        super().__init__()
        self.layers = nn.ModuleList([
            OmniEncoderLayer(hidden_size, num_heads, num_kv_heads,
                             head_dim, intermediate_size, rms_norm_eps,
                             quant_config=quant_config,
                             prefix=f"{prefix}.layers.{i}")
            for i in range(num_layers)
        ])

    @staticmethod
    def _max_seqlen(cu: torch.Tensor) -> int:
        if cu.numel() <= 1:
            return 0
        return int((cu[1:] - cu[:-1]).max().item())

    def forward(self, hidden_states: torch.Tensor,
                cu_seqlens, rotary_freqs: torch.Tensor) -> torch.Tensor:
        per_layer_cu = isinstance(cu_seqlens, (list, tuple))
        per_layer_rotary = torch.is_tensor(rotary_freqs) and rotary_freqs.dim() == 3

        if per_layer_cu and len(cu_seqlens) != len(self.layers):
            raise ValueError(
                f"per-layer cu_seqlens length {len(cu_seqlens)} != "
                f"num_layers {len(self.layers)}")
        if per_layer_rotary and rotary_freqs.size(0) != len(self.layers):
            raise ValueError(
                f"per-layer rotary length {rotary_freqs.size(0)} != "
                f"num_layers {len(self.layers)}")

        shared_max = None if per_layer_cu else self._max_seqlen(cu_seqlens)

        for i, layer in enumerate(self.layers):
            layer_cu = cu_seqlens[i] if per_layer_cu else cu_seqlens
            layer_max = self._max_seqlen(layer_cu) if per_layer_cu else shared_max
            layer_rotary = rotary_freqs[i] if per_layer_rotary else rotary_freqs
            hidden_states = layer(hidden_states, layer_cu, layer_max, layer_rotary)
        return hidden_states


# ===========================================================================
# Frontend Embeddings
# ===========================================================================


class VisionEmbeddings(nn.Module):
    """Linear patch projection: [N_patches, patch_dim] -> [N_patches, hidden_size]."""

    def __init__(self, patch_dim: int, hidden_size: int):
        super().__init__()
        self.patch_embedding = nn.Linear(patch_dim, hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.patch_embedding(pixel_values)


def _get_feat_extract_output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
    """Compute output lengths after 3×Conv2d with n_window=50 chunking.

    Matches HF YoutuVITACNNAudioEmbeddings._get_feat_extract_output_lengths.
    Each 100-frame chunk produces 13 tokens; tail chunk computed separately.
    """
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


class AudioEmbeddings(nn.Module):
    """3×Conv2d stride-2 with chunked processing + linear projection.

    Matches HF YoutuVITACNNAudioEmbeddings exactly:
    - Splits mel by n_window*2 (=100) frames into chunks
    - Pads chunks to equal length, applies Conv2d with GELU
    - Extracts valid tokens using mask
    - Projects to hidden_size
    """

    def __init__(self, num_mel_bins: int, downsample_hidden_size: int,
                 hidden_size: int, conv_chunksize: int = 500,
                 n_window: int = 50):
        super().__init__()
        self.num_mel_bins = num_mel_bins
        self.downsample_hidden_size = downsample_hidden_size
        self.conv_chunksize = conv_chunksize
        self.n_window = n_window  # chunk size = n_window * 2

        # 3 Conv2d layers, each stride=2 (padding=1)
        self.conv2d1 = nn.Conv2d(1, downsample_hidden_size, 3, stride=2, padding=1)
        self.conv2d2 = nn.Conv2d(downsample_hidden_size, downsample_hidden_size, 3, stride=2, padding=1)
        self.conv2d3 = nn.Conv2d(downsample_hidden_size, downsample_hidden_size, 3, stride=2, padding=1)

        # Linear projection: D * (mel_bins after 3 convs) -> hidden_size
        # After 3 stride-2 convs on mel dim: mel_bins//8
        self.linear_proj = nn.Linear(
            downsample_hidden_size * (num_mel_bins // 8), hidden_size, bias=False)

    def forward(self, audios: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audios: list of [T_i, num_mel_bins] tensors

        Returns:
            features: [sum_output_tokens, hidden_size] packed features
            lengths: [num_audios] output token counts per audio
        """
        if not audios:
            device = self.conv2d1.weight.device
            return torch.empty(0, self.linear_proj.out_features, device=device), \
                   torch.zeros(0, dtype=torch.long, device=device)

        device = self.conv2d1.weight.device
        dtype = self.conv2d1.weight.dtype

        # Concatenate all audios: [sum_T, mel_bins]
        feature_lens = torch.tensor([a.shape[0] for a in audios], dtype=torch.long, device=device)
        input_features = torch.cat(audios, dim=0).to(device=device, dtype=dtype)
        # input_features: [sum_T, mel_bins]

        # Compute output lengths
        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)

        # Chunk by n_window * 2
        chunk_size = self.n_window * 2  # 100
        chunk_num = torch.ceil(feature_lens.float() / chunk_size).long()

        # Build chunk lengths
        chunk_lengths = torch.tensor(
            [chunk_size] * int(chunk_num.sum().item()),
            dtype=torch.long, device=device)
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % chunk_size
        chunk_lengths[chunk_lengths == 0] = chunk_size

        # Split mel into chunks: input_features.T is [mel_bins, sum_T]
        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=1)
        # Each chunk: [mel_bins, chunk_len_i]

        # Pad chunks to same length and stack
        # pad_sequence works on dim=0, so transpose each chunk
        padded_feature = torch.nn.utils.rnn.pad_sequence(
            [c.T for c in chunk_list], batch_first=True).transpose(1, 2)
        # padded_feature: [num_chunks, mel_bins, max_chunk_len]

        # Compute per-chunk output lengths for mask
        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
        padded_mask_after_cnn = torch.nn.utils.rnn.pad_sequence(
            [torch.ones(int(length), dtype=torch.bool, device=device) 
             for length in feature_lens_after_cnn],
            batch_first=True,
        )

        # Add channel dim: [num_chunks, 1, mel_bins, max_chunk_len]
        padded_feature = padded_feature.unsqueeze(1)

        # Apply 3 Conv2d with GELU (matching HF), split by conv_chunksize to control memory
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            padded_embed = F.gelu(self.conv2d1(chunk))
            padded_embed = F.gelu(self.conv2d2(padded_embed))
            padded_embed = F.gelu(self.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)
        # padded_embed: [num_chunks, D, mel//8, max_chunk_len//8]

        b, c, f, t = padded_embed.size()
        padded_embed = padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        # padded_embed: [num_chunks, max_t, D * mel//8]

        # Extract valid tokens using mask
        hidden_states = padded_embed[padded_mask_after_cnn]
        # hidden_states: [total_valid_tokens, D * mel//8]

        # Linear projection
        hidden_states = self.linear_proj(hidden_states)
        # hidden_states: [total_valid_tokens, hidden_size]

        return hidden_states, aftercnn_lens


# ===========================================================================
# Patch Mergers
# ===========================================================================


class VisionPatchMerger(nn.Module):
    """2×2 spatial merge + LayerNorm + 2-layer MLP to LM hidden.

    Weight keys: norm.weight, linear_fc1.weight, linear_fc2.weight
    """

    def __init__(self, encoder_hidden: int, spatial_merge_size: int, out_hidden_size: int):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        in_dim = encoder_hidden * (spatial_merge_size ** 2)
        self.norm = OmniRMSNorm(in_dim)
        self.linear_fc1 = nn.Linear(in_dim, out_hidden_size, bias=False)
        self.linear_fc2 = nn.Linear(out_hidden_size, out_hidden_size, bias=False)

    def forward(self, encoder_output: torch.Tensor,
                image_grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_output: [N_patches_total, H_enc]
                NOTE: The encoder output is already in merge-compatible order
                because vision_rot_pos_emb arranged positions by merge pattern.
                So we just reshape every (merge^2) consecutive tokens into one.
            image_grid_thw: [N_images, 3] (T, H_patches, W_patches) - unused here

        Returns:
            merged: [N_merged_total, H_lm]
        """
        # HF implementation: x.reshape(-1, self.hidden_size) where hidden_size = enc_dim * merge^2
        # This works because vision_rot_pos_emb already arranged tokens in merge-grouped order
        in_dim = self.spatial_merge_size ** 2 * (encoder_output.shape[-1])
        x = encoder_output.reshape(-1, in_dim)
        x = self.norm(x)
        x = self.linear_fc2(F.gelu(self.linear_fc1(x)))
        return x


class AudioPatchMerger(nn.Module):
    """2× temporal merge + LayerNorm + 2-layer MLP to LM hidden.

    Weight keys: norm.weight, linear_fc1.weight, linear_fc2.weight
    """

    def __init__(self, encoder_hidden: int, temporal_merge_size: int, out_hidden_size: int):
        super().__init__()
        self.temporal_merge_size = temporal_merge_size
        in_dim = encoder_hidden * temporal_merge_size
        self.norm = OmniRMSNorm(in_dim)
        self.linear_fc1 = nn.Linear(in_dim, out_hidden_size, bias=False)
        self.linear_fc2 = nn.Linear(out_hidden_size, out_hidden_size, bias=False)

    def forward(self, x: torch.Tensor,
                audio_lengths: torch.Tensor) -> torch.Tensor:
        """
        Matches HF YoutuVITAOmniAudioPatchMerger.forward:
        1. pad_and_reshape(x, temporal_merge_size) → [B, S//merge, merge*H]
        2. norm → linear_fc1 → GELU → linear_fc2

        Args:
            x: [B, S, H_enc] padded batch of encoder outputs
            audio_lengths: [B] per-audio lengths (before merge)

        Returns:
            merged: [B, S_merged, H_lm]
        """
        merge = self.temporal_merge_size
        if merge > 1:
            # pad_and_reshape: pad S to multiple of merge, then reshape
            B, S, D = x.shape
            pad_size = (merge - (S % merge)) % merge
            if pad_size > 0:
                x = F.pad(x, (0, 0, 0, pad_size))
            new_S = S + pad_size
            x = x.view(B, new_S // merge, merge, D).flatten(2)
            # x: [B, S//merge, merge*D]

        # norm expects [B, *, in_dim]
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = self.norm(x)
        x = F.gelu(self.linear_fc1(x))
        x = self.linear_fc2(x)
        return x


# ===========================================================================
# Top-level OmniEncoder
# ===========================================================================


class VITAOmniEncoder(nn.Module):
    """Complete OmniEncoder: frontends + shared body + mergers."""

    def __init__(self, omni_config, text_hidden_size: int,
                 quant_config=None, prefix: str = "omni_encoder"):
        super().__init__()
        hidden_size = omni_config.hidden_size
        num_heads = omni_config.num_attention_heads
        num_kv_heads = omni_config.num_key_value_heads
        head_dim = getattr(omni_config, 'head_dim', hidden_size // num_heads)
        intermediate_size = omni_config.intermediate_size
        num_layers = omni_config.num_hidden_layers
        rms_norm_eps = getattr(omni_config, 'rms_norm_eps', 1e-6)
        patch_size = omni_config.patch_size
        spatial_merge_size = omni_config.spatial_merge_size
        temporal_merge_size = omni_config.temporal_merge_size
        num_mel_bins = omni_config.num_mel_bins
        downsample_hidden_size = omni_config.downsample_hidden_size
        conv_chunksize = getattr(omni_config, 'conv_chunksize', 500)
        self.rope_theta = getattr(omni_config, 'rope_theta', 10000.0)

        # Frontends
        patch_dim = patch_size ** 2 * 3  # C=3, patch²
        temporal_patch = getattr(omni_config, 'temporal_patch_size', 1)
        patch_dim *= temporal_patch
        self.vision_embeddings = VisionEmbeddings(patch_dim, hidden_size)
        self.audio_embeddings = AudioEmbeddings(
            num_mel_bins, downsample_hidden_size, hidden_size, conv_chunksize)

        # Shared Transformer (tensor-parallel body)
        self.encoder = SharedTransformerEncoder(
            hidden_size, num_heads, num_kv_heads, head_dim,
            intermediate_size, num_layers, rms_norm_eps,
            quant_config=quant_config, prefix=f"{prefix}.encoder")

        # Mergers
        self.vision_merger = VisionPatchMerger(
            hidden_size, spatial_merge_size, text_hidden_size)
        self.audio_merger = AudioPatchMerger(
            hidden_size, temporal_merge_size, text_hidden_size)

        # Config cache
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.spatial_merge_size = spatial_merge_size
        self.temporal_merge_size = temporal_merge_size
        # Rotary dim: head_dim // 2 for the base embedding (each 2D pos uses half)
        # The actual rotary_dim in config is head_dim//2 = 64 for head_dim=128
        self.rotary_dim = head_dim // 2
        self.num_layers = num_layers

        # ---- youtu_vita new omni features ----
        self.video_group_attention = bool(getattr(omni_config, 'video_group_attention', False))
        self.video_omni_chunked_mthw_rope = bool(getattr(omni_config, 'video_omni_chunked_mthw_rope', False))
        self.rope_m_dim = int(getattr(omni_config, 'rope_m_dim', 0))
        self.rope_theta_m = float(getattr(omni_config, 'rope_theta_m', 100.0))
        self._fusion_pattern = self._normalize_fusion_pattern(
            getattr(omni_config, 'video_fusion_layer_freq', None), num_layers)
        if self.video_omni_chunked_mthw_rope:
            self._4d_dims = split_4d_dims(self.rotary_dim, self.rope_m_dim)
        else:
            self._4d_dims = None

    @staticmethod
    def _normalize_fusion_pattern(freq, num_layers):
        """None->all-fusion; int N->i%N==0; list[0/1]->explicit. Mirrors HF."""
        if freq is None:
            return [True] * num_layers
        if isinstance(freq, bool):
            raise ValueError(f"video_fusion_layer_freq must be int/list/None, got bool {freq}")
        if isinstance(freq, int):
            n = max(int(freq), 1)
            return [(i % n == 0) for i in range(num_layers)]
        if isinstance(freq, (list, tuple)):
            if len(freq) != num_layers:
                raise ValueError(
                    f"video_fusion_layer_freq length {len(freq)} != num_layers {num_layers}")
            return [bool(v) for v in freq]
        raise ValueError(f"video_fusion_layer_freq must be int/list/None, got {type(freq)}")

    def _compute_cu_seqlens(self, lengths: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Compute cu_seqlens from a list of lengths."""
        cu_seqlens = F.pad(lengths.cumsum(0), (1, 0)).to(torch.int32)
        max_seqlen = int(lengths.max().item())
        return cu_seqlens, max_seqlen

    def forward_vision(self, pixel_values: torch.Tensor,
                       image_grid_thw: torch.Tensor) -> torch.Tensor:
        """Encode images.

        Args:
            pixel_values: [N_patches_total, patch_dim]
            image_grid_thw: [N_images, 3]

        Returns:
            image_embeds: [N_merged_total, text_hidden_size]
        """
        # Vision embedding
        x = self.vision_embeddings(pixel_values)  # [N_patches, hidden_size]

        # RoPE. When the checkpoint enables the 4D (m,t,h,w) rope
        # (video_omni_chunked_mthw_rope=True — the case for this model), the
        # NATIVE vision path uses vision_rot_pos_emb_4d(grid, m_id=M_IMAGE), NOT
        # the legacy 2D rope. Using 2D here silently corrupts every image's
        # positional encoding -> the vision encoder attends wrong -> the model
        # "cannot see" the image (image VQA describes unrelated content).
        # Mirror native forward_vision (modeling_youtu_vita.py:2950-2951).
        if self.video_omni_chunked_mthw_rope and self._4d_dims is not None:
            pos_list = []
            for i in range(image_grid_thw.shape[0]):
                pos, _ = build_frame_thw_pos_ids(
                    image_grid_thw[i], self.spatial_merge_size,
                    0.0, M_IMAGE, x.device)
                pos_list.append(pos)
            pos_ids = torch.cat(pos_list, dim=0)
            rotary_freqs = chunked_mthw_rotary(
                pos_ids, self._4d_dims, self.rope_theta_m, self.rope_theta)
        else:
            # 2D RoPE with spatial merge pattern (legacy path)
            rotary_freqs = _get_2d_vision_rotary(
                image_grid_thw, self.rotary_dim, self.spatial_merge_size, self.rope_theta)
        rotary_freqs = rotary_freqs.to(x.device)

        # cu_seqlens from grid_thw
        lengths = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).to(x.device)
        cu_seqlens, _ = self._compute_cu_seqlens(lengths)

        # Shared encoder
        x = self.encoder(x, cu_seqlens, rotary_freqs)

        # Vision merger
        return self.vision_merger(x, image_grid_thw)

    def forward_audio(self, audios: list[torch.Tensor]
                      ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode audio.

        Matches HF YoutuVITAOmniModel.forward_audio exactly:
        1. AudioEmbeddings (chunked Conv)
        2. 1D RoPE + packed encoder
        3. Split → pad_sequence → AudioMerger (padded batch)
        4. Return flat packed merged features

        Args:
            audios: list of [T_i, num_mel_bins] tensors

        Returns:
            audio_embeds: [sum_merged, text_hidden_size] flat packed
            merged_lengths: [num_audios]
        """
        # Audio embedding (chunked Conv)
        x, feature_lens = self.audio_embeddings(audios)  # [sum_tokens, hidden]

        if x.shape[0] == 0:
            device = self.audio_merger.linear_fc1.weight.device
            return torch.empty(0, self.audio_merger.linear_fc2.out_features, device=device), \
                   torch.zeros(0, dtype=torch.long, device=device)

        feature_lens = feature_lens.to(x.device)

        # RoPE. When the checkpoint enables the 4D (M|T|H|W) rope
        # (video_omni_chunked_mthw_rope=True — the case for this model), the
        # NATIVE audio path uses audio_rot_pos_emb_4d(feature_lens, m_id=M_AUDIO),
        # NOT the legacy 1D rope. Using 1D here silently corrupts every audio
        # chunk's positional encoding (1D: all 64 dims non-zero with inv_freq
        # based on dim=64; 4D/rope_m_dim=0 -> 3D [T|H|W]=(0,21,21,22): only the
        # 21-dim T segment is non-zero, inv_freq based on 2*t_dim=42, h=w=0).
        # The video-audio path (_forward_video_interleaved -> _audio_4d_rotary)
        # already uses 4D; this was the only path still on 1D.
        if self.video_omni_chunked_mthw_rope:
            rotary_freqs = self._audio_rot_pos_emb_4d(feature_lens)
        else:
            rotary_freqs = _get_1d_audio_rotary(
                feature_lens, self.rotary_dim, self.rope_theta)
        rotary_freqs = rotary_freqs.to(x.device)

        # cu_seqlens
        cu_seqlens, _ = self._compute_cu_seqlens(feature_lens)

        # Shared encoder
        x = self.encoder(x, cu_seqlens, rotary_freqs)

        # Re-batch to [B, S, H] for audio merger (matches HF)
        features = x.split(feature_lens.tolist(), dim=0)
        features_padded = torch.nn.utils.rnn.pad_sequence(
            features, batch_first=True, padding_value=0.0)
        # features_padded: [B, max_S, H]

        # Audio merger (operates on padded batch)
        merged = self.audio_merger(features_padded, feature_lens)
        # merged: [B, max_S_merged, H_lm]

        # Compute merged lengths
        merged_lens = -(-feature_lens // self.temporal_merge_size)

        # Unpad and flatten back to packed
        result_parts = []
        for i, ml in enumerate(merged_lens.tolist()):
            result_parts.append(merged[i, :ml])
        result = torch.cat(result_parts, dim=0)

        return result, merged_lens

    def forward_video(self, video_images: torch.Tensor,
                      video_image_grid_thw: torch.Tensor,
                      video_audios: Optional[list[torch.Tensor]],
                      video_split: torch.Tensor,
                      video_interleave_orders: Optional[torch.Tensor] = None,
                      ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """Encode video: interleaved omni fusion (vision frames + audio chunks
        share one attention window inside the shared encoder).

        Args:
            video_images: [N_patches_all, patch_dim]
            video_image_grid_thw: [N_frames_total, 3]
            video_audios: list of [T_i, mel_bins] per audio chunk
            video_split: [N_videos, 2] each row is (num_images, num_audio_chunks)
            video_interleave_orders: optional [N_events_total, 2] flat tensor of
                ``(modality, idx)`` rows in real temporal order, produced by the
                processor. ``idx`` is GLOBAL (across all videos); modality
                0=vision, 1=audio. When present, each video's rows drive the
                true interleave order (mirrors HF ``use_indices``). When None,
                falls back to the divmod heuristic (only correct for uniform
                1-image:N-audio distribution).

        Returns:
            video_image_embeds: [N_merged_visual, text_hidden_size]
            video_audio_result: (audio_embeds, audio_lengths) or None
        """
        if video_split is None or video_split.numel() == 0:
            # Defensive fallback (video_split is normally always present): no
            # interleave info -> encode vision / audio independently.
            img_embeds = self.forward_vision(video_images, video_image_grid_thw)
            if video_audios:
                aud_embeds, aud_lens = self.forward_audio(video_audios)
                return img_embeds, (aud_embeds, aud_lens)
            return img_embeds, None

        # Interleaved omni fusion
        return self._forward_video_interleaved(
            video_images, video_image_grid_thw, video_audios, video_split,
            video_interleave_orders)

    # ------------------------------------------------------------------
    # 4D RoPE builders (mirror HF _build_video_frame_4d_rotary /
    # _build_video_audio_4d_rotary). Return (raw_angles[S, dim_rot], t_adv).
    # ------------------------------------------------------------------
    def _frame_4d_rotary(self, grid_row, t_base, device):
        pos, t_adv = build_frame_thw_pos_ids(
            grid_row, self.spatial_merge_size, t_base, M_VIDEO_FRAME, device)
        rot = chunked_mthw_rotary(
            pos, self._4d_dims, self.rope_theta_m, self.rope_theta)
        return rot, t_adv

    def _audio_4d_rotary(self, chunk_len, t_base, device):
        pos, t_adv = build_audio_thw_pos_ids(
            chunk_len, t_base, M_VIDEO_AUDIO, device)
        rot = chunked_mthw_rotary(
            pos, self._4d_dims, self.rope_theta_m, self.rope_theta)
        return rot, t_adv

    def _audio_rot_pos_emb_4d(self, feature_lens: torch.Tensor) -> torch.Tensor:
        """4D RoPE for the pure-audio (packed) path.

        Mirrors HF ``YoutuVITAOmniEncoder.audio_rot_pos_emb_4d``: within each
        chunk ``pos = [M_AUDIO, t=0..n-1, h=0, w=0]``; ``t`` restarts at 0 per
        chunk because each chunk is its own attention segment. Returns
        ``[total_tokens, dim_rot]`` raw angles (``dim_rot == rotary_dim``).
        """
        device = feature_lens.device
        pos_list = []
        for length in feature_lens.tolist():
            n = int(length)
            if n <= 0:
                continue
            # t_base=0 per chunk (independent attention segments)
            pos, _ = build_audio_thw_pos_ids(n, 0.0, M_AUDIO, device)
            pos_list.append(pos)
        if not pos_list:
            return torch.zeros(
                (0, self.rotary_dim), dtype=torch.float32, device=device)
        pos_ids = torch.cat(pos_list, dim=0)  # [total, 4]
        return chunked_mthw_rotary(
            pos_ids, self._4d_dims, self.rope_theta_m, self.rope_theta)

    def _forward_video_interleaved(
        self,
        video_images: torch.Tensor,
        video_image_grid_thw: torch.Tensor,
        video_audios: Optional[list[torch.Tensor]],
        video_split: torch.Tensor,
        video_interleave_orders: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """Joint video omni-fusion (interleaved vision + audio). Mirrors HF
        ``YoutuVITAOmniModel.forward_video``.

        Implements the three youtu_vita video features:
        - ``video_group_attention``: partition each video's interleaved
          image/audio events into attention groups at every ``A -> I``
          boundary (leading audio absorbed into the first group).
        - ``video_fusion_layer_freq``: per-layer attention window -- fusion
          layers use the group-level ``cu_seqlens`` while non-fusion layers
          isolate every image / audio chunk into its own window.
        - ``video_omni_chunked_mthw_rope`` (rope_m_dim=0 -> 3D T|H|W):
          fusion layers use group-shared time cursors; non-fusion layers use
          per-chunk-local time (t restarts at 0 per chunk).

        Interleave order: when ``video_interleave_orders`` is provided (the
        processor always produces it for the vision+audio combination), each
        video's rows give the TRUE temporal order of image/audio events
        (mirrors HF ``use_indices`` -- merge & sort by seq_pos). Without it
        we fall back to the ``divmod`` heuristic (only correct for uniform
        N-image:M-audio distribution; otherwise the 4D RoPE time cursor and
        attention grouping are both wrong).
        """
        device = video_images.device

        use_4d = self.video_omni_chunked_mthw_rope
        has_nofusion = not all(self._fusion_pattern)
        build_nofusion_rotary = use_4d and has_nofusion

        # 1. Per-modality frontends.
        vision_features = self.vision_embeddings(video_images)  # [N_all, H_enc]
        if video_audios:
            audio_features, audio_token_lens = self.audio_embeddings(video_audios)
            audio_token_lens = audio_token_lens.to(device)
        else:
            audio_features = vision_features.new_zeros((0, vision_features.size(-1)))
            audio_token_lens = torch.zeros((0,), dtype=torch.long, device=device)

        # Flatten interleave orders to a list of (modality, idx) once.
        orders_list: list[tuple[int, int]] = []
        if video_interleave_orders is not None:
            _orders = video_interleave_orders
            while isinstance(_orders, (list, tuple)):
                _orders = _orders[0] if len(_orders) == 1 else _orders[0]
            if isinstance(_orders, torch.Tensor):
                orders_list = [(int(r[0].item()), int(r[1].item()))
                               for r in _orders]
            else:
                orders_list = [(int(m), int(i)) for m, i in _orders]

        num_videos = int(video_split.shape[0])
        image_cursor = 0
        audio_chunk_cursor = 0
        vision_patch_cursor = 0
        audio_token_cursor = 0
        order_cursor = 0  # advances over orders_list across videos

        segment_features = []
        segment_rotary = []
        segment_rotary_nofusion = [] if build_nofusion_rotary else None
        segment_modality_masks = []       # True = vision, False = audio
        segment_lengths = []
        nofusion_segment_lengths = []     # one entry per image / audio chunk
        per_video_audio_chunk_lens = []

        for vid in range(num_videos):
            num_images = int(video_split[vid, 0].item())
            num_audio_chunks = int(video_split[vid, 1].item())

            # Snapshot global offsets BEFORE this video's slices are consumed,
            # so interleave-order global idxs can be remapped to per-video idxs.
            img_global_start = image_cursor
            aud_global_start = audio_chunk_cursor

            # ---- vision slice ----
            video_grid_thw = video_image_grid_thw[image_cursor:image_cursor + num_images]
            if num_images > 0:
                num_patch_rows = int(
                    (video_grid_thw[:, 0] * video_grid_thw[:, 1] * video_grid_thw[:, 2])
                    .sum().item())
            else:
                num_patch_rows = 0
            vid_vision_feats = vision_features[
                vision_patch_cursor:vision_patch_cursor + num_patch_rows]
            image_cursor += num_images
            vision_patch_cursor += num_patch_rows

            # ---- audio slice ----
            vid_audio_chunk_lens = audio_token_lens[
                audio_chunk_cursor:audio_chunk_cursor + num_audio_chunks]
            vid_audio_total = (
                int(vid_audio_chunk_lens.sum().item()) if num_audio_chunks > 0 else 0)
            vid_audio_feats = audio_features[
                audio_token_cursor:audio_token_cursor + vid_audio_total]
            audio_chunk_cursor += num_audio_chunks
            audio_token_cursor += vid_audio_total

            # ---- per-frame vision feature chunks + (legacy 2D rotary) ----
            if num_images > 0:
                tokens_per_frame = (
                    video_grid_thw[:, 0] * video_grid_thw[:, 1] * video_grid_thw[:, 2]
                ).tolist()
                vision_feature_chunks = list(vid_vision_feats.split(tokens_per_frame, dim=0))
                if use_4d:
                    vision_rotary_chunks = [None] * num_images
                else:
                    vr = _get_2d_vision_rotary(
                        video_grid_thw.to(device), self.rotary_dim,
                        self.spatial_merge_size, self.rope_theta)
                    vision_rotary_chunks = list(vr.split(tokens_per_frame, dim=0))
            else:
                vision_feature_chunks, vision_rotary_chunks = [], []

            # ---- per-chunk audio feature chunks + (legacy 1D rotary) ----
            if num_audio_chunks > 0:
                tokens_per_audio = vid_audio_chunk_lens.tolist()
                audio_feature_chunks = list(vid_audio_feats.split(tokens_per_audio, dim=0))
                if use_4d:
                    audio_rotary_chunks = [None] * num_audio_chunks
                else:
                    ar = _get_1d_audio_rotary(
                        vid_audio_chunk_lens, self.rotary_dim, self.rope_theta)
                    audio_rotary_chunks = list(ar.split(tokens_per_audio, dim=0))
            else:
                audio_feature_chunks, audio_rotary_chunks = [], []

            num_image_chunks = len(vision_feature_chunks)
            num_audio_chunks_actual = len(audio_feature_chunks)
            if num_image_chunks == 0 and num_audio_chunks_actual == 0:
                continue

            # ---- interleave events in temporal order ----
            # modality_kind: 0 = vision, 1 = audio.
            #
            # Preferred: the processor-provided ``video_interleave_orders``
            # (mirrors HF ``use_indices`` -- merge & sort by real seq_pos).
            # Each row is GLOBAL (modality, idx); remap to per-video idx by
            # subtracting this video's offset, so it indexes the per-video
            # ``vision_feature_chunks`` / ``audio_feature_chunks`` sliced above.
            #
            # Fallback: ``divmod`` heuristic (only correct for uniform
            # N-image:M-audio distribution). Single-modality videos use the
            # trivial all-one-kind order (no interleave needed).
            ordered_events: list[tuple[int, int]] = []
            use_indices = (
                len(orders_list) > 0
                and num_image_chunks > 0
                and num_audio_chunks_actual > 0
            )
            if use_indices:
                n_events_this_video = num_image_chunks + num_audio_chunks_actual
                for _ in range(n_events_this_video):
                    modality, gidx = orders_list[order_cursor]
                    order_cursor += 1
                    # The processor builds video_interleave_orders with
                    # PER-VIDEO-LOCAL idx (frame/audio idx resets each video,
                    # matching ref's use_indices `enumerate` which is per-video).
                    # vision_feature_chunks / audio_feature_chunks here are
                    # already THIS video's local slices, so use gidx directly.
                    # Do NOT subtract img_global_start/aud_global_start -- that
                    # remap (assuming global idx) makes video 2's idx negative
                    # -> indexes the wrong frame from the end of the slice
                    # -> wrong rotary/feat -> video 2 vision embeddings diverge
                    # (cos 0.71) and IndexError when un-chunked.
                    ordered_events.append((modality, gidx))
            elif num_image_chunks == 0:
                ordered_events = [(1, k) for k in range(num_audio_chunks_actual)]
            elif num_audio_chunks_actual == 0:
                ordered_events = [(0, k) for k in range(num_image_chunks)]
            else:
                audios_per_image, extra = divmod(num_audio_chunks_actual, num_image_chunks)
                ai = 0
                for ii in range(num_image_chunks):
                    ordered_events.append((0, ii))
                    take = audios_per_image + (1 if ii < extra else 0)
                    for _ in range(take):
                        ordered_events.append((1, ai))
                        ai += 1

            # ---- materialise attention groups (video_group_attention) ----
            has_img = any(m == 0 for m, _ in ordered_events)
            has_aud = any(m == 1 for m, _ in ordered_events)
            if self.video_group_attention and has_img and has_aud:
                event_groups: list[list[tuple[int, int]]] = []
                cur: list[tuple[int, int]] = []
                prev = None
                for m, idx in ordered_events:
                    if m == 0 and prev == 1:  # new group at every A -> I boundary
                        event_groups.append(cur)
                        cur = []
                    cur.append((m, idx))
                    prev = m
                if cur:
                    event_groups.append(cur)
            else:
                event_groups = [ordered_events]

            # ---- build packed segments (per group) ----
            for group_events in event_groups:
                if not group_events:
                    continue
                g_feats, g_rot, g_mask = [], [], []
                g_rot_nofusion = [] if build_nofusion_rotary else None
                t_cursor_vision = 0.0
                t_cursor_audio = 0.0
                for m, idx in group_events:
                    rot_nf = None
                    if m == 0:  # vision
                        feat = vision_feature_chunks[idx]
                        mask_val = True
                        if use_4d:
                            rot, t_adv = self._frame_4d_rotary(
                                video_grid_thw[idx], t_cursor_vision, device)
                            t_cursor_vision += t_adv
                            if build_nofusion_rotary:
                                rot_nf, _ = self._frame_4d_rotary(
                                    video_grid_thw[idx], 0.0, device)
                        else:
                            rot = vision_rotary_chunks[idx]
                    else:       # audio
                        feat = audio_feature_chunks[idx]
                        mask_val = False
                        if use_4d:
                            clen = int(feat.size(0))
                            rot, t_adv = self._audio_4d_rotary(
                                clen, t_cursor_audio, device)
                            t_cursor_audio += t_adv
                            if build_nofusion_rotary:
                                rot_nf, _ = self._audio_4d_rotary(clen, 0.0, device)
                        else:
                            rot = audio_rotary_chunks[idx]
                    g_feats.append(feat)
                    g_rot.append(rot)
                    if build_nofusion_rotary:
                        g_rot_nofusion.append(rot_nf)
                    g_mask.append(torch.full(
                        (feat.size(0),), mask_val, dtype=torch.bool, device=device))
                    # Each event is its own attention window in the non-fusion path.
                    nofusion_segment_lengths.append(int(feat.size(0)))

                segment_features.append(torch.cat(g_feats, dim=0))
                segment_rotary.append(torch.cat(g_rot, dim=0))
                if build_nofusion_rotary:
                    segment_rotary_nofusion.append(torch.cat(g_rot_nofusion, dim=0))
                segment_modality_masks.append(torch.cat(g_mask, dim=0))
                segment_lengths.append(segment_features[-1].size(0))

            per_video_audio_chunk_lens.append(vid_audio_chunk_lens)

        if not segment_features:
            empty = torch.empty(
                0, self.vision_merger.linear_fc2.out_features, device=device)
            return empty, None

        # ---- concatenate per-video segments into one packed sequence ----
        packed_features = torch.cat(segment_features, dim=0)
        packed_rotary = torch.cat(segment_rotary, dim=0)
        packed_mask = torch.cat(segment_modality_masks, dim=0)
        packed_rotary_nofusion = (
            torch.cat(segment_rotary_nofusion, dim=0)
            if build_nofusion_rotary else None)

        cu_fusion = F.pad(
            torch.tensor(segment_lengths, dtype=torch.int32, device=device)
            .cumsum(0, dtype=torch.int32), (1, 0), value=0)

        # ---- per-layer cu_seqlens dispatch (video_fusion_layer_freq) ----
        if has_nofusion:
            cu_nofusion = F.pad(
                torch.tensor(nofusion_segment_lengths, dtype=torch.int32, device=device)
                .cumsum(0, dtype=torch.int32), (1, 0), value=0)
            cu_seqlens = [
                cu_fusion if fusion else cu_nofusion
                for fusion in self._fusion_pattern
            ]
        else:
            cu_seqlens = cu_fusion

        # ---- per-layer rotary dispatch (4D RoPE fusion vs non-fusion) ----
        if build_nofusion_rotary:
            rotary_arg = torch.stack(
                [packed_rotary if fusion else packed_rotary_nofusion
                 for fusion in self._fusion_pattern],
                dim=0)
        else:
            rotary_arg = packed_rotary

        # ---- run the shared transformer once over the packed sequence ----
        encoded = self.encoder(packed_features, cu_seqlens, rotary_arg)

        # ---- split encoder output back via the modality mask ----
        vision_output = encoded[packed_mask]
        audio_output_packed = encoded[~packed_mask]

        if vision_output.size(0) > 0:
            vis_embeds = self.vision_merger(vision_output, video_image_grid_thw)
        else:
            vis_embeds = encoded.new_zeros(
                (0, self.vision_merger.linear_fc2.out_features))

        if audio_output_packed.size(0) > 0 and per_video_audio_chunk_lens:
            all_audio_chunk_lens = torch.cat(per_video_audio_chunk_lens, dim=0)
            aud_split = audio_output_packed.split(all_audio_chunk_lens.tolist(), dim=0)
            aud_padded = torch.nn.utils.rnn.pad_sequence(
                aud_split, batch_first=True, padding_value=0.0)
            merged = self.audio_merger(aud_padded, all_audio_chunk_lens)
            aud_merged_lens = -(-all_audio_chunk_lens // self.temporal_merge_size)
            parts = [merged[i, :ml] for i, ml in enumerate(aud_merged_lens.tolist())]
            aud_embeds = torch.cat(parts, dim=0)
            return vis_embeds, (aud_embeds, aud_merged_lens)

        return vis_embeds, None
