"""4D (M|T|H|W) chunked MTHW RoPE for youtu_vita video omni-fusion path.

Mirrors YoutuVITAOmniChunkedMTHWRotaryEmbedding + _split_4d_dims in
modular_youtu_vita.py. With rope_m_dim=0 it degrades to pure 3D (T|H|W).

Released checkpoint: chunked_mthw_rope=True, rope_m_dim=0, head_dim=128
-> dim_rot=64, split_4d_dims(64,0)=(0,21,21,22).
"""
from __future__ import annotations

import torch

M_IMAGE = 0
M_AUDIO = 1
M_VIDEO_FRAME = 2
M_VIDEO_AUDIO = 3

# Audio tokens per second of video (100Hz mel / 8x downsample).
VIDEO_FRAME_T_STEP = 100.0 / 8  # = 12.5


def split_4d_dims(dim_rot: int, m_dim: int = 4):
    """Split dim_rot into (m,t,h,w); remainder goes to W."""
    if m_dim < 0:
        raise ValueError(f"m_dim must be >= 0, got {m_dim}")
    if dim_rot < m_dim + 3:
        raise ValueError(f"dim_rot {dim_rot} too small for m_dim {m_dim}")
    rest = dim_rot - m_dim
    per = rest // 3
    rem = rest - per * 3
    t_dim, h_dim, w_dim = per, per, per + rem
    assert m_dim + t_dim + h_dim + w_dim == dim_rot
    return m_dim, t_dim, h_dim, w_dim


def _make_inv_freq(d, base, device):
    """Per-segment locally-normalized inv_freq of length d (matches HF _make_inv).

    inv_freq is computed on CPU on purpose: CPU and CUDA ``pow`` differ by a few
    ulp, which accumulates across layers and changes the vision outputs.
    """
    inv = 1.0 / (base ** (torch.arange(0, d * 2, 2, dtype=torch.float32, device='cpu') / (d * 2)))
    return inv.to(device)


def chunked_mthw_rotary(pos_ids, dims, theta_m=100.0, theta=10000.0):
    """pos_ids [S,4]=(m,t,h,w) -> raw rotary angles [S, dim_rot] (NOT cos/sin).

    m_dim=0 skips the M segment (pure 3D T|H|W layout).
    """
    m_dim, t_dim, h_dim, w_dim = dims
    device = pos_ids.device
    pos = pos_ids.float()
    segs = []
    if m_dim > 0:
        segs.append(torch.outer(pos[:, 0], _make_inv_freq(m_dim, theta_m, device)))
    segs.append(torch.outer(pos[:, 1], _make_inv_freq(t_dim, theta, device)))
    segs.append(torch.outer(pos[:, 2], _make_inv_freq(h_dim, theta, device)))
    segs.append(torch.outer(pos[:, 3], _make_inv_freq(w_dim, theta, device)))
    return torch.cat(segs, dim=-1)


def build_frame_thw_pos_ids(grid_row, spatial_merge_size, t_base, m_id, device):
    """[T*H*W,4] (m,t,h,w) pos ids for ONE video frame. Returns (pos, t_advance)."""
    T_, H_, W_ = (int(v) for v in grid_row.tolist())
    merge = spatial_merge_size
    hpos = torch.arange(H_, device=device).unsqueeze(1).expand(-1, W_)
    hpos = hpos.reshape(H_ // merge, merge, W_ // merge, merge).permute(0, 2, 1, 3).flatten()
    wpos = torch.arange(W_, device=device).unsqueeze(0).expand(H_, -1)
    wpos = wpos.reshape(H_ // merge, merge, W_ // merge, merge).permute(0, 2, 1, 3).flatten()
    hw = hpos.numel()
    n = T_ * hw
    pos = torch.zeros(n, 4, dtype=torch.float32, device=device)
    pos[:, 0] = float(m_id)
    step = VIDEO_FRAME_T_STEP
    for ti in range(T_):
        s, e = ti * hw, (ti + 1) * hw
        pos[s:e, 1] = t_base + ti * step
        pos[s:e, 2] = hpos.float()
        pos[s:e, 3] = wpos.float()
    return pos, T_ * step


def build_audio_thw_pos_ids(a_len, t_base, m_id, device):
    """[a_len,4] (m,t,h,w) pos ids for ONE audio chunk. Returns (pos, t_advance)."""
    pos = torch.zeros(int(a_len), 4, dtype=torch.float32, device=device)
    pos[:, 0] = float(m_id)
    pos[:, 1] = t_base + torch.arange(int(a_len), dtype=torch.float32, device=device)
    return pos, float(a_len)
