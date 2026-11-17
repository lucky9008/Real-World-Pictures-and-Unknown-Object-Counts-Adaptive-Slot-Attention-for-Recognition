"""Classical building blocks: Slot Attention, multi-head self-attention, and a
spatial-broadcast decoder."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    """Competitive Slot Attention routing module."""

    def __init__(
        self,
        num_slots: int = 7,
        dim: int = 64,
        iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: int = 128,
        input_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        in_d = input_dim or dim

        # Gaussian slot distribution parameters
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.slots_mu)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(in_d, dim, bias=False)
        self.to_v = nn.Linear(in_d, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

        self.norm_input = nn.LayerNorm(in_d)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

    def forward(
        self,
        inputs: torch.Tensor,
        num_slots: Optional[int] = None,
        iters: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        b, n, _ = inputs.shape
        k = num_slots or self.num_slots
        t_iters = iters or self.iters

        # Adaptable slot count (K) sampling
        mu = self.slots_mu.expand(b, k, -1)
        sigma = self.slots_log_sigma.exp().expand(b, k, -1)
        if seed is not None:
            gen = torch.Generator(device=inputs.device).manual_seed(seed)
            slots = mu + sigma * torch.randn(mu.shape, generator=gen, device=inputs.device)
        else:
            slots = mu + sigma * torch.randn_like(mu)

        inputs_norm = self.norm_input(inputs)
        k_proj = self.to_k(inputs_norm)
        v_proj = self.to_v(inputs_norm)

        all_attns: List[torch.Tensor] = []
        attn = torch.zeros(b, k, n, device=inputs.device)

        for _ in range(t_iters):
            slots_norm = self.norm_slots(slots)
            q_proj = self.to_q(slots_norm)

            # Competitive Softmax across slots (dim=1)
            dots = torch.einsum("bkd,bnd->bkn", q_proj, k_proj) * self.scale
            attn = F.softmax(dots, dim=1)
            all_attns.append(attn)

            # Weighted mean
            attn_norm = attn + self.eps
            attn_norm = attn_norm / attn_norm.sum(dim=-1, keepdim=True)
            updates = torch.einsum("bkn,bnd->bkd", attn_norm, v_proj)

            # GRU update + Residual MLP refinement
            slots = self.gru(updates.reshape(-1, self.dim), slots.reshape(-1, self.dim))
            slots = slots.reshape(b, k, self.dim)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn, all_attns


class MultiHeadSelfAttention(nn.Module):
    """Standard Multi-Head Self-Attention over visual patch tokens."""

    def __init__(self, dim: int = 192, num_heads: int = 3, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        # Standard dot-product attention over tokens (dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        return out, attn


class SpatialBroadcastDecoder(nn.Module):
    def __init__(self, slot_dim: int = 64, hidden_dim: int = 64, initial_size: Tuple[int, int] = (14, 14)):
        super().__init__()
        self.initial_size = initial_size
        self.net = nn.Sequential(
            nn.ConvTranspose2d(slot_dim + 4, hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 4, kernel_size=3, padding=1),
        )

    def forward(self, slots: torch.Tensor, target_size: Tuple[int, int] = (224, 224)):
        b, k, d = slots.shape
        h0, w0 = self.initial_size

        # 1. Broadcast slots over spatial grid
        slots_flat = slots.reshape(b * k, d, 1, 1).expand(-1, -1, h0, w0)

        # 2. Append linear coordinate grids [-1, 1]
        y = torch.linspace(-1.0, 1.0, h0, device=slots.device)
        x = torch.linspace(-1.0, 1.0, w0, device=slots.device)
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([gx, gy, 1.0 - gx, 1.0 - gy], dim=0).unsqueeze(0).expand(b * k, -1, -1, -1)

        dec_in = torch.cat([slots_flat, coords], dim=1)
        dec_out = self.net(dec_in)
        dec_out = dec_out.reshape(b, k, 4, target_size[0], target_size[1])

        slot_rgb = dec_out[:, :, :3]
        mask_logits = dec_out[:, :, 3:]
        alpha = F.softmax(mask_logits, dim=1)  # Softmax competition across slots
        recon = (slot_rgb * alpha).sum(dim=1)

        return recon, slot_rgb, alpha
