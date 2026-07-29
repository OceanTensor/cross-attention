"""Lightweight Embedding — 센서 시계열 & 이미지 패치 → d_model 벡터."""
import torch
import torch.nn as nn
import math


class SensorEmbedding(nn.Module):
    """다변량 센서 시계열 (B, T, C) → (B, T, d_model).

    positional encoding을 더해 시간 순서 정보도 포함.
    """

    def __init__(self, sensor_dim: int, d_model: int, max_len: int = 512):
        super().__init__()
        self.proj = nn.Linear(sensor_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

        # 고정 sinusoidal PE
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) → (B, T, d_model)"""
        T = x.size(1)
        return self.norm(self.proj(x) + self.pe[:, :T])


class PatchEmbedding(nn.Module):
    """이미지 (B, C_img, H, W) → 패치 토큰 (B, N_patches, d_model).

    patch_size 크기로 이미지를 격자 분할 후 선형 투영.
    """

    def __init__(
        self,
        img_channels: int = 3,
        patch_size: int = 16,
        d_model: int = 128,
        img_size: int = 64,
    ):
        super().__init__()
        self.patch_size = patch_size
        n_patches = (img_size // patch_size) ** 2
        patch_dim = img_channels * patch_size * patch_size

        self.proj = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, d_model) * 0.02)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, N+1, d_model)"""
        B, C, H, W = img.shape
        p = self.patch_size
        # reshape into patches
        patches = img.unfold(2, p, p).unfold(3, p, p)  # (B, C, nh, nw, p, p)
        B_, C_, nh, nw, _, _ = patches.shape
        patches = patches.contiguous().view(B_, nh * nw, C_ * p * p)
        x = self.proj(patches)                          # (B, N, d_model)
        cls = self.cls_token.expand(B_, -1, -1)
        x = torch.cat([cls, x], dim=1)                 # (B, N+1, d_model)
        return x + self.pos_embed
