"""Cross-Modal Fusion — 센서 토큰과 이미지 패치 토큰을 융합."""
import torch
import torch.nn as nn
import math


class CrossModalAttention(nn.Module):
    """센서 시계열 토큰(Query)이 이미지 패치 토큰(Key/Value)에 attend.

    이미지 정보를 센서 스트림에 주입하는 단방향 cross-attention.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def forward(
        self,
        sensor_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sensor_tokens: (B, Ts, d_model)
            image_tokens:  (B, Ti, d_model)

        Returns:
            fused: (B, Ts, d_model)
        """
        B = sensor_tokens.size(0)
        Q = self._split(self.q_proj(sensor_tokens))    # (B, H, Ts, D)
        K = self._split(self.k_proj(image_tokens))     # (B, H, Ti, D)
        V = self._split(self.v_proj(image_tokens))

        attn = (Q @ K.transpose(-2, -1)) / self.scale  # (B, H, Ts, Ti)
        attn = self.dropout(torch.softmax(attn, dim=-1))

        out = (attn @ V).transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_head)
        out = self.out_proj(out)
        return self.norm(out + sensor_tokens)           # residual


class CrossModalFusion(nn.Module):
    """센서 + 이미지 두 모달리티를 완전 융합.

    1. CrossModalAttention (sensor attends image)
    2. 게이팅으로 원본 센서 스트림 보호
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.cross_attn = CrossModalAttention(d_model, n_heads)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(
        self,
        sensor_tokens: torch.Tensor,
        image_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        """이미지 없으면 센서 토큰 그대로 반환."""
        if image_tokens is None:
            return sensor_tokens
        fused = self.cross_attn(sensor_tokens, image_tokens)
        gate = self.gate(torch.cat([sensor_tokens, fused], dim=-1))
        return gate * fused + (1 - gate) * sensor_tokens
