"""Dynamic FFN — 중요 토큰만 풀 연산, 나머지는 경량 linear."""
import torch
import torch.nn as nn


class DynamicFFN(nn.Module):
    """Attention score 기반 중요도 마스크를 받아 선택적으로 연산 비용을 배분.

    - 중요 토큰: d_ff 크기 2-layer FFN (풀 연산)
    - 나머지:   단순 linear projection (경량 연산)
    """

    def __init__(self, d_model: int, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.full_ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.light_ff = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        importance_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:               (B, T, d_model)
            importance_mask: (B, T) bool — True인 토큰이 '중요 토큰'

        Returns:
            out: (B, T, d_model)
        """
        out = self.light_ff(x)                                  # 기본: 전체 경량 연산
        if importance_mask.any():
            out[importance_mask] = self.full_ff(x[importance_mask])  # 중요 토큰 풀 연산
        return self.norm(out + x)                               # residual
