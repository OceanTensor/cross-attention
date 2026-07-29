"""Token Importance Scorer — Attention 이전에 불필요 토큰 제거 (Top-K 선택)."""
import torch
import torch.nn as nn


class TokenImportanceScorer(nn.Module):
    """각 토큰의 중요도를 0~1 스칼라로 평가해 Top-K 토큰만 남긴다.

    Args:
        d_model: 토큰 임베딩 차원
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        k_ratio: float = 0.3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Top-K 토큰 선택.

        Args:
            x: (B, T, d_model)
            k_ratio: 유지할 토큰 비율 (default 30%)

        Returns:
            pruned:    (B, K, d_model)  — 선택된 토큰
            topk_idx:  (B, K)           — 원본 위치 인덱스
            scores:    (B, T)           — 중요도 점수 (XAI 활용)
        """
        scores = self.scorer(x).squeeze(-1)              # (B, T)
        k = max(1, int(x.size(1) * k_ratio))
        topk_idx = torch.topk(scores, k, dim=1).indices  # (B, K)
        pruned = torch.gather(
            x, 1, topk_idx.unsqueeze(-1).expand(-1, -1, x.size(-1))
        )
        return pruned, topk_idx, scores
