"""Sparse Multi-Head Attention — 중요 토큰 간 attention만 계산."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SparseMHA(nn.Module):
    """Sparse Multi-Head Attention.

    표준 MHA 위에 top-p sparsemax를 적용해 attention weight를 희소화한다.
    XAI를 위해 attention_weights도 함께 반환.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, D)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:    (B, T, d_model)
            mask: (B, T) — True면 무시할 토큰

        Returns:
            out:          (B, T, d_model)
            attn_weights: (B, H, T, T)  — XAI 활용
        """
        B, T, _ = x.shape

        Q = self._split_heads(self.q_proj(x))  # (B, H, T, D)
        K = self._split_heads(self.k_proj(x))
        V = self._split_heads(self.v_proj(x))

        scale = math.sqrt(self.d_head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, T, T)

        if mask is not None:
            # mask: (B, T) → (B, 1, 1, T)
            scores = scores.masked_fill(mask[:, None, None, :], float("-inf"))

        # Sparsemax 근사: top-k softmax (k=ceil(sqrt(T)))
        k = max(1, int(math.ceil(math.sqrt(T))))
        topk_scores, _ = scores.topk(k, dim=-1)
        threshold = topk_scores[..., -1:].expand_as(scores)
        sparse_scores = scores.masked_fill(scores < threshold, float("-inf"))
        attn_weights = F.softmax(sparse_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(out)
        return out, attn_weights
