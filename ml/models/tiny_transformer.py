"""Tiny Transformer v1.0 — Edge 추론용 경량 황백화 탐지 모델.

목표 스펙: ~2M Params, <30ms 지연, CPU/Jetson Orin Nano 동작
입력:  센서 시계열 (B, T, sensor_dim) + 선택적 이미지 (B, C, H, W)
출력:  anomaly_score(0~1), severity_pct(%), stage(0~4), attn_weights
"""
import torch
import torch.nn as nn

from ml.models.token_scorer import TokenImportanceScorer
from ml.models.sparse_attention import SparseMHA
from ml.models.dynamic_ffn import DynamicFFN
from ml.models.embedding import SensorEmbedding, PatchEmbedding
from ml.models.cross_modal import CrossModalFusion


class TinyTransformerBlock(nn.Module):
    """Sparse Attention + Dynamic FFN 블록 1개."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SparseMHA(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = DynamicFFN(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with pre-norm
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(normed)
        x = x + attn_out

        # Dynamic FFN — 평균 attention이 높은 토큰이 '중요 토큰'
        importance = attn_weights.mean(dim=1).max(dim=-1).values > 0.1  # (B, T)
        x = self.ffn(self.norm2(x), importance)
        return x, attn_weights


class TinyTransformer(nn.Module):
    """Tiny Transformer v1.0.

    Architecture:
        SensorEmbedding → TokenImportanceScorer(Top-30%)
        → 2× TinyTransformerBlock
        → CrossModalFusion (이미지 있을 때)
        → Mean Pooling → [anomaly_head, severity_head, stage_head]

    Args:
        sensor_dim:  입력 센서 채널 수 (기본 8: 수온·DO·DIN·DIP·N:P·염분·강수·Chl)
        d_model:     임베딩 차원 (기본 128)
        n_heads:     attention head 수 (기본 4)
        n_layers:    transformer block 수 (기본 2)
        d_ff:        FFN 내부 차원 (기본 256)
        n_stages:    황백화 단계 수 (0=정상 ~ 4=심각, 기본 5)
        k_ratio:     Token Pruning 비율 (기본 0.3 = 30% 유지)
    """

    def __init__(
        self,
        sensor_dim: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        n_stages: int = 5,
        k_ratio: float = 0.3,
        img_size: int = 64,
        patch_size: int = 16,
    ):
        super().__init__()
        self.k_ratio = k_ratio
        self.n_stages = n_stages

        # Embedding
        self.sensor_embed = SensorEmbedding(sensor_dim, d_model)
        self.patch_embed = PatchEmbedding(3, patch_size, d_model, img_size)

        # Token Pruning
        self.token_scorer = TokenImportanceScorer(d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TinyTransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )

        # Cross-modal fusion
        self.cross_modal = CrossModalFusion(d_model, n_heads)

        # Output heads
        self.norm = nn.LayerNorm(d_model)
        self.anomaly_head = nn.Linear(d_model, 1)     # 이상 점수 0~1
        self.severity_head = nn.Linear(d_model, 1)    # 심각도 % 0~100
        self.stage_head = nn.Linear(d_model, n_stages) # 단계 분류 logits

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        sensor_seq: torch.Tensor,
        img: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            sensor_seq: (B, T, sensor_dim)
            img:        (B, 3, H, W) 또는 None

        Returns dict with keys:
            anomaly_score:  (B,)        0~1
            severity_pct:   (B,)        0~100
            stage_logits:   (B, n_stages)
            stage:          (B,)        int 0~4
            token_scores:   (B, T)      각 타임스텝 중요도 (XAI)
            attn_weights:   (B, H, K, K) 마지막 블록 attention (XAI)
        """
        # 1. Sensor embedding
        x = self.sensor_embed(sensor_seq)              # (B, T, d_model)

        # 2. Token Pruning — 중요 타임스텝만 선택
        x_pruned, topk_idx, token_scores = self.token_scorer(x, self.k_ratio)

        # 3. Transformer blocks
        attn_weights = None
        for block in self.blocks:
            x_pruned, attn_weights = block(x_pruned)

        # 4. Cross-Modal Fusion (이미지 있을 때)
        if img is not None:
            img_tokens = self.patch_embed(img)          # (B, N+1, d_model)
            x_pruned = self.cross_modal(x_pruned, img_tokens)
        else:
            x_pruned = self.cross_modal(x_pruned, None)

        # 5. Pooling & Heads
        x_norm = self.norm(x_pruned)
        pooled = x_norm.mean(dim=1)                    # (B, d_model)

        anomaly_score = torch.sigmoid(self.anomaly_head(pooled)).squeeze(-1)
        severity_pct = torch.sigmoid(self.severity_head(pooled)).squeeze(-1) * 100
        stage_logits = self.stage_head(pooled)
        stage = stage_logits.argmax(dim=-1)

        return {
            "anomaly_score": anomaly_score,
            "severity_pct": severity_pct,
            "stage_logits": stage_logits,
            "stage": stage,
            "token_scores": token_scores,
            "attn_weights": attn_weights,
        }

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
