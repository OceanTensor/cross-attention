"""ST-MMT (Spatio-Temporal Multi-Modal Transformer) — 클라우드 학습용.

Ocean Tensor Cube [B, T, C, H, W] 입력을 받아 황백화 단계(0~4)를 공간적으로 예측.
TinyTransformer의 교사 모델(Teacher) 역할.

Architecture:
    PatchSpatialEmbed → TemporalEmbed
    → N× ST-Block (Spatial Attn + Temporal Attn + FFN)
    → UNet-style decoder
    → [B, T, H, W, n_stages] pixel-wise classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class SpatialAttention(nn.Module):
    """공간 축(H×W)에 대한 Multi-Head Attention.

    각 타임스텝 t마다 HW 위치들 간 attention.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*T, HW, d_model)"""
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)


class TemporalAttention(nn.Module):
    """시간 축(T)에 대한 Multi-Head Attention.

    각 공간 위치 p마다 T 타임스텝들 간 causal attention.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B*HW, T, d_model)"""
        out, _ = self.attn(x, x, x, attn_mask=causal_mask)
        return self.norm(x + out)


class STBlock(nn.Module):
    """Spatial-Temporal Block: SpatialAttn → TemporalAttn → FFN."""

    def __init__(self, d_model: int, n_heads: int = 4, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.spatial_attn = SpatialAttention(d_model, n_heads, dropout)
        self.temporal_attn = TemporalAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, HW, d_model)"""
        B, T, HW, D = x.shape

        # Spatial attention
        x_s = x.view(B * T, HW, D)
        x_s = self.spatial_attn(x_s)
        x = x_s.view(B, T, HW, D)

        # Temporal attention
        x_t = x.permute(0, 2, 1, 3).contiguous().view(B * HW, T, D)
        x_t = self.temporal_attn(x_t)
        x = x_t.view(B, HW, T, D).permute(0, 2, 1, 3).contiguous()

        # FFN
        ffn_out = self.drop(self.ffn(x))
        return self.norm(x + ffn_out)


# ---------------------------------------------------------------------------
# Ocean Cube Dataset helper
# ---------------------------------------------------------------------------

class OceanCubeDataset(torch.utils.data.Dataset):
    """Ocean Tensor Cube [T, H, W, C] 로부터 슬라이딩 윈도우 샘플 생성.

    Args:
        cube:   (T, H, W, C) float32 정규화 텐서
        labels: (T, H, W)    int64 — 0~4 단계
        t_in:   입력 타임스텝 수 (기본 24h)
        stride: 슬라이딩 스트라이드 (기본 6h)
    """

    def __init__(
        self,
        cube: torch.Tensor,
        labels: torch.Tensor,
        t_in: int = 24,
        stride: int = 6,
    ):
        self.cube = cube    # (T, H, W, C)
        self.labels = labels
        self.t_in = t_in
        self.stride = stride
        T = cube.size(0)
        self.indices = list(range(0, T - t_in, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        t0 = self.indices[idx]
        x = self.cube[t0: t0 + self.t_in]      # (T_in, H, W, C)
        y = self.labels[t0 + self.t_in - 1]    # (H, W)
        # (T, H, W, C) → (T, C, H, W)
        x = x.permute(0, 3, 1, 2).float()
        return x, y.long()


# ---------------------------------------------------------------------------
# ST-MMT
# ---------------------------------------------------------------------------

class STMMT(nn.Module):
    """ST-MMT v1.0 — Cloud Training Model.

    Args:
        in_channels: Ocean Cube 채널 수 (기본 16)
        d_model:     임베딩 차원 (기본 256)
        n_heads:     attention head 수 (기본 8)
        n_layers:    ST-Block 수 (기본 4)
        d_ff:        FFN 내부 차원 (기본 512)
        n_stages:    황백화 단계 수 (기본 5)
        patch_size:  공간 패치 크기 (기본 4 — 64→16 token grid)
        dropout:     dropout 비율
    """

    def __init__(
        self,
        in_channels: int = 16,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        n_stages: int = 5,
        patch_size: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_stages = n_stages

        # Patch projection: (B*T, C, H, W) → (B*T, HW/p², d_model)
        self.patch_proj = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )
        self.patch_norm = nn.LayerNorm(d_model)

        # Temporal PE (sinusoidal)
        max_T = 128
        pe_t = torch.zeros(max_T, d_model)
        pos = torch.arange(max_T).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe_t[:, 0::2] = torch.sin(pos * div)
        pe_t[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe_t", pe_t)  # (max_T, d_model)

        # ST-Blocks
        self.blocks = nn.ModuleList(
            [STBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        # Decoder: upsample to original H×W, then pixel-wise classify
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model, d_model // 2, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.Conv2d(d_model // 2, n_stages, kernel_size=1),
        )

        # Physics-informed spatial smoothness loss 계수
        self.spatial_loss_weight = 0.05

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, C, H, W)

        Returns:
            logits:     (B, T, n_stages, H, W)
            pred:       (B, T, H, W) int — 예측 단계
        """
        B, T, C, H, W = x.shape

        # Patch embedding per timestep
        x_flat = x.view(B * T, C, H, W)
        patches = self.patch_proj(x_flat)                # (B*T, d_model, h, w)
        _, D, h, w = patches.shape
        patches = patches.view(B * T, D, h * w).permute(0, 2, 1)  # (B*T, HW, d)
        patches = self.patch_norm(patches)
        patches = patches.view(B, T, h * w, D)

        # Add temporal PE
        pe = self.pe_t[:T].unsqueeze(1)                 # (T, 1, D)
        patches = patches + pe.unsqueeze(0)              # broadcast over (B, HW)

        # ST-Blocks
        feat = patches                                    # (B, T, HW, D)
        for block in self.blocks:
            feat = block(feat)

        # Decode: take last timestep features → pixel-wise logits
        last = feat[:, -1]                              # (B, HW, D)
        last = last.permute(0, 2, 1).view(B, D, h, w)  # (B, D, h, w)
        logits = self.decoder(last)                     # (B, n_stages, H, W)

        # Expand to (B, T, n_stages, H, W) — broadcast over T for loss
        logits_full = logits.unsqueeze(1).expand(-1, T, -1, -1, -1)

        return {
            "logits": logits_full,
            "pred": logits_full.argmax(dim=2),          # (B, T, H, W)
            "last_logits": logits,                      # (B, n_stages, H, W)
        }

    def compute_loss(
        self,
        outputs: dict,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            outputs: forward() 반환 dict
            labels:  (B, H, W) — 마지막 타임스텝 라벨

        Returns:
            total, ce_loss, spatial_loss
        """
        logits = outputs["last_logits"]  # (B, n_stages, H, W)

        # Cross-entropy
        ce_loss = F.cross_entropy(logits, labels, label_smoothing=0.05)

        # Spatial smoothness: TV-norm (황백화는 공간적으로 연속)
        dy = torch.abs(logits[:, :, 1:, :] - logits[:, :, :-1, :]).mean()
        dx = torch.abs(logits[:, :, :, 1:] - logits[:, :, :, :-1]).mean()
        spatial_loss = dy + dx

        total = ce_loss + self.spatial_loss_weight * spatial_loss
        return {
            "total": total,
            "ce_loss": ce_loss,
            "spatial_loss": spatial_loss,
        }

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
