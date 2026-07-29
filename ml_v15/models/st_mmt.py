"""ST-MMT (Spatio-Temporal Multi-Modal Transformer) — 클라우드 학습용.

Ocean Tensor Cube [B, T, C, H, W] 입력을 받아 미래 t_out일 연속 ADI(0~10)를 공간적으로 회귀.
TinyTransformer의 교사 모델(Teacher) 역할.

Architecture:
    PatchSpatialEmbed → TemporalEmbed
    → N× ST-Block (Spatial Attn + Temporal Attn + FFN)
    → 공유 upsample trunk
    → 3-헤드: ① ADI 회귀(B,t_out,H,W) ② warn 로짓(B,H,W) ③ severe 로짓(B,H,W)
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
        t_out: int = 7,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_stages = n_stages
        self.t_out = t_out   # 미래 예측 지평(일). 디코더가 t_out일치 라벨을 한 번에 출력

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

        # Decoder: 공유 upsample trunk → 3-헤드 (멀티태스크 회귀)
        #   ① adi_head    : 미래 t_out일 연속 ADI(0~10) 회귀
        #   ② warn_head   : 7일창 P(max ADI ≥ 5.0=발생) 이진 로짓 (사용자 "7일 위험도")
        #   ③ severe_head : 7일창 P(max ADI ≥ 8.0=심화) 이진 로짓 (사용자 "심각 전이")
        self.dec_trunk = nn.Sequential(
            nn.ConvTranspose2d(d_model, d_model // 2, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
        )
        self.adi_head    = nn.Conv2d(d_model // 2, t_out, kernel_size=1)
        self.warn_head   = nn.Conv2d(d_model // 2, 1, kernel_size=1)
        self.severe_head = nn.Conv2d(d_model // 2, 1, kernel_size=1)
        self.adi_max = 10.0   # ADI 상한 (sigmoid 스케일링)

        # 손실 가중치
        self.spatial_loss_weight = 0.05   # ADI 공간 평활(TV)
        self.warn_loss_weight    = 1.0
        self.severe_loss_weight  = 1.0

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, C, H, W)

        Returns:
            adi:          (B, t_out, H, W) — 미래 t_out일 연속 ADI 예측 ∈ [0, 10]
            warn_logit:   (B, H, W) — P(7일내 ADI≥5=발생) 로짓
            severe_logit: (B, H, W) — P(7일내 ADI≥8=심화) 로짓
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

        # Decode: 마지막 타임스텝 feature → 공유 trunk → 3-헤드
        last = feat[:, -1]                              # (B, HW, D)
        last = last.permute(0, 2, 1).view(B, D, h, w)  # (B, D, h, w)
        trunk = self.dec_trunk(last)                    # (B, D/2, Hf, Wf)

        adi = torch.sigmoid(self.adi_head(trunk)) * self.adi_max   # (B, t_out, Hf, Wf) ∈ [0, 10]
        warn_logit   = self.warn_head(trunk).squeeze(1)            # (B, Hf, Wf)
        severe_logit = self.severe_head(trunk).squeeze(1)          # (B, Hf, Wf)

        return {
            "adi":          adi,           # (B, t_out, H, W) 연속 ADI 예측
            "warn_logit":   warn_logit,    # (B, H, W) P(7일내 ADI≥5=발생) 로짓
            "severe_logit": severe_logit,  # (B, H, W) P(7일내 ADI≥8=심화) 로짓
        }

    # warn/severe 이진 타깃 임계 (연속 ADI 기준) — 라벨러 등급경계 정합(발생=5, 심화=8). eval.py와 공유(SSOT)
    WARN_THRESH = 5.0
    SEVERE_THRESH = 8.0

    def compute_loss(
        self,
        outputs: dict,
        adi_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """멀티헤드 손실 = Huber(ADI) + BCE(warn) + BCE(severe) + λ·spatial.

        Args:
            outputs:    forward() dict — adi, warn_logit, severe_logit
            adi_target: (B, t_out, H, W) 연속 ADI 타깃. IGNORE=-1(마스크)

        Returns:
            dict(total, huber, bce_warn, bce_severe, spatial)
        """
        adi          = outputs["adi"]           # (B, t_out, H, W)
        warn_logit   = outputs["warn_logit"]    # (B, H, W)
        severe_logit = outputs["severe_logit"]  # (B, H, W)

        valid = adi_target >= 0                  # (B, t_out, H, W) — 유효(비-IGNORE)일

        # ── ① ADI 회귀 (Huber, 유효 픽셀만) ──
        if valid.any():
            huber = F.smooth_l1_loss(adi[valid], adi_target[valid])
        else:
            huber = adi.sum() * 0.0   # 전부 IGNORE → 그래프 유지 0 loss

        # ── ②③ warn/severe 이진 타깃: 7일창 "유효일 중" max ADI ──
        neg = torch.full_like(adi_target, -1.0)
        adi_valid_only = torch.where(valid, adi_target, neg)
        pix_max   = adi_valid_only.amax(dim=1)   # (B, H, W) — 유효일 중 최대 ADI
        pix_valid = valid.any(dim=1)             # (B, H, W) — 하루라도 유효하면 픽셀 유효
        warn_tgt   = (pix_max >= self.WARN_THRESH).float()
        severe_tgt = (pix_max >= self.SEVERE_THRESH).float()
        if pix_valid.any():
            bce_warn   = F.binary_cross_entropy_with_logits(warn_logit[pix_valid],   warn_tgt[pix_valid])
            bce_severe = F.binary_cross_entropy_with_logits(severe_logit[pix_valid], severe_tgt[pix_valid])
        else:
            bce_warn   = warn_logit.sum() * 0.0
            bce_severe = severe_logit.sum() * 0.0

        # ── ④ ADI 공간 평활(TV) — Hf/Wf==1이면 빈 차분(→NaN) 방어 ──
        dy = torch.abs(adi[:, :, 1:, :] - adi[:, :, :-1, :]).mean() if adi.shape[2] > 1 else adi.sum() * 0.0
        dx = torch.abs(adi[:, :, :, 1:] - adi[:, :, :, :-1]).mean() if adi.shape[3] > 1 else adi.sum() * 0.0
        spatial = dy + dx

        total = (huber
                 + self.warn_loss_weight   * bce_warn
                 + self.severe_loss_weight * bce_severe
                 + self.spatial_loss_weight * spatial)
        return {
            "total": total,
            "huber": huber,
            "bce_warn": bce_warn,
            "bce_severe": bce_severe,
            "spatial": spatial,
        }

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
