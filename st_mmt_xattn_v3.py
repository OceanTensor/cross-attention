"""
st_mmt_xattn_v3.py — ST-MMT 잔차 결합 하이브리드 (Joint Self-Attn 백본 + Cross-Attn 보강)
════════════════════════════════════════════════════════════════════
xattn_v1(3그룹)·xattn_v2(4그룹) 둘 다 v19(단일 Self-Attention, 32채널
한꺼번에 처리)보다 재현율100% F1이 낮았다. 공통 원인 가설: 채널을
그룹으로 미리 쪼개면, 각 그룹 인코더가 서로를 전혀 못 본 채 학습되어
"여러 채널의 결합 패턴"(예: 특정 SST-영양염 조합)을 초기 단계에서
포착 못 할 수 있다.

v3 설계: 두 경로를 잔차로 결합한다.
  경로1(JointEncoder, 백본): v19와 동일하게 32채널을 한꺼번에
    패치임베딩 → Self-Attention. v19가 갖던 강점을 구조적으로 보존.
  경로2(3그룹 Cross-Attention, 보강): xattn_v1과 동일한 구조
    (물리+영양염 통합 A / 대기 B / 위성 C, 허브 방식).
    Cross-Attention 요구사항(구현제안서 명세, XAI attn_weights)을
    계속 충족.
  최종 = LayerNorm(경로1 + α·경로2) — α는 학습 가능한 스칼라
    (초기값 0에 가깝게 시작해, 학습이 진행되며 보강 경로의
    기여도를 모델이 스스로 조절하게 함)

채널 그룹 (물리+영양염을 다시 하나로 합침, v1과 동일 A 정의로 복귀):
  A. 현장 해양실측(20ch): point_env(10)+khoa해류(2)+해양파생(8)
  B. 대기(7ch): kma(6)+Days_Since_Rain(1)
  C. 위성(3ch): satellite(2)+NIR_daily_change(1)
  정적(2ch): Month_sin/cos

사용 예:
  from st_mmt_xattn_v3 import STMMTCrossAttnV3
  model = STMMTCrossAttnV3(d_model=256, n_heads=8, n_layers=2, n_stages=2)
  out = model(x)
"""
import math
import torch
import torch.nn as nn

# ── 채널 그룹 정의 (xattn_v1과 동일한 A/B/C, CHANNEL_MASTER 대조 완료) ──
GROUP_A_INSITU    = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 23, 27, 28, 30, 31, 20, 21, 22]
GROUP_B_ATMOS     = [6, 7, 14, 16, 17, 18, 19]
GROUP_C_SATELLITE = [15, 24, 29]
GROUP_STATIC      = [25, 26]

_all = GROUP_A_INSITU + GROUP_B_ATMOS + GROUP_C_SATELLITE + GROUP_STATIC
assert len(set(_all)) == 32 and sorted(_all) == list(range(32)), "채널 그룹이 32채널을 정확히 커버하지 않음"


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query_feat, kv_feat):
        out, attn_weights = self.attn(query_feat, kv_feat, kv_feat)
        return self.norm(query_feat + out), attn_weights


class STBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.spatial_attn = SelfAttention(d_model, n_heads, dropout)
        self.temporal_attn = SelfAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, HW, D = x.shape
        x_s = x.view(B * T, HW, D)
        x_s = self.spatial_attn(x_s)
        x = x_s.view(B, T, HW, D)

        x_t = x.permute(0, 2, 1, 3).contiguous().view(B * HW, T, D)
        x_t = self.temporal_attn(x_t)
        x = x_t.view(B, HW, T, D).permute(0, 2, 1, 3).contiguous()

        ffn_out = self.drop(self.ffn(x))
        return self.norm(x + ffn_out)


def _positional_encoding(max_T, d_model):
    pe_t = torch.zeros(max_T, d_model)
    pos = torch.arange(max_T).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe_t[:, 0::2] = torch.sin(pos * div)
    pe_t[:, 1::2] = torch.cos(pos * div)
    return pe_t


class JointEncoder(nn.Module):
    """v19와 동일한 구조 — 32채널을 한꺼번에 패치임베딩 후 Self-Attention.
    그룹 분리 없이 모든 채널이 처음부터 서로를 보며 학습된다(백본 경로)."""
    def __init__(self, in_channels=32, d_model=256, n_heads=8, n_layers=4, patch_size=4, max_T=32):
        super().__init__()
        self.patch_proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.patch_norm = nn.LayerNorm(d_model)
        self.register_buffer("pe_t", _positional_encoding(max_T, d_model))
        self.blocks = nn.ModuleList([STBlock(d_model, n_heads, d_model * 2) for _ in range(n_layers)])

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        patches = self.patch_proj(x_flat)
        _, D, h, w = patches.shape
        patches = patches.view(B * T, D, h * w).permute(0, 2, 1)
        patches = self.patch_norm(patches)
        patches = patches.view(B, T, h * w, D)
        pe = self.pe_t[:T].unsqueeze(1)
        patches = patches + pe.unsqueeze(0)
        feat = patches
        for block in self.blocks:
            feat = block(feat)
        return feat, h, w


class GroupEncoder(nn.Module):
    """그룹별 인코더(보강 경로용) — xattn_v1과 동일 구조."""
    def __init__(self, in_channels, d_model=256, n_heads=8, n_layers=2, patch_size=4, max_T=32):
        super().__init__()
        self.patch_proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.patch_norm = nn.LayerNorm(d_model)
        self.register_buffer("pe_t", _positional_encoding(max_T, d_model))
        self.blocks = nn.ModuleList([STBlock(d_model, n_heads, d_model * 2) for _ in range(n_layers)])

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        patches = self.patch_proj(x_flat)
        _, D, h, w = patches.shape
        patches = patches.view(B * T, D, h * w).permute(0, 2, 1)
        patches = self.patch_norm(patches)
        patches = patches.view(B, T, h * w, D)
        pe = self.pe_t[:T].unsqueeze(1)
        patches = patches + pe.unsqueeze(0)
        feat = patches
        for block in self.blocks:
            feat = block(feat)
        return feat, h, w


class STMMTCrossAttnV3(nn.Module):
    """Joint Self-Attention 백본(v19 강점 보존) + 3그룹 Cross-Attention 보강
    (명세충족·XAI유지) 경로를 잔차로 결합."""
    def __init__(self, d_model=256, n_heads=8, n_layers_joint=4, n_layers_group=2,
                 n_stages=2, patch_size=4, init_alpha=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_stages = n_stages
        self.patch_size = patch_size

        # 경로1: 백본(32채널 통합, v19와 동일 규모)
        self.joint_encoder = JointEncoder(32, d_model, n_heads, n_layers_joint, patch_size)

        # 경로2: 3그룹 Cross-Attention 보강
        self.encoder_A = GroupEncoder(len(GROUP_A_INSITU), d_model, n_heads, n_layers_group, patch_size)
        self.encoder_B = GroupEncoder(len(GROUP_B_ATMOS), d_model, n_heads, n_layers_group, patch_size)
        self.encoder_C = GroupEncoder(len(GROUP_C_SATELLITE), d_model, n_heads, n_layers_group, patch_size)
        self.xattn_B = CrossAttentionBlock(d_model, n_heads)
        self.xattn_C = CrossAttentionBlock(d_model, n_heads)

        # 잔차 결합 가중치(학습 가능, 0 근처에서 시작 — 처음엔 백본 위주로 안전하게 출발)
        self.residual_alpha = nn.Parameter(torch.tensor(init_alpha))

        self.static_proj = nn.Linear(len(GROUP_STATIC), d_model)
        self.fusion_norm = nn.LayerNorm(d_model)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model, d_model // 2, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.Conv2d(d_model // 2, n_stages, kernel_size=1),
        )
        self.spatial_loss_weight = 0.05

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        B, T, C, H, W = x.shape
        assert C == 32, f"입력 채널은 32여야 함, 받은 값: {C}"

        # 경로1: 백본(전체 32채널 통합)
        feat_joint, h, w = self.joint_encoder(x)

        # 경로2: 3그룹 Cross-Attention 보강
        x_A = x[:, :, GROUP_A_INSITU]
        x_B = x[:, :, GROUP_B_ATMOS]
        x_C = x[:, :, GROUP_C_SATELLITE]
        x_static = x[:, :, GROUP_STATIC]

        feat_A, _, _ = self.encoder_A(x_A)
        feat_B, _, _ = self.encoder_B(x_B)
        feat_C, _, _ = self.encoder_C(x_C)

        BT_A = feat_A.view(B * T, h * w, self.d_model)
        BT_B = feat_B.view(B * T, h * w, self.d_model)
        BT_C = feat_C.view(B * T, h * w, self.d_model)

        refine, w_b = self.xattn_B(BT_A, BT_B)
        refine, w_c = self.xattn_C(refine, BT_C)
        refine = refine.view(B, T, h * w, self.d_model)

        # 잔차 결합: 백본 + α·보강경로
        fused = self.fusion_norm(feat_joint + self.residual_alpha * refine)

        static_mean = x_static.mean(dim=(3, 4))
        static_feat = self.static_proj(static_mean)
        fused = fused + static_feat.unsqueeze(2)

        last = fused[:, -1]
        last = last.permute(0, 2, 1).view(B, self.d_model, h, w)
        logits = self.decoder(last)
        logits_full = logits.unsqueeze(1).expand(-1, T, -1, -1, -1)

        return {
            "logits": logits_full,
            "pred": logits_full.argmax(dim=2),
            "last_logits": logits,
            "xattn_weights": (w_b, w_c),
            "residual_alpha": self.residual_alpha.item(),
        }
