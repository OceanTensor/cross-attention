"""
st_mmt_xattn_v2.py — ST-MMT Cross-Attention 4그룹 확장판 (신규 파일, st_mmt.py/xattn_v1 미수정)
════════════════════════════════════════════════════════════════════
16.6.2절 발견(목포 8/19 vs 완도 8월말 오탐 사례에서 DIP·SiO2·DIN_DIP_ratio가
뚜렷이 다른데도 SST_anomaly 같은 압도적 공통신호에 가려짐)에 근거해,
기존 3그룹(현장실측/대기/위성)에서 영양염 채널을 별도 그룹(N)으로 분리하고,
허브(물리해양 A)가 영양염(N)에 가장 먼저·강하게 주목하도록 순서를 재설계했다.

채널 그룹 (silver_channels.py CHANNEL_MASTER 실측 대조, 32채널 전체 커버 검증됨):
  A. 물리 해양(13ch): SST, Salinity, Chlorophyll, DO, Current_U/V, SST_anomaly,
     SST_7d_avg, SST_gradient, Current_Speed, Chl_7d_avg, SST_30d_avg, MLD
  N. 영양염(7ch, 신규 분리): DIN, DIP, SiO2, DIN_DIP_ratio, pH, NO3, NH4
  B. 대기(7ch): Precipitation, Solar_Radiation, Days_Since_Rain, Wind_Speed,
     Wind_Dir_sin/cos, Air_Temp
  C. 위성(3ch): Turbidity, NIR_idx, NIR_daily_change
  정적(2ch, 인코더 밖): Month_sin/cos

허브 순서: A(물리해양) ← N(영양염) ← B(대기) ← C(위성)
  — 영양염을 가장 먼저 결합해 대기·위성 신호에 가려지지 않도록 설계

사용 예:
  from st_mmt_xattn_v2 import STMMTCrossAttnV2
  model = STMMTCrossAttnV2(d_model=256, n_heads=8, n_layers=2, n_stages=2)
  out = model(x)  # x: (B, T, 32, H, W)
  out["last_logits"]     # (B, n_stages, H, W) — 기존과 동일 출력 형식
  out["xattn_weights"]   # (w_nutrient, w_atmos, w_satellite) — XAI 재활용 가능
"""
import math
import torch
import torch.nn as nn

# ── 채널 그룹 정의 (CHANNEL_MASTER 실측 대조 완료, 32채널 전체 커버 검증됨) ──
GROUP_A_PHYSICAL  = [0, 5, 8, 9, 10, 11, 12, 13, 23, 27, 28, 30, 31]  # 물리해양
GROUP_N_NUTRIENT  = [1, 2, 3, 4, 20, 21, 22]                          # 영양염(신규분리)
GROUP_B_ATMOS     = [6, 7, 14, 16, 17, 18, 19]                        # 대기
GROUP_C_SATELLITE = [15, 24, 29]                                      # 위성
GROUP_STATIC      = [25, 26]                                          # 정적

assert len(GROUP_A_PHYSICAL) == 13
assert len(GROUP_N_NUTRIENT) == 7
assert len(GROUP_B_ATMOS) == 7
assert len(GROUP_C_SATELLITE) == 3
assert len(GROUP_STATIC) == 2
_all = GROUP_A_PHYSICAL + GROUP_N_NUTRIENT + GROUP_B_ATMOS + GROUP_C_SATELLITE + GROUP_STATIC
assert len(set(_all)) == 32 and sorted(_all) == list(range(32)), "채널 그룹이 32채널을 정확히 커버하지 않음"


class SelfAttention(nn.Module):
    """기존 STMMT의 SpatialAttention/TemporalAttention과 동일(Q=K=V=x)."""
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        return self.norm(x + out)


class CrossAttentionBlock(nn.Module):
    """Query는 한 그룹, Key/Value는 다른 그룹 — 진짜 Cross-Attention."""
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query_feat, kv_feat):
        out, attn_weights = self.attn(query_feat, kv_feat, kv_feat)
        return self.norm(query_feat + out), attn_weights


class STBlockGroup(nn.Module):
    """그룹별 인코더 하나 — 공간 Self-Attn → 시간 Self-Attn → FFN."""
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


class GroupEncoder(nn.Module):
    """채널 그룹 하나를 패치임베딩 + STBlock 여러 층으로 인코딩."""
    def __init__(self, in_channels, d_model=256, n_heads=8, n_layers=2, patch_size=4, max_T=32):
        super().__init__()
        self.patch_proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.patch_norm = nn.LayerNorm(d_model)

        pe_t = torch.zeros(max_T, d_model)
        pos = torch.arange(max_T).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe_t[:, 0::2] = torch.sin(pos * div)
        pe_t[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe_t", pe_t)

        self.blocks = nn.ModuleList([
            STBlockGroup(d_model, n_heads, d_model * 2) for _ in range(n_layers)
        ])

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


class STMMTCrossAttnV2(nn.Module):
    """4개 그룹 인코더(물리해양/영양염/대기/위성) + 허브 방식 Cross-Attention
    (A←N←B←C, 영양염을 최우선 결합) + 기존과 동일한 디코더."""
    def __init__(self, d_model=256, n_heads=8, n_layers=2, n_stages=2, patch_size=4):
        super().__init__()
        self.d_model = d_model
        self.n_stages = n_stages
        self.patch_size = patch_size

        self.encoder_A = GroupEncoder(len(GROUP_A_PHYSICAL), d_model, n_heads, n_layers, patch_size)
        self.encoder_N = GroupEncoder(len(GROUP_N_NUTRIENT), d_model, n_heads, n_layers, patch_size)
        self.encoder_B = GroupEncoder(len(GROUP_B_ATMOS), d_model, n_heads, n_layers, patch_size)
        self.encoder_C = GroupEncoder(len(GROUP_C_SATELLITE), d_model, n_heads, n_layers, patch_size)

        # 허브 순서: 영양염을 가장 먼저(우선순위 최고), 그다음 대기, 위성
        self.xattn_N = CrossAttentionBlock(d_model, n_heads)
        self.xattn_B = CrossAttentionBlock(d_model, n_heads)
        self.xattn_C = CrossAttentionBlock(d_model, n_heads)

        self.static_proj = nn.Linear(len(GROUP_STATIC), d_model)

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

        x_A = x[:, :, GROUP_A_PHYSICAL]
        x_N = x[:, :, GROUP_N_NUTRIENT]
        x_B = x[:, :, GROUP_B_ATMOS]
        x_C = x[:, :, GROUP_C_SATELLITE]
        x_static = x[:, :, GROUP_STATIC]

        feat_A, h, w = self.encoder_A(x_A)
        feat_N, _, _ = self.encoder_N(x_N)
        feat_B, _, _ = self.encoder_B(x_B)
        feat_C, _, _ = self.encoder_C(x_C)

        BT_A = feat_A.view(B * T, h * w, self.d_model)
        BT_N = feat_N.view(B * T, h * w, self.d_model)
        BT_B = feat_B.view(B * T, h * w, self.d_model)
        BT_C = feat_C.view(B * T, h * w, self.d_model)

        # 영양염 → 대기 → 위성 순으로 결합 (영양염 최우선)
        fused, w_n = self.xattn_N(BT_A, BT_N)
        fused, w_b = self.xattn_B(fused, BT_B)
        fused, w_c = self.xattn_C(fused, BT_C)
        fused = fused.view(B, T, h * w, self.d_model)

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
            "xattn_weights": (w_n, w_b, w_c),
        }
