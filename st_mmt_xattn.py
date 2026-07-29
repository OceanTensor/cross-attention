"""
st_mmt_xattn.py — ST-MMT Cross-Attention 확장 (신규 파일, st_mmt.py 미수정)
════════════════════════════════════════════════════════════════════
32채널을 물리적 출처 3그룹(현장실측/대기/위성)으로 분리해 각각
독립 인코더(기존 STMMT의 공간·시간 Self-Attention 블록 재사용)에
통과시킨 뒤, 현장실측을 허브로 삼아 대기·위성에 순차적으로
Cross-Attention(Query≠Key/Value)을 적용한다.

채널 그룹 (silver_channels.py CHANNEL_MASTER 실측 대조로 확정):
  A. 현장 해양실측(20ch): point_env(10) + khoa해류(2) +
     해양파생(8: DIN_DIP_ratio,SST_anomaly,SST_7d_avg,SST_gradient,
               Current_Speed,Chl_7d_avg,SST_30d_avg,MLD)
  B. 대기(7ch): kma(6) + Days_Since_Rain(1, 강수파생)
  C. 위성(3ch): satellite(2: Turbidity,NIR_idx) + NIR_daily_change(1)
  정적(2ch, 인코더 밖): Month_sin/cos — 최종 특징에 위치인코딩처럼 가산

사용 예:
  from st_mmt_xattn import STMMTCrossAttn
  model = STMMTCrossAttn(d_model=256, n_heads=8, n_layers=4, n_stages=2)
  out = model(x)  # x: (B, T, 32, H, W)
  out["last_logits"]  # (B, n_stages, H, W) — 기존 STMMT와 동일 출력 형식
  out["xattn_weights"]  # (w_atmosphere, w_satellite) — XAI 재활용 가능
"""
import math
import torch
import torch.nn as nn

# ── 채널 그룹 정의 (silver_channels.py CHANNEL_MASTER 실측 대조 완료) ──
GROUP_A_INSITU = [0, 1, 2, 3, 5, 8, 9, 20, 21, 22, 10, 11, 4, 12, 13, 23, 27, 28, 30, 31]
GROUP_B_ATMOS = [6, 7, 16, 17, 18, 19, 14]
GROUP_C_SATELLITE = [15, 24, 29]
GROUP_STATIC = [25, 26]

assert len(GROUP_A_INSITU) == 20
assert len(GROUP_B_ATMOS) == 7
assert len(GROUP_C_SATELLITE) == 3
assert len(GROUP_STATIC) == 2
assert len(set(GROUP_A_INSITU + GROUP_B_ATMOS + GROUP_C_SATELLITE + GROUP_STATIC)) == 32


class SelfAttention(nn.Module):
    """기존 STMMT의 SpatialAttention/TemporalAttention과 동일한 Self-Attention
    (Q=K=V=x). 각 그룹 인코더 내부에서 재사용."""
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        return self.norm(x + out)


class CrossAttentionBlock(nn.Module):
    """Query는 한 그룹, Key/Value는 다른 그룹에서 산출 — 진짜 Cross-Attention.
    기존 STMMT의 Self-Attention(Q=K=V=x)과 명확히 구분됨."""
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query_feat, kv_feat):
        # query_feat, kv_feat: (B*T, HW, d_model)
        out, attn_weights = self.attn(query_feat, kv_feat, kv_feat)
        return self.norm(query_feat + out), attn_weights


class STBlockGroup(nn.Module):
    """그룹별 인코더 하나 — 공간 Self-Attn → 시간 Self-Attn → FFN.
    기존 STMMT의 STBlock과 동일 구조(파라미터는 그룹마다 독립)."""
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
        # x: (B, T, HW, d_model)
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
        # x: (B, T, C_group, H, W)
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
        return feat, h, w  # (B, T, HW, D)


class STMMTCrossAttn(nn.Module):
    """3개 그룹 인코더 + 허브 방식 Cross-Attention(A←B, A←C) + 기존과 동일한 디코더."""
    def __init__(self, d_model=256, n_heads=8, n_layers=2, n_stages=2, patch_size=4):
        super().__init__()
        self.d_model = d_model
        self.n_stages = n_stages
        self.patch_size = patch_size

        self.encoder_A = GroupEncoder(len(GROUP_A_INSITU), d_model, n_heads, n_layers, patch_size)
        self.encoder_B = GroupEncoder(len(GROUP_B_ATMOS), d_model, n_heads, n_layers, patch_size)
        self.encoder_C = GroupEncoder(len(GROUP_C_SATELLITE), d_model, n_heads, n_layers, patch_size)

        self.xattn_B = CrossAttentionBlock(d_model, n_heads)
        self.xattn_C = CrossAttentionBlock(d_model, n_heads)

        # 정적 채널(Month_sin/cos)을 위치인코딩처럼 더하기 위한 소형 투영
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
        # x: (B, T, 32, H, W)
        B, T, C, H, W = x.shape
        assert C == 32, f"입력 채널은 32여야 함, 받은 값: {C}"

        x_A = x[:, :, GROUP_A_INSITU]
        x_B = x[:, :, GROUP_B_ATMOS]
        x_C = x[:, :, GROUP_C_SATELLITE]
        x_static = x[:, :, GROUP_STATIC]  # (B, T, 2, H, W)

        feat_A, h, w = self.encoder_A(x_A)  # (B, T, HW, D)
        feat_B, _, _ = self.encoder_B(x_B)
        feat_C, _, _ = self.encoder_C(x_C)

        BT_A = feat_A.view(B * T, h * w, self.d_model)
        BT_B = feat_B.view(B * T, h * w, self.d_model)
        BT_C = feat_C.view(B * T, h * w, self.d_model)

        fused, w1 = self.xattn_B(BT_A, BT_B)
        fused, w2 = self.xattn_C(fused, BT_C)
        fused = fused.view(B, T, h * w, self.d_model)

        # 정적 채널 가산 (전체 격자 평균값 사용 — 정적 채널은 공간적으로 균일)
        static_mean = x_static.mean(dim=(3, 4))  # (B, T, 2)
        static_feat = self.static_proj(static_mean)  # (B, T, D)
        fused = fused + static_feat.unsqueeze(2)

        last = fused[:, -1]  # (B, HW, D)
        last = last.permute(0, 2, 1).view(B, self.d_model, h, w)
        logits = self.decoder(last)  # (B, n_stages, H, W)
        logits_full = logits.unsqueeze(1).expand(-1, T, -1, -1, -1)

        return {
            "logits": logits_full,
            "pred": logits_full.argmax(dim=2),
            "last_logits": logits,
            "xattn_weights": (w1, w2),
        }
