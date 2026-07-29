"""
st_mmt_xattn_v4.py — ST-MMT 전체쌍(pairwise) 양방향 Cross-Attention (신규 파일)
════════════════════════════════════════════════════════════════════
v1~v3까지 전부 "허브 방식"(A가 중심, 나머지가 A로만 순차 연결)이었다.
이번엔 모든 그룹 쌍이 서로 양방향으로 정보를 주고받는 구조를 시도한다:
  A↔B, A↔C, B↔C (총 6개 Cross-Attention: 3쌍×양방향)

채널 그룹 (v1과 동일한 A/B/C 구성):
  A. 현장 해양실측(20ch): point_env+khoa해류+해양파생(영양염 포함)
  B. 대기(7ch)
  C. 위성(3ch)
  정적(2ch): 인코더 밖에서 가산

최종 결합: 각 그룹은 "원본 + 다른 두 그룹으로부터 받은 정보"를 합친 뒤,
세 그룹의 최종 표현을 concat+projection으로 하나의 특징맵으로 합친다.

사용 예:
  from st_mmt_xattn_v4 import STMMTCrossAttnV4
  model = STMMTCrossAttnV4(d_model=256, n_heads=8, n_layers=2, n_stages=2)
  out = model(x)
"""
import math
import torch
import torch.nn as nn

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


class GroupEncoder(nn.Module):
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


class STMMTCrossAttnV4(nn.Module):
    """3그룹(A/B/C) 전체쌍 양방향 Cross-Attention — 허브 방식이 아닌
    모든 그룹이 서로에게 주목하는 구조."""
    def __init__(self, d_model=256, n_heads=8, n_layers=2, n_stages=2, patch_size=4):
        super().__init__()
        self.d_model = d_model
        self.n_stages = n_stages
        self.patch_size = patch_size

        self.encoder_A = GroupEncoder(len(GROUP_A_INSITU), d_model, n_heads, n_layers, patch_size)
        self.encoder_B = GroupEncoder(len(GROUP_B_ATMOS), d_model, n_heads, n_layers, patch_size)
        self.encoder_C = GroupEncoder(len(GROUP_C_SATELLITE), d_model, n_heads, n_layers, patch_size)

        # 전체쌍 양방향: A↔B, A↔C, B↔C (6개)
        self.xattn_A_from_B = CrossAttentionBlock(d_model, n_heads)
        self.xattn_B_from_A = CrossAttentionBlock(d_model, n_heads)
        self.xattn_A_from_C = CrossAttentionBlock(d_model, n_heads)
        self.xattn_C_from_A = CrossAttentionBlock(d_model, n_heads)
        self.xattn_B_from_C = CrossAttentionBlock(d_model, n_heads)
        self.xattn_C_from_B = CrossAttentionBlock(d_model, n_heads)

        self.fusion_proj = nn.Linear(d_model * 3, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)

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

        x_A = x[:, :, GROUP_A_INSITU]
        x_B = x[:, :, GROUP_B_ATMOS]
        x_C = x[:, :, GROUP_C_SATELLITE]
        x_static = x[:, :, GROUP_STATIC]

        feat_A, h, w = self.encoder_A(x_A)
        feat_B, _, _ = self.encoder_B(x_B)
        feat_C, _, _ = self.encoder_C(x_C)

        BT_A = feat_A.view(B * T, h * w, self.d_model)
        BT_B = feat_B.view(B * T, h * w, self.d_model)
        BT_C = feat_C.view(B * T, h * w, self.d_model)

        # 전체쌍 양방향 Cross-Attention
        A_from_B, w_ab = self.xattn_A_from_B(BT_A, BT_B)
        B_from_A, w_ba = self.xattn_B_from_A(BT_B, BT_A)
        A_from_C, w_ac = self.xattn_A_from_C(BT_A, BT_C)
        C_from_A, w_ca = self.xattn_C_from_A(BT_C, BT_A)
        B_from_C, w_bc = self.xattn_B_from_C(BT_B, BT_C)
        C_from_B, w_cb = self.xattn_C_from_B(BT_C, BT_B)

        # 각 그룹의 최종 표현 = 두 방향에서 받은 정보의 평균
        A_final = (A_from_B + A_from_C) / 2
        B_final = (B_from_A + B_from_C) / 2
        C_final = (C_from_A + C_from_B) / 2

        # 세 그룹을 concat 후 projection으로 결합
        fused = torch.cat([A_final, B_final, C_final], dim=-1)
        fused = self.fusion_norm(self.fusion_proj(fused))
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
            "xattn_weights": (w_ab, w_ba, w_ac, w_ca, w_bc, w_cb),
        }
