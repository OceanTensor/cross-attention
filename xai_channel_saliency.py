"""
xai_channel_saliency.py — 그래디언트 기반 채널별 기여도(Saliency) 계산
════════════════════════════════════════════════════════════════════
v13 체크포인트로, "경계(양성)" 클래스 예측에 각 입력 채널이 
얼마나 민감하게 기여하는지를 그래디언트 크기로 정량화한다.

방법: 입력 x에 대해 그래디언트 추적을 켜고, 양성 클래스 로짓의
합을 스칼라 손실로 삼아 역전파. |dL/dx|의 채널별 평균이
그 채널의 "국소적 민감도"(saliency)다. 모델 구조(self/cross-attention
여부)와 무관하게 적용 가능한 모델-비종속적(model-agnostic) 방법.

사용:
  uv run xai_channel_saliency.py \
      --tile_spec 32:data/wando_..._anom.zarr \
      --tile_spec 53:data/mokpo_..._nirchg.zarr \
      --sample_csv h100_transfer_v13_wando/gold_sample.csv \
      --sample_csv h100_transfer_mokpo/gold_sample.csv \
      --label_csv h100_transfer_v13_wando/gold_label.csv \
      --label_csv h100_transfer_mokpo/gold_label.csv \
      --checkpoint checkpoints/v13/best_model.pt
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gold_train_multitile import MultiTileGoldDataset, VALID_INDICES, VALID_CHANNELS, parse_tile_spec
from silver_channels import CHANNEL_MASTER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile_spec", action="append", required=True)
    ap.add_argument("--sample_csv", action="append", required=True)
    ap.add_argument("--label_csv", action="append", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_stages", type=int, default=2)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    args = ap.parse_args()

    import torch
    from ml.models.st_mmt import STMMT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tile_spec = parse_tile_spec(args.tile_spec)
    tile_cube_ids = list(tile_spec.keys())
    label_pairs = list(zip(tile_cube_ids, args.label_csv))

    ds = MultiTileGoldDataset(tile_spec, args.sample_csv, label_pairs,
                               args.split, patch_size=args.patch, augment=False)
    print(f"{args.split} 샘플: {len(ds)}개, 채널: {len(VALID_INDICES)}개")

    model = STMMT(
        in_channels=len(VALID_INDICES), d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_model * 2, n_stages=args.n_stages, patch_size=4,
    ).to(device)
    sd = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.eval()  # 가중치는 고정, 다만 입력 x에는 그래디언트 필요
    print(f"체크포인트 로드 완료: {args.checkpoint}")

    n_channels = len(VALID_INDICES)
    channel_saliency_sum = np.zeros(n_channels, dtype=np.float64)
    n_samples_used = 0

    print("  채널별 표준편차 계산 중 (스케일 보정용)...")
    all_x_for_std = []
    dl_std = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    with torch.no_grad():
        for x, y in dl_std:
            all_x_for_std.append(x.numpy())
    all_x_stack = np.concatenate(all_x_for_std, axis=0)
    channel_std = all_x_stack.std(axis=(0, 1, 3, 4))
    del all_x_for_std, all_x_stack

    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    for i, (x, y) in enumerate(dl):
        x = x.to(device)
        x.requires_grad_(True)

        out = model(x)
        logits = out["last_logits"]
        pos_logit_sum = logits[:, 1].sum()

        model.zero_grad(set_to_none=True)
        pos_logit_sum.backward()

        grad = x.grad.detach().abs()
        per_channel = grad.mean(dim=(0, 1, 3, 4)).cpu().numpy()
        channel_saliency_sum += per_channel
        n_samples_used += 1

        if (i + 1) % 20 == 0:
            print(f"  진행: {i+1}/{len(ds)}")

    channel_saliency_avg_raw = channel_saliency_sum / n_samples_used
    channel_saliency_avg = channel_saliency_avg_raw * channel_std

    # 채널명 매핑 + 정렬 출력
    results = []
    for idx, ch_key in enumerate(VALID_CHANNELS):
        ch_name = CHANNEL_MASTER[ch_key][0]
        results.append((ch_key, ch_name, channel_saliency_avg[idx]))
    results.sort(key=lambda r: -r[2])

    print(f"\n{'='*60}")
    print(f"  채널별 기여도(Saliency, 스케일보정) 순위 — {n_samples_used}개 표본 평균")
    print(f"{'='*60}")
    for rank, (ch_key, ch_name, val) in enumerate(results, 1):
        print(f"  {rank:2d}. {ch_key:6s} {ch_name:22s} {val:.6f}")


if __name__ == "__main__":
    main()
