"""
gold_identify_hard_negatives.py — train셋에서 모델이 헷갈리는 "어려운 정상 샘플" 식별
════════════════════════════════════════════════════════════════════
val(검증셋)의 오탐 샘플을 직접 재사용하는 것은 검증 오염이라 불가능하다는 것을
확인했다(완도 8/20~9/5 전부 val 소속). 대신 train셋 안에서 현재 체크포인트가
실제로 헷갈려하는 "정상인데 확률이 높은" 샘플을 진짜 OHEM 방식으로 찾는다.

사용:
  uv run gold_identify_hard_negatives.py \
      --tile_spec 128:data/wando_..._nirchg.zarr \
      --tile_spec 135:data/mokpo_..._nirchg.zarr \
      --sample_csv h100_transfer_v19_wando/gold_sample.csv \
      --sample_csv h100_transfer_v19_mokpo/gold_sample.csv \
      --label_csv h100_transfer_v19_wando/gold_label.csv \
      --label_csv h100_transfer_v19_mokpo/gold_label.csv \
      --checkpoint checkpoints/xattn_v2/best_model.pt \
      --out hard_negative_weights.json
"""
import argparse, json, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile_spec", action="append", required=True)
    ap.add_argument("--sample_csv", action="append", required=True)
    ap.add_argument("--label_csv", action="append", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="hard_negative_weights.json")
    ap.add_argument("--hard_weight", type=float, default=3.0, help="어려운 샘플에 줄 배수")
    ap.add_argument("--top_frac", type=float, default=0.2, help="정상 샘플 중 상위 몇 %를 어렵다고 볼지")
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_stages", type=int, default=2)
    args = ap.parse_args()

    import torch
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gold_train_multitile import MultiTileGoldDataset, VALID_INDICES, parse_tile_spec
    from st_mmt_xattn_v3 import STMMTCrossAttnV3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tile_spec = parse_tile_spec(args.tile_spec)
    tile_cube_ids = list(tile_spec.keys())
    label_pairs = list(zip(tile_cube_ids, args.label_csv))

    # ★ train split으로 로드 (val 아님 — 검증 오염 방지)
    ds_train = MultiTileGoldDataset(tile_spec, args.sample_csv, label_pairs,
                                     "train", patch_size=args.patch, augment=False)
    print(f"train 샘플: {len(ds_train)}개")

    model = STMMTCrossAttnV3(d_model=args.d_model, n_heads=args.n_heads,
                              n_layers_joint=4, n_layers_group=2, n_stages=args.n_stages, patch_size=4).to(device)
    sd = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"체크포인트 로드 완료: {args.checkpoint}")

    probs_per_sample = []
    labels_per_sample = []
    with torch.no_grad():
        for idx in range(len(ds_train)):
            x, y = ds_train[idx]
            x = x.unsqueeze(0).to(device)
            out = model(x)
            prob = torch.softmax(out["last_logits"], dim=1)[0, 1].mean().item()
            label = int(y[0, 0].item())
            probs_per_sample.append(prob)
            labels_per_sample.append(label)

    probs = np.array(probs_per_sample)
    labels = np.array(labels_per_sample)

    # 실제 라벨=0(정상)인데 확률이 높은 것 = hard negative
    neg_idx = np.where(labels == 0)[0]
    neg_probs = probs[neg_idx]
    n_hard = max(1, int(len(neg_idx) * args.top_frac))
    hard_local_idx = np.argsort(-neg_probs)[:n_hard]  # 확률 높은 순 상위 n_hard개
    hard_global_idx = set(neg_idx[hard_local_idx].tolist())

    print(f"\n정상 샘플 {len(neg_idx)}개 중 상위 {args.top_frac*100:.0f}%({n_hard}개)를 hard negative로 지정")
    print(f"hard negative 확률 범위: {neg_probs[hard_local_idx].min():.4f} ~ {neg_probs[hard_local_idx].max():.4f}")
    print(f"전체 정상 확률 평균: {neg_probs.mean():.4f}")

    weights = [args.hard_weight if i in hard_global_idx else 1.0 for i in range(len(ds_train))]

    with open(args.out, "w") as f:
        json.dump({
            "weights": weights,
            "hard_weight": args.hard_weight,
            "top_frac": args.top_frac,
            "n_hard_negatives": n_hard,
            "checkpoint_source": args.checkpoint,
        }, f, indent=2)
    print(f"\n✓ 저장: {args.out} (샘플별 가중치 {len(weights)}개)")


if __name__ == "__main__":
    main()
