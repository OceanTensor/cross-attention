"""
gold_identify_false_positives.py — 재현율100% 임계값에서 오탐되는 
                                     정확한 샘플(날짜·지역) 규명
════════════════════════════════════════════════════════════════════
v19/v20/v21이 전부 소수점까지 동일한 confusion_matrix
([[126976,36864],[0,204800]])를 냈다 — 어떤 모델을 써도
36,864픽셀(=9개 샘플×4096픽셀)이 항상 오탐된다.
이 9개 샘플이 정확히 어느 지역·어느 날짜인지 규명한다.

사용:
  uv run gold_identify_false_positives.py \
      --tile_spec 128:data/wando_..._nirchg.zarr \
      --tile_spec 135:data/mokpo_..._nirchg.zarr \
      --sample_csv h100_transfer_v19_wando/gold_sample.csv \
      --sample_csv h100_transfer_v19_mokpo/gold_sample.csv \
      --label_csv h100_transfer_v19_wando/gold_label.csv \
      --label_csv h100_transfer_v19_mokpo/gold_label.csv \
      --checkpoint checkpoints/v19/best_model.pt
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gold_train_multitile import MultiTileGoldDataset, VALID_INDICES, parse_tile_spec


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
    args = ap.parse_args()

    import torch
    from st_mmt_xattn import STMMTCrossAttn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tile_spec = parse_tile_spec(args.tile_spec)
    tile_cube_ids = list(tile_spec.keys())
    label_pairs = list(zip(tile_cube_ids, args.label_csv))

    ds_val = MultiTileGoldDataset(tile_spec, args.sample_csv, label_pairs,
                                   "val", patch_size=args.patch, augment=False)
    print(f"val 샘플: {len(ds_val)}개")

    model = STMMTCrossAttn(
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=2, n_stages=args.n_stages, patch_size=4,
    ).to(device)
    sd = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.eval()

    # 배치 없이 샘플별로 하나씩 평가 (어느 샘플인지 추적하기 위해)
    per_sample_results = []
    with torch.no_grad():
        for idx in range(len(ds_val)):
            x, y = ds_val[idx]
            x = x.unsqueeze(0).to(device)  # (1, T, C, H, W)
            out = model(x)
            probs = torch.softmax(out["last_logits"], dim=1)[0, 1]  # (H, W)
            mean_prob = probs.mean().item()
            has_event = bool(y[0, 0].item())  # 패치 전체가 동일 라벨이므로 첫 픽셀만 확인
            s = ds_val.samples[idx]
            per_sample_results.append({
                "idx": idx,
                "cube_id": s["cube_id"],
                "target_date": s["target_date"],
                "has_event": has_event,
                "mean_prob": mean_prob,
            })

    # 재현율100% 임계값 계산 (전체 픽셀 기준과 동일한 원리, 샘플 평균확률로 근사)
    pos_probs = [r["mean_prob"] for r in per_sample_results if r["has_event"]]
    threshold = max(0.0, min(pos_probs) - 1e-6)
    print(f"\n재현율100% 임계값(샘플평균 기준 근사): {threshold:.6f}")

    # 오탐(음성인데 threshold 넘는 것) 찾기
    false_positives = [r for r in per_sample_results
                        if not r["has_event"] and r["mean_prob"] >= threshold]
    false_positives.sort(key=lambda r: -r["mean_prob"])

    print(f"\n{'='*70}")
    print(f"  오탐 의심 샘플 — {len(false_positives)}개 (재현율100% 임계값 초과)")
    print(f"{'='*70}")
    for r in false_positives:
        tile_name = {v: k for k, v in tile_spec.items()}.get(r["cube_id"], r["cube_id"])
        print(f"  cube_id={r['cube_id']:4d}  날짜={r['target_date']}  "
              f"평균확률={r['mean_prob']:.4f}  (임계값={threshold:.4f})")

    print(f"\n{'='*70}")
    print(f"  참고 — 가장 확률 낮은 양성 샘플(임계값을 결정한 표본)")
    print(f"{'='*70}")
    pos_sorted = sorted([r for r in per_sample_results if r["has_event"]],
                         key=lambda r: r["mean_prob"])
    for r in pos_sorted[:5]:
        print(f"  cube_id={r['cube_id']:4d}  날짜={r['target_date']}  "
              f"평균확률={r['mean_prob']:.4f}")


if __name__ == "__main__":
    main()
