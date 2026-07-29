"""
gold_infer_external.py — 학습에 안 쓰인 새 지역·샘플에 대한 순수 추론(외부검증)
════════════════════════════════════════════════════════════════════
v19(완도+목포로만 학습된 모델)가 한 번도 본 적 없는 지역(서천)·
시점(2025-10-14, 실제 공식 사건일)에 대해 무엇을 예측하는지 확인.
재학습 없음 — 순수 추론(inference)만 수행.

사용:
  uv run gold_infer_external.py \
      --tile_spec 144:data/seocheon_30m_full32_2025_ffilled_turb_ma.zarr \
      --sample_csv h100_transfer_seocheon_infer/gold_sample.csv \
      --label_csv h100_transfer_seocheon_infer/gold_label.csv \
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
    from ml.models.st_mmt import STMMT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tile_spec = parse_tile_spec(args.tile_spec)
    tile_cube_ids = list(tile_spec.keys())
    label_pairs = list(zip(tile_cube_ids, args.label_csv))

    print(f"타일: {tile_spec}")
    print(f"라벨: {label_pairs}")

    ds = MultiTileGoldDataset(tile_spec, args.sample_csv, label_pairs,
                               "val", patch_size=args.patch, augment=False)
    print(f"추론 샘플: {len(ds)}개, 채널: {len(VALID_INDICES)}개")

    model = STMMT(
        in_channels=len(VALID_INDICES), d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_model * 2, n_stages=args.n_stages, patch_size=4,
    ).to(device)
    sd = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"체크포인트 로드 완료: {args.checkpoint} (v19, 완도+목포로만 학습됨)")

    with torch.no_grad():
        for idx in range(len(ds)):
            x, y = ds[idx]
            s = ds.samples[idx]
            x = x.unsqueeze(0).to(device)
            out = model(x)
            probs = torch.softmax(out["last_logits"], dim=1)[0, 1]  # (H, W)
            mean_prob = probs.mean().item()
            max_prob = probs.max().item()
            min_prob = probs.min().item()
            actual_label = "사건(실제)" if bool(y[0, 0].item()) else "정상(실제)"

            print(f"\n{'='*60}")
            print(f"  외부검증 결과 — cube_id={s['cube_id']}, 날짜={s['target_date']}")
            print(f"{'='*60}")
            print(f"  실제 라벨: {actual_label}")
            print(f"  v19 예측 확률(사건일 확률): 평균={mean_prob:.4f}, "
                  f"최소={min_prob:.4f}, 최대={max_prob:.4f}")
            print(f"  참고 — v13/v19 재현율100% 임계값(완도+목포 기준): 약 0.13~0.66 범위였음")
            if mean_prob >= 0.5:
                print(f"  → v19는 이 지역·시점을 '위험'으로 판단함 (확률 {mean_prob*100:.1f}%)")
            else:
                print(f"  → v19는 이 지역·시점을 '정상'으로 판단함 (확률 {mean_prob*100:.1f}%)")


if __name__ == "__main__":
    main()
