"""
diagnose_auc_mismatch.py — 학습스크립트 vs 스캔스크립트의 AUC 불일치 진단
════════════════════════════════════════════════════════════════════
같은 체크포인트, 같은 val 데이터인데 AUC가 다르게 나오는 원인을
확률 배열 자체를 비교해서 찾는다.

사용:
  uv run diagnose_auc_mismatch.py \
      --tile_spec 118:data/wando_...anomMY.zarr \
      --tile_spec 119:data/mokpo_...anomMY.zarr \
      --sample_csv h100_transfer_v18_wando/gold_sample.csv \
      --sample_csv h100_transfer_v18_mokpo/gold_sample.csv \
      --label_csv h100_transfer_v18_wando/gold_label.csv \
      --label_csv h100_transfer_v18_mokpo/gold_label.csv \
      --checkpoint checkpoints/v18/best_model.pt \
      --batch_size 8
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
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_stages", type=int, default=2)
    args = ap.parse_args()

    import torch
    from sklearn.metrics import roc_auc_score
    from ml.models.st_mmt import STMMT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tile_spec = parse_tile_spec(args.tile_spec)
    tile_cube_ids = list(tile_spec.keys())
    label_pairs = list(zip(tile_cube_ids, args.label_csv))

    # ★ 핵심 확인 1: ds_val을 새로 만들 때마다 anchors가 같은지
    print("=== 진단 1: Dataset 재구성 시 anchors 일관성 ===")
    ds_val_a = MultiTileGoldDataset(tile_spec, args.sample_csv, label_pairs,
                                     "val", patch_size=args.patch, augment=False)
    ds_val_b = MultiTileGoldDataset(tile_spec, args.sample_csv, label_pairs,
                                     "val", patch_size=args.patch, augment=False)
    anchors_match = ds_val_a.anchors == ds_val_b.anchors
    samples_match = [s["sample_id"] for s in ds_val_a.samples] == [s["sample_id"] for s in ds_val_b.samples]
    print(f"  anchors 일치: {anchors_match}")
    print(f"  samples 순서 일치: {samples_match}")

    # 모델 로드 (한 번만)
    model = STMMT(
        in_channels=len(VALID_INDICES), d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_model * 2, n_stages=args.n_stages, patch_size=4,
    ).to(device)
    sd = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.eval()

    def compute_probs(ds, batch_size):
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
        all_true, all_prob = [], []
        with torch.no_grad():
            for x, y in dl:
                x = x.to(device)
                out = model(x)
                probs = torch.softmax(out["last_logits"], dim=1)[:, 1]
                all_true.append(y.numpy().ravel())
                all_prob.append(probs.cpu().numpy().ravel())
        return np.concatenate(all_true), np.concatenate(all_prob)

    print("\n=== 진단 2: batch_size에 따른 AUC 변화 ===")
    for bs in [1, 8, args.batch_size]:
        y_true, y_prob = compute_probs(ds_val_a, bs)
        auc = roc_auc_score(y_true, y_prob)
        print(f"  batch_size={bs}: AUC={auc:.4f}, 확률평균={y_prob.mean():.4f}, "
              f"확률표준편차={y_prob.std():.4f}")

    print("\n=== 진단 3: 동일 batch_size, 반복 실행 시 재현성 ===")
    for i in range(3):
        y_true, y_prob = compute_probs(ds_val_a, args.batch_size)
        auc = roc_auc_score(y_true, y_prob)
        print(f"  실행{i+1}: AUC={auc:.4f}")

    print("\n=== 진단 4: model.training 상태 확인 ===")
    print(f"  model.training = {model.training} (False여야 정상, eval() 호출 후)")


if __name__ == "__main__":
    main()
