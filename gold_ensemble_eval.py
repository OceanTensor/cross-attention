"""
gold_ensemble_eval.py — 여러 체크포인트의 확률을 평균내 재현율100% 재평가
════════════════════════════════════════════════════════════════════
v21: seed가 다른 3개 독립 학습 결과(체크포인트)의 확률을 평균내서,
"재현율100%를 깨는 취약 표본 하나"의 순위가 앙상블로 바뀌는지 확인.

사용:
  uv run gold_ensemble_eval.py \
      --tile_spec 128:data/wando_..._nirchg.zarr \
      --tile_spec 135:data/mokpo_..._nirchg.zarr \
      --sample_csv h100_transfer_v19_wando/gold_sample.csv \
      --sample_csv h100_transfer_v19_mokpo/gold_sample.csv \
      --label_csv h100_transfer_v19_wando/gold_label.csv \
      --label_csv h100_transfer_v19_mokpo/gold_label.csv \
      --checkpoint checkpoints/v21_run1/best_model.pt \
      --checkpoint checkpoints/v21_run2/best_model.pt \
      --checkpoint checkpoints/v21_run3/best_model.pt
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
    ap.add_argument("--checkpoint", action="append", required=True,
                     help="여러 체크포인트 경로, 반복 지정")
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_stages", type=int, default=2)
    args = ap.parse_args()

    import torch
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                   f1_score, confusion_matrix, precision_score, recall_score)
    from ml.models.st_mmt import STMMT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tile_spec = parse_tile_spec(args.tile_spec)
    tile_cube_ids = list(tile_spec.keys())
    label_pairs = list(zip(tile_cube_ids, args.label_csv))

    ds_val = MultiTileGoldDataset(tile_spec, args.sample_csv, label_pairs,
                                   "val", patch_size=args.patch, augment=False)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=8, shuffle=False)
    print(f"val 샘플: {len(ds_val)}개, 채널: {len(VALID_INDICES)}개")
    print(f"앙상블 체크포인트 {len(args.checkpoint)}개: {args.checkpoint}")

    y_true = None
    all_model_probs = []  # 모델별 [N] 확률 배열들의 리스트

    for ckpt_path in args.checkpoint:
        model = STMMT(
            in_channels=len(VALID_INDICES), d_model=args.d_model,
            n_heads=args.n_heads, n_layers=args.n_layers,
            d_ff=args.d_model * 2, n_stages=args.n_stages, patch_size=4,
        ).to(device)
        sd = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(sd, strict=True)
        model.eval()

        labels_list, probs_list = [], []
        with torch.no_grad():
            for x, y in dl_val:
                x = x.to(device)
                out = model(x)
                probs = torch.softmax(out["last_logits"], dim=1)[:, 1]
                labels_list.append(y.numpy().ravel())
                probs_list.append(probs.cpu().numpy().ravel())
        y_this = np.concatenate(labels_list)
        prob_this = np.concatenate(probs_list)

        if y_true is None:
            y_true = y_this
        else:
            assert np.array_equal(y_true, y_this), "라벨 불일치 - 데이터셋 순서 문제"

        auc_single = roc_auc_score(y_true, prob_this)
        print(f"  {ckpt_path}: 단독 AUC={auc_single:.4f}")
        all_model_probs.append(prob_this)

    # 앙상블: 모델별 확률의 단순 평균
    y_prob_ensemble = np.mean(all_model_probs, axis=0)

    print(f"\n{'='*55}")
    print(f"  앙상블 평가 ({len(args.checkpoint)}개 모델 평균)")
    print(f"{'='*55}")
    auc = roc_auc_score(y_true, y_prob_ensemble)
    ap = average_precision_score(y_true, y_prob_ensemble)
    print(f"AUC(ROC): {auc:.4f}  PR-AUC: {ap:.4f}")

    pos_probs = y_prob_ensemble[y_true == 1]
    threshold = max(0.0, pos_probs.min() - 1e-6)
    pred = (y_prob_ensemble >= threshold).astype(int)
    recall = recall_score(y_true, pred)
    precision = precision_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, pred)

    print(f"\n재현율100% 보장 임계값: {threshold:.6f}")
    print(f"실제 재현율: {recall*100:.2f}%  정밀도: {precision*100:.2f}%")
    print(f"F1(macro): {f1:.4f}")
    print(f"confusion_matrix: {cm.tolist()}")


if __name__ == "__main__":
    main()
