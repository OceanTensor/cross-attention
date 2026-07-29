"""
gold_compare_checkpoints.py — 여러 체크포인트를 동일 조건에서 재평가·비교
════════════════════════════════════════════════════════════════════
trainer.fit()이 반환한 report가 체크포인트 간 의심스럽게 동일했던 문제를
검증하기 위해, 각 체크포인트를 독립적으로 새로 로드해서
확률 기반(AUC/PR-AUC) + 클래스 예측을 직접 재계산한다.

사용:
  uv run gold_compare_checkpoints.py \
      --cube_path data/wando_30m_full32_2021_ffilled.zarr \
      --sample_csv h100_transfer/gold_sample.csv \
      --label_csv h100_transfer/gold_label.csv \
      --checkpoints checkpoints/gold_v2_binary/best_model.pt checkpoints/gold_v3_balanced/best_model.pt
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gold_train_stmmt_h100 import OceanTensorGoldDatasetH100, VALID_INDICES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube_path", required=True)
    ap.add_argument("--sample_csv", required=True)
    ap.add_argument("--label_csv", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=8)
    args = ap.parse_args()

    import torch
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
    from ml.models.st_mmt import STMMT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds_val = OceanTensorGoldDatasetH100(args.cube_path, args.sample_csv, args.label_csv,
                                          "val", patch_size=args.patch, augment=False)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=8, shuffle=False)
    print(f"val 샘플: {len(ds_val)}개\n")

    for ckpt_path in args.checkpoints:
        model = STMMT(in_channels=len(VALID_INDICES), d_model=args.d_model,
                       n_heads=args.n_heads, n_layers=args.n_layers,
                       d_ff=args.d_model*2, n_stages=2, patch_size=4).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        # 로드 검증: 가중치 일부 체크섬 (진짜 다른 가중치인지 재확인)
        first_param = next(iter(model.state_dict().values())).flatten()[:5]

        all_labels, all_probs, all_preds = [], [], []
        with torch.no_grad():
            for x, y in dl_val:
                x = x.to(device)
                out = model(x)
                probs = torch.softmax(out["last_logits"], dim=1)
                pred = probs.argmax(dim=1)
                all_labels.append(y.numpy().ravel())
                all_probs.append(probs[:, 1].cpu().numpy().ravel())
                all_preds.append(pred.cpu().numpy().ravel())

        y_true = np.concatenate(all_labels)
        y_prob = np.concatenate(all_probs)
        y_pred = np.concatenate(all_preds)

        auc = roc_auc_score(y_true, y_prob)
        ap_score = average_precision_score(y_true, y_prob)
        f1 = f1_score(y_true, y_pred, average="macro")
        cm = confusion_matrix(y_true, y_pred)

        print(f"=== {ckpt_path} ===")
        print(f"  첫 파라미터 샘플(가중치 확인용): {first_param.tolist()}")
        print(f"  AUC(ROC): {auc:.4f}  PR-AUC: {ap_score:.4f}  F1(macro): {f1:.4f}")
        print(f"  y_prob 평균: {y_prob.mean():.4f}, 표준편차: {y_prob.std():.4f}")
        print(f"  confusion_matrix:\n{cm}")
        print()


if __name__ == "__main__":
    main()
