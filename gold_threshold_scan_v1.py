"""
gold_threshold_scan_v1.py — v1.0 계열(v1~v20) 체크포인트용 임계값 스캔
════════════════════════════════════════════════════════════════════
gold_train_stmmt_h100.py가 사용하는 것과 동일한 모델(ml.models.st_mmt,
단일 헤드 last_logits, n_stages 이진분류)을 그대로 재사용한다.

(gold_threshold_scan_v15.py는 v13 전이학습 계열(3헤드 adi/warn/severe,
33채널) 전용이라 이 프로젝트의 주력 계열(v1~v20)과는 다른 모델이다 —
이번 스캔은 v1~v20 계열 전용으로 새로 작성.)

사용:
  uv run gold_threshold_scan_v1.py \
      --cube_path data/wando_30m_full32_2021_ffilled_ma_mld_nir.zarr \
      --sample_csv h100_transfer_v4/gold_sample.csv \
      --label_csv h100_transfer_v4/gold_label.csv \
      --checkpoint checkpoints/v6/best_model.pt
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
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_stages", type=int, default=2)
    args = ap.parse_args()

    import torch
    from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_auc_score, average_precision_score
    from ml.models.st_mmt import STMMT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds_val = OceanTensorGoldDatasetH100(args.cube_path, args.sample_csv, args.label_csv,
                                          "val", patch_size=args.patch, augment=False)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=8, shuffle=False)
    print(f"val 샘플: {len(ds_val)}개, 채널: {len(VALID_INDICES)}개")

    model = STMMT(
        in_channels=len(VALID_INDICES), d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_model * 2, n_stages=args.n_stages, patch_size=4,
    ).to(device)
    sd = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"체크포인트 로드 완료: {args.checkpoint}")

    all_true, all_prob = [], []
    with torch.no_grad():
        for x, y in dl_val:
            x = x.to(device)
            out = model(x)
            probs = torch.softmax(out["last_logits"], dim=1)[:, 1]  # 양성(경계) 확률
            all_true.append(y.numpy().ravel())
            all_prob.append(probs.cpu().numpy().ravel())

    y_true = np.concatenate(all_true)
    y_prob = np.concatenate(all_prob)

    print(f"\n확률 분포: min={y_prob.min():.4f} max={y_prob.max():.4f} "
          f"평균={y_prob.mean():.4f} 중앙값={np.median(y_prob):.4f}")
    print(f"양성 비율(실제): {y_true.mean()*100:.1f}%")

    try:
        auc = roc_auc_score(y_true, y_prob)
        ap_score = average_precision_score(y_true, y_prob)
        print(f"AUC(ROC): {auc:.4f}  PR-AUC: {ap_score:.4f}")
    except ValueError as e:
        print(f"AUC 계산 실패: {e}")

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s[:-1])
    best_thresh = thresholds[best_idx]
    print(f"\n최적 임계값(F1 최대): {best_thresh:.4f} (F1={f1s[best_idx]:.4f})")

    print(f"\n{'='*50}")
    print(f"  임계값별 성능 비교")
    print(f"{'='*50}")
    for th in sorted(set([0.5, 0.3, 0.2, 0.1, float(best_thresh)])):
        pred = (y_prob >= th).astype(int)
        f1 = f1_score(y_true, pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_true, pred)
        label = " ← 최적" if abs(th - best_thresh) < 1e-6 else ""
        print(f"  threshold={th:.4f}: F1(macro)={f1:.4f}, confusion_matrix={cm.tolist()}{label}")


if __name__ == "__main__":
    main()
