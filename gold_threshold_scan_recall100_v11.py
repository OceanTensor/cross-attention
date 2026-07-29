"""
gold_threshold_scan_recall100.py — 재현율 100% 보장 임계값 탐색
════════════════════════════════════════════════════════════════════
gold_threshold_scan_v1.py는 F1을 최대화하는 임계값을 찾았으나,
이 프로젝트의 원칙(조기경보 = 사건을 놓치지 않는 것이 최우선)과
맞지 않을 수 있다 — v7에서 F1 최적 임계값이 재현율을 94.6%로
떨어뜨린 사례가 실제로 발생했다.

이 스크립트는 순서를 뒤집는다:
  1순위: 재현율 100%(양성을 하나도 놓치지 않음)를 만족하는 임계값만 후보로 삼는다
  2순위: 그 후보들 중에서 F1(또는 정밀도)이 가장 높은 임계값을 선택한다

이게 v2~v6까지 우리가 실제로 지켜온 원칙과 일치하며,
v7 이후 모든 버전에 동일하게 적용해 일관된 비교표를 만드는 데 쓴다.

사용:
  uv run gold_threshold_scan_recall100.py \
      --cube_path data/wando_30m_full32_2021_ffilled_ma_mld_nir_anom.zarr \
      --sample_csv h100_transfer_v5/gold_sample.csv \
      --label_csv h100_transfer_v5/gold_label.csv \
      --checkpoint checkpoints/v7b/best_model.pt
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gold_train_stmmt_h100_v11 import OceanTensorGoldDatasetH100, VALID_INDICES


def find_recall100_threshold(y_true, y_prob):
    """재현율 100%를 만족하는 임계값 중 가장 높은(정밀도가 가장 좋아지는) 것을 찾는다.
    양성(경계=1) 클래스의 확률 중 최솟값보다 살짝 낮은 지점이
    이론적으로 재현율 100%를 만족하는 최대 임계값이다."""
    pos_probs = y_prob[y_true == 1]
    if len(pos_probs) == 0:
        return 0.5, None
    # 양성 샘플 중 확률이 가장 낮은 값 = 이 값 이하로 임계값을 내려야 그 샘플도 잡힘
    min_pos_prob = pos_probs.min()
    # 부동소수점 경계 문제 방지용 아주 작은 여유
    threshold = max(0.0, min_pos_prob - 1e-6)
    return threshold, min_pos_prob


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
    ap.add_argument("--min_recall", type=float, default=1.0,
                     help="보장할 최소 재현율 (기본 1.0 = 100%%)")
    args = ap.parse_args()

    import torch
    from sklearn.metrics import (f1_score, confusion_matrix, roc_auc_score,
                                   average_precision_score, precision_score, recall_score)
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
          f"평균={y_prob.mean():.4f}")
    print(f"양성 비율(실제): {y_true.mean()*100:.1f}%")

    try:
        auc = roc_auc_score(y_true, y_prob)
        ap_score = average_precision_score(y_true, y_prob)
        print(f"AUC(ROC): {auc:.4f}  PR-AUC: {ap_score:.4f}")
    except ValueError as e:
        print(f"AUC 계산 실패: {e}")

    # ★ 재현율 100% 보장 임계값 (원칙: 안전 최우선)
    recall100_th, min_pos_prob = find_recall100_threshold(y_true, y_prob)
    print(f"\n{'='*55}")
    print(f"  재현율 100% 보장 임계값: {recall100_th:.6f}")
    print(f"  (양성 샘플 중 최저 확률 = {min_pos_prob:.6f})")
    print(f"{'='*55}")

    pred_r100 = (y_prob >= recall100_th).astype(int)
    recall_check = recall_score(y_true, pred_r100)
    precision_check = precision_score(y_true, pred_r100, zero_division=0)
    f1_r100 = f1_score(y_true, pred_r100, average="macro", zero_division=0)
    cm_r100 = confusion_matrix(y_true, pred_r100)
    print(f"  실제 재현율: {recall_check*100:.2f}% (검증: {'✓ 100%' if recall_check==1.0 else '✗ 미달'})")
    print(f"  정밀도: {precision_check*100:.2f}%")
    print(f"  F1(macro): {f1_r100:.4f}")
    print(f"  confusion_matrix: {cm_r100.tolist()}")

    # 참고 — F1 최대화 임계값과 비교 (대조용)
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s[:-1])
    f1_best_th = thresholds[best_idx]
    pred_f1best = (y_prob >= f1_best_th).astype(int)
    recall_f1best = recall_score(y_true, pred_f1best)

    print(f"\n  [대조] F1 최대화 임계값: {f1_best_th:.4f} "
          f"(F1={f1s[best_idx]:.4f}, 재현율={recall_f1best*100:.1f}%)")
    if recall_f1best < 1.0:
        print(f"  ⚠ F1 최대화 임계값은 재현율 100% 미달 — 이 프로젝트 원칙상 채택 안 함")
    print(f"\n  ★ 최종 채택(재현율 우선): threshold={recall100_th:.6f}, F1={f1_r100:.4f}")


if __name__ == "__main__":
    main()
