"""
gold_threshold_scan_recall100_multitile.py — 다중타일 재현율 100% 임계값 탐색
════════════════════════════════════════════════════════════════════
gold_train_multitile.py의 MultiTileGoldDataset을 그대로 재사용해
v13(완도+목포 통합) 체크포인트를 재현율 100% 원칙으로 재평가.

사용:
  uv run gold_threshold_scan_recall100_multitile.py \
      --tile_spec 32:data/wando_30m_full32_2021_ffilled_ma_mld_nir_anom.zarr \
      --tile_spec 53:data/mokpo_30m_full32_2021_ffilled_nir_turb_ma_anom_nirchg.zarr \
      --sample_csv h100_transfer_v13_wando/gold_sample.csv \
      --sample_csv h100_transfer_mokpo/gold_sample.csv \
      --label_csv h100_transfer_v13_wando/gold_label.csv \
      --label_csv h100_transfer_mokpo/gold_label.csv \
      --checkpoint checkpoints/v13/best_model.pt
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gold_train_multitile import MultiTileGoldDataset, VALID_INDICES, parse_tile_spec
from st_mmt_xattn_v2 import STMMTCrossAttnV2


def find_recall100_threshold(y_true, y_prob):
    """양성 샘플 중 최저 확률보다 살짝 낮은 지점 = 재현율 100%를 보장하는 임계값."""
    pos_probs = y_prob[y_true == 1]
    if len(pos_probs) == 0:
        return 0.5, None
    min_pos_prob = pos_probs.min()
    threshold = max(0.0, min_pos_prob - 1e-6)
    return threshold, min_pos_prob


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
    from sklearn.metrics import (f1_score, confusion_matrix, roc_auc_score,
                                   average_precision_score, precision_score,
                                   recall_score, precision_recall_curve)
    from ml.models.st_mmt import STMMT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tile_spec = parse_tile_spec(args.tile_spec)

    tile_cube_ids = list(tile_spec.keys())
    if len(tile_cube_ids) != len(args.label_csv):
        raise ValueError(f"--tile_spec 개수와 --label_csv 개수가 다름 — "
                          f"같은 순서로 1:1 대응해야 함")
    label_pairs = list(zip(tile_cube_ids, args.label_csv))

    ds_val = MultiTileGoldDataset(tile_spec, args.sample_csv, label_pairs,
                                   "val", patch_size=args.patch, augment=False)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=8, shuffle=False)
    print(f"val 샘플: {len(ds_val)}개, 채널: {len(VALID_INDICES)}개")

    model = STMMTCrossAttnV2(
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=2, n_stages=args.n_stages, patch_size=4,
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
            probs = torch.softmax(out["last_logits"], dim=1)[:, 1]
            all_true.append(y.numpy().ravel())
            all_prob.append(probs.cpu().numpy().ravel())

    y_true = np.concatenate(all_true)
    y_prob = np.concatenate(all_prob)

    print(f"\n확률 분포: min={y_prob.min():.4f} max={y_prob.max():.4f} 평균={y_prob.mean():.4f}")
    print(f"양성 비율(실제): {y_true.mean()*100:.1f}%")

    try:
        auc = roc_auc_score(y_true, y_prob)
        ap_score = average_precision_score(y_true, y_prob)
        print(f"AUC(ROC): {auc:.4f}  PR-AUC: {ap_score:.4f}")
    except ValueError as e:
        print(f"AUC 계산 실패: {e}")

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
