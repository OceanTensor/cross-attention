"""
gold_threshold_scan_v15.py — 저장된 v15 체크포인트로 최적 임계값 탐색
════════════════════════════════════════════════════════════════════
재학습 없이 best_model.pt(epoch1, AUC 0.9563)를 로드해
0.5 고정 대신 val PR곡선에서 F1을 최대화하는 임계값을 찾는다.

사용:
  uv run gold_threshold_scan_v15.py \
      --cube_path data/wando_30m_full32_2021_ffilled_ma.zarr \
      --sample_csv h100_transfer_v2/gold_sample.csv \
      --label_csv h100_transfer_v2/gold_label.csv \
      --checkpoint checkpoints/v15/best_model.pt
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gold_finetune_v15 import Full32ChannelDataset, N_CHANNELS, T_OUT, WARN_THRESH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube_path", required=True)
    ap.add_argument("--sample_csv", required=True)
    ap.add_argument("--label_csv", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--patch", type=int, default=64)
    args = ap.parse_args()

    import torch
    from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve
    from ml_v15.models.st_mmt import STMMT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds_val = Full32ChannelDataset(args.cube_path, args.sample_csv, args.label_csv,
                                    "val", patch_size=args.patch, augment=False)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=8, shuffle=False)
    print(f"val 샘플: {len(ds_val)}개")

    model = STMMT(in_channels=N_CHANNELS, d_model=256, n_heads=8, n_layers=4,
                   d_ff=512, patch_size=4, t_out=T_OUT).to(device)
    sd = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"체크포인트 로드 완료: {args.checkpoint}")

    all_true, all_prob = [], []
    with torch.no_grad():
        for x, adi_target, sev in dl_val:
            x = x.to(device)
            out = model(x)
            prob = torch.sigmoid(out["warn_logit"]).cpu().numpy().ravel()
            pix_max = adi_target.amax(dim=1)
            Hf, Wf = out["warn_logit"].shape[1], out["warn_logit"].shape[2]
            if (Hf, Wf) != tuple(pix_max.shape[1:]):
                pix_max = torch.nn.functional.adaptive_max_pool2d(pix_max.unsqueeze(1), (Hf, Wf)).squeeze(1)
            y_true = (pix_max >= WARN_THRESH).float().numpy().ravel()
            all_true.append(y_true)
            all_prob.append(prob)

    y_true = np.concatenate(all_true)
    y_prob = np.concatenate(all_prob)

    print(f"\n확률 분포: min={y_prob.min():.4f} max={y_prob.max():.4f} "
          f"평균={y_prob.mean():.4f} 중앙값={np.median(y_prob):.4f}")
    print(f"양성 비율(실제): {y_true.mean()*100:.1f}%")

    # PR 곡선에서 F1 최대 임계값 탐색
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s[:-1])  # 마지막은 threshold 없음
    best_thresh = thresholds[best_idx]
    print(f"\n최적 임계값(F1 최대): {best_thresh:.4f} (F1={f1s[best_idx]:.4f})")

    print(f"\n{'='*50}")
    print(f"  임계값별 성능 비교")
    print(f"{'='*50}")
    for th in [0.5, 0.3, 0.2, 0.1, best_thresh]:
        pred = (y_prob >= th).astype(int)
        f1 = f1_score(y_true, pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_true, pred)
        label = " ← 최적" if abs(th - best_thresh) < 1e-6 else ""
        print(f"  threshold={th:.4f}: F1(macro)={f1:.4f}, confusion_matrix={cm.tolist()}{label}")


if __name__ == "__main__":
    main()
