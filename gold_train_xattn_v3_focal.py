"""
gold_train_xattn_v3_focal.py — xattn_v3(잔차구조)를 Focal Loss로 재학습
════════════════════════════════════════════════════════════════════
Cost-sensitive Focal Loss: 이미 잘 맞추는 "쉬운" 샘플의 손실 기여도를
크게 낮추고, 여전히 틀리는 "어려운" 샘플에 학습을 집중시킨다.
class_weights(0.5,1.5)와 결합해 사용(alpha 역할은 class_weights가 담당).

FL(pt) = -alpha_t * (1-pt)^gamma * log(pt)

사용:
  uv run gold_train_xattn_v3_focal.py \
      --tile_spec 128:data/wando_..._nirchg.zarr \
      --tile_spec 135:data/mokpo_..._nirchg.zarr \
      --sample_csv h100_transfer_v19_wando/gold_sample.csv \
      --sample_csv h100_transfer_v19_mokpo/gold_sample.csv \
      --label_csv h100_transfer_v19_wando/gold_label.csv \
      --label_csv h100_transfer_v19_mokpo/gold_label.csv \
      --epochs 30 --batch_size 8 --class_weights 0.5 1.5 --gamma 2.0 \
      --save_dir checkpoints/xattn_v3_focal
"""
import argparse, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gold_train_multitile import MultiTileGoldDataset, VALID_INDICES, parse_tile_spec


def focal_loss(logits, targets, weight, gamma=2.0):
    """다중클래스 Focal Loss. logits:(B,n_stages,H,W), targets:(B,H,W)"""
    import torch.nn.functional as F
    ce_per_pixel = F.cross_entropy(logits, targets, weight=weight, reduction="none")
    pt = (-ce_per_pixel).exp()
    focal = ((1 - pt) ** gamma) * ce_per_pixel
    return focal.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile_spec", action="append", required=True)
    ap.add_argument("--sample_csv", action="append", required=True)
    ap.add_argument("--label_csv", action="append", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--save_dir", type=str, default="checkpoints/xattn_v3_focal")
    ap.add_argument("--class_weights", type=float, nargs=2, default=[0.5, 1.5])
    ap.add_argument("--n_stages", type=int, default=2)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=2.0, help="Focal Loss 감쇠 지수")
    args = ap.parse_args()

    import torch
    from st_mmt_xattn_v3 import STMMTCrossAttnV3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tile_spec = parse_tile_spec(args.tile_spec)

    print("="*60)
    print(f"  xattn_v3 Focal Loss 재학습(gamma={args.gamma}) — 자체루프")
    print(f"  device={device}" + (f" ({torch.cuda.get_device_name(0)})" if device=="cuda" else ""))
    print("="*60)

    tile_cube_ids = list(tile_spec.keys())
    label_pairs = list(zip(tile_cube_ids, args.label_csv))

    ds_train = MultiTileGoldDataset(tile_spec, args.sample_csv, label_pairs,
                                     "train", patch_size=args.patch, augment=True)
    ds_val = MultiTileGoldDataset(tile_spec, args.sample_csv, label_pairs,
                                   "val", patch_size=args.patch, augment=False)
    print(f"  train: {len(ds_train)}샘플 / val: {len(ds_val)}샘플")

    model = STMMTCrossAttnV3(d_model=args.d_model, n_heads=args.n_heads,
                              n_layers_joint=4, n_layers_group=2,
                              n_stages=args.n_stages, patch_size=4).to(device)
    print(f"  파라미터: {model.count_params():,}")

    cw_tensor = torch.tensor(args.class_weights, dtype=torch.float32, device=device)
    print(f"  class_weights(alpha): {args.class_weights}, Focal gamma: {args.gamma}")

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_path = os.path.join(args.save_dir, "best_model.pt")
    patience_counter = 0

    print("학습 시작 (Focal Loss, 자체 학습루프)")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            logits = out["last_logits"]
            fl = focal_loss(logits, y, cw_tensor, gamma=args.gamma)
            dy = torch.abs(logits[:, :, 1:, :] - logits[:, :, :-1, :]).mean()
            dx = torch.abs(logits[:, :, :, 1:] - logits[:, :, :, :-1]).mean()
            spatial_loss = dy + dx
            loss = fl + model.spatial_loss_weight * spatial_loss
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = sum(train_losses) / len(train_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in dl_val:
                x, y = x.to(device), y.to(device)
                out = model(x)
                logits = out["last_logits"]
                loss = focal_loss(logits, y, cw_tensor, gamma=args.gamma)
                val_losses.append(loss.item())
        val_loss = sum(val_losses) / len(val_losses)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d}/{args.epochs} | train={train_loss:.4f} | "
              f"val={val_loss:.4f} | lr={lr_now:.2e}")
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\n  ✓ 완료 — best_path={best_path}")

    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()
    all_labels, all_probs_pos = [], []
    with torch.no_grad():
        for x, y in dl_val:
            x = x.to(device)
            out = model(x)
            probs = torch.softmax(out["last_logits"], dim=1)[:, 1]
            all_labels.append(y.numpy().ravel())
            all_probs_pos.append(probs.cpu().numpy().ravel())
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs_pos)
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    print(f"  binary AUC(ROC): {auc:.4f}  PR-AUC(AP): {ap:.4f}")

    pos_probs = y_prob[y_true == 1]
    threshold = max(0.0, pos_probs.min() - 1e-6)
    pred = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_true, pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, pred)
    recall = (pred[y_true == 1] == 1).mean()
    precision = pred[pred == 1].size and (y_true[pred == 1] == 1).mean() or 0.0
    print(f"\n  재현율100% 임계값: {threshold:.6f}")
    print(f"  실제 재현율: {recall*100:.2f}%  정밀도: {precision*100:.2f}%")
    print(f"  F1(macro): {f1:.4f}")
    print(f"  confusion_matrix: {cm.tolist()}")


if __name__ == "__main__":
    main()
