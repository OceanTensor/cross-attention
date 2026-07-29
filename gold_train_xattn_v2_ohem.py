"""
gold_train_xattn_v2_ohem.py — xattn_v2를 Hard Negative 가중샘플링으로 재학습
════════════════════════════════════════════════════════════════════
gold_identify_hard_negatives.py가 찾은 "train 내 어려운 정상 샘플"에
WeightedRandomSampler로 더 자주 노출시켜 재학습한다. 팀원의 Trainer
클래스는 커스텀 샘플러를 지원하는지 확인되지 않아, v20과 동일하게
자체 학습루프를 사용해 완전한 제어권을 확보한다.

사용:
  uv run gold_train_xattn_v2_ohem.py \
      --tile_spec 128:data/wando_..._nirchg.zarr \
      --tile_spec 135:data/mokpo_..._nirchg.zarr \
      --sample_csv h100_transfer_v19_wando/gold_sample.csv \
      --sample_csv h100_transfer_v19_mokpo/gold_sample.csv \
      --label_csv h100_transfer_v19_wando/gold_label.csv \
      --label_csv h100_transfer_v19_mokpo/gold_label.csv \
      --hard_neg_weights hard_negative_weights.json \
      --epochs 30 --batch_size 8 --class_weights 0.5 1.5 \
      --save_dir checkpoints/xattn_v2_ohem
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gold_train_multitile import MultiTileGoldDataset, VALID_INDICES, parse_tile_spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile_spec", action="append", required=True)
    ap.add_argument("--sample_csv", action="append", required=True)
    ap.add_argument("--label_csv", action="append", required=True)
    ap.add_argument("--hard_neg_weights", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--save_dir", type=str, default="checkpoints/xattn_v2_ohem")
    ap.add_argument("--class_weights", type=float, nargs=2, default=[0.5, 1.5])
    ap.add_argument("--n_stages", type=int, default=2)
    ap.add_argument("--patience", type=int, default=10)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from st_mmt_xattn_v2 import STMMTCrossAttnV2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tile_spec = parse_tile_spec(args.tile_spec)

    print("="*60)
    print(f"  xattn_v2 Hard Negative 가중샘플링 재학습(OHEM) — 자체루프")
    print(f"  device={device}" + (f" ({torch.cuda.get_device_name(0)})" if device=="cuda" else ""))
    print("="*60)

    tile_cube_ids = list(tile_spec.keys())
    label_pairs = list(zip(tile_cube_ids, args.label_csv))

    ds_train = MultiTileGoldDataset(tile_spec, args.sample_csv, label_pairs,
                                     "train", patch_size=args.patch, augment=True)
    ds_val = MultiTileGoldDataset(tile_spec, args.sample_csv, label_pairs,
                                   "val", patch_size=args.patch, augment=False)
    print(f"  train: {len(ds_train)}샘플 / val: {len(ds_val)}샘플")

    with open(args.hard_neg_weights) as f:
        hn_data = json.load(f)
    weights = hn_data["weights"]
    assert len(weights) == len(ds_train), \
        f"가중치 개수({len(weights)})와 train 샘플 수({len(ds_train)}) 불일치"
    n_hard = hn_data["n_hard_negatives"]
    print(f"  Hard Negative 가중샘플링 적용: {n_hard}개 샘플에 {hn_data['hard_weight']}배 가중치")

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights, num_samples=len(ds_train), replacement=True)

    model = STMMTCrossAttnV2(d_model=args.d_model, n_heads=args.n_heads,
                              n_layers=2, n_stages=args.n_stages, patch_size=4).to(device)
    print(f"  파라미터: {model.count_params():,}")

    cw_tensor = torch.tensor(args.class_weights, dtype=torch.float32, device=device)

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_path = os.path.join(args.save_dir, "best_model.pt")
    patience_counter = 0

    print("학습 시작 (Hard Negative 가중샘플링, 자체 학습루프)")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            logits = out["last_logits"]
            ce_loss = F.cross_entropy(logits, y, weight=cw_tensor)
            dy = torch.abs(logits[:, :, 1:, :] - logits[:, :, :-1, :]).mean()
            dx = torch.abs(logits[:, :, :, 1:] - logits[:, :, :, :-1]).mean()
            spatial_loss = dy + dx
            loss = ce_loss + model.spatial_loss_weight * spatial_loss
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
                loss = F.cross_entropy(logits, y, weight=cw_tensor)
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
