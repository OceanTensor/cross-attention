"""
gold_train_multitile_v20.py — confidence 가중 손실 학습 (자체 학습루프)
════════════════════════════════════════════════════════════════════
gold_train_multitile.py를 베이스로 두 가지를 변경:

1. confidence 가중 손실: bleaching_event의 confidence(사건 확신도, 
   day+0=1.0 근처, 사건 후반부 애매한 날짜=0.4 근처)를 픽셀별 
   손실 가중치로 반영. "애매한 경계일"을 "명확한 사건일"과 
   똑같은 무게로 강제 학습시키지 않도록 함.
   목적: 재현율100% 임계값을 깨는 표본이 대개 이런 애매한 
   경계일일 것이라는 가설 검증 (v13:0.9193, v19:0.8953 대비 개선 시도).

2. ★ Trainer 클래스(블랙박스) 대신 자체 학습루프 사용.
   v18 조사에서 Trainer.fit() 내부 로직이 완전히 투명하지 않고
   재현 안 되는 결과를 낸 전례가 있어(부록 13.5 참고), 
   confidence라는 새로운 정보를 안전하게 통과시키기 위해서라도
   이번엔 처음부터 우리가 전체를 통제하는 루프로 작성한다.
   LR 스케줄은 기존 로그(9.97e-05→...→여러 에폭에 걸쳐 감소)의
   패턴을 참고해 CosineAnnealingLR로 근사했다(완전히 동일한 
   스케줄이라는 보장은 없음, 정직하게 명시).

사용:
  uv run gold_train_multitile_v20.py \
      --tile_spec 128:data/wando_..._nirchg.zarr \
      --tile_spec 135:data/mokpo_..._nirchg.zarr \
      --sample_csv h100_transfer_v19_wando/gold_sample.csv \
      --sample_csv h100_transfer_v19_mokpo/gold_sample.csv \
      --label_csv h100_transfer_v19_wando/gold_label.csv \
      --label_csv h100_transfer_v19_mokpo/gold_label.csv \
      --epochs 30 --batch_size 8 --class_weights 0.5 1.5 \
      --save_dir checkpoints/v20
"""
import argparse, csv, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from silver_channels import CHANNEL_ORDER

EMPTY_CHANNELS = set()
VALID_CHANNELS = [ch for ch in CHANNEL_ORDER if ch not in EMPTY_CHANNELS]
VALID_INDICES = [int(ch[2:]) for ch in VALID_CHANNELS]


def _to_bool(v):
    return str(v).strip().lower() in ("true", "t", "1", "yes")


def load_samples_csv_multi(paths, split):
    rows = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["split"] != split:
                    continue
                rows.append({
                    "sample_id": int(r["sample_id"]),
                    "cube_id": int(r["cube_id"]),
                    "input_start_frame": int(r["input_start_frame"]),
                    "input_len": int(r["input_len"]),
                    "target_frame": int(r["target_frame"]),
                    "target_date": r["target_date"],
                    "has_event": _to_bool(r["has_event"]),
                    "severity": int(r["severity"]) if r["severity"] not in ("", None) else 0,
                })
    return rows


def load_labels_csv_multi_conf(cube_id_path_pairs):
    """(cube_id, frame_idx) -> (has_event_int, confidence) 튜플로 저장.
    ★ 신규: confidence 컬럼도 함께 읽음. has_event=False인 날은
    '음성이라는 확신도 100%'로 보고 confidence=1.0 고정."""
    out = {}
    for cube_id, path in cube_id_path_pairs:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                has_event = _to_bool(r["has_event"])
                frame_idx = int(r["frame_idx"])
                if has_event:
                    conf_raw = r.get("confidence", "")
                    conf = float(conf_raw) if conf_raw not in ("", None) else 1.0
                else:
                    conf = 1.0
                out[(cube_id, frame_idx)] = (1 if has_event else 0, conf)
    return out


class MultiTileGoldDatasetConf:
    """confidence까지 함께 반환하는 Dataset. (x, y, conf) 3-tuple 반환."""

    def __init__(self, tile_spec, sample_csvs, label_csvs, split,
                 patch_size=64, augment=True, seed=42):
        self.cube_paths = tile_spec
        self.patch = patch_size
        self.augment = augment

        self.samples = load_samples_csv_multi(sample_csvs, split)
        if not self.samples:
            raise ValueError(f"샘플 없음: split={split}")
        self.frame_info = load_labels_csv_multi_conf(label_csvs)

        used_cube_ids = set(s["cube_id"] for s in self.samples)
        missing = used_cube_ids - set(self.cube_paths.keys())
        if missing:
            raise ValueError(f"tile_spec에 없는 cube_id 참조됨: {missing}")

        import zarr
        self._zarrs = {}
        for cube_id, path in self.cube_paths.items():
            z = zarr.open(path, mode="r")
            self._zarrs[cube_id] = z

        rng = np.random.default_rng(seed)
        H, W = next(iter(self._zarrs.values())).shape[2:4]
        self.H, self.W = H, W
        self.anchors = [
            (int(rng.integers(0, max(1, H - self.patch + 1))),
             int(rng.integers(0, max(1, W - self.patch + 1))))
            for _ in self.samples
        ]

        from collections import Counter
        cnt = Counter(s["cube_id"] for s in self.samples)
        print(f"    지역별({split}) 샘플 분포: {dict(cnt)}")

        confs = [self.frame_info.get((s["cube_id"], s["target_frame"]), (0, 1.0))[1]
                 for s in self.samples if s["has_event"]]
        if confs:
            print(f"    ({split}) 양성표본 confidence: "
                  f"min={min(confs):.2f} max={max(confs):.2f} 평균={sum(confs)/len(confs):.2f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torch
        s = self.samples[idx]
        cube_id = s["cube_id"]
        z = self._zarrs[cube_id]
        t0, t_in = s["input_start_frame"], s["input_len"]
        rh, rw = self.anchors[idx]
        ph = pw = self.patch
        x = np.asarray(z[t0:t0+t_in, :, rh:rh+ph, rw:rw+pw])
        x = x[:, VALID_INDICES, :, :]
        x = np.nan_to_num(x, nan=0.0).astype(np.float32)
        x = torch.from_numpy(x)

        sev, conf = self.frame_info.get((cube_id, s["target_frame"]), (0, 1.0))
        y = torch.full((ph, pw), sev, dtype=torch.int64)
        conf_map = torch.full((ph, pw), conf, dtype=torch.float32)

        if self.augment and torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[3])
            y = torch.flip(y, dims=[1])
            conf_map = torch.flip(conf_map, dims=[1])
        return x, y, conf_map


def parse_tile_spec(spec_list):
    out = {}
    for spec in spec_list:
        cube_id_str, path = spec.split(":", 1)
        out[int(cube_id_str)] = path
    return out


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
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--save_dir", type=str, default="checkpoints/v20")
    ap.add_argument("--class_weights", type=float, nargs=2, default=[0.5, 1.5])
    ap.add_argument("--n_stages", type=int, default=2)
    ap.add_argument("--patience", type=int, default=10)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from ml.models.st_mmt import STMMT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tile_spec = parse_tile_spec(args.tile_spec)

    print("="*60)
    print(f"  Gold STMMT confidence가중 학습(v20) — {len(tile_spec)}개 지역, 자체루프")
    print(f"  타일: {tile_spec}")
    print(f"  device={device}" + (f" ({torch.cuda.get_device_name(0)})" if device=="cuda" else ""))
    print("="*60)

    tile_cube_ids = list(tile_spec.keys())
    if len(tile_cube_ids) != len(args.label_csv):
        raise ValueError("--tile_spec 개수와 --label_csv 개수가 다름 — 순서대로 1:1 대응 필요")
    label_pairs = list(zip(tile_cube_ids, args.label_csv))
    print(f"  label_csv 매칭: {label_pairs}")

    ds_train = MultiTileGoldDatasetConf(tile_spec, args.sample_csv, label_pairs,
                                         "train", patch_size=args.patch, augment=True)
    ds_val = MultiTileGoldDatasetConf(tile_spec, args.sample_csv, label_pairs,
                                       "val", patch_size=args.patch, augment=False)
    print(f"  train: {len(ds_train)}샘플 / val: {len(ds_val)}샘플")
    print(f"  patch: {args.patch}x{args.patch}, 채널: {len(VALID_INDICES)}개")

    model = STMMT(
        in_channels=len(VALID_INDICES), d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_model*2, n_stages=args.n_stages, patch_size=4,
    ).to(device)
    print(f"  파라미터: {model.count_params():,}")

    cw_tensor = torch.tensor(args.class_weights, dtype=torch.float32, device=device)
    print(f"  class_weights: {args.class_weights}, confidence 가중 손실 적용")

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_path = os.path.join(args.save_dir, "best_model.pt")
    patience_counter = 0

    print("학습 시작 (confidence 가중, 자체 학습루프)")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y, conf in dl_train:
            x, y, conf = x.to(device), y.to(device), conf.to(device)
            optimizer.zero_grad()
            out = model(x)
            logits = out["last_logits"]  # (B, n_stages, H, W)
            # ★ confidence 가중 CrossEntropy: reduction='none'으로 픽셀별 손실을 낸 뒤
            # confidence를 곱해서 평균 (애매한 표본은 손실 기여도가 줄어듦)
            ce_per_pixel = F.cross_entropy(logits, y, weight=cw_tensor, reduction="none")
            ce_loss = (ce_per_pixel * conf).sum() / conf.sum().clamp(min=1e-6)
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
            for x, y, conf in dl_val:
                x, y = x.to(device), y.to(device)
                out = model(x)
                logits = out["last_logits"]
                # val loss는 confidence 가중 없이 순수 평가(공정한 비교 기준 유지)
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

    # 최종 평가 (기존 스크립트와 동일한 이진 AUC/PR-AUC 계산)
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()
    all_labels, all_probs_pos = [], []
    with torch.no_grad():
        for x, y, conf in dl_val:
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
