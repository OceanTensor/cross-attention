"""
gold_train_multitile.py — 여러 타일(지역)을 하나의 학습으로 통합 (v13)
════════════════════════════════════════════════════════════════════
gold_train_stmmt_h100.py의 검증된 학습 루프(Trainer, 클래스가중치 몽키패치,
이진 AUC 보완계산)는 그대로 유지하고, Dataset 부분만 다중타일 지원으로 교체.

기존 코드의 두 가지 버그를 수정:
  1. load_samples_csv가 CSV의 cube_id 컬럼을 무시함 → 이제 읽어서 보존
  2. load_labels_csv가 frame_idx만으로 키를 만들어 서로 다른 지역의
     같은 날짜지수(frame_idx)가 충돌함 → (cube_id, frame_idx) 튜플로 변경

사용 (완도+목포 통합):
  uv run gold_train_multitile.py \
      --tile_spec 32:data/wando_30m_full32_2021_ffilled_ma_mld_nir_anom.zarr \
      --tile_spec 53:data/mokpo_30m_full32_2021_ffilled_nir_turb_ma_anom_nirchg.zarr \
      --sample_csv h100_transfer_v7/gold_sample.csv \
      --sample_csv h100_transfer_mokpo/gold_sample.csv \
      --label_csv h100_transfer_v7/gold_label.csv \
      --label_csv h100_transfer_mokpo/gold_label.csv \
      --epochs 30 --batch_size 8 --class_weights 0.5 1.5 \
      --save_dir checkpoints/v13
"""
import argparse, csv, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from silver_channels import CHANNEL_ORDER

EMPTY_CHANNELS = set()  # 32/32 완전체 기준. 지역마다 빈 채널 다르면 조정 필요.
VALID_CHANNELS = [ch for ch in CHANNEL_ORDER if ch not in EMPTY_CHANNELS]
VALID_INDICES = [int(ch[2:]) for ch in VALID_CHANNELS]


def _to_bool(v):
    return str(v).strip().lower() in ("true", "t", "1", "yes")


def load_samples_csv_multi(paths, split):
    """여러 sample_csv를 이어붙임. cube_id를 각 행에서 그대로 보존(원본 버그 수정)."""
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


def load_labels_csv_multi(cube_id_path_pairs):
    """여러 label_csv를 이어붙임. (cube_id, frame_idx) 튜플로 키 —
    서로 다른 지역의 같은 frame_idx 충돌 방지(원본 버그 수정).

    ★ label_csv 파일 자체엔 cube_id 컬럼이 없음(파일 하나=큐브 하나 설계).
    호출측에서 --tile_spec 순서와 --label_csv 순서를 1:1 매칭해
    (cube_id, path) 쌍으로 넘겨줘야 한다."""
    out = {}
    for cube_id, path in cube_id_path_pairs:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                has_event = _to_bool(r["has_event"])
                frame_idx = int(r["frame_idx"])
                out[(cube_id, frame_idx)] = 1 if has_event else 0
    return out


class MultiTileGoldDataset:
    """여러 cube_id -> zarr 파일 매핑을 지원하는 Dataset.
    __getitem__ 반환 형식은 원본 OceanTensorGoldDatasetH100과 완전히 동일
    (Trainer가 그대로 재사용 가능)."""

    def __init__(self, tile_spec, sample_csvs, label_csvs, split,
                 patch_size=64, augment=True, seed=42):
        self.cube_paths = tile_spec  # {cube_id: zarr_path}
        self.patch = patch_size
        self.augment = augment

        self.samples = load_samples_csv_multi(sample_csvs, split)
        if not self.samples:
            raise ValueError(f"샘플 없음: split={split}")
        # label_csvs는 이미 (cube_id, path) 쌍 리스트로 전달됨
        self.frame_severity = load_labels_csv_multi(label_csvs)

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
        sev = self.frame_severity.get((cube_id, s["target_frame"]), 0)
        y = torch.full((ph, pw), sev, dtype=torch.int64)
        if self.augment and torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[3])
            y = torch.flip(y, dims=[1])
        return x, y


def parse_tile_spec(spec_list):
    out = {}
    for spec in spec_list:
        cube_id_str, path = spec.split(":", 1)
        out[int(cube_id_str)] = path
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile_spec", action="append", required=True,
                     help="'cube_id:zarr경로' 형태, 지역마다 반복 지정")
    ap.add_argument("--sample_csv", action="append", required=True)
    ap.add_argument("--label_csv", action="append", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--save_dir", type=str, default="checkpoints/v13")
    ap.add_argument("--class_weights", type=float, nargs=2, default=[0.5, 1.5],
                     metavar=("W0", "W1"))
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--n_stages", type=int, default=2)
    args = ap.parse_args()

    import torch
    from st_mmt_xattn import STMMTCrossAttn
    from ml.training.trainer import Trainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tile_spec = parse_tile_spec(args.tile_spec)

    print("="*60)
    print(f"  Gold STMMT 다중타일 학습(v13) — {len(tile_spec)}개 지역 통합")
    print(f"  타일: {tile_spec}")
    print(f"  device={device}" + (f" ({torch.cuda.get_device_name(0)})" if device=="cuda" else ""))
    print("="*60)

    # --tile_spec 순서와 --label_csv 순서가 1:1 대응한다고 가정하고 페어링
    tile_cube_ids = list(tile_spec.keys())
    if len(tile_cube_ids) != len(args.label_csv):
        raise ValueError(f"--tile_spec 개수({len(tile_cube_ids)})와 "
                          f"--label_csv 개수({len(args.label_csv)})가 다름 — "
                          f"같은 순서로 1:1 대응해서 지정해야 함")
    label_pairs = list(zip(tile_cube_ids, args.label_csv))
    print(f"  label_csv 매칭: {label_pairs}")

    ds_train = MultiTileGoldDataset(tile_spec, args.sample_csv, label_pairs,
                                     "train", patch_size=args.patch, augment=True)
    ds_val   = MultiTileGoldDataset(tile_spec, args.sample_csv, label_pairs,
                                     "val", patch_size=args.patch, augment=False)
    print(f"  train: {len(ds_train)}샘플 / val: {len(ds_val)}샘플")
    print(f"  patch: {args.patch}x{args.patch}, 채널: {len(VALID_INDICES)}개")

    model = STMMTCrossAttn(
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=2, n_stages=args.n_stages, patch_size=4,
    )
    # ★ Cross-Attention 버전: in_channels 인자 없음(내부에서 32채널 
    # 고정 분리). VALID_INDICES(32채널 완전체) 사용 전제 — 
    # 채널 제외(EMPTY_CHANNELS) 실험과는 호환 안 됨, 별도 확인 필요
    print(f"  파라미터: {model.count_params():,}")

    # ★ 클래스 가중치 반영 — 원본 gold_train_stmmt_h100.py와 완전히 동일한 몽키패치
    import types
    import torch.nn.functional as F
    cw_tensor = torch.tensor(args.class_weights, dtype=torch.float32)

    def _weighted_compute_loss(self, outputs, labels):
        logits = outputs["last_logits"]
        w = cw_tensor.to(logits.device)
        ce_loss = F.cross_entropy(logits, labels, weight=w, label_smoothing=0.05)
        dy = torch.abs(logits[:, :, 1:, :] - logits[:, :, :-1, :]).mean()
        dx = torch.abs(logits[:, :, :, 1:] - logits[:, :, :, :-1]).mean()
        spatial_loss = dy + dx
        total = ce_loss + self.spatial_loss_weight * spatial_loss
        return {"total": total, "ce_loss": ce_loss, "spatial_loss": spatial_loss}

    model.compute_loss = types.MethodType(_weighted_compute_loss, model)
    print(f"  class_weights 적용: {args.class_weights} (몽키패치, st_mmt.py 원본 미수정)")

    trainer = Trainer(
        model=model, dataset=ds_train, val_dataset=ds_val,
        save_dir=args.save_dir, device=device, lr=args.lr,
        batch_size=args.batch_size, epochs=args.epochs, patience=10,
        n_stages=args.n_stages, class_weights=list(args.class_weights),
        focal_gamma=args.focal_gamma, thresholds=None, use_wandb=False,
    )
    result = trainer.fit()

    print(f"\n  ✓ 완료 — best_path={result['best_path']}")
    ev = {k: v for k, v in result["eval"].items() if k not in ("report", "confusion_matrix")}
    print(f"  최종 지표: {json.dumps(ev, indent=2, ensure_ascii=False)}")

    if args.n_stages == 2:
        print(f"\n  보완 계산: 이진 AUC/PR-AUC (auc_ovo NaN 대응)")
        from sklearn.metrics import roc_auc_score, average_precision_score
        model.load_state_dict(torch.load(result["best_path"], map_location=device))
        model.eval()
        dl_val = torch.utils.data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)
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
        print(f"  binary AUC(ROC): {auc:.4f}  PR-AUC(AP): {ap:8.4f}  "
              f"(클래스 불균형 상황엔 이 지표가 더 정보량 많음)")


if __name__ == "__main__":
    main()
