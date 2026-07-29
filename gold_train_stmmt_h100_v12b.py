"""
gold_train_stmmt_h100.py — H100 전용 (DB 없이 CSV+Zarr만으로 학습)
════════════════════════════════════════════════════════════════════
네이버클라우드 DB에 접속하지 않고, gold_export_for_h100.py가 뽑은 CSV와
scp로 옮긴 Zarr 큐브만으로 학습. gold_train_stmmt.py의 DB 의존 부분을
CSV 읽기로 교체한 버전.

디렉토리 구조 기대값 (~/gold_wando/):
  gold_train_stmmt_h100.py
  ml/models/st_mmt.py, ml/training/trainer.py, ml/training/eval.py
  silver_channels.py
  data_cube/wando_30m_full32_2021_ffilled.zarr
  h100_transfer/gold_sample.csv, gold_label.csv

사용:
  uv run gold_train_stmmt_h100.py \
      --cube_path data_cube/wando_30m_full32_2021_ffilled.zarr \
      --sample_csv h100_transfer/gold_sample.csv \
      --label_csv h100_transfer/gold_label.csv \
      --epochs 30 --batch_size 8
"""
import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from silver_channels import CHANNEL_MASTER, CHANNEL_ORDER

# ch13,28,30은 이동평균 파생 (gold_add_ma_channels.py) → 유효채널
# ch31(MLD)은 HYCOM mixed_layer_thickness (gold_add_mld_channel.py) → 유효채널
# ch24(NIR_idx)는 GOCI-II(팀원 kosc_ml.py 재사용, ffill) → 유효채널
# ch12(SST_anomaly)는 2021 vs 2021+2024 2개년 평균 편차 → 유효채널
# ch29(NIR_daily_change)는 ch24의 일별 변화량 (gold_add_nirchange_channel.py) → 유효채널
#   ※ 원 정의는 7일 이동평균이었으나 정보 중복(v8, AUC 0.850)으로 확인되어
#     일별 변화량으로 개정(v9, AUC 0.917). silver_channels.py 마스터도 함께 개정됨.
#   ※ 정직한 한계: ch24 자체가 ffill(계단식)이라 평활 효과는 약함
# 남은 1개만 진짜 빈 채널: ch15 Turbidity (위성 알고리즘 미확보, 32/32 중 마지막)
# ch13,28,30은 이동평균 파생 (gold_add_ma_channels.py) → 유효채널
# ch31(MLD)은 HYCOM mixed_layer_thickness (gold_add_mld_channel.py) → 유효채널
# ch24(NIR_idx)는 GOCI-II(팀원 kosc_ml.py 재사용, ffill) → 유효채널
# ch12(SST_anomaly)는 2021 vs 2021+2024 2개년 평균 편차 → 유효채널
# ch29(NIR_daily_change)는 ch24의 일별 변화량 (마스터 정의 개정됨, gold_add_nirchange_channel.py) → 유효채널
# ch15(Turbidity)는 KOEM 실측 부유물질(fltng_mttr_sfclyr, ffill) → 유효채널
#   ※ 위성 반사도-탁도 변환 보정계수 문제를 실측값으로 회피 (가장 정직한 소스)
# 32/32채널 전부 확보 완료 (v10)
EMPTY_CHANNELS = {"ch30"}
VALID_CHANNELS = [ch for ch in CHANNEL_ORDER if ch not in EMPTY_CHANNELS]  # 32개
VALID_INDICES = [int(ch[2:]) for ch in VALID_CHANNELS]


def _to_bool(s):
    return str(s).strip().lower() in ("true", "t", "1", "yes")


def load_samples_csv(path, split):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] != split:
                continue
            rows.append({
                "sample_id": int(r["sample_id"]),
                "input_start_frame": int(r["input_start_frame"]),
                "input_len": int(r["input_len"]),
                "target_frame": int(r["target_frame"]),
                "target_date": r["target_date"],
                "has_event": _to_bool(r["has_event"]),
                "severity": int(r["severity"]) if r["severity"] not in ("", None) else 0,
            })
    return rows


def load_labels_csv(path):
    """frame_idx -> binary label(0=정상,1=발생) dict.
    실측 확인: severity는 데이터에 0과 2만 존재(1=초기 없음) → 순수 이진 문제.
    (SQL 검증: severity=0 314건, severity=2 51건, severity=1 0건)"""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            has_event = _to_bool(r["has_event"])
            out[int(r["frame_idx"])] = 1 if has_event else 0
    return out


class OceanTensorGoldDatasetH100:
    """CSV + Zarr 기반 (DB 없음). gold_dataset의 H100 버전."""

    def __init__(self, cube_path, sample_csv, label_csv, split,
                 patch_size=64, augment=True, seed=42):
        self.file_path = cube_path
        self.patch = patch_size
        self.augment = augment

        self.samples = load_samples_csv(sample_csv, split)
        if not self.samples:
            raise ValueError(f"샘플 없음: split={split} (csv={sample_csv})")
        self.frame_severity = load_labels_csv(label_csv)

        import zarr
        z = zarr.open(cube_path, mode="r")
        self.H, self.W = z.shape[2], z.shape[3]
        self._zarr = None

        rng = np.random.default_rng(seed)
        self.anchors = [
            (int(rng.integers(0, max(1, self.H - self.patch + 1))),
             int(rng.integers(0, max(1, self.W - self.patch + 1))))
            for _ in self.samples
        ]

    def _get_zarr(self):
        if self._zarr is None:
            import zarr
            self._zarr = zarr.open(self.file_path, mode="r")
        return self._zarr

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torch
        s = self.samples[idx]
        z = self._get_zarr()
        t0, t_in = s["input_start_frame"], s["input_len"]
        rh, rw = self.anchors[idx]
        ph = pw = self.patch

        x = np.asarray(z[t0:t0+t_in, :, rh:rh+ph, rw:rw+pw])  # [t_in, 32, ph, pw]
        x = x[:, VALID_INDICES, :, :]                          # [t_in, 24, ph, pw]
        x = np.nan_to_num(x, nan=0.0).astype(np.float32)
        x = torch.from_numpy(x)

        sev = self.frame_severity.get(s["target_frame"], 0)
        y = torch.full((ph, pw), sev, dtype=torch.int64)

        if self.augment and torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[3])
            y = torch.flip(y, dims=[1])

        return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube_path", required=True)
    ap.add_argument("--sample_csv", required=True)
    ap.add_argument("--label_csv", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--save_dir", type=str, default="checkpoints/gold_v1")
    ap.add_argument("--class_weights", type=float, nargs=2, default=[0.3, 3.0],
                     metavar=("W0", "W1"), help="CrossEntropy weight [정상 발생] — 클래스 불균형 대응")
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--n_stages", type=int, default=2, help="실측 라벨은 0/2뿐(초기 없음) → 기본 이진(2)")
    args = ap.parse_args()

    import torch
    from ml.models.st_mmt import STMMT
    from ml.training.trainer import Trainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("="*60)
    print(f"  Gold STMMT(v1.0) 학습 [H100/CSV] — {args.cube_path}")
    print(f"  device={device}" + (f" ({torch.cuda.get_device_name(0)})" if device=="cuda" else ""))
    print("="*60)

    ds_train = OceanTensorGoldDatasetH100(args.cube_path, args.sample_csv, args.label_csv,
                                            "train", patch_size=args.patch, augment=True)
    ds_val   = OceanTensorGoldDatasetH100(args.cube_path, args.sample_csv, args.label_csv,
                                            "val", patch_size=args.patch, augment=False)
    print(f"  train: {len(ds_train)}샘플 / val: {len(ds_val)}샘플")
    print(f"  patch: {args.patch}x{args.patch}, 채널: {len(VALID_INDICES)}개")

    model = STMMT(
        in_channels=len(VALID_INDICES), d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_model*2, n_stages=args.n_stages, patch_size=4,
    )
    print(f"  파라미터: {model.count_params():,}")

    # ★ 클래스 가중치 반영 (팀원 st_mmt.py는 수정하지 않고 인스턴스 메서드만 교체)
    # 원본 STMMT.compute_loss는 가중치 없는 단순 CrossEntropy만 사용 —
    # 우리 데이터는 양성(발생) 비율이 낮아(train 8.7%) 가중치가 필요.
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
        # 참고: Trainer의 class_weights는 TinyTransformer(stage_logits) 경로에만 쓰임.
        # STMMT(last_logits) 경로는 위에서 몽키패치한 model.compute_loss가 실제 가중치를 적용함.
        focal_gamma=args.focal_gamma, thresholds=None, use_wandb=False,
    )
    result = trainer.fit()

    print(f"\n  ✓ 완료 — best_path={result['best_path']}")
    ev = {k: v for k, v in result["eval"].items() if k not in ("report", "confusion_matrix")}
    print(f"  최종 지표: {json.dumps(ev, indent=2, ensure_ascii=False)}")

    # ★ 보완 계산: eval.py의 auc_ovo는 multi_class='ovo'를 쓰는데
    # n_stages=2(이진)에선 sklearn이 예외를 던져 NaN이 된다
    # (팀원 eval.py는 원래 4/5단계 체계용으로 설계됨 — 파일은 안 건드리고
    #  여기서 이진 전용 AUC/PR-AUC를 별도로 계산)
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
                probs = torch.softmax(out["last_logits"], dim=1)[:, 1]  # 발생(1) 확률
                all_labels.append(y.numpy().ravel())
                all_probs_pos.append(probs.cpu().numpy().ravel())
        y_true = np.concatenate(all_labels)
        y_prob = np.concatenate(all_probs_pos)
        try:
            auc_binary = roc_auc_score(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            print(f"  binary AUC(ROC): {auc_binary:.4f}")
            print(f"  PR-AUC(AP):      {ap:.4f}  (클래스 불균형 상황엔 이 지표가 더 정보량 많음)")
        except ValueError as e:
            print(f"  AUC 계산 실패: {e}")


if __name__ == "__main__":
    main()
