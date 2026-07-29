"""
gold_train_stmmt.py — STMMT(v1.0) 학습 (완도 2021, 24채널)
════════════════════════════════════════════════════════════════════
팀원 제공 모델(ml/models/st_mmt.py, STMMT v1.0)을 우리 Gold 데이터로 학습.

주의 — 라벨 해상도:
  STMMT는 픽셀별 공간라벨(H,W)을 기대하지만, 우리 라벨(NIFS 사건)은
  지역 단위(완도연안 전체) 스칼라다. 실제 관측 해상도가 그렇기 때문에
  (지역 단위 모니터링), 타일 전체에 같은 severity를 broadcast한다.
  → 서브타일 세부정보를 조작해 만들지 않고, 있는 라벨 해상도를 정직하게 반영.

라벨: severity 0(정상)/1(경고)/2(발생) → n_stages=3
입력: 24채널(ffilled), patch 랜덤크롭 (기본 128×128, 512에서 크롭)
분할: gold_sample.split (train/val, 클래스 균형 확인된 분할)

사용:
  uv run gold_train_stmmt.py --cube_id 24 --epochs 30 --batch_size 8
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np
import psycopg2, psycopg2.extras

try:
    from dotenv import load_dotenv
    load_dotenv("/home/attention/prj/.env")
except ImportError:
    pass

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from silver_channels import CHANNEL_MASTER, CHANNEL_ORDER

DB_CONFIG = {
    "host":     os.getenv("PG_HOST",     os.getenv("PGHOST", "localhost")),
    "port":     os.getenv("PG_PORT",     os.getenv("PGPORT", "5432")),
    "dbname":   os.getenv("PG_DATABASE", os.getenv("PGDATABASE", "oceantensor_db")),
    "user":     os.getenv("PG_USER",     os.getenv("PGUSER", "oceantensor_user")),
    "password": os.getenv("PG_PASSWORD", os.getenv("PGPASSWORD", "")),
}

EMPTY_CHANNELS = {"ch12", "ch13", "ch15", "ch24", "ch28", "ch29", "ch30", "ch31"}
VALID_CHANNELS = [ch for ch in CHANNEL_ORDER if ch not in EMPTY_CHANNELS]  # 24개
VALID_INDICES = [int(ch[2:]) for ch in VALID_CHANNELS]


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def fetch_cube_info(cube_id):
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT file_path, file_format, n_frames, grid_h, grid_w FROM grid_cube WHERE cube_id=%s", (cube_id,))
            return cur.fetchone()
    finally:
        conn.close()


def fetch_samples(cube_id, split):
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT sample_id, input_start_frame, input_len, target_frame,
                       target_date, has_event, severity
                FROM gold_sample WHERE cube_id=%s AND split=%s ORDER BY sample_id
            """, (cube_id, split))
            return cur.fetchall()
    finally:
        conn.close()


def fetch_all_labels(cube_id):
    """frame_idx -> severity (0/1/2). 전체 프레임에 대해 dict로."""
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT frame_idx, has_event, severity FROM gold_label WHERE cube_id=%s ORDER BY frame_idx", (cube_id,))
            rows = cur.fetchall()
        out = {}
        for r in rows:
            sev = r["severity"] if (r["has_event"] and r["severity"]) else 0
            sev = min(max(int(sev), 0), 2)  # 0~2로 클램프
            out[r["frame_idx"]] = sev
        return out
    finally:
        conn.close()


class OceanTensorGoldDataset:
    """gold_sample 인덱스 + ffilled Zarr 큐브 → STMMT(v1.0) 학습용 Dataset.
    RealCubeDataset과 동일 관례: (t_in, ph, pw, C) → permute (t_in, C, ph, pw)."""

    def __init__(self, cube_id, split, patch_size=128, augment=True, seed=42):
        info = fetch_cube_info(cube_id)
        if info is None:
            raise ValueError(f"cube_id={cube_id} 없음")
        self.file_path = info["file_path"]
        self.H, self.W = info["grid_h"], info["grid_w"]
        self.patch = min(patch_size, self.H, self.W)
        self.augment = augment

        self.samples = fetch_samples(cube_id, split)
        if not self.samples:
            raise ValueError(f"샘플 없음: cube_id={cube_id}, split={split}")

        self.frame_severity = fetch_all_labels(cube_id)

        self._zarr = None
        rng = np.random.default_rng(seed)
        # 샘플마다 결정론적 patch 앵커 하나씩 (재현성)
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

        x = np.asarray(z[t0:t0+t_in, :, rh:rh+ph, rw:rw+pw])   # [t_in, 32, ph, pw]
        x = x[:, VALID_INDICES, :, :]                           # [t_in, 24, ph, pw]
        x = np.nan_to_num(x, nan=0.0).astype(np.float32)
        x = torch.from_numpy(x)                                 # (T, C, ph, pw) — STMMT 기대 형태와 일치

        # 라벨: 타깃 프레임의 지역단위 severity를 (ph,pw) 전체에 broadcast
        # (실제 관측이 지역단위라 서브타일 세부정보 없음 — 정직한 처리)
        sev = self.frame_severity.get(s["target_frame"], 0)
        y = torch.full((ph, pw), sev, dtype=torch.int64)

        if self.augment and torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[3])
            y = torch.flip(y, dims=[1])

        return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube_id", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--save_dir", type=str, default="checkpoints/gold_v1")
    ap.add_argument("--class_weights", type=float, nargs=3, default=[0.3, 2.0, 3.0],
                     metavar=("W0", "W1", "W2"), help="FocalLoss alpha [정상 경고 발생]")
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    args = ap.parse_args()

    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ml.models.st_mmt import STMMT
    from ml.training.trainer import Trainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("="*60)
    print(f"  Gold STMMT(v1.0) 학습 — cube_id={args.cube_id}")
    print(f"  device={device}" + (f" ({torch.cuda.get_device_name(0)})" if device=="cuda" else ""))
    print("="*60)

    ds_train = OceanTensorGoldDataset(args.cube_id, "train", patch_size=args.patch, augment=True)
    ds_val   = OceanTensorGoldDataset(args.cube_id, "val",   patch_size=args.patch, augment=False)
    print(f"  train: {len(ds_train)}샘플 / val: {len(ds_val)}샘플")
    print(f"  patch: {args.patch}x{args.patch}, 채널: {len(VALID_INDICES)}개")

    model = STMMT(
        in_channels=len(VALID_INDICES), d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_model*2, n_stages=3, patch_size=4,
    )
    print(f"  파라미터: {model.count_params():,}")

    trainer = Trainer(
        model=model,
        dataset=ds_train,
        val_dataset=ds_val,
        save_dir=args.save_dir,
        device=device,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=10,
        n_stages=3,
        class_weights=args.class_weights,
        focal_gamma=args.focal_gamma,
        thresholds=None,
        use_wandb=False,
    )
    result = trainer.fit()

    print(f"\n  ✓ 완료 — best_path={result['best_path']}")
    print(f"  최종 지표: {json.dumps({k:v for k,v in result['eval'].items() if k!='report' and k!='confusion_matrix'}, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
