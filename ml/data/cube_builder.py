"""Zarr OceanTensorCube 저장 + 로더.

저장: [T, H, W, C] float32 + labels [T, H, W] int8 + 메타데이터
로드: OceanCubeDataset (ST-MMT 학습용 슬라이딩 윈도우)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import zarr
    _ZARR_OK = True
except ImportError:
    _ZARR_OK = False
    print("[cube_builder] zarr 없음 — numpy .npz fallback 사용")


def save_cube(result: dict[str, Any], output_dir: str, version: str = "v1") -> Path:
    """channel_builder 결과를 Zarr 큐브로 저장.

    output_dir/cube_{version}/
      ├── data.zarr    [T, H, W, C] float32
      ├── labels.zarr  [T, H, W]    int8
      └── meta.json
    """
    cube: np.ndarray   = result["cube"]    # (T, H, W, C)
    labels: np.ndarray = result["labels"]  # (T, H, W)

    out = Path(output_dir) / f"cube_{version}"
    out.mkdir(parents=True, exist_ok=True)

    if _ZARR_OK:
        z_data = zarr.open(
            str(out / "data.zarr"), mode="w",
            shape=cube.shape, dtype="float32",
            chunks=(min(72, cube.shape[0]), cube.shape[1], cube.shape[2], cube.shape[3]),
        )
        z_data[:] = cube
        z_labels = zarr.open(
            str(out / "labels.zarr"), mode="w",
            shape=labels.shape, dtype="int8",
            chunks=(min(72, labels.shape[0]), labels.shape[1], labels.shape[2]),
        )
        z_labels[:] = labels
        print(f"[cube_builder] Zarr 저장 완료: {out}")
    else:
        np.save(str(out / "data.npy"), cube)
        np.save(str(out / "labels.npy"), labels)
        print(f"[cube_builder] .npy fallback 저장 완료: {out}")

    meta = {
        "version":      version,
        "shape":        list(cube.shape),
        "labels_shape": list(labels.shape),
        "channel_names": result["channel_names"],
        "dates":         result["dates"],
        "norm_stats":    result["norm_stats"],
        "grid_meta":     result["grid_meta"],
        "T_resolution":  "1day",
        "label_schema":  {0: "정상", 1: "초기", 2: "경계", 3: "진행", 4: "심각"},
        "wbi_formula":   "0.38*din_risk+0.27*temp_risk+0.19*np_risk+0.10*do_risk+0.06*sal_risk",
    }
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    label_dist = np.bincount(labels.ravel(), minlength=5).tolist()
    meta["label_distribution"] = dict(zip(["정상","초기","경계","진행","심각"], label_dist))
    print(f"[cube_builder] 메타 저장: {out / 'meta.json'}")
    return out


def load_cube(cube_dir: str):
    """저장된 큐브 로드 → (cube, labels, meta).

    cube  : zarr array (lazy, 메모리 로드 없음) 또는 np.ndarray mmap
    labels: np.ndarray int8 (27MB — 전체 로드해도 무방)
    """
    p = Path(cube_dir)
    with open(p / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)

    if _ZARR_OK and (p / "data.zarr").exists():
        _z   = zarr.open_array(str(p / "data.zarr"), mode="r")
        print(f"  큐브 RAM 로드 중 ({_z.nbytes/1e9:.1f}GB)...", flush=True)
        cube = _z[:]   # 전체 로드 → __getitem__ RAM slice로 가속
        labels = zarr.open_array(str(p / "labels.zarr"), mode="r")[:].astype(np.int8)
    else:
        cube   = np.load(str(p / "data.npy"),   mmap_mode="r")
        labels = np.load(str(p / "labels.npy")).astype(np.int8)

    return cube, labels, meta


class RealCubeDataset:
    """실데이터 OceanTensorCube → ST-MMT 학습용 Dataset (torch lazy import).

    - 시간 슬라이딩 윈도우 (t_in 일)
    - 공간 랜덤 크롭 (patch_h × patch_w)
    - 채널 정규화 (norm_stats 기반 z-score)

    __getitem__ 반환:
        x: (t_in, patch_h, patch_w, C)  float32
        y: (patch_h, patch_w)            int64  (WBI 단계)
    """

    def __init__(
        self,
        cube,                        # zarr array 또는 np.ndarray (lazy OK)
        labels: np.ndarray,
        norm_stats: dict,
        channel_names: list[str],
        t_in: int = 24,
        stride: int = 6,
        patch_h: int = 64,
        patch_w: int = 64,
        augment: bool = True,
        t_indices: list[int] | None = None,
        din_mask: np.ndarray | None = None,  # (T,) float32 — DIN 관측일 마스크
    ):
        T, H, W, C = cube.shape
        assert labels.shape == (T, H, W)
        self.T, self.H, self.W, self.C = T, H, W, C
        self.t_in    = t_in
        self.stride  = stride
        self.patch_h = min(patch_h, H)
        self.patch_w = min(patch_w, W)
        self.augment = augment

        import torch
        # cube는 zarr lazy array — __getitem__에서 slice별 로드
        self.cube   = cube
        self.labels = torch.from_numpy(labels.astype(np.int64))  # (T, H, W) — 27MB OK

        # z-score 파라미터: cube 실채널(din_mask 제외)만
        n_cube_ch = cube.shape[3]
        cube_ch_names = [n for n in channel_names if n != "din_mask"][:n_cube_ch]
        means = np.array([norm_stats.get(n, {}).get("mean", 0.0) for n in cube_ch_names], dtype=np.float32)
        stds  = np.array([norm_stats.get(n, {}).get("std",  1.0) for n in cube_ch_names], dtype=np.float32)
        stds[stds < 1e-8] = 1.0
        self.mean = torch.from_numpy(means)   # (C,)
        self.std  = torch.from_numpy(stds)    # (C,)

        # DIN 마스크 (별도 채널 — concat은 __getitem__에서)
        self.din_mask = torch.from_numpy(din_mask) if din_mask is not None else None

        self.t_indices = t_indices if t_indices is not None else list(range(0, T - t_in, stride))

        # 결정론적 공간 크롭 앵커
        rng = np.random.default_rng(42)
        n_spatial = max(1, (H // patch_h) * (W // patch_w) * 4)
        self.spatial_anchors = [
            (rng.integers(0, max(1, H - self.patch_h + 1)),
             rng.integers(0, max(1, W - self.patch_w + 1)))
            for _ in range(n_spatial)
        ]
        self._len = len(self.t_indices) * len(self.spatial_anchors)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        import torch
        ti_idx = idx // len(self.spatial_anchors)
        sp_idx = idx %  len(self.spatial_anchors)

        t0     = self.t_indices[ti_idx]
        rh, rw = self.spatial_anchors[sp_idx]
        ph, pw = self.patch_h, self.patch_w

        # zarr lazy slice → numpy → tensor  (메모리: batch 크기만큼만)
        x = torch.from_numpy(
            self.cube[t0:t0+self.t_in, rh:rh+ph, rw:rw+pw, :].astype(np.float32)
        )  # (t_in, ph, pw, C)

        # z-score 정규화 (채널별, in-place broadcast)
        x = (x - self.mean) / self.std   # mean/std: (C,) → auto-broadcast

        # DIN 마스크 채널 추가
        if self.din_mask is not None:
            m = self.din_mask[t0:t0+self.t_in].reshape(-1, 1, 1).expand(self.t_in, ph, pw)
            x = torch.cat([x, m.unsqueeze(-1)], dim=-1)  # (t_in, ph, pw, C+1)

        y = self.labels[t0 + self.t_in - 1, rh:rh+ph, rw:rw+pw]   # (ph, pw)

        # NHWC → NCHW: (t_in, ph, pw, C) → (t_in, C, ph, pw)
        x = x.permute(0, 3, 1, 2).contiguous()

        if self.augment and torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[3])  # 좌우 반전 (W 축)
            y = torch.flip(y, dims=[1])

        return x, y
