"""
gold_finetune_v15.py — v13 사전학습 백본 → 32채널 완도 데이터 전체 파인튜닝 (v15)
════════════════════════════════════════════════════════════════════════════
팀원 제공 v13(3헤드 ST-MMT, in_channels=33)의 사전학습 가중치를 백본으로 재사용하되,
우리 파이프라인 원칙(32채널 고정)을 지키기 위해 din_mask(33번째 채널)를 넣지 않는다.

전이학습 설계:
  - patch_proj.weight만 새로 초기화 (33→32채널로 입력 차원이 바뀌어 형태가 안 맞음)
  - 나머지 84/85 텐서(pe_t, patch_norm, ST-Block×4, dec_trunk,
    adi_head, warn_head, severe_head)는 v13 사전학습 가중치를 그대로 이식
    → 전체 파라미터의 97.4%가 사전학습에서 옴
  - 전체 파인튜닝(freeze 없음) — 모든 파라미터 학습 가능

라벨 설계:
  - 우리 심각도 4단계(0정상/1주의/2경계/3심각)를 ADI 프록시(0~10 스케일)로 매핑
    0→0.0, 1→3.0, 2→6.0, 3→9.0
  - v13의 WARN_THRESH(5.0) 기준을 그대로 사용 → 우리 실측 severity=2(경계)가
    정확히 warn=1로 유도됨 (프록시 설계가 v13 원 임계값과 자연스럽게 정합)
  - warn_head만 학습(BCE). adi_head/severe_head는 로드된 사전학습 가중치를
    그대로 유지만 하고 손실에서 제외 — 우리 데이터는 진짜 연속 ADI가 아니라
    계단형 프록시이므로, 이를 회귀 학습시키면 사전학습된 회귀 품질을 오염시킬
    위험이 있음. severe(심각=3) 실측 사례가 전혀 없어 학습 신호로 쓸 수 없음.

사용 (H100):
  uv run gold_finetune_v15.py \
      --cube_path data/wando_30m_full32_2021_ffilled_ma.zarr \
      --sample_csv h100_transfer_v2/gold_sample.csv \
      --label_csv h100_transfer_v2/gold_label.csv \
      --v13_checkpoint checkpoints/v13/best_model.pt \
      --epochs 30 --batch_size 8
"""
import argparse, csv, json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── 심각도(0~3) → ADI 프록시(0~10) ───
SEVERITY_TO_ADI = {0: 0.0, 1: 3.0, 2: 6.0, 3: 9.0}
WARN_THRESH = 5.0
SEVERE_THRESH = 8.0
N_CHANNELS = 32  # din_mask 없이 32채널 고정 (원칙 유지)
T_OUT = 7


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
    """frame_idx -> severity(0~3) dict. 실측은 0/2뿐이나 향후 1/3 대비해 전 범위 지원."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            has_event = _to_bool(r["has_event"])
            sev_raw = r["severity"]
            sev = int(sev_raw) if (has_event and sev_raw not in ("", None)) else 0
            sev = min(max(sev, 0), 3)
            out[int(r["frame_idx"])] = sev
    return out


class Full32ChannelDataset:
    """32채널 전체(0~31)를 그대로 사용. 미확보 채널(5개)은 0으로 채움.
    (VALID_INDICES 서브셋을 쓰던 이전 버전과 달리, v13 patch_proj가 기대하는
     '32채널 전체 슬롯' 구조를 그대로 따른다.)"""

    def __init__(self, cube_path, sample_csv, label_csv, split,
                 patch_size=64, t_out=T_OUT, augment=True, seed=42):
        self.file_path = cube_path
        self.patch = patch_size
        self.t_out = t_out
        self.augment = augment

        self.samples = load_samples_csv(sample_csv, split)
        if not self.samples:
            raise ValueError(f"샘플 없음: split={split}")
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

        x = np.asarray(z[t0:t0+t_in, :N_CHANNELS, rh:rh+ph, rw:rw+pw])  # [t_in, 32, ph, pw]
        x = np.nan_to_num(x, nan=0.0).astype(np.float32)
        x = torch.from_numpy(x)

        # ADI 프록시 타깃: (t_out, ph, pw) 전부 같은 값 (단일 타깃일 라벨을 7일창에 broadcast)
        # 주의: 이건 v13 원래의 "일자별 다른 연속 ADI"가 아니라 단순화된 계단형 프록시.
        sev = self.frame_severity.get(s["target_frame"], 0)
        adi_val = SEVERITY_TO_ADI[sev]
        adi_target = torch.full((self.t_out, ph, pw), adi_val, dtype=torch.float32)

        if self.augment and torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[3])
            adi_target = torch.flip(adi_target, dims=[2])

        return x, adi_target, sev  # sev도 반환 (평가용 원본 라벨)


def partial_load_v13_backbone(model, checkpoint_path, device):
    """v13(33채널) 체크포인트를 32채널 모델에 이식.
    patch_proj.weight만 형태 불일치로 제외, 나머지 84/85 텐서 전부 로드."""
    import torch
    sd33 = torch.load(checkpoint_path, map_location=device, weights_only=True)
    skip = {"patch_proj.weight"}
    compatible = {k: v for k, v in sd33.items() if k not in skip}
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    total = sum(p.numel() for p in model.parameters())
    transferred = sum(v.numel() for k, v in compatible.items())
    print(f"  전이학습 로드: {len(compatible)}/{len(sd33)}개 텐서")
    print(f"    이식된 파라미터: {transferred:,} ({transferred/total*100:.1f}%)")
    print(f"    새로 초기화: {missing} ({total-transferred:,}개, {(total-transferred)/total*100:.2f}%)")
    assert missing == ["patch_proj.weight"], f"예상 밖 missing 키: {missing}"
    assert not unexpected, f"예상 밖 unexpected 키: {unexpected}"
    return model


def warn_only_loss(outputs, adi_target, warn_thresh=WARN_THRESH):
    """adi_head/severe_head는 학습에서 제외 (프록시 타깃이라 회귀 오염 위험,
    심각=3 실측 없어 severe 학습 근거 없음). warn_head만 BCE로 파인튜닝."""
    import torch.nn.functional as F
    warn_logit = outputs["warn_logit"]              # (B, Hf, Wf)
    pix_max = adi_target.amax(dim=1)                 # (B, ph, pw) — 프록시라 전부 유효
    # warn_logit 해상도(Hf,Wf=ph/patch_size)에 맞춰 타깃 다운샘플(최댓값 풀링)
    import torch
    ph, pw = pix_max.shape[1], pix_max.shape[2]
    Hf, Wf = warn_logit.shape[1], warn_logit.shape[2]
    if (Hf, Wf) != (ph, pw):
        pix_max = torch.nn.functional.adaptive_max_pool2d(pix_max.unsqueeze(1), (Hf, Wf)).squeeze(1)
    warn_tgt = (pix_max >= warn_thresh).float()
    bce_warn = F.binary_cross_entropy_with_logits(warn_logit, warn_tgt)
    return bce_warn, warn_tgt


def evaluate(model, loader, device):
    import torch
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
    model.eval()
    all_true, all_prob, all_pred = [], [], []
    with torch.no_grad():
        for x, adi_target, sev in loader:
            x = x.to(device)
            out = model(x)
            prob = torch.sigmoid(out["warn_logit"]).cpu().numpy().ravel()
            # 타깃도 warn_logit 해상도로 맞춰 비교
            pix_max = adi_target.amax(dim=1)
            Hf, Wf = out["warn_logit"].shape[1], out["warn_logit"].shape[2]
            if (Hf, Wf) != tuple(pix_max.shape[1:]):
                pix_max = torch.nn.functional.adaptive_max_pool2d(pix_max.unsqueeze(1), (Hf, Wf)).squeeze(1)
            y_true = (pix_max >= WARN_THRESH).float().numpy().ravel()
            all_true.append(y_true)
            all_prob.append(prob)
            all_pred.append((prob >= 0.5).astype(int))
    y_true = np.concatenate(all_true)
    y_prob = np.concatenate(all_prob)
    y_pred = np.concatenate(all_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
    except ValueError:
        auc, ap = float("nan"), float("nan")
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {"auc": auc, "pr_auc": ap, "f1_macro": f1, "confusion_matrix": cm.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube_path", required=True)
    ap.add_argument("--sample_csv", required=True)
    ap.add_argument("--label_csv", required=True)
    ap.add_argument("--v13_checkpoint", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)  # 파인튜닝이라 v1.0 때보다 낮게
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--save_dir", type=str, default="checkpoints/v15")
    ap.add_argument("--patience", type=int, default=8)
    args = ap.parse_args()

    import torch
    from ml_v15.models.st_mmt import STMMT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("="*60)
    print(f"  v15 파인튜닝 — v13 백본 전이학습 (32채널, warn_head만 학습)")
    print(f"  device={device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print("="*60)

    ds_train = Full32ChannelDataset(args.cube_path, args.sample_csv, args.label_csv,
                                      "train", patch_size=args.patch, augment=True)
    ds_val = Full32ChannelDataset(args.cube_path, args.sample_csv, args.label_csv,
                                    "val", patch_size=args.patch, augment=False)
    print(f"  train: {len(ds_train)}샘플 / val: {len(ds_val)}샘플")
    print(f"  patch: {args.patch}x{args.patch}, 채널: {N_CHANNELS}개(고정)")

    model = STMMT(in_channels=N_CHANNELS, d_model=256, n_heads=8, n_layers=4,
                   d_ff=512, patch_size=4, t_out=T_OUT).to(device)
    partial_load_v13_backbone(model, args.v13_checkpoint, device)

    # 손실 몽키패치: warn_head만 학습 (adi/severe는 사전학습 값 유지, 손실 미반영)
    import types
    model.compute_loss_warn_only = types.MethodType(
        lambda self, out, adi_target: warn_only_loss(out, adi_target), model)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)

    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0
    best_epoch = 0
    history = []
    t0 = time.time()

    print(f"\n  학습 시작 (전체 파인튜닝, lr={args.lr})")
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        for x, adi_target, sev in dl_train:
            x, adi_target = x.to(device), adi_target.to(device)
            out = model(x)
            loss, _ = warn_only_loss(out, adi_target)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        ep_loss /= max(len(dl_train), 1)

        ev = evaluate(model, dl_val, device)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{args.epochs} | loss={ep_loss:.4f} | "
              f"val_AUC={ev['auc']:.4f} val_PR-AUC={ev['pr_auc']:.4f} "
              f"val_F1={ev['f1_macro']:.4f} | {elapsed:.0f}s")
        history.append({"epoch": epoch, "train_loss": ep_loss, **{k: v for k, v in ev.items() if k != "confusion_matrix"}})

        if ev["auc"] > best_auc:
            best_auc = ev["auc"]; best_epoch = epoch
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            with open(save_dir / "best_eval.json", "w") as f:
                json.dump(ev, f, indent=2)
        elif epoch - best_epoch >= args.patience:
            print(f"  Early stopping at epoch {epoch} (best epoch {best_epoch}, AUC={best_auc:.4f})")
            break

    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # 최종 베스트 모델로 재평가 (요약용)
    model.load_state_dict(torch.load(save_dir / "best_model.pt", map_location=device, weights_only=True))
    final_ev = evaluate(model, dl_val, device)

    print(f"\n{'='*60}")
    print(f"  ✓ v15 파인튜닝 완료 — best epoch {best_epoch}")
    print(f"  최종 val AUC(ROC): {final_ev['auc']:.4f}")
    print(f"  최종 val PR-AUC:   {final_ev['pr_auc']:.4f}")
    print(f"  최종 val F1(macro):{final_ev['f1_macro']:.4f}")
    print(f"  confusion_matrix: {final_ev['confusion_matrix']}")
    print(f"  저장: {save_dir}/best_model.pt")

    # ── v15 기법 요약 (10줄 이내) ──
    print(f"\n{'='*60}")
    print("  v15 적용 기법 요약")
    print(f"{'='*60}")
    print("""
  1. v13 사전학습 ST-MMT 백본(전체 파라미터의 97.4%)을 32채널 모델로 이식
  2. patch_proj만 재초기화(33→32채널, din_mask 미사용 원칙 유지), 나머지 전량 전이
  3. 전체 파인튜닝(freeze 없음), 저학습률(5e-5)로 사전학습 표현 보존하며 적응
  4. 심각도 4단계(0~3)를 ADI 프록시(0~10)로 매핑해 v13 임계값(WARN=5.0) 그대로 재사용
  5. warn_head만 학습, adi_head/severe_head는 손실에서 제외(실측 없는 헤드 오염 방지)
  6. 평가: 자체 sklearn 기반 AUC/PR-AUC/F1(v13 eval.py 미사용, 버전 불일치 리스크 회피)
    """)

    print("  교훈:")
    print("""
  - 체크포인트·코드·문서 3자 버전이 실제로 어긋날 수 있음 → 반드시 strict 로드로 실측 검증
  - 부분 전이학습(shape 불일치 텐서만 skip)으로 원칙(32채널)과 사전학습 재사용을 동시 만족 가능
  - 라벨 스키마가 다른 두 모델을 억지로 맞추기보다, 상대 모델의 임계값 로직에 맞춰
    우리 라벨을 정직하게 프록시 변환하는 것이 안전 (근거 없는 헤드는 학습에서 제외)
    """)


if __name__ == "__main__":
    main()
