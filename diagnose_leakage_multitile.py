"""
diagnose_leakage_multitile.py — 다중타일 val 샘플의 단일채널 분리력 진단
════════════════════════════════════════════════════════════════════
gold_train_multitile.py의 MultiTileGoldDataset을 재사용해,
v14(완도+목포+진도) val 135개에서 각 채널이 얼마나 강하게
"어느 지역인지"를 알려주는지(=지역판별 지름길 위험) 확인.

핵심 목적: jindo만 MLD(ch31)가 0으로 고정돼 있는데, 
이게 모델이 "MLD=0 → 진도"라는 지름길을 학습했는지 확인.

사용:
  uv run diagnose_leakage_multitile.py \
      --tile_spec 32:data/wando_30m_full32_2021_ffilled_ma_mld_nir_anom.zarr \
      --tile_spec 53:data/mokpo_30m_full32_2021_ffilled_nir_turb_ma_anom_nirchg.zarr \
      --tile_spec 70:data/jindo_30m_full32_2021_ffilled_nir_turb_ma_anom_nirchg.zarr \
      --sample_csv h100_transfer_v13_wando/gold_sample.csv \
      --sample_csv h100_transfer_mokpo/gold_sample.csv \
      --sample_csv h100_transfer_jindo/gold_sample.csv \
      --label_csv h100_transfer_v13_wando/gold_label.csv \
      --label_csv h100_transfer_mokpo/gold_label.csv \
      --label_csv h100_transfer_jindo/gold_label.csv
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gold_train_multitile import parse_tile_spec, load_samples_csv_multi
from silver_channels import CHANNEL_MASTER, CHANNEL_ORDER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile_spec", action="append", required=True)
    ap.add_argument("--sample_csv", action="append", required=True)
    ap.add_argument("--label_csv", action="append", required=True)
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score
    import zarr

    tile_spec = parse_tile_spec(args.tile_spec)
    samples = load_samples_csv_multi(args.sample_csv, "val")
    print(f"val 샘플: {len(samples)}개 (다중타일)")

    zarrs = {cid: zarr.open(path, mode="r") for cid, path in tile_spec.items()}

    # 지역(cube_id)별로 프레임 평균값을 뽑아 "이 채널만으로 지역이 갈리는지" 확인
    cube_ids_present = sorted(set(s["cube_id"] for s in samples))
    print(f"참여 지역 cube_id: {cube_ids_present}")
    if len(cube_ids_present) < 2:
        print("지역이 1개뿐 — 지역판별 진단 불가")
        return

    print(f"\n  {'채널':10s} {'이름':20s} {'지역간AUC(최대쌍)':>18s}  판정")
    print(f"  {'-'*65}")

    suspects = []
    for ch_idx, ch_key in enumerate(CHANNEL_ORDER):
        ch_name = CHANNEL_MASTER[ch_key][0]

        vals_by_cube = {}
        for s in samples:
            cid = s["cube_id"]
            z = zarrs[cid]
            frame = np.asarray(z[s["target_frame"], ch_idx])
            v = np.nanmean(frame)
            vals_by_cube.setdefault(cid, []).append(v)

        # 모든 지역 쌍에 대해 "이 채널 값만으로 두 지역을 구분하는 AUC" 계산, 최댓값 기록
        max_auc = 0.5
        for i in range(len(cube_ids_present)):
            for j in range(i+1, len(cube_ids_present)):
                a, b = cube_ids_present[i], cube_ids_present[j]
                va, vb = vals_by_cube.get(a, []), vals_by_cube.get(b, [])
                if not va or not vb:
                    continue
                y = np.array([0]*len(va) + [1]*len(vb))
                x = np.array(va + vb)
                if np.nanstd(x) < 1e-9 or np.isnan(x).all():
                    continue
                try:
                    auc = roc_auc_score(y, np.nan_to_num(x, nan=np.nanmean(x)))
                    auc = max(auc, 1 - auc)
                    max_auc = max(max_auc, auc)
                except ValueError:
                    continue

        flag = ""
        if max_auc >= 0.98:
            flag = "🚨 의심 (지역판별 지름길 가능)"
            suspects.append((ch_key, ch_name, max_auc))
        elif max_auc >= 0.90:
            flag = "⚠ 주의"

        print(f"  {ch_key:10s} {ch_name:20s} {max_auc:18.4f}  {flag}")

    print(f"\n{'='*65}")
    if suspects:
        print(f"  🚨 지역판별 지름길 의심 채널 {len(suspects)}개:")
        for ch_key, ch_name, auc in suspects:
            print(f"    {ch_key} ({ch_name}): 최대 지역간AUC={auc:.4f}")
        print(f"  → 모델이 실제 해양신호 대신 이 채널로 '어느 지역인지'만")
        print(f"    학습했을 위험이 있다. 이 채널들의 실제 예측 기여도 재검토 필요.")
    else:
        print(f"  단일 채널로 지역이 완벽히 갈리는 경우 없음")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
