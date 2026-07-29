"""
gold_infer_batch_pilot.py — xattn_v3_focal로 파일럿 여러 타일 일괄 추론
════════════════════════════════════════════════════════════════════
각 타일의 ffilled 큐브에서 "가장 최근 관측 가능한 날짜"(입력창 14일이
확보되는 마지막 날짜)를 자동으로 찾아 그 시점의 위험도를 추론한다.
서천 외부검증과 동일하게 재학습 없이 순수 추론만 수행.

사용:
  uv run gold_infer_batch_pilot.py \
      --tiles cluster_057:166 cluster_081:171 cluster_033:175 cluster_005:179 cluster_092:183 \
      --checkpoint checkpoints/xattn_v3_focal/best_model.pt
"""
import argparse, os, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="+", required=True,
                     help="tile_id:cube_id 형식으로 여러 개 (예: cluster_057:166)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input_len", type=int, default=14)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_stages", type=int, default=2)
    ap.add_argument("--data_dir", default="data",
                     help="H100 로컬 데이터 디렉토리")
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--patch", type=int, default=64,
                     help="512x512에서 중앙 크롭할 패치 크기 — 학습·기존 추론과 동일하게 64 사용 (512 그대로 넣으면 어텐션 메모리 폭발)")
    args = ap.parse_args()

    import torch
    import zarr

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from st_mmt_xattn_v3 import STMMTCrossAttnV3

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = STMMTCrossAttnV3(d_model=args.d_model, n_heads=args.n_heads,
                              n_layers_joint=4, n_layers_group=2,
                              n_stages=args.n_stages, patch_size=4).to(device)
    sd = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"체크포인트 로드 완료: {args.checkpoint}\n")

    results = []
    with torch.no_grad():
        for spec in args.tiles:
            tile_id, cube_id = spec.split(":")
            cube_id = int(cube_id)

            # 파일명 접미사가 지역마다 다를 수 있어(예: cluster_057은 _ffilled 없이 _turb만)
            # 패턴 검색으로 실제 파일을 자동 탐색
            import glob
            candidates = sorted(
                glob.glob(os.path.join(args.data_dir, f"{tile_id}_full32_{args.year}*turb.zarr")),
                key=len, reverse=True,  # 이름이 더 긴(=더 많은 처리단계를 거친) 파일 우선
            )
            if not candidates:
                print(f"[{tile_id}] 매칭되는 로컬 파일 없음(패턴: {tile_id}_full32_{args.year}*turb.zarr), 건너뜀")
                continue
            local_path = candidates[0]
            print(f"[{tile_id}] 사용 파일: {local_path}")

            z = zarr.open(local_path, mode="r")
            T = z.shape[0]
            # 가장 최근 "실측이 실제로 반영된" 시점을 목표일로 삼는다 (12/31 등 
            # 순수 ffill 계단 구간을 목표로 잡으면 정보 없는 입력이 되므로 회피)

            x_full = np.asarray(z[:])  # 전체 365일 로드해서 실측 마지막 날짜 탐색
            sst_full = x_full[:, 0, 256, 256]  # 중앙 픽셀 SST로 변화 시점 탐색(가벼움)

            # 뒤에서부터 스캔하며 "값이 이전 프레임과 다른" 마지막 지점(=마지막 실측 반영일)을 찾음
            last_change_idx = args.input_len  # 최소값 보장
            for i in range(len(sst_full) - 1, args.input_len, -1):
                if not np.isnan(sst_full[i]) and not np.isnan(sst_full[i-1]) and abs(sst_full[i] - sst_full[i-1]) > 1e-6:
                    last_change_idx = i
                    break

            target_idx = last_change_idx
            start_idx = target_idx - args.input_len

            x = x_full[start_idx:target_idx].astype(np.float32)  # (14, 32, 512, 512)
            x = np.nan_to_num(x, nan=0.0)

            # 512x512 중앙에서 patch 크기만큼 크롭 (학습·기존 추론과 동일 관례)
            H, W = x.shape[-2], x.shape[-1]
            p = args.patch
            top, left = (H - p) // 2, (W - p) // 2
            x = x[:, :, top:top+p, left:left+p]  # (14, 32, patch, patch)

            x_t = torch.from_numpy(x).unsqueeze(0).to(device)  # (1, 14, 32, patch, patch)

            out = model(x_t)
            probs = torch.softmax(out["last_logits"], dim=1)[0, 1]
            mean_prob = probs.mean().item()
            max_prob = probs.max().item()

            from datetime import date, timedelta
            d0 = date(args.year, 1, 1)
            target_date = d0 + timedelta(days=target_idx)

            status = "위험" if mean_prob >= 0.5 else "정상"
            print(f"[{tile_id}] cube_id={cube_id}, 목표일자={target_date}, "
                  f"평균확률={mean_prob*100:.1f}%, 최대={max_prob*100:.1f}% → {status}")
            results.append({"tile_id": tile_id, "cube_id": cube_id, "target_date": str(target_date),
                             "mean_prob": mean_prob, "max_prob": max_prob, "status": status})

    print(f"\n{'='*60}\n  파일럿 {len(results)}개 지역 추론 완료\n{'='*60}")
    for r in results:
        print(f"  {r['tile_id']}: {r['mean_prob']*100:.1f}% ({r['status']})")


if __name__ == "__main__":
    main()
