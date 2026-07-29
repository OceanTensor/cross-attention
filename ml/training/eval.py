"""평가 지표 모음 — F1, AUC, Latency, Confusion Matrix."""
import time
import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def threshold_predict(probs: np.ndarray, thresholds: dict) -> np.ndarray:
    """argmax 대신 클래스별 threshold로 예측.

    thresholds 예: {3: 0.18, 1: 0.15, 2: 0.30}
    우선순위: 진행(3) > 초기(1) > 경계(2) > 정상(0)
    """
    preds = np.zeros(len(probs), dtype=int)
    for i, p in enumerate(probs):
        if p[3] >= thresholds.get(3, 0.18):
            preds[i] = 3
        elif p[1] >= thresholds.get(1, 0.15):
            preds[i] = 1
        elif p[2] >= thresholds.get(2, 0.30):
            preds[i] = 2
        else:
            preds[i] = 0
    return preds


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_stages: int = 5,
    thresholds: dict | None = None,
) -> dict:
    """ST-MMT 또는 TinyTransformer 평가.

    Returns:
        dict with keys: f1_macro, f1_weighted, auc_ovo,
                        accuracy, confusion_matrix, report,
                        avg_latency_ms
    """
    model.eval()
    all_preds, all_labels = [], []
    all_probs = []
    latencies = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch
            else:
                raise ValueError("Dataloader must yield (x, y) pairs.")

            x = x.to(device)
            y_np = y.view(-1).cpu().numpy()

            t0 = time.perf_counter()
            out = model(x)
            latencies.append((time.perf_counter() - t0) * 1000)

            # TinyTransformer 출력
            if "stage_logits" in out:
                logits = out["stage_logits"]                # (B, n_stages)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = logits.argmax(dim=-1).cpu().numpy()
            # STMMT 출력 — last_logits (B, n_stages, H, W)
            elif "last_logits" in out:
                logits = out["last_logits"]                 # (B, n_stages, H, W)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1).view(-1).cpu().numpy()
                probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, n_stages).cpu().numpy()
                y_np = y.view(-1).cpu().numpy()
            else:
                raise ValueError("Unsupported model output format.")

            all_preds.extend(preds.flatten().tolist())
            all_labels.extend(y_np.tolist())
            all_probs.extend(probs.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # -1(IGNORE) 라벨 제거: 평가 대상에서 제외
    valid_mask = (all_labels != -1)
    all_preds = all_preds[valid_mask]
    all_labels = all_labels[valid_mask]
    all_probs = all_probs[valid_mask]

    target_names = ["정상", "초기", "경계", "진행"][:n_stages]

    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    accuracy = (all_preds == all_labels).mean()

    try:
        auc_ovo = roc_auc_score(
            all_labels, all_probs,
            multi_class="ovo", average="macro",
            labels=list(range(n_stages)),
        )
    except ValueError:
        auc_ovo = float("nan")

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(n_stages)))
    report = classification_report(
        all_labels, all_preds,
        labels=list(range(n_stages)),
        target_names=target_names,
        zero_division=0,
    )

    result = {
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "auc_ovo": float(auc_ovo),
        "accuracy": float(accuracy),
        "confusion_matrix": cm.tolist(),
        "report": report,
        "avg_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
    }

    # threshold 기반 평가 (제공된 경우)
    if thresholds is not None:
        thr_preds = threshold_predict(all_probs, thresholds)
        thr_f1 = f1_score(all_labels, thr_preds, average="macro", zero_division=0)
        thr_cm = confusion_matrix(all_labels, thr_preds, labels=list(range(n_stages)))
        thr_report = classification_report(
            all_labels, thr_preds,
            labels=list(range(n_stages)),
            target_names=target_names,
            zero_division=0,
        )
        result["threshold_f1_macro"] = float(thr_f1)
        result["threshold_confusion_matrix"] = thr_cm.tolist()
        result["threshold_report"] = thr_report
        result["thresholds_used"] = thresholds

    return result


def benchmark_latency(
    model: torch.nn.Module,
    input_shape: tuple,
    device: torch.device,
    n_runs: int = 100,
    warmup: int = 10,
) -> dict:
    """순수 추론 지연시간 벤치마크."""
    model.eval()
    dummy = torch.randn(*input_shape, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)

    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy)
            latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
    }
