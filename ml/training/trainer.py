"""ST-MMT / TinyTransformer 학습 루프."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset

from ml.training.eval import evaluate_model

try:
    import wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False

try:
    from rich.progress import (
        Progress, BarColumn, TextColumn,
        TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn,
    )
    from rich.console import Console
    from rich.table import Table
    _RICH_OK = sys.stdout.isatty()  # nohup/파일 리다이렉트 시 비활성화
except ImportError:
    _RICH_OK = False


class FocalLoss(nn.Module):
    """Multi-class Focal Loss — 희귀 이벤트 클래스 학습 강화.

    alpha는 cross_entropy weight가 아닌 샘플별 별도 곱셈으로 적용.
    (weight를 CE에 넣으면 pt = exp(-weighted_ce)가 되어 focal factor 왜곡됨)
    """

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0, ignore_index: int = -1):
        super().__init__()
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # F.cross_entropy에 ignore_index를 전달하여 무시 대상 픽셀의 CE loss가 0이 되도록 함
        ce = F.cross_entropy(logits, targets, reduction="none", ignore_index=self.ignore_index)
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        
        mask = (targets != self.ignore_index)
        if self.alpha is not None:
            # targets에 ignore_index(-1)가 있을 때 index out of bounds 방지를 위해 safe_targets 생성
            safe_targets = targets.clone()
            safe_targets[~mask] = 0
            focal = self.alpha[safe_targets] * focal
            
        valid_pixels = mask.sum()
        if valid_pixels > 0:
            return (focal * mask.float()).sum() / valid_pixels
        else:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)


class MulticlassDiceLoss(nn.Module):
    """Multi-class Dice Loss — 픽셀 영역의 시공간적 겹침(Dice) 극대화."""

    def __init__(self, ignore_index: int = -1, eps: float = 1e-6):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=1)
        
        # ignore_index 마스크 생성
        mask = (targets != self.ignore_index).float()  # (N,)
        
        # targets 원핫 벡터 변환
        safe_targets = targets.clone()
        safe_targets[targets == self.ignore_index] = 0
        targets_onehot = F.one_hot(safe_targets, num_classes=num_classes).float()  # (N, C)
        
        # 마스크 적용
        probs = probs * mask.unsqueeze(1)
        targets_onehot = targets_onehot * mask.unsqueeze(1)
        
        # Dice 계산
        intersection = (probs * targets_onehot).sum(dim=0)
        union = probs.sum(dim=0) + targets_onehot.sum(dim=0)
        
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        
        # Dice Loss = 1 - mean(dice)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Focal Loss + Dice Loss 1:1 블렌딩 손실 함수."""

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0, ignore_index: int = -1):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index)
        self.dice = MulticlassDiceLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal(logits, targets) + self.dice(logits, targets)


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class Trainer:
    """범용 학습기 — ST-MMT와 TinyTransformer 모두 지원."""

    def __init__(
        self,
        model: nn.Module,
        dataset: torch.utils.data.Dataset,
        save_dir: str = "checkpoints",
        device: str = "cpu",
        lr: float = 3e-4,
        batch_size: int = 8,
        epochs: int = 30,
        val_ratio: float = 0.2,
        patience: int = 5,
        n_stages: int = 5,
        val_dataset: torch.utils.data.Dataset | None = None,
        use_wandb: bool = False,
        wandb_project: str = "hwangbaek",
        wandb_name: str | None = None,
        wandb_config: dict | None = None,
        class_weights: list[float] | None = None,
        focal_gamma: float = 1.0,
        thresholds: dict | None = None,
    ):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.epochs = epochs
        self.n_stages = n_stages
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if val_dataset is not None:
            train_ds, val_ds = dataset, val_dataset
        else:
            n_val = max(1, int(len(dataset) * val_ratio))
            n_train = len(dataset) - n_val
            train_ds, val_ds = random_split(dataset, [n_train, n_val])

        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self.val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        cw = torch.tensor(
            class_weights if class_weights is not None else [0.25, 2.0, 1.5, 3.0],
            dtype=torch.float32,
            device=self.device,
        )
        self.criterion = CombinedLoss(alpha=cw, gamma=focal_gamma, ignore_index=-1)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )
        self.early_stop = EarlyStopping(patience=patience)
        self.thresholds = thresholds
        self.history: list[dict] = []

        # wandb 초기화
        self.use_wandb = use_wandb and _WANDB_OK
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_name,
                config=wandb_config or {},
                reinit=True,
            )
            wandb.watch(model, log="gradients", log_freq=50)
        elif use_wandb and not _WANDB_OK:
            print("[trainer] wandb 미설치 — 로그 없이 진행")

    def _forward_loss(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        out = self.model(x)

        if "stage_logits" in out:
            logits = out["stage_logits"]
            loss = self.criterion(logits, y.view(-1))
        elif "last_logits" in out:
            loss_dict = self.model.compute_loss(out, y)
            loss = loss_dict["total"]
        else:
            raise ValueError("Unsupported model output.")

        return loss, out

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            loss, _ = self._forward_loss(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def val_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                loss, _ = self._forward_loss(batch)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def fit(self) -> dict:
        """학습 실행."""
        n_params = self.model.count_params()
        print(f"학습 시작 | device={self.device} | params={n_params:,}")
        best_val = float("inf")
        best_path = self.save_dir / "best_model.pt"

        if _RICH_OK:
            console = Console()
            progress = Progress(
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(bar_width=35),
                MofNCompleteColumn(),
                TextColumn("train=[green]{task.fields[train]:.4f}[/]"),
                TextColumn("val=[red]{task.fields[val]:.4f}[/]"),
                TextColumn("best=[yellow]{task.fields[best]:.4f}[/]"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
                refresh_per_second=2,
            )
            task = progress.add_task(
                "Epoch", total=self.epochs,
                train=0.0, val=0.0, best=float("inf"),
            )
            progress.start()
        else:
            progress = None

        print("Epoch loop 시작", flush=True)
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            print(f"  [Epoch {epoch}] train 시작...", flush=True)
            try:
                train_loss = self.train_epoch()
            except Exception as e:
                import traceback
                print(f"  [Epoch {epoch}] train_epoch 예외: {e}", flush=True)
                traceback.print_exc()
                raise
            print(f"  [Epoch {epoch}] val 시작 (train_loss={train_loss:.4f})", flush=True)
            val_loss   = self.val_epoch()
            self.scheduler.step()
            elapsed = time.time() - t0
            lr = self.scheduler.get_last_lr()[0]

            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), best_path)

            row = {
                "epoch":      epoch,
                "train_loss": round(train_loss, 4),
                "val_loss":   round(val_loss, 4),
                "lr":         lr,
                "elapsed_s":  round(elapsed, 1),
            }
            self.history.append(row)

            if self.use_wandb:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss,
                           "lr": lr, "best_val": best_val}, step=epoch)

            if progress:
                progress.update(task, advance=1,
                                train=train_loss, val=val_loss, best=best_val)
            else:
                print(
                    f"  Epoch {epoch:3d}/{self.epochs} | "
                    f"train={train_loss:.4f} | val={val_loss:.4f} | "
                    f"lr={lr:.2e} | {elapsed:.1f}s"
                )

            if self.early_stop.step(val_loss):
                if progress:
                    progress.stop()
                print(f"  Early stopping at epoch {epoch}")
                break

        if progress and not progress.finished:
            progress.stop()

        self.model.load_state_dict(torch.load(best_path, weights_only=True))
        eval_result = evaluate_model(
            self.model, self.val_loader, self.device, self.n_stages,
            thresholds=self.thresholds,
        )

        history_path = self.save_dir / "history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

        print(f"\n  최종 val F1 (macro): {eval_result['f1_macro']:.4f}")
        print(f"  최종 val AUC  (ovo): {eval_result['auc_ovo']:.4f}")
        print(f"  평균 추론 지연: {eval_result['avg_latency_ms']:.1f} ms")
        print(f"\n{eval_result['report']}")

        if self.use_wandb:
            wandb.log({
                "final/f1_macro":  eval_result["f1_macro"],
                "final/auc_ovo":   eval_result["auc_ovo"],
                "final/accuracy":  eval_result["accuracy"],
            })
            wandb.finish()

        return {"history": self.history, "eval": eval_result, "best_path": str(best_path)}
