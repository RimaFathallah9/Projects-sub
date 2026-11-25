"""Training engine: AMP, LR scheduling, early stopping, checkpointing, TensorBoard.

Model selection uses the VALIDATION split only (default monitor: val macro-F1, which is more
informative than accuracy on this imbalanced dataset). The test split is never seen here.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import SGD, Adam, AdamW, RMSprop, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from fer.constants import CLASSES
from fer.evaluation.metrics import compute_metrics
from fer.training.early_stopping import EarlyStopping
from fer.training.losses import build_loss


def build_optimizer(model: nn.Module, tcfg: dict) -> torch.optim.Optimizer:
    lr, wd = float(tcfg["lr"]), float(tcfg.get("weight_decay", 0.0))
    mult = float(tcfg.get("backbone_lr_mult", 1.0))
    if mult != 1.0 and hasattr(model, "backbone_parameters"):
        groups = [{"params": model.backbone_parameters(), "lr": lr * mult},
                  {"params": model.head_parameters(), "lr": lr}]
        params = [g for g in groups if len(g["params"]) > 0]
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    name = tcfg.get("optimizer", "adamw").lower()
    if name == "adam":
        return Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return SGD(params, lr=lr, momentum=float(tcfg.get("momentum", 0.9)), nesterov=True, weight_decay=wd)
    if name == "rmsprop":
        return RMSprop(params, lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f"Unknown optimizer '{name}'")


def build_scheduler(opt, tcfg: dict, steps_per_epoch: int, mode: str):
    """Returns (scheduler|None, when) with when in {'epoch','batch','plateau'}."""
    name = tcfg.get("scheduler", "cosine").lower()
    epochs = int(tcfg["epochs"])
    warm = int(tcfg.get("warmup_epochs", 0))
    if name == "none":
        return None, "epoch"
    if name == "cosine":
        main = lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - warm), eta_min=1e-6)
        if warm > 0:
            warmup = lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warm)
            return lr_scheduler.SequentialLR(opt, [warmup, main], milestones=[warm]), "epoch"
        return main, "epoch"
    if name == "step":
        return lr_scheduler.StepLR(opt, step_size=int(tcfg.get("step_size", 20)), gamma=float(tcfg.get("gamma", 0.1))), "epoch"
    if name == "plateau":
        return lr_scheduler.ReduceLROnPlateau(opt, mode=mode, factor=0.5, patience=int(tcfg.get("plateau_patience", 3))), "plateau"
    if name == "onecycle":
        return lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in opt.param_groups],
                                       epochs=epochs, steps_per_epoch=steps_per_epoch), "batch"
    raise ValueError(f"Unknown scheduler '{name}'")


class Trainer:
    def __init__(self, model: nn.Module, loaders: dict, cfg: dict, run_dir: Path,
                 class_counts, device: torch.device):
        self.model, self.loaders, self.cfg, self.run_dir, self.device = model, loaders, cfg, Path(run_dir), device
        self.t = cfg["train"]
        self.use_amp = bool(self.t.get("amp", True)) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.criterion = build_loss(self.t, class_counts, device)
        self.optimizer = build_optimizer(model, self.t)
        self.monitor = self.t.get("monitor", "val_macro_f1")
        self.mode = "min" if self.monitor == "val_loss" else "max"
        self.scheduler, self.sched_when = build_scheduler(self.optimizer, self.t, len(loaders["train"]), self.mode)
        self.stopper = EarlyStopping(int(self.t.get("patience", 10)), self.mode, float(self.t.get("min_delta", 0.0)))
        self.writer = SummaryWriter(str(self.run_dir / "tb"))
        self.history: list[dict] = []
        self.best_path = self.run_dir / "best.pt"

    # ------------------------------------------------------------------ loops
    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        loss_sum, correct, n = 0.0, 0, 0
        clip = float(self.t.get("grad_clip", 0.0))
        for x, y in tqdm(self.loaders["train"], desc=f"epoch {epoch} train", leave=False):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                out = self.model(x)
                loss = self.criterion(out, y)
            self.scaler.scale(loss).backward()
            if clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.sched_when == "batch":
                self.scheduler.step()
            bs = y.size(0)
            loss_sum += loss.item() * bs
            correct += (out.argmax(1) == y).sum().item()
            n += bs
        return {"train_loss": loss_sum / n, "train_acc": correct / n}

    @torch.no_grad()
    def _eval_epoch(self) -> dict:
        self.model.eval()
        loss_sum, n = 0.0, 0
        preds, ys = [], []
        for x, y in self.loaders["val"]:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                out = self.model(x)
                loss = self.criterion(out, y)
            loss_sum += loss.item() * y.size(0)
            n += y.size(0)
            preds.append(out.argmax(1).cpu())
            ys.append(y.cpu())
        p, t = torch.cat(preds).numpy(), torch.cat(ys).numpy()
        return {"val_loss": loss_sum / n, "val_acc": float((p == t).mean()),
                "val_macro_f1": float(f1_score(t, p, average="macro", zero_division=0))}

    # ------------------------------------------------------------------ main
    def fit(self) -> dict:
        epochs = int(self.t["epochs"])
        t_start = time.time()
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            rec = {"epoch": epoch, **self._train_epoch(epoch), **self._eval_epoch()}
            rec["lr"] = self.optimizer.param_groups[-1]["lr"]
            rec["epoch_time_s"] = time.time() - t0
            monitored = rec[self.monitor]

            if self.scheduler is not None and self.sched_when == "epoch":
                self.scheduler.step()
            elif self.sched_when == "plateau":
                self.scheduler.step(monitored)

            improved = self.stopper.step(monitored, epoch)
            if improved:
                torch.save({"model": self.model.state_dict(), "config": self.cfg, "epoch": epoch,
                            "val_metrics": {k: float(v) for k, v in rec.items()}, "class_names": CLASSES},
                           self.best_path)
            self.history.append(rec)
            for k, v in rec.items():
                if k != "epoch":
                    self.writer.add_scalar(k, v, epoch)
            (self.run_dir / "history.json").write_text(json.dumps(self.history, indent=1))
            print(f"[{epoch:03d}/{epochs}] train_loss {rec['train_loss']:.4f} acc {rec['train_acc']:.4f} | "
                  f"val_loss {rec['val_loss']:.4f} acc {rec['val_acc']:.4f} f1 {rec['val_macro_f1']:.4f} | "
                  f"lr {rec['lr']:.2e} | {rec['epoch_time_s']:.1f}s{'  *best*' if improved else ''}")
            if self.stopper.should_stop:
                print(f"Early stopping at epoch {epoch} (best epoch {self.stopper.best_epoch}).")
                break
        train_time = time.time() - t_start
        self.writer.close()

        # Re-evaluate the best checkpoint on the validation split with the full metric set.
        ckpt = torch.load(self.best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        val_metrics = self._full_val_metrics()
        return {
            "best_epoch": self.stopper.best_epoch, "monitor": self.monitor,
            "best_monitor_value": float(self.stopper.best), "epochs_run": len(self.history),
            "train_time_s": train_time,
            "mean_epoch_time_s": float(np.mean([h["epoch_time_s"] for h in self.history])),
            "val_metrics": val_metrics,
        }

    @torch.no_grad()
    def _full_val_metrics(self) -> dict:
        self.model.eval()
        preds, ys = [], []
        for x, y in self.loaders["val"]:
            out = self.model(x.to(self.device))
            preds.append(out.argmax(1).cpu())
            ys.append(y)
        return compute_metrics(torch.cat(ys).numpy(), torch.cat(preds).numpy())
