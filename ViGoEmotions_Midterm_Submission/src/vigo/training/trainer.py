"""Generic PyTorch training loop shared by the BiLSTM and transformer models.

* model(**batch) -> logits (B, C); batch["labels"] is the multi-hot target.
* Model selection: best VALIDATION epoch by `select_metric` (default macro-F1 @ tau=0.5),
  early stopping with `patience`. Test data is never touched here.
* Mixed precision on CUDA (fp16 or bf16); fp32 on MPS/CPU.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from tqdm.auto import tqdm

from vigo.evaluation.thresholds import tune_global


def move(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def predict_probs(model, loader, device, amp_dtype=None) -> np.ndarray:
    model.eval()
    out = []
    for batch in loader:
        batch = move(batch, device)
        batch.pop("labels", None)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(**batch)
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out, 0)


def val_scores(y, probs) -> dict:
    pred = (probs >= 0.5).astype(np.int8)
    tau = tune_global(y, probs)
    return {
        "val_macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "val_micro_f1": f1_score(y, pred, average="micro", zero_division=0),
        "val_map": average_precision_score(y, probs, average="macro"),
        "val_macro_f1_tuned": f1_score(y, (probs >= tau).astype(np.int8), average="macro", zero_division=0),
        "val_best_global_tau": tau,
    }


def fit(
    model,
    train_loader,
    val_loader,
    y_val: np.ndarray,
    loss_fn,
    optimizer,
    scheduler,
    device,
    epochs: int,
    select_metric: str = "val_macro_f1",
    patience: int | None = None,
    amp: str | None = None,  # None | "fp16" | "bf16"
    grad_clip: float = 1.0,
    grad_accum: int = 1,
    log_every: int = 0,
    max_steps: int | None = None,  # smoke-test option: stop each epoch after this many batches
) -> dict:
    amp_dtype = None
    if amp and device.type == "cuda":
        amp_dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    loss_fn = loss_fn.to(device)
    best = {"score": -math.inf, "epoch": -1, "state": None}
    history = []
    bad_epochs = 0
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False, mininterval=5)
        for step, batch in enumerate(pbar, 1):
            if max_steps and step > max_steps:
                break
            batch = move(batch, device)
            labels = batch.pop("labels").float()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(**batch)
            loss = loss_fn(logits.float(), labels) / grad_accum
            scaler.scale(loss).backward()
            if step % grad_accum == 0 or step == len(train_loader) or step == max_steps:
                if grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
            running += loss.item() * grad_accum
            n_batches += 1
            if log_every and step % log_every == 0:
                pbar.set_postfix(loss=f"{running / n_batches:.4f}")
        train_time = time.perf_counter() - t0
        probs = predict_probs(model, val_loader, device, amp_dtype)
        scores = val_scores(y_val, probs)
        rec = {"epoch": epoch, "train_loss": running / max(n_batches, 1), "epoch_time_s": train_time, **scores}
        history.append(rec)
        improved = scores[select_metric] > best["score"] + 1e-6
        print(
            f"epoch {epoch:>2} loss={rec['train_loss']:.4f} val_macroF1@.5={scores['val_macro_f1']:.4f} "
            f"microF1@.5={scores['val_micro_f1']:.4f} mAP={scores['val_map']:.4f} "
            f"macroF1@tau*={scores['val_macro_f1_tuned']:.4f} (tau*={scores['val_best_global_tau']:.2f}) "
            f"[{train_time:.0f}s]{' *' if improved else ''}",
            flush=True,
        )
        if improved:
            best = {"score": scores[select_metric], "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if patience is not None and bad_epochs >= patience:
                print(f"early stopping at epoch {epoch} (best epoch {best['epoch']})")
                break
    total_time = time.perf_counter() - t_start
    model.load_state_dict(best["state"])
    return {"history": history, "best_epoch": best["epoch"], "best_score": best["score"],
            "train_time_s": total_time, "amp_dtype": amp_dtype}
