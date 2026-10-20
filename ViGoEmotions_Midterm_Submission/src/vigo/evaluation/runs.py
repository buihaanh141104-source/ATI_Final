"""Run bookkeeping: every model writes the same artefacts to results/runs/<run_name>/.

  config.json         resolved configuration (hyper-parameters, preprocessing, seed)
  env.json            hardware + package versions
  val.npz / test.npz  ids, y_true, probs (continuous, BEFORE thresholding)
  summary.json        params, wall-clock train time, val/test metrics @ tau=0.5
  history.csv         per-epoch log (neural models)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vigo.evaluation.metrics import evaluate
from vigo.utils import RUNS_DIR, environment_info, load_json, save_json


def run_dir(name: str) -> Path:
    d = RUNS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_probs(name: str, split: str, ids, y_true, probs) -> None:
    np.savez_compressed(
        run_dir(name) / f"{split}.npz",
        ids=np.asarray(ids, dtype=str),
        y_true=np.asarray(y_true, dtype=np.int8),
        probs=np.asarray(probs, dtype=np.float32),
    )


def load_probs(name: str, split: str) -> dict:
    z = np.load(RUNS_DIR / name / f"{split}.npz", allow_pickle=False)
    return {"ids": z["ids"], "y_true": z["y_true"], "probs": z["probs"]}


def finalize_run(
    name: str,
    cfg: dict,
    label_names: list[str],
    data_ids: dict,
    y: dict,
    probs: dict,
    params: dict,
    train_time_s: float,
    extra: dict | None = None,
    history: list[dict] | None = None,
) -> dict:
    d = run_dir(name)
    save_json(cfg, d / "config.json")
    save_json(environment_info(), d / "env.json")
    summary = {"run": name, **params, "train_time_s": float(train_time_s)}
    for split in ("val", "test"):
        save_probs(name, split, data_ids[split], y[split], probs[split])
        m = evaluate(y[split], probs[split], 0.5, label_names)
        summary.update({f"{split}_{k}": v for k, v in m.items()})
    if extra:
        summary.update(extra)
    save_json(summary, d / "summary.json")
    if history:
        pd.DataFrame(history).to_csv(d / "history.csv", index=False)
    print(
        f"[{name}] val macro-F1={summary['val_macro_f1']:.4f} micro-F1={summary['val_micro_f1']:.4f} | "
        f"test macro-F1={summary['test_macro_f1']:.4f} micro-F1={summary['test_micro_f1']:.4f} "
        f"| params={params.get('total_params')} | time={train_time_s:.1f}s"
    )
    return summary


def load_summary(name: str) -> dict:
    return load_json(RUNS_DIR / name / "summary.json")


def available_runs() -> list[str]:
    return sorted(p.name for p in RUNS_DIR.glob("*") if (p / "summary.json").exists())
