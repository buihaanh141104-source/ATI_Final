"""Seeding, device selection, parameter counting and environment capture.

    python -m vigo env      # writes hardware_and_versions.txt at the project root
"""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import contextmanager

import numpy as np

from vigo.utils.paths import PROJECT_ROOT

DEFAULT_SEED = 42

CHECKPOINTS = {
    "PhoBERT (required)": "vinai/phobert-base-v2",
    "ViSoBERT": "uitnlp/visobert",
    "ViSoBERT-14GB": "5CD-AI/visobert-14gb-corpus",
    "CafeBERT (optional, GPU)": "uitnlp/CafeBERT",
    "PhoBERT-large (optional, GPU)": "vinai/phobert-large",
    "Word segmenter": "VnCoreNLP-1.2 RDRSegmenter (fallback: pyvi)",
}


def seed_everything(seed: int = DEFAULT_SEED, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(prefer: str | None = None):
    import torch

    if prefer and prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": int(total), "trainable_params": int(trainable)}


@contextmanager
def timer():
    """Usage: `with timer() as t: ...; t()` returns elapsed seconds."""
    start = time.perf_counter()
    elapsed = [None]
    yield lambda: elapsed[0] if elapsed[0] is not None else time.perf_counter() - start
    elapsed[0] = time.perf_counter() - start


def environment_info() -> dict:
    """Hardware + package versions, recorded next to every run for reproducibility."""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "in_colab": "google.colab" in sys.modules,
    }
    if sys.platform == "darwin":
        try:
            info["cpu_brand"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            pass
    from importlib import metadata

    for p in ["numpy", "pandas", "scikit-learn", "scipy", "torch", "transformers", "pyvi", "emoji"]:
        try:
            info[f"pkg_{p}"] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
        info["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except ImportError:
        pass
    return info


def main(argv=None) -> None:
    """Write hardware_and_versions.txt (hardware, package versions, seed, checkpoint revisions)."""
    lines = ["# Hardware, package versions, seeds", "", json.dumps(environment_info(), indent=2), "",
             f"global seed: {DEFAULT_SEED} (python/numpy/torch; every config sets seed: 42)", "",
             "pre-trained checkpoints (Hugging Face ids, main branch at download time):"]
    lines += [f"  {k}: {v}" for k, v in CHECKPOINTS.items()]
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        lines += ["", "resolved checkpoint revisions (commit sha):"]
        for v in CHECKPOINTS.values():
            if "/" in v and " " not in v:
                try:
                    lines.append(f"  {v}: {api.model_info(v).sha}")
                except Exception as e:  # offline
                    lines.append(f"  {v}: (unresolved: {e.__class__.__name__})")
    except ImportError:
        pass
    (PROJECT_ROOT / "hardware_and_versions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
