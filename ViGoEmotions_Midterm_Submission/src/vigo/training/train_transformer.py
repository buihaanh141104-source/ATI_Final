"""Train M3 - a pre-trained transformer (PhoBERT by default) with an independent sigmoid head.

    python -m vigo train-transformer --config configs/phobert.yaml
    python -m vigo train-transformer --config configs/visobert.yaml train.epochs=8 seed=13

Checkpoint selection uses VALIDATION only; probabilities for val and test are saved before any
thresholding (results/runs/<run_name>/{val,test}.npz).
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from vigo.data.dataset import load_data
from vigo.data.loaders import EncodedDataset, LengthGroupedSampler, make_transformer_collate
from vigo.data.preprocessing import preprocess_texts
from vigo.evaluation.runs import finalize_run
from vigo.models.losses import build_loss
from vigo.models.transformer import TransformerMultiLabel, llrd_param_groups, load_tokenizer
from vigo.training.trainer import fit, predict_probs
from vigo.utils import RESULTS_DIR, count_parameters, ensure_dirs, get_device, load_config, seed_everything


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vigo train-transformer")
    ap.add_argument("--config", default="configs/phobert.yaml")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    ensure_dirs()
    seed = cfg.get("seed", 42)
    seed_everything(seed)
    device = get_device(cfg.get("device"))
    name = cfg.get("run_name", "phobert")
    m, t = cfg["model"], cfg["train"]
    print(f"device={device} model={m['name']} run={name}")

    data = load_data(args.data_dir, cfg.get("split", "official"))
    texts = {s: preprocess_texts(data.texts(s), cfg["preprocess"]) for s in ("train", "val", "test")}
    tok = load_tokenizer(m["name"], m.get("use_fast", True))
    max_len = m.get("max_len", 192)
    enc = {s: tok(texts[s], truncation=True, max_length=max_len, add_special_tokens=True)["input_ids"] for s in texts}
    lens = np.array([len(x) for x in enc["train"]])
    trunc = np.mean([len(x) >= max_len for x in enc["train"]])
    unk_rate = np.mean([i == tok.unk_token_id for x in enc["val"] for i in x])
    print(f"subword length p50={np.median(lens):.0f} p95={np.percentile(lens, 95):.0f} "
          f"p99={np.percentile(lens, 99):.0f} truncated={trunc:.2%} val UNK rate={unk_rate:.3%}")

    bs = t["batch_size"]
    collate = make_transformer_collate(tok.pad_token_id)
    ebs = t.get("eval_batch_size", 128)
    loaders = {
        "train": DataLoader(EncodedDataset(enc["train"], data.y["train"]), batch_size=bs,
                            sampler=LengthGroupedSampler(lens, bs, seed), collate_fn=collate),
        "val": DataLoader(EncodedDataset(enc["val"], data.y["val"]), batch_size=ebs, collate_fn=collate),
        "test": DataLoader(EncodedDataset(enc["test"], data.y["test"]), batch_size=ebs, collate_fn=collate),
    }

    model = TransformerMultiLabel(
        m["name"], data.n_labels, pooling=m.get("pooling", "cls"), dropout=m.get("dropout", 0.1),
        multi_sample_dropout=m.get("multi_sample_dropout", 1), hidden_dropout=m.get("hidden_dropout"),
        attention_dropout=m.get("attention_dropout"),
        label_prior=data.y["train"].mean(0) if m.get("prior_bias_init", True) else None,
        gradient_checkpointing=m.get("gradient_checkpointing", False),
    ).to(device)
    params = count_parameters(model)
    print(params)

    groups = llrd_param_groups(model, t["lr"], t.get("head_lr", t["lr"]), t.get("llrd", 1.0), t.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(groups, lr=t["lr"], eps=t.get("adam_eps", 1e-6), betas=(0.9, 0.999))
    grad_accum = t.get("grad_accum", 1)
    steps = (len(loaders["train"]) // grad_accum + 1) * t["epochs"]
    sched_fn = get_cosine_schedule_with_warmup if t.get("scheduler", "linear") == "cosine" else get_linear_schedule_with_warmup
    scheduler = sched_fn(optimizer, int(t.get("warmup_ratio", 0.1) * steps), steps)
    loss_fn = build_loss(cfg.get("loss"), data.y["train"])

    res = fit(model, loaders["train"], loaders["val"], data.y["val"], loss_fn, optimizer, scheduler, device,
              epochs=t["epochs"], select_metric=t.get("select_metric", "val_macro_f1"), patience=t.get("patience"),
              amp=t.get("amp", "fp16"), grad_clip=t.get("grad_clip", 1.0), grad_accum=grad_accum, log_every=20,
              max_steps=t.get("max_steps"))
    probs = {s: predict_probs(model, loaders[s], device, res["amp_dtype"]) for s in ("val", "test")}
    if cfg.get("save_model", False):
        ckpt = RESULTS_DIR / "checkpoints" / name
        ckpt.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt / "model.pt")
        tok.save_pretrained(ckpt)
    finalize_run(
        name, {**cfg, "subword_len_p99": float(np.percentile(lens, 99)), "truncated_frac": float(trunc)},
        data.label_names, {s: data.ids(s) for s in ("val", "test")}, data.y, probs, params, res["train_time_s"],
        extra={"best_epoch": res["best_epoch"], "device": str(device),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        history=res["history"],
    )


if __name__ == "__main__":
    main()
