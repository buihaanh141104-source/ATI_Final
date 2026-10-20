"""Train M2 - BiLSTM (or BiGRU) baseline.

    python -m vigo train-rnn --config configs/bilstm.yaml

Tokenisation: normalised text -> VnCoreNLP word segmentation -> whitespace tokens; emojis and
emoticons are kept as their own tokens. Vocabulary is fitted on TRAIN only (min_freq=2).
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from vigo.data.dataset import load_data
from vigo.data.loaders import SeqDataset, rnn_collate
from vigo.data.preprocessing import preprocess_texts
from vigo.data.vocab import Vocab, load_word_vectors
from vigo.evaluation.runs import finalize_run
from vigo.models.losses import build_loss
from vigo.models.rnn import RNNClassifier
from vigo.training.trainer import fit, predict_probs
from vigo.utils import count_parameters, ensure_dirs, get_device, load_config, seed_everything


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vigo train-rnn")
    ap.add_argument("--config", default="configs/bilstm.yaml")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    ensure_dirs()
    seed = cfg.get("seed", 42)
    seed_everything(seed)
    device = get_device(cfg.get("device"))
    if device.type == "mps" and not cfg.get("allow_mps", False):
        device = torch.device("cpu")  # cuDNN-less LSTM on MPS is ~2x slower than CPU on Apple Silicon
    name = cfg.get("run_name", "bilstm")
    print("device:", device)

    data = load_data(args.data_dir, cfg.get("split", "official"))
    texts = {s: preprocess_texts(data.texts(s), cfg["preprocess"]) for s in ("train", "val", "test")}
    toks = {s: [t.split() for t in texts[s]] for s in texts}
    m = cfg["model"]
    vocab = Vocab(toks["train"], min_freq=m.get("min_freq", 2), max_size=m.get("max_vocab", 50000))
    max_len = m.get("max_len", 128)
    ids = {s: [vocab.encode(t, max_len) for t in toks[s]] for s in toks}
    oov = np.mean([i == 1 for doc in ids["val"] for i in doc])
    print(f"vocab={len(vocab):,}  val OOV rate={oov:.3%}")

    pretrained, coverage = None, None
    if m.get("pretrained_vectors"):
        pretrained, coverage = load_word_vectors(m["pretrained_vectors"], vocab, m["emb_dim"], seed)
        print(f"pretrained vectors coverage: {coverage:.1%}")

    g = torch.Generator().manual_seed(seed)
    bs = cfg["train"]["batch_size"]
    loaders = {
        "train": DataLoader(SeqDataset(ids["train"], data.y["train"]), batch_size=bs, shuffle=True,
                            collate_fn=rnn_collate, generator=g),
        "val": DataLoader(SeqDataset(ids["val"], data.y["val"]), batch_size=256, collate_fn=rnn_collate),
        "test": DataLoader(SeqDataset(ids["test"], data.y["test"]), batch_size=256, collate_fn=rnn_collate),
    }
    model = RNNClassifier(
        len(vocab), data.n_labels, emb_dim=m["emb_dim"], hidden=m["hidden"], layers=m["layers"],
        cell=m.get("cell", "lstm"), emb_dropout=m["emb_dropout"], rnn_dropout=m["rnn_dropout"],
        dropout=m["dropout"], pretrained=pretrained, label_prior=data.y["train"].mean(0),
    ).to(device)
    params = count_parameters(model)
    print(params)

    t = cfg["train"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 0.0))
    steps = len(loaders["train"]) * t["epochs"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=t["lr"], total_steps=steps,
                                                    pct_start=t.get("warmup_ratio", 0.1), anneal_strategy="cos")
    loss_fn = build_loss(cfg.get("loss"), data.y["train"])
    res = fit(model, loaders["train"], loaders["val"], data.y["val"], loss_fn, optimizer, scheduler, device,
              epochs=t["epochs"], select_metric=t.get("select_metric", "val_macro_f1"), patience=t.get("patience"),
              grad_clip=t.get("grad_clip", 1.0), max_steps=t.get("max_steps"))
    probs = {s: predict_probs(model, loaders[s], device) for s in ("val", "test")}
    finalize_run(name, {**cfg, "vocab_size": len(vocab), "pretrained_coverage": coverage}, data.label_names,
                 {s: data.ids(s) for s in ("val", "test")}, data.y, probs, params, res["train_time_s"],
                 extra={"best_epoch": res["best_epoch"], "device": str(device)}, history=res["history"])


if __name__ == "__main__":
    main()
