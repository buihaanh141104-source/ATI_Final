"""Single command-line entry point:  python -m vigo <command> [options]   (or `vigo <command>`).

Each command maps to the `main(argv)` function of one module, so the modules can also be run
directly, e.g. `python -m vigo.training.train_tfidf --config configs/tfidf_lr.yaml`.
"""
from __future__ import annotations

import importlib
import sys

COMMANDS: dict[str, tuple[str, str, str]] = {
    # command            module                              function       help
    "eda":               ("vigo.analysis.eda",               "main",        "dataset analysis tables and figures (WP1)"),
    "split":             ("vigo.data.split",                 "main",        "write split ids and audit multi-label stratification (WP2)"),
    "train-tfidf":       ("vigo.training.train_tfidf",       "main",        "train M1: TF-IDF + One-vs-Rest Logistic Regression"),
    "train-rnn":         ("vigo.training.train_rnn",         "main",        "train M2: BiLSTM / BiGRU"),
    "train-transformer": ("vigo.training.train_transformer", "main",        "train M3: PhoBERT / ViSoBERT / CafeBERT ..."),
    "ensemble":          ("vigo.training.ensemble",          "main",        "average the saved probabilities of several runs"),
    "ensemble-select":   ("vigo.training.ensemble",          "select_main", "compare ensemble compositions on validation only"),
    "analyze":           ("vigo.analysis.midterm",           "main",        "result tables, Midterm questions Q1-Q5, error analysis"),
    "env":               ("vigo.utils.reproducibility",      "main",        "write hardware_and_versions.txt"),
}


def usage() -> str:
    lines = ["usage: python -m vigo <command> [options]", "", "commands:"]
    lines += [f"  {name:<18} {help_}" for name, (_, _, help_) in COMMANDS.items()]
    lines += ["", "run `python -m vigo <command> -h` for the options of a command"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(usage())
        return
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"unknown command: {cmd}\n\n{usage()}", file=sys.stderr)
        sys.exit(2)
    module, func, _ = COMMANDS[cmd]
    getattr(importlib.import_module(module), func)(rest)
