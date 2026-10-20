"""Text-style subsets for Midterm Q4. Rules are FROZEN here (decided from training-set
EDA before looking at any model output) and computed on the RAW comment text.

  short        <= 5 whitespace tokens            (~17% of train; train p25 = 7 tokens)
  long         >= 30 whitespace tokens           (~10% of train)
  emoji        >= 1 emoji                        (~25%)
  emoji_heavy  >= 2 emoji (repeats counted)      (~10%)
  emoticon     >= 1 ASCII emoticon (":))", "=))", ":v", "^^" ...)   (~11%)
  teencode     >= 1 multi-character token from the teencode dictionary
               (single letters such as "k", "m", "a" are excluded as too ambiguous) (~23%)
  plain        none of short / emoji / emoticon / teencode
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vigo.data.preprocessing import _emoticon_regex, count_emoji, teencode_hits

RULES = {
    "short": "<= 5 whitespace tokens",
    "long": ">= 30 whitespace tokens",
    "emoji": ">= 1 emoji",
    "emoji_heavy": ">= 2 emoji characters",
    "emoticon": ">= 1 ASCII emoticon",
    "teencode": ">= 1 multi-character teencode-dictionary token",
    "plain": "not short, no emoji/emoticon/teencode",
}


def style_flags(texts: list[str]) -> pd.DataFrame:
    emo_re = _emoticon_regex()
    rows = []
    for t in texts:
        n_tok = len(str(t).split())
        n_emoji = count_emoji(t)
        n_emoticon = len(emo_re.findall(str(t).lower()))
        n_teen = len([h for h in teencode_hits(t) if len(h) > 1])
        rows.append(
            {
                "n_tokens": n_tok,
                "n_chars": len(str(t)),
                "n_emoji": n_emoji,
                "n_emoticon": n_emoticon,
                "n_teencode": n_teen,
                "short": n_tok <= 5,
                "long": n_tok >= 30,
                "emoji": n_emoji >= 1,
                "emoji_heavy": n_emoji >= 2,
                "emoticon": n_emoticon >= 1,
                "teencode": n_teen >= 1,
            }
        )
    df = pd.DataFrame(rows)
    df["plain"] = ~(df["short"] | df["emoji"] | df["emoticon"] | df["teencode"])
    return df


def subset_masks(texts: list[str]) -> dict[str, np.ndarray]:
    f = style_flags(texts)
    masks = {"all": np.ones(len(texts), dtype=bool)}
    for k in RULES:
        masks[k] = f[k].to_numpy()
    return masks
