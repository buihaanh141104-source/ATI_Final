"""Fast sanity tests: python -m pytest -q tests"""
import unicodedata

import numpy as np

from vigo.evaluation.metrics import evaluate
from vigo.data.preprocessing import TextNormalizer, replace_teencode
from vigo.evaluation.thresholds import fit_all, tune_global, tune_per_label


def test_nfc_prevents_teencode_corruption():
    nfd = unicodedata.normalize("NFD", "cảm thấy tự hào về đất nước mik")
    assert TextNormalizer({"emoji": "keep"})(nfd) == "cảm thấy tự hào về đất nước mình"


def test_tone_and_elongation():
    n = TextNormalizer({"emoji": "keep", "teencode": False})
    assert n("Hòa bình khỏe quáaaa đẹppppp") == "hoà bình khoẻ quá đẹp"
    assert n("xoong nồi") == "xoong nồi"


def test_emoji_modes_and_emoticons():
    assert TextNormalizer({"emoji": "keep"})("vui 😂😂😂 =)))))") == "vui 😂 =))"
    assert TextNormalizer({"emoji": "text"})("buồn 😭") == "buồn khóc lớn"
    assert TextNormalizer({"emoji": "remove"})("buồn 😭 :((") == "buồn"
    assert ":3" not in TextNormalizer({"emoji": "keep"})("lúc 10:30")


def test_teencode_token_level():
    assert replace_teencode("ko biết, dc ko") == "không biết, được không"
    assert replace_teencode("kocó") == "kocó"


def test_metrics_perfect_and_threshold_tuning():
    rng = np.random.default_rng(0)
    y = (rng.random((500, 5)) < 0.2).astype(int)
    y[y.sum(1) == 0, 0] = 1
    probs = np.clip(y * 0.35 + rng.random(y.shape) * 0.3, 0, 1)  # positives in [0.35, 0.65]
    m = evaluate(y, (y > 0).astype(float), 0.5)
    assert m["macro_f1"] == 1.0 and m["hamming_loss"] == 0.0 and m["lrap"] == 1.0
    tau = tune_global(y, probs)
    assert tau < 0.5  # optimal global threshold is below 0.5 for these scores
    taus = tune_per_label(y, probs)
    assert taus.shape == (5,)
    th = fit_all(y, probs)
    assert evaluate(y, probs, th["per_label"])["macro_f1"] >= evaluate(y, probs, 0.5)["macro_f1"]
