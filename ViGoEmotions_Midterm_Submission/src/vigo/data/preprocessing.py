"""Vietnamese social-media text normalisation.

Pipeline (each step is switchable from the YAML config and documented in the README):

  1. Unicode NFC + strip emoji variation selectors / skin-tone modifiers.
     (The official ViGoEmotions notebooks skip NFC; on decomposed (NFD) comments their
     regex teencode replacement corrupts words, e.g. "cảm" -> "cảmày". We fix that.)
  2. HTML entities / <br> -> newline -> sentence break; @mentions -> "@user"; URLs -> "url".
  3. Lower-casing (optional).
  4. Vietnamese tone-mark placement normalisation with VinAI's table (old -> new style:
     "hòa" -> "hoà", "khỏe" -> "khoẻ", "thủy" -> "thuỷ"), the convention of PhoBERT's
     pre-training corpus, so one word is not split into two vocabulary entries.
  5. Elongation squeeze: a letter repeated >= 3 times -> 1 ("đẹppppp" -> "đẹp");
     laughter "kkkk" -> "haha"; punctuation / emoticon runs collapsed (":)))))" -> ":))").
  6. Emoji: `keep` (deduplicate consecutive repeats, space-separate), `text`
     (emoji and ASCII emoticons -> Vietnamese description, e.g. 😭 -> "khóc lớn"),
     or `remove`.
  7. Teencode / abbreviation expansion with a token-level dictionary
     (official ViGoEmotions list + `resources/teencode_extra.txt`).
  8. Punctuation spacing + whitespace cleanup.
  9. (PhoBERT / word-level models only) Vietnamese word segmentation that joins
     multi-syllable words with "_" ("cuộc đời" -> "cuộc_đời"). Default: VnCoreNLP
     RDRSegmenter, the segmenter used to build PhoBERT's pre-training data (needs Java);
     falls back to pyvi (used by the ViGoEmotions paper) if Java is unavailable.
     Emoji / emoticon tokens bypass the segmenter (VnCoreNLP corrupts astral-plane chars).
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import emoji as emoji_lib

from vigo.utils import CACHE_DIR, RESOURCES_DIR

DEFAULT_PREPROCESS = {
    "nfc": True,
    "lowercase": True,
    "mask_user_url": True,
    "tone_normalize": True,
    "squeeze_repeats": True,
    "emoji": "keep",  # keep | text | remove
    "teencode": True,
    "segment": False,  # word segmentation (PhoBERT, BiLSTM, TF-IDF word n-grams)
    "segmenter": "vncorenlp",  # vncorenlp (RDRSegmenter, PhoBERT's own) | pyvi
}

_VS_RE = re.compile("[︎️\U0001F3FB-\U0001F3FF]")
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_BR_RE = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)
_LETTER_RUN_RE = re.compile(r"([^\W\d_])\1{2,}")
_LAUGH_K_RE = re.compile(r"\bk{3,}\b")
_PUNCT_RUNS = [
    (re.compile(r"\){2,}"), "))"),
    (re.compile(r"\({2,}"), "(("),
    (re.compile(r"\]{2,}"), "]]"),
    (re.compile(r"\[{2,}"), "[["),
    (re.compile(r">{2,}"), ">"),
    (re.compile(r"<{2,}"), "<"),
    (re.compile(r"={2,}"), "="),
    (re.compile(r":{2,}"), ":"),
    (re.compile(r"\?{2,}"), "?"),
    (re.compile(r"!{2,}"), "!"),
    (re.compile(r"\.{3,}"), "..."),
    (re.compile(r",{2,}"), ","),
]
# Tone-mark placement normalisation, VinAI table (old style -> new style, "òa" -> "oà"),
# the convention used for PhoBERT / BARTpho / PhoW2V pre-training data:
# https://github.com/VinAIResearch/BARTpho/blob/main/VietnameseToneNormalization.md
_TONE_FIX = {
    "òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã", "ọa": "oạ",
    "òe": "oè", "óe": "oé", "ỏe": "oẻ", "õe": "oẽ", "ọe": "oẹ",
    "ùy": "uỳ", "úy": "uý", "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ",
}
_TONE_FIX.update({k.capitalize(): v.capitalize() for k, v in list(_TONE_FIX.items())})
_TONE_FIX.update({k.upper(): v.upper() for k, v in list(_TONE_FIX.items())})
_TONE_RE = re.compile("|".join(map(re.escape, _TONE_FIX)))
_PUNCT_SPACE_RE = re.compile(r"([.,!?;:\"“”()\[\]])")
_WS_RE = re.compile(r"\s+")
_VN_DIACRITIC_CHARS = set(
    "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
)


def strip_variation(s: str) -> str:
    return _VS_RE.sub("", s)


def _base(ch: str) -> str:
    return unicodedata.normalize("NFD", ch)[0]


def squeeze_accented_elongation(text: str) -> str:
    """'quáaaa' -> 'quá', 'khoẻee' -> 'khoẻ': drop plain vowels repeating an accented vowel."""
    out = []
    for ch in text:
        if out and ch in "aeiouy" and out[-1] != ch and _base(out[-1]) == ch:
            continue
        out.append(ch)
    return "".join(out)


@lru_cache(maxsize=1)
def load_teencode() -> dict[str, str]:
    table: dict[str, str] = {}
    for fname in ("teencode_official.txt", "teencode_extra.txt"):
        path = RESOURCES_DIR / fname
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and parts[0].strip():
                key = unicodedata.normalize("NFC", parts[0].strip().lower())
                table.setdefault(key, unicodedata.normalize("NFC", parts[1].strip()))
    return table


@lru_cache(maxsize=1)
def load_emoji_vi() -> tuple[dict[str, str], dict[str, str]]:
    """Return (emoji -> vi description, ascii emoticon -> vi description)."""
    official = json.loads((RESOURCES_DIR / "emoji_official.json").read_text(encoding="utf-8"))
    extra_path = RESOURCES_DIR / "emoji_vi_extra.json"
    extra = json.loads(extra_path.read_text(encoding="utf-8")) if extra_path.exists() else {}
    emojis: dict[str, str] = {}
    emoticons: dict[str, str] = {}
    for src in (official, extra):  # extra overrides official (fixes e.g. 😵 -> "hôn")
        for k, v in src.items():
            key = strip_variation(unicodedata.normalize("NFC", k.strip()))
            if not key:
                continue
            desc = v.replace("_", " ").strip()
            if emoji_lib.emoji_count(key) > 0 or any(ord(c) > 0x2000 for c in key):
                emojis[key] = desc
            else:
                emoticons[key.lower()] = desc
    return emojis, emoticons


@lru_cache(maxsize=1)
def load_cldr_vi() -> dict[str, str]:
    """Unicode CLDR Vietnamese emoji names (1,975 entries) - fallback for rare emoji."""
    path = RESOURCES_DIR / "cldr_annotations_vi.xml"
    if not path.exists():
        return {}
    import xml.etree.ElementTree as ET

    table = {}
    for a in ET.parse(path).getroot().iter("annotation"):
        if a.get("type") == "tts" and a.text:
            table[strip_variation(a.get("cp"))] = a.text.strip()
    return table


@lru_cache(maxsize=1)
def _emoticon_regex() -> re.Pattern:
    _, emoticons = load_emoji_vi()
    keys = sorted(emoticons, key=len, reverse=True)
    parts = []
    for k in keys:
        pat = re.escape(k)
        if k[-1].isalnum():  # ":v", ":d", ":3" must not be followed by a letter
            pat += r"(?![^\W_])"
        parts.append(pat)
    return re.compile(r"(?<!\d)(" + "|".join(parts) + ")")


def emoji_to_text(text: str) -> str:
    emojis, emoticons = load_emoji_vi()
    last = None
    pieces = []
    pos = 0
    for m in emoji_lib.emoji_list(text):
        pieces.append(text[pos : m["match_start"]])
        e = strip_variation(m["emoji"])
        desc = emojis.get(e)
        if desc is None:
            base = e.split("‍")[0]
            desc = emojis.get(base)
        if desc is None:
            desc = load_cldr_vi().get(e) or load_cldr_vi().get(e.split("\u200d")[0])
        if desc is None:
            desc = emoji_lib.demojize(e, delimiters=("", "")).replace("_", " ")
        if desc != last:  # drop consecutive duplicates (😭😭😭 -> one description)
            pieces.append(f" {desc} ")
        last = desc
        pos = m["match_end"]
    pieces.append(text[pos:])
    text = "".join(pieces)
    text = _emoticon_regex().sub(lambda m: f" {emoticons[m.group(1).lower()]} ", text)
    return text


def dedupe_and_space_emoji(text: str) -> str:
    pieces, pos, last = [], 0, None
    for m in emoji_lib.emoji_list(text):
        between = text[pos : m["match_start"]]
        e = strip_variation(m["emoji"])
        if between.strip():
            last = None
        pieces.append(between)
        if e != last:
            pieces.append(f" {e} ")
        last = e
        pos = m["match_end"]
    pieces.append(text[pos:])
    return "".join(pieces)


def remove_emoji(text: str) -> str:
    return emoji_lib.replace_emoji(text, replace=" ")


def replace_teencode(text: str) -> str:
    table = load_teencode()
    # token-level lookup: split on whitespace, keep punctuation attached tokens intact
    out = []
    for tok in text.split(" "):
        low = tok.lower()
        if low in table:
            out.append(table[low])
            continue
        # token with trailing punctuation, e.g. "ko," -> "không,"
        m = re.match(r"^([^\W_]+)([^\w\s]+)$", tok)
        if m and m.group(1).lower() in table:
            out.append(table[m.group(1).lower()] + m.group(2))
            continue
        out.append(tok)
    return " ".join(out)


class TextNormalizer:
    def __init__(self, cfg: dict | None = None):
        self.cfg = {**DEFAULT_PREPROCESS, **(cfg or {})}
        if self.cfg["emoji"] not in {"keep", "text", "remove"}:
            raise ValueError("preprocess.emoji must be keep|text|remove")

    def __call__(self, text: str) -> str:
        c = self.cfg
        text = str(text)
        if c["nfc"]:
            text = unicodedata.normalize("NFC", text)
        text = strip_variation(text)
        text = html.unescape(text)
        text = _BR_RE.sub("\n", text)
        if c["mask_user_url"]:
            text = _URL_RE.sub(" url ", text)
            text = _MENTION_RE.sub(" @user ", text)
            text = _HASHTAG_RE.sub(r" \1 ", text)
        if c["lowercase"]:
            text = text.lower()
        if c["tone_normalize"]:
            text = _TONE_RE.sub(lambda m: _TONE_FIX[m.group(0)], text)
        if c["squeeze_repeats"]:
            text = _LAUGH_K_RE.sub(" haha ", text)
            text = _LETTER_RUN_RE.sub(r"\1", text)
            text = squeeze_accented_elongation(text)
            for pat, rep in _PUNCT_RUNS:
                text = pat.sub(rep, text)
        # newline -> sentence boundary (as in the official pipeline)
        text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
        text = re.sub(r"(?<![.,!?;:\n])\n+", ". ", text)
        text = text.replace("\n", " ")
        if c["emoji"] == "text":
            text = emoji_to_text(text)
        elif c["emoji"] == "remove":
            text = remove_emoji(text)
        else:
            text = dedupe_and_space_emoji(text)
        # protect ASCII emoticons (":))", ":v", "=((") from punctuation spacing
        protected: list[str] = []

        def _protect(m: re.Match) -> str:
            if c["emoji"] == "remove":
                return " "
            protected.append(m.group(1))
            return f" {len(protected) - 1} "

        if c["emoji"] != "text":
            text = _emoticon_regex().sub(_protect, text)
        text = _WS_RE.sub(" ", text).strip()
        if c["teencode"]:
            text = replace_teencode(text)
        text = _PUNCT_SPACE_RE.sub(r" \1 ", text)
        if protected:
            text = re.sub("(\\d+)", lambda m: protected[int(m.group(1))], text)
        text = _WS_RE.sub(" ", text).strip()
        return text if text else "."


# ----------------------------------------------------------------------------- segmentation
VNCORENLP_URL = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master"
VNCORENLP_FILES = ("VnCoreNLP-1.2.jar", "models/wordsegmenter/vi-vocab", "models/wordsegmenter/wordsegmenter.rdr")


def _ensure_vncorenlp(save_dir: Path) -> None:
    import urllib.request

    for rel in VNCORENLP_FILES:
        dst = save_dir / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading {rel} ...")
            urllib.request.urlretrieve(f"{VNCORENLP_URL}/{rel}", dst)


@lru_cache(maxsize=1)
def _vncorenlp():
    import os

    import py_vncorenlp

    save_dir = (Path(CACHE_DIR) / "vncorenlp").resolve()
    _ensure_vncorenlp(save_dir)
    cwd = os.getcwd()
    try:  # py_vncorenlp chdir()s into save_dir; restore afterwards
        return py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(save_dir))
    finally:
        os.chdir(cwd)


def _segment_chunk(chunk: str, segmenter: str) -> str:
    if segmenter == "vncorenlp":
        return " ".join(_vncorenlp().word_segment(chunk))
    from pyvi import ViTokenizer

    return ViTokenizer.tokenize(chunk)


def resolve_segmenter(segmenter: str) -> str:
    if segmenter != "vncorenlp":
        return segmenter
    try:
        _vncorenlp()
        return "vncorenlp"
    except Exception as e:  # no Java / no pyjnius -> documented fallback
        print(f"[preprocess] VnCoreNLP unavailable ({e.__class__.__name__}: {e}); falling back to pyvi")
        return "pyvi"


def segment_texts(texts: list[str], segmenter: str = "vncorenlp") -> list[str]:
    """Word segmentation ("cuộc đời" -> "cuộc_đời"); emoji/emoticons bypass the segmenter."""
    segmenter = resolve_segmenter(segmenter)
    emo_re = _emoticon_regex()

    def is_symbolic(tok: str) -> bool:
        return (emoji_lib.emoji_count(tok) > 0 or bool(emo_re.fullmatch(tok)) or tok == "@user"
                or any(ord(ch) > 0xFFFF for ch in tok))

    out = []
    for t in texts:
        chunks, buf = [], []
        for tok in t.split():
            if is_symbolic(tok):
                if buf:
                    chunks.append(("w", " ".join(buf)))
                    buf = []
                chunks.append(("s", tok))
            else:
                buf.append(tok)
        if buf:
            chunks.append(("w", " ".join(buf)))
        seg = []
        for kind, chunk in chunks:
            if kind == "s":
                seg.append(chunk)
                continue
            try:
                seg.append(_segment_chunk(chunk, segmenter))
            except Exception:
                seg.append(chunk)
        out.append(_WS_RE.sub(" ", " ".join(seg)).strip())
    return out


def preprocess_texts(texts: list[str], cfg: dict | None = None, use_cache: bool = True) -> list[str]:
    """Normalise (+ optionally segment) a list of texts, cached on disk by config hash."""
    cfg = {**DEFAULT_PREPROCESS, **(cfg or {})}
    if cfg.get("segment"):
        cfg["segmenter"] = resolve_segmenter(cfg.get("segmenter", "vncorenlp"))
    key = hashlib.md5(
        (json.dumps(cfg, sort_keys=True) + "\x00".join(texts) + _resource_signature()).encode("utf-8")
    ).hexdigest()
    cache_path = Path(CACHE_DIR) / f"prep_{key}.json"
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    norm = TextNormalizer(cfg)
    out = [norm(t) for t in texts]
    if cfg.get("segment"):
        out = segment_texts(out, cfg.get("segmenter", "vncorenlp"))
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


@lru_cache(maxsize=1)
def _resource_signature() -> str:
    h = hashlib.md5()
    for p in sorted(RESOURCES_DIR.glob("*")):
        if p.is_file():
            h.update(p.read_bytes())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- text features
def count_emoji(text: str) -> int:
    return emoji_lib.emoji_count(strip_variation(unicodedata.normalize("NFC", str(text))))


def has_diacritics(text: str) -> bool:
    return any(ch in _VN_DIACRITIC_CHARS for ch in unicodedata.normalize("NFC", str(text)).lower())


def teencode_hits(text: str) -> list[str]:
    table = load_teencode()
    toks = re.findall(r"[^\W_]+", unicodedata.normalize("NFC", str(text)).lower())
    return [t for t in toks if t in table]
