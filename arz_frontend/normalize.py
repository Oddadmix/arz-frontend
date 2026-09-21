"""Text normalization for the Egyptian Arabic TTS front-end.

Every decision below was verified empirically against the ``arz`` voice
(``ESPEAK_DATA_PATH=<fork-build> <fork-build>/src/espeak-ng -xq --ipa -v arz``):

* Bracketed citations ``[1]`` / ``[فر]`` are never voiced by a narrator, so
  they are dropped (Nabra precedent).
* Parenthesized *source tags* — ``(رويترز)``, ``(أ ف ب)``, ``(AP)`` — are
  dropped via a small denylist (policy knob: extend ``_SOURCE_TAGS``). Other
  parentheticals are kept and voiced (Nabra precedent: parens content is
  voiced by espeak).
* Latin-script runs are dropped by default (``keep_latin=False``): a narrator
  reading Egyptian does not voice "I love" in English mid-sentence, so keeping
  the phonemes would misalign text and audio. Email/URL-like tokens are
  dropped wholesale first so ``@`` / ``.`` residue never reaches espeak.
  Pass ``keep_latin=True`` to preserve code-switching; espeak auto-tags the
  spans ``(en)…(arz)`` and the cleaner strips the markers but keeps the
  English phonemes (all in-vocab).
* Clause/sentence punctuation (``، ؛ ؟ ? ! . … :`` and friends) is mapped to
  a plain space. Verified phoneme-neutral: ``أهلا، وسهلا`` and ``أهلا وسهلا``
  produce byte-identical phonemes apart from the clause newline, which the
  cleaner turns into a space anyway — and in ``-xq --ipa`` mode espeak never
  emits pause tokens for punctuation, so nothing is lost. This also gives the
  batch phonemizer its 1:1 input→output line guarantee (see ``phonemize.py``).
* ``%`` and ``$`` are KEPT: espeak verbalizes them (``50%`` → ``xamsiːn
  filmijja``, ``100$`` → ``mijja dullar``) — dropping them would lose spoken
  content. ``&`` is mapped to space (espeak mangles it).
* Tatweel ``ـ`` is stripped (pure decoration; espeak drops it silently).
* Alef/hamza forms (``أ إ آ ء ؤ ئ``) are deliberately NOT normalized: the voice
  handles every form, and folding them would change phonemization
  (``آ`` → ``ʔaː`` vs ``ا`` → ``a``).
* Tashkeel (diacritics), when present in the input, is kept as-is: it is
  signal, and the voice was validated on diacritized Egyptian (89.1% types).
"""

import re

# Bracketed citation/footnote markers — e.g. [1], [132][133], [فر].
# A narrator does not voice these (Nabra precedent).
_CITATION = re.compile(r"\[[^\]]*\]")

# Parenthesized source tags: news-agency credits etc. A narrator does not
# voice "(رويترز)". POLICY KNOB: extend the alternation for new agencies.
_SOURCE_TAGS = (
    r"رويترز|أ\s?ف\s?ب|ا\s?ف\s?ب|فرانس\s?برس|أسوشيتد\s?برس|"
    r"reuters|associated\s?press|ap|afp"
)
_SOURCE_TAG = re.compile(rf"\((?:{_SOURCE_TAGS})\.?\)", re.IGNORECASE)

# Email / URL-like tokens, dropped wholesale when keep_latin=False so that
# "@" / "." / "/" residue never reaches espeak.
_EMAIL_URL = re.compile(
    r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+|(?:https?://|www\.)\S+", re.IGNORECASE
)

# Runs of Latin letters = code-switched spans espeak would voice in English.
_LATIN_RUN = re.compile(r"[A-Za-z]+")

_TATWEEL = "ـ"  # U+0640 ARABIC TATWEEL

# Punctuation mapped to space. The first group MUST be mapped: these are the
# characters that make espeak emit clause/sentence newlines in -xq mode, which
# would break the batch phonemizer's 1:1 line alignment. The rest are noise
# espeak handles poorly or not at all. % and $ are deliberately ABSENT:
# espeak verbalizes them (filmijja / dullar).
_PUNCT_TO_SPACE = re.compile(
    "["
    "،؛؟"          # Arabic clause/sentence punctuation (MUST map)
    "?!.…:"         # ASCII sentence punctuation + ellipsis (MUST map)
    "«»“”\"'’‘"    # quotes
    "()\\[\\]{}"   # brackets/parens residue
    "*#+="         # symbol noise
    "/\\\\_~^|&@"  # symbol noise (& is mangled by espeak, @ is email residue)
    "–—―‹›《》"     # dashes / CJK quotes
    "]"
)

_WS = re.compile(r"\s+")


def normalize_text(text: str, keep_latin: bool = False) -> tuple[str, dict]:
    """Normalize Egyptian Arabic text for the G2P pipeline.

    Args:
        text: raw input text (one transcript / sentence).
        keep_latin: if False (default, training policy), Latin-script runs
            and email/URL tokens are dropped and counted. If True they are
            preserved for code-switched inference.

    Returns:
        (cleaned_text, stats) where stats has at least ``latin_dropped``.
        ``cleaned_text`` contains no clause/sentence punctuation and no
        newlines, which is what lets :func:`phonemize` batch lines 1:1.
    """
    stats = {"latin_dropped": 0}

    text = _CITATION.sub(" ", text)
    text = _SOURCE_TAG.sub(" ", text)

    if not keep_latin:
        text = _EMAIL_URL.sub(" ", text)
        latin = _LATIN_RUN.findall(text)
        if latin:
            stats["latin_dropped"] = len(latin)
            text = _LATIN_RUN.sub(" ", text)

    text = text.replace(_TATWEEL, "")
    text = _PUNCT_TO_SPACE.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text, stats
