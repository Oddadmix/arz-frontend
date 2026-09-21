"""Vocabulary for the Egyptian Arabic TTS front-end.

The known symbol set is Kokoro's 178-token table as shipped in
``oddadmix/Nabra-82M-v0.1`` ``config.json`` (114 populated symbols; the rest
are unused gap indices), PLUS the two free slots Nabra claimed:

    EXTRA_SYMBOLS = {"\u0295": 7, "\u0127": 8}   # \u0295 (u+0295), \u0127 (u+0127)

``validate_vocab`` fails LOUD on any other symbol: a phoneme outside this set
has no embedding in the model, so silently passing it through would corrupt
training. The one deliberate exception is the pharyngealization mark ``\u02e4``,
which the cleaner strips before validation (see ``clean.py``).
"""

KOKORO_BASE_SYMBOLS = frozenset({
    ' ', '!', '"', '(', ')', ',', '.', ':',
    ';', '?', 'A', 'I', 'O', 'Q', 'S', 'T',
    'W', 'Y', 'a', 'b', 'c', 'd', 'e', 'f',
    'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
    'x', 'y', 'z', 'æ', 'ç', 'ð', 'ø', 'ŋ',
    'œ', 'ɐ', 'ɑ', 'ɒ', 'ɔ', 'ɕ', 'ɖ', 'ə',
    'ɚ', 'ɛ', 'ɜ', 'ɟ', 'ɡ', 'ɣ', 'ɤ', 'ɥ',
    'ɨ', 'ɪ', 'ɯ', 'ɰ', 'ɲ', 'ɳ', 'ɴ', 'ɸ',
    'ɹ', 'ɻ', 'ɽ', 'ɾ', 'ʁ', 'ʂ', 'ʃ', 'ʈ',
    'ʊ', 'ʋ', 'ʌ', 'ʎ', 'ʒ', 'ʔ', 'ʝ', 'ʣ',
    'ʤ', 'ʥ', 'ʦ', 'ʧ', 'ʨ', 'ʰ', 'ʲ', 'ˈ',
    'ˌ', 'ː', '̃', 'β', 'θ', 'χ', 'ᵊ', 'ᵝ',
    'ᵻ', '—', '“', '”', '…', '→', '↓', '↗',
    '↘', 'ꭧ',
})

# ── Out-of-vocab phonemes KEPT via free Kokoro embedding slots ────────────────
# Indices 7 & 8 are confirmed free in Kokoro-82M config.json (Nabra's slots).
# The Egyptian model reuses them unchanged: zero new slots needed.
EXTRA_SYMBOLS = {
    "\u0295": 7,  # \u0295 voiced pharyngeal fricative (distinct from \u0621/\u0294)
    "\u0127": 8,  # \u0127 voiceless pharyngeal fricative (distinct from \u0647/h)
}

KNOWN_VOCAB = frozenset(KOKORO_BASE_SYMBOLS) | frozenset(EXTRA_SYMBOLS)


def validate_vocab(text: str, source: str = "") -> None:
    """Raise ValueError if any character of *text* is outside KNOWN_VOCAB.

    *source* is a short excerpt of the input text, included in the error to
    make the offending row findable in a 135k-row corpus.
    """
    bad = sorted(set(text) - KNOWN_VOCAB)
    if bad:
        excerpt = (source[:80] + "…") if len(source) > 80 else source
        raise ValueError(
            f"symbols outside Kokoro vocab {bad!r} in phonemes {text!r} "
            f"(from input {excerpt!r})"
        )
