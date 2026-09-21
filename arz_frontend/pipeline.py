"""End-to-end pipeline: Egyptian Arabic text -> Kokoro-ready phonemes.

Pipeline (note: NO diacritizer — the ``arz`` voice predicts Egyptian vowels
itself, 97.2% token accuracy on undiacritized text, so the camel-tools stage
Nabra needs for MSA is unnecessary here):

    raw text
      │  normalize_text  (citations, Latin policy, punctuation -> space, …)
      ▼
    normalized text ──► display_text (for metadata.csv: Latin-stripped,
      │                   but NOT otherwise transformed)
      │  phonemize     (fork's espeak-ng, -xq --ipa -v arz, batched)
      ▼
    raw IPA
      │  clean_phonemes (markers/fixups stripped, ˤ collapsed, vocab-validated)
      ▼
    phonemes  (every char in KNOWN_VOCAB, or an exception — never garbage)
"""

from functools import lru_cache

from .clean import clean_phonemes
from .normalize import normalize_text
from .phonemize import resolve_espeak, verify_arz_voice
from .vocab import EXTRA_SYMBOLS, KNOWN_VOCAB, validate_vocab

__all__ = [
    "EgyptianG2P",
    "text_to_phonemes",
    "normalize_text",
    "clean_phonemes",
    "validate_vocab",
    "EXTRA_SYMBOLS",
    "KNOWN_VOCAB",
]


class EgyptianG2P:
    """Text -> IPA phonemes for Cairene Egyptian Arabic.

    Stateful counters (``latin_dropped``) accumulate across calls so the
    caller can print a single summary at the end of a corpus run.

    Args:
        espeak_root: fork build directory; falls back to ``ARZ_ESPEAK_ROOT``
            env, then the baked-in default. MUST be this fork's build —
            verified to contain the ``arz`` voice at construction.
        keep_latin: False (default, training policy) drops Latin-script runs;
            True preserves code-switched spans for inference.
    """

    def __init__(self, espeak_root: str | None = None, keep_latin: bool = False):
        self.espeak_root, _ = resolve_espeak(espeak_root)
        verify_arz_voice(self.espeak_root)  # fail fast on a stock espeak-ng
        self.keep_latin = keep_latin
        self.latin_dropped = 0

    def process(self, raw_text: str) -> tuple[str, str]:
        """Full pipeline for one text. Returns (display_text, phonemes).

        ``display_text`` is the normalized, Latin-stripped text suitable for
        ``metadata.csv``; ``phonemes`` are derived from the same normalized
        text (no diacritizer — the voice predicts vowels).
        """
        return self.process_batch([raw_text])[0]

    def process_batch(self, raw_texts: list[str]) -> list[tuple[str, str]]:
        """Full pipeline for many texts. One batched espeak call internally."""
        raw_texts = list(raw_texts)
        displays: list[str] = []
        for raw in raw_texts:
            text, stats = normalize_text(raw, keep_latin=self.keep_latin)
            self.latin_dropped += stats["latin_dropped"]
            displays.append(text)
        from .phonemize import phonemize

        ipas = phonemize(displays, espeak_root=self.espeak_root)
        return [
            (display, clean_phonemes(ipa, source=raw))
            for display, ipa, raw in zip(displays, ipas, raw_texts)
        ]

    def __call__(self, raw_text: str) -> str:
        """Convenience: return just the phonemes."""
        return self.process(raw_text)[1]

    def summary(self) -> str:
        return (
            f"Latin runs dropped: {self.latin_dropped:,} | "
            f"diacritizer: off (arz voice predicts vowels) | "
            f"espeak_root: {self.espeak_root}"
        )


@lru_cache(maxsize=4)
def _default_g2p(espeak_root: str | None, keep_latin: bool) -> EgyptianG2P:
    return EgyptianG2P(espeak_root=espeak_root, keep_latin=keep_latin)


def text_to_phonemes(
    text: str,
    keep_latin: bool = False,
    espeak_root: str | None = None,
) -> str:
    """Convenience: normalize -> phonemize -> clean for one text."""
    return _default_g2p(espeak_root, keep_latin)(text)
