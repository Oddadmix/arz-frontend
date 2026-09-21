"""Phoneme cleaning: espeak-ng ``arz`` IPA -> Kokoro token stream.

What the cleaner does, in order:

1. Strip espeak's language-switch markers ``(en)`` / ``(arz)``. These are
   markup, not phonemes; the English phonemes inside a kept Latin span stay
   (they are all in-vocab — relevant only with ``keep_latin=True``).
2. Strip the pharyngealization mark ``ˤ``: ``sˤ→s, dˤ→d, tˤ→t, zˤ→z`` (Nabra
   precedent). This is lossy by design and SAFE: the emphatic contrast is
   carried by the backed vowels, which are KEPT — ``صار`` → ``sɑːr`` vs
   ``سار`` → ``saːr``. This keep-the-backed-vowels choice is the one
   deliberate divergence from a naive Nabra port (Nabra's MSA voice emits no
   backed-vowel cue, so it collapses fully).
3. Strip syllable dots ``.``. ``normalize_text`` already maps every ``.`` in
   the *input* to space, so any ``.`` reaching the cleaner can only be an
   espeak syllable-boundary artifact — unlike Nabra we do NOT need the
   keep-sentence-final-dot carve-out. (In ``-xq --ipa`` mode espeak never
   emits pause tokens for punctuation anyway; verified empirically.)
4. Defensive strips (Nabra precedent; the ``arz`` voice was verified never
   to emit these, but the cleaner must not depend on that): combining bridge
   ``◌̪``, ``[ ] { }`` untranslatable-token brackets.
5. Collapse whitespace, then FAIL LOUD:
   * any ASCII or Arabic-Indic digit surviving in the output → ValueError
     (espeak is supposed to verbalize all digits: ``3`` → ``talaːta``);
   * any character outside :data:`KNOWN_VOCAB` → ValueError naming the
     symbol and the input it came from. A symbol with no embedding would
     silently corrupt training — never let it through.

Deliberate deviation from the first draft of this spec: stress marks ``ˈ``
and ``ˌ`` are KEPT, not stripped. Rationale: (a) Nabra's ``clean_phonemes``
keeps them; (b) Kokoro's 178-token table has dedicated slots (156/157)
because upstream Kokoro trains on English espeak output *with* stress marks —
keeping them is the English-parity choice; (c) the fork's Cairene stress rule
(``STRESSPOSN_CAIRENE``, 99.7% on the stress suite) is its crown jewel —
stripping ``ˈ`` would throw that supervision away.
"""

import re

from .vocab import EXTRA_SYMBOLS, KNOWN_VOCAB, validate_vocab

__all__ = ["EXTRA_SYMBOLS", "clean_phonemes", "validate_vocab"]

# espeak language-switch artifacts, e.g. "(en)aɪ lʌv(arz) masˤr".
_LANG_MARK = re.compile(r"\((?:en|arz)\)")

# Out-of-vocab / markup artifacts remapped away. NOTE: ʕ and ħ are
# intentionally absent — they are preserved via EXTRA_SYMBOLS.
_PHONEME_FIXUPS = {
    "ˤ": "",  # pharyngealization → strip; emphatics collapse onto the base
              # consonant, contrast carried by backed vowels (ɑ/e/o kept)
    ".": "",  # syllable-boundary dot (sentence-final dots never reach us:
              # normalize_text maps input "." to space first)
    "̪": "",  # combining bridge below (dental) → strip (Nabra precedent)
    "[": "",  # espeak untranslatable-token bracket → strip
    "]": "",
    "{": "",  # espeak parenthetical/markup bracket → strip
    "}": "",
}

_DIGIT = re.compile(r"[0-9\u0660-\u0669]")  # ASCII + Arabic-Indic digits
_WS = re.compile(r"\s+")


def clean_phonemes(ipa: str, source: str = "") -> str:
    """Clean raw espeak ``arz`` IPA into the final Kokoro token string.

    Args:
        ipa: raw IPA output of :func:`phonemize` for one record.
        source: short excerpt of the input text, used in error messages.

    Returns:
        The cleaned phoneme string. Every character is guaranteed to be in
        :data:`KNOWN_VOCAB` (raises otherwise).

    Raises:
        ValueError: if a digit survived G2P, or any symbol is outside the
            known vocabulary. Fail loud, never silent garbage.
    """
    ph = _LANG_MARK.sub("", ipa)
    for old, new in _PHONEME_FIXUPS.items():
        ph = ph.replace(old, new)
    ph = _WS.sub(" ", ph).strip()

    m = _DIGIT.search(ph)
    if m:
        raise ValueError(
            f"digit {m.group()!r} survived G2P (from input {source[:80]!r}): {ph!r}"
        )
    validate_vocab(ph, source)
    return ph
