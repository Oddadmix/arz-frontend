# arz-frontend

Egyptian Arabic (Cairene) grapheme-to-phoneme front-end for training a
Kokoro/StyleTTS2 TTS model. Mirrors Nabra's `arabic_g2p.py` pipeline, adapted
for the `arz` voice of the [Oddadmix/espeak-ng fork](https://github.com/Oddadmix/espeak-ng) —
**with no diacritizer stage**: the `arz` voice predicts Egyptian vowels itself
(97.2% token accuracy on undiacritized text), so the camel-tools step Nabra
needs for MSA is unnecessary here.

## Pipeline

```
raw text
  │  normalize_text      citations [1] / (رويترز) dropped · Latin runs dropped
  │                      (training policy) · punctuation -> space · %/$ kept
  ▼                      (espeak verbalizes them) · tatweel stripped
normalized text ──► display_text   (for metadata.csv)
  │  phonemize           fork's espeak-ng, -xq --ipa -v arz, batched
  ▼
raw IPA
  │  clean_phonemes       (en)/(arz) markers stripped · ˤ collapsed · syllable
  ▼                      dots stripped · digits asserted absent · vocab-validated
phonemes  (every char ∈ KNOWN_VOCAB, or an exception — never silent garbage)
```

## The one deliberate divergence from a naive Nabra port: keep the backed vowels

Nabra strips `ˤ`, collapsing ص→`s`, ض→`d`, ط→`t`, ظ→`z`. We do the same —
**but we keep the backed-vowel allophones `ɑ e o`** that the `arz` voice emits
after emphatics:

| word | raw IPA | cleaned |
|------|---------|---------|
| صار | `sˤˈaːr` | `sˈaːr` |
| سار | `sˈaːr` | `sˈaːr` |
| صوم | `sˤˈoːm` | `sˈoːm` |
| صاحبة | `sˤˈɑħba` | `sˈɑħba` |

Where the voice provides the backing, the emphatic contrast survives the `ˤ`
strip (`sˈɑħba` vs `sˈahba`). Nabra's MSA voice emits no backed-vowel cue, so
it collapses fully; ours preserves the cue the voice worked hard to produce
(`ph_arz`: *"this vowel colouring is the last cue distinguishing صار from سار"*).

**Honest caveat:** the voice emits backing inconsistently today — the F-context
rules back correctly through rule-derived words (`صاحبة` → `sˤˈɑħba`, nonsense
`صافق` → `sˤˈɑːfiʔ`), but several lexicon (`arz_listx`) entries carry plain
vowels after emphatics (`صار` → `sˤˈaːr`, `طوب` → `tˤˈuːb`). The cleaner is a
faithful function of the voice: it preserves backing where emitted. Closing
the lexicon gaps is voice work for a later round (it must go through the
repo's round discipline: `setrow.py` → regenerate → `ctest` → corpus snapshot).

## Stress marks are kept (deliberate)

An early draft of this spec stripped `ˈ`/`ˌ`. We keep them:

1. Nabra's `clean_phonemes` keeps them;
2. Kokoro's 178-token table has dedicated slots (156/157) because upstream
   Kokoro trains on English espeak output *with* stress marks — keeping them
   is the English-parity choice;
3. the fork's Cairene stress rule (`STRESSPOSN_CAIRENE`, 99.7% on the stress
   suite) is its crown jewel — stripping `ˈ` would discard that supervision.

## `ARZ_ESPEAK_ROOT` — read this

`phonemize()` resolves the espeak-ng binary as: explicit `espeak_root` arg →
`ARZ_ESPEAK_ROOT` env → built-in default (`/tmp/espeak-ng/build`).

**It MUST point at this fork's build directory** (containing `src/espeak-ng`
and `espeak-ng-data`). The Cairene stress rule lives in the fork's
`dictionary.c`, so a stock espeak-ng install would silently produce worse
phonemes. `EgyptianG2P` verifies the `arz` voice is listed at construction
and raises `RuntimeError` otherwise. Never hand-edit the fork — pronunciation
changes go through `tools-arz/` on its `v3` branch.

## Install & usage

```bash
pip install ./arz-frontend            # or: pip install -e ./arz-frontend
export ARZ_ESPEAK_ROOT=/path/to/fork/build   # optional if default fits
```

```python
from arz_frontend import EgyptianG2P, text_to_phonemes

# one-liner
print(text_to_phonemes("أهلاً وسهلاً بيكم"))
# ʔˈahlan wisˈahlan bˈiːkum

# corpus run (batched, one espeak call per chunk)
g2p = EgyptianG2P()                      # keep_latin=False: training policy
pairs = g2p.process_batch(transcripts)   # [(display_text, phonemes), ...]
print(g2p.summary())

# code-switched inference: keep Latin spans (espeak voices them in English)
g2p_en = EgyptianG2P(keep_latin=True)
print(g2p_en("I love مصر"))              # aɪ lˈʌv mˈasr
```

`EXTRA_SYMBOLS = {"ʕ": 7, "ħ": 8}` reuses Nabra's vocab extension unchanged —
zero new embedding slots needed (see `arz_vocab_mapping.md` in the parent
project). Train with `kmodel.vocab.update(EXTRA_SYMBOLS)` exactly as Nabra does.

## Normalize decisions (all probed against the voice)

| Input | Decision | Why |
|-------|----------|-----|
| `[1]`, `[فر]` | dropped | never voiced (Nabra precedent) |
| `(رويترز)`, `(أ ف ب)`, `(AP)` | dropped (denylist `_SOURCE_TAG`) | source tags, not voiced; policy knob — extend the list |
| other `(…)` | kept, voiced | narrator voices them (Nabra precedent) |
| Latin runs | dropped by default, counted | training text/audio alignment; `keep_latin=True` preserves |
| emails/URLs | dropped wholesale | so `@`/`.` residue never reaches espeak |
| `؟ ? ! . … ، ؛ :` | → space | **phoneme-neutral** (verified): espeak emits no pause tokens in `-xq --ipa`; the clause newline becomes a space after cleaning. Also gives the batcher its 1:1 line guarantee |
| `«»""''()[]{}*&@#+=/\_~^|–—` | → space | noise espeak handles poorly (`&` is mangled) |
| `%`, `$` | **kept** | espeak verbalizes: `50%`→`xamsiːn filmijja`, `100$`→`mijja dullar`; dropping loses spoken content |
| digits (`3`, `١٢`) | kept for espeak | espeak expands (`3`→`talaːta`); cleaner **raises** if any digit survives |
| tatweel `ـ` | stripped | decoration; espeak drops it silently |
| alef/hamza `أإآءؤئ` | **not** normalized | voice handles all forms; folding would change phonemization (`آ`→`ʔaː` vs `ا`→`a`) |
| tashkeel | kept | signal, not noise; voice validated on diacritized input |

Batching detail: espeak chunks long inputs internally (~400+ chars, measured),
so records over 350 normalized chars go through one subprocess each (chunk
lines joined with space — espeak chunks on its own word tokens); shorter
records batch 1:1 with blank-line separators. A count assertion plus
per-record fallback guards every batch.

## Frozen G2P

**Once training starts, this package's version is pinned with the model.**
The phoneme stream is part of the model's input distribution: any change to
normalization, the voice, or the cleaner silently breaks compatibility with
already-trained weights. Tag the release used for a training run
(`git tag`), record the tag with the model card, and treat any later change
as requiring retraining or a compatibility audit.

## Tests

```bash
python -m unittest discover -s tests   # needs the fork built + ARZ_ESPEAK_ROOT
```

Includes a 200-row wild-corpus test (Egyptian translation dataset, Arabic
column) asserting zero unknown symbols and zero surviving digits through the
full pipeline.
