"""Batch phonemization through the fork's espeak-ng ``arz`` voice.

Batch protocol (verified empirically):
* Records are joined with a BLANK line (``"\\n\\n"``). A blank line is a
  paragraph break for espeak: each record yields exactly one output line.
* NOTE: via *stdin* espeak echoes one empty line per blank-line separator
  (via ``-f`` it does not — stdin/-f differ here); empty output lines are
  filtered, which is safe because records are pre-filtered non-empty.
* espeak chunks long INPUTS internally (~400+ chars, measured): a long
  record yields several output lines, one per espeak-internal word-chunk.
  Records longer than ``_BATCH_CHAR_LIMIT`` (350, conservative) therefore go
  through one subprocess each and their lines are joined with a space —
  espeak chunks on its own word tokens (verified: ``ومعانا`` -> ``wimaʕˈaːna``
  vs ``و معانا`` -> ``wˈi maʕˈaːna``), so space-joining faithfully
  reconstructs its word sequence.
* PRECONDITION: records must already be normalized (see ``normalize.py``).
  Un-normalized clause punctuation (``، ؛``) and sentence terminators
  (``؟ ? ! .``) make espeak emit extra newlines mid-record, and a record
  without terminal punctuation merges with the next one — both silently break
  1:1 alignment. ``normalize_text`` maps all of these to space, which is
  phoneme-neutral (verified: identical phonemes, newline becomes space).
* After the batch call we assert ``len(outputs) == len(records)``. On a
  mismatch we fall back to one subprocess per record for that chunk and warn
  — slow but correct, and it surfaces the offending input instead of
  silently misaligning a 135k-row corpus.
* Empty/whitespace-only records are never sent; they map to ``""``.

``espeak_root`` resolution: explicit argument, else the ``ARZ_ESPEAK_ROOT``
environment variable, else the baked-in default. It MUST point at THIS fork's
build directory (the one containing ``src/espeak-ng`` and ``espeak-ng-data``):
the Cairene stress rule lives in the fork's ``dictionary.c``, so a stock
espeak-ng install would silently produce worse phonemes. The voice is verified
present via ``espeak-ng --voices`` at ``EgyptianG2P`` construction time.
"""

import os
import subprocess
import warnings

DEFAULT_ESPEAK_ROOT = "/tmp/espeak-ng/build"

# Records at or below this normalized length go through the fast batch path.
# Measured: espeak starts chunking long inputs into multiple output lines
# somewhere past ~400 chars; 350 keeps a safe margin.
_BATCH_CHAR_LIMIT = 350


def resolve_espeak(espeak_root: str | None = None) -> tuple[str, str]:
    """Return (root, binary_path) for the fork's espeak-ng build."""
    root = espeak_root or os.environ.get("ARZ_ESPEAK_ROOT") or DEFAULT_ESPEAK_ROOT
    binary = os.path.join(root, "src", "espeak-ng")
    if not (os.path.isfile(binary) and os.access(binary, os.X_OK)):
        raise RuntimeError(
            f"espeak-ng binary not found/executable at {binary!r}. "
            "Set espeak_root or the ARZ_ESPEAK_ROOT environment variable to "
            "this fork's build directory (containing src/espeak-ng and "
            "espeak-ng-data). A stock espeak-ng install will NOT do: the "
            "Cairene stress rule lives in the fork's dictionary.c."
        )
    return root, binary


def verify_arz_voice(espeak_root: str | None = None) -> None:
    """Raise RuntimeError unless the ``arz`` voice is listed by the binary."""
    root, binary = resolve_espeak(espeak_root)
    env = {**os.environ, "ESPEAK_DATA_PATH": root}
    try:
        proc = subprocess.run(
            [binary, "--voices"],
            capture_output=True, text=True, env=env, timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise RuntimeError(f"could not run {binary} --voices: {e}")
    if not _has_arz_voice(proc.stdout):
        raise RuntimeError(
            f"voice 'arz' not listed by {binary}. ARZ_ESPEAK_ROOT={root!r} "
            "probably points at a stock espeak-ng install, not this fork's build."
        )


def _has_arz_voice(voices_output: str) -> bool:
    """True if any --voices line lists the arz language/voice."""
    for line in voices_output.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[1] == "arz":
            return True
    return False


def _run_espeak(payload: str, binary: str, env: dict) -> list[str]:
    proc = subprocess.run(
        [binary, "-xq", "--ipa", "-v", "arz"],
        input=payload, capture_output=True, text=True, env=env, timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"espeak-ng failed (rc={proc.returncode}): {proc.stderr[:500]}")
    lines = proc.stdout.split("\n")
    # Via stdin, espeak echoes one EMPTY line per blank-line record separator
    # (via -f it does not — stdin/-f differ here). Records are pre-filtered
    # non-empty, so every remaining line is exactly one record's output, in
    # order. The caller still asserts the count and falls back per-record on
    # any mismatch.
    return [ln.strip() for ln in lines if ln.strip()]


def phonemize(
    texts: list[str],
    espeak_root: str | None = None,
    chunk_size: int = 10000,
) -> list[str]:
    """Phonemize normalized texts with the ``arz`` voice.

    One subprocess call per *chunk* (default 10k records); records inside a
    chunk go through a single espeak invocation joined by blank lines.

    Args:
        texts: list of NORMALIZED strings (see ``normalize_text``).
        espeak_root: fork build dir; defaults per :func:`resolve_espeak`.
        chunk_size: records per subprocess call.

    Returns:
        List of IPA strings, aligned 1:1 with *texts* (``""`` for empty
        inputs). Newlines inside a record's output are collapsed to spaces.
    """
    root, binary = resolve_espeak(espeak_root)
    env = {**os.environ, "ESPEAK_DATA_PATH": root}

    texts = list(texts)
    results: list[str] = [""] * len(texts)
    nonempty = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
    if not nonempty:
        return results

    short = [(i, t) for i, t in nonempty if len(t) <= _BATCH_CHAR_LIMIT]
    long = [(i, t) for i, t in nonempty if len(t) > _BATCH_CHAR_LIMIT]

    # Long records: one subprocess each; espeak chunks them internally, so
    # join the chunk lines with a space (see module docstring).
    for i, t in long:
        results[i] = _phonemize_one(t, binary, env)

    # Short records: batched, one subprocess per chunk.
    for start in range(0, len(short), chunk_size):
        chunk = short[start:start + chunk_size]
        payload = "\n\n".join(t for _, t in chunk)
        lines = _run_espeak(payload, binary, env)
        if len(lines) != len(chunk):
            warnings.warn(
                f"espeak batch misalignment: {len(chunk)} records -> "
                f"{len(lines)} output lines; falling back to per-record calls"
            )
            lines = [_phonemize_one(t, binary, env) for _, t in chunk]
        for (i, _), line in zip(chunk, lines):
            results[i] = " ".join(line.split())
    return results


def _phonemize_one(text: str, binary: str, env: dict) -> str:
    lines = _run_espeak(text, binary, env)
    return " ".join(" ".join(lines).split())
