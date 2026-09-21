"""Tests for the Egyptian Arabic TTS front-end.

Expected phoneme strings are grounded by running the pipeline against the
fork's built ``arz`` voice (v3); they are regression pins, not guesses.
The corpus test needs pandas+pyarrow and the dataset parquet.
"""

import os
import random
import re
import unittest

from arz_frontend import (
    EXTRA_SYMBOLS,
    EgyptianG2P,
    clean_phonemes,
    normalize_text,
    text_to_phonemes,
    validate_vocab,
)

ESPEAK_ROOT = os.environ.get("ARZ_ESPEAK_ROOT", "/tmp/espeak-ng/build")
PARQUET = os.environ.get(
    "ARZ_CORPUS_PARQUET", "/home/hatch/workspace/data/egyptian.parquet"
)


class TestNormalize(unittest.TestCase):
    def test_citation_and_source_tag_stripped(self):
        text, _ = normalize_text("النص [1] (رويترز) تمام")
        self.assertEqual(text, "النص تمام")

    def test_latin_dropped_by_default(self):
        text, stats = normalize_text("I love مصر")
        self.assertEqual(text, "مصر")
        self.assertEqual(stats["latin_dropped"], 2)

    def test_latin_kept_on_request(self):
        text, stats = normalize_text("I love مصر", keep_latin=True)
        self.assertEqual(text, "I love مصر")
        self.assertEqual(stats["latin_dropped"], 0)

    def test_punctuation_to_space(self):
        text, _ = normalize_text("إنت فين؟")
        self.assertEqual(text, "إنت فين")
        text, _ = normalize_text("أهلاً، وسهلاً؛ تمام…")
        self.assertNotRegex(text, r"[،؛؟…]")

    def test_tatweel_stripped(self):
        text, _ = normalize_text("كلمةـتطويل")
        self.assertNotIn("ـ", text)

    def test_alef_hamza_forms_preserved(self):
        # Deliberate: the voice handles every alef/hamza form; folding them
        # would change phonemization (آ -> ʔaː vs ا -> a).
        text, _ = normalize_text("أإآءؤئ")
        self.assertEqual(text, "أإآءؤئ")

    def test_percent_and_dollar_kept(self):
        # espeak verbalizes them (filmijja / dullar) — dropping loses content.
        text, _ = normalize_text("خصم 50% وسعر 100$")
        self.assertIn("%", text)
        self.assertIn("$", text)

    def test_empty_and_punct_only(self):
        self.assertEqual(normalize_text("")[0], "")
        self.assertEqual(normalize_text("؟!،")[0], "")

    def test_cjk_stripped(self):
        # Audit row 21603: 散 (U+6563) made espeak voice-switch to Mandarin.
        text, stats = normalize_text("من散هم")
        self.assertEqual(text, "من هم")
        self.assertEqual(stats["script_stripped"], 1)

    def test_cyrillic_and_thai_stripped(self):
        text, stats = normalize_text("مصر москва กรุงเทพ")
        self.assertEqual(text, "مصر")
        self.assertGreater(stats["script_stripped"], 0)

    def test_salawat_ligature_kept(self):
        # ﷺ (U+FDFA) is an Arabic presentation form the voice handles
        # (v3 lexicon fix) — the normalizer must not strip it.
        text, stats = normalize_text("قال ﷺ")
        self.assertEqual(text, "قال ﷺ")
        self.assertEqual(stats["script_stripped"], 0)


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g2p = EgyptianG2P(espeak_root=ESPEAK_ROOT)

    def test_clean_sentence(self):
        self.assertEqual(
            self.g2p("أهلاً وسهلاً بيكم"), "ʔˈahlan wisˈahlan bˈiːkum"
        )

    def test_digits_expanded_no_digits_survive(self):
        ph = self.g2p("عندي 3 تفاحات")
        self.assertEqual(ph, "ʕˈandi talˈaːta tifaħˈaːt")
        self.assertNotRegex(ph, r"[0-9\u0660-\u0669]")

    def test_latin_dropped_default(self):
        self.assertEqual(self.g2p("I love مصر"), "mˈasr")

    def test_latin_kept(self):
        g2p = EgyptianG2P(espeak_root=ESPEAK_ROOT, keep_latin=True)
        ph = g2p("I love مصر")
        self.assertEqual(ph, "aɪ lˈʌv mˈasr")  # (en)/(arz) markers stripped

    def test_arabic_punctuation_single_line(self):
        ph = self.g2p("إنت فين؟")
        self.assertEqual(ph, "ˈinta fˈeːn")
        self.assertNotIn("\n", ph)

    def test_hamza_forms(self):
        self.assertEqual(self.g2p("أإآ"), "ʔˈaʔaʔaː")

    def test_empty(self):
        self.assertEqual(self.g2p(""), "")
        self.assertEqual(self.g2p("؟!،"), "")

    def test_citation_not_voiced(self):
        ph = self.g2p("النص [1] (رويترز) تمام")
        self.assertEqual(ph, "ʔinnˈuss tamˈaːm")

    def test_batch_alignment(self):
        # Sentences with ؟ ، ؛ must stay 1:1 (normalize maps them to space).
        texts = ["ازيك؟ عامل ايه؟", "أهلاً، وسهلاً؛ تمام.", "عندي 3 تفاحات"]
        pairs = self.g2p.process_batch(texts)
        self.assertEqual(len(pairs), 3)
        for (display, ph), raw in zip(pairs, texts):
            self.assertNotIn("\n", ph, f"newline leaked for {raw!r}")
            self.assertNotRegex(ph, r"[0-9\u0660-\u0669]")

    def test_process_returns_display_text(self):
        display, ph = self.g2p.process("I love مصر [1]")
        self.assertEqual(display, "مصر")
        self.assertEqual(ph, "mˈasr")

    def test_text_to_phonemes_convenience(self):
        self.assertEqual(text_to_phonemes("إنت فين؟"), "ˈinta fˈeːn")

    def test_salawat_ligature_voiced(self):
        # Audit anomaly: ﷺ used to be spelled aloud as its hex codepoint.
        # v3 lexicon fix expands it to the salawat phrase.
        self.assertEqual(
            self.g2p("ﷺ"), "sallaʔallaːhʕaleːwisˈallam"
        )

    def test_urdu_dal_mapped(self):
        # Audit anomaly: ڈ used to be spelled as "ستة تمانية تمانية".
        # v3 FOREIGN fix maps it to د.
        self.assertEqual(self.g2p("ڈال"), "dˈaːl")

    def test_leading_dash_via_stdin(self):
        # Audit harness artifact: 2 rows starting with "- " produced empty
        # output because argv parsing treated the dash as an option flag.
        # The front-end always passes text via stdin, so these phonemize.
        ph = self.g2p("- قال أهلا")
        self.assertEqual(ph, "ʔˈaːl ʔˈahlan")

    def test_mandarin_row_no_foreign_phonemes(self):
        # Audit row 21603: CJK char caused a (cmn) voice switch leaking a
        # tone digit into the IPA. The normalizer strips the script now.
        ph = self.g2p("من散هم")
        self.assertEqual(ph, "mˈin hˈumma")
        self.assertNotRegex(ph, r"[0-9\u0660-\u0669]")


class TestCleaner(unittest.TestCase):
    def test_emphatic_marker_stripped_backed_vowel_kept(self):
        # sˤ -> s, but a voice-emitted ɑ survives (contrast cue).
        self.assertEqual(clean_phonemes("sˤˈaːr"), "sˈaːr")
        self.assertEqual(clean_phonemes("sˤˈoːm"), "sˈoːm")
        self.assertEqual(clean_phonemes("sˤˈuːrɑ"), "sˈuːrɑ")

    def test_language_markers_stripped(self):
        self.assertEqual(clean_phonemes("(en)aɪ lˈʌv(arz) mˈasr"), "aɪ lˈʌv mˈasr")

    def test_syllable_dots_stripped(self):
        self.assertEqual(clean_phonemes("a.b"), "ab")

    def test_digit_surviving_raises(self):
        with self.assertRaises(ValueError):
            clean_phonemes("ab1cd", source="test")
        with self.assertRaises(ValueError):
            clean_phonemes("ab٣cd", source="test")

    def test_unknown_symbol_raises(self):
        with self.assertRaises(ValueError) as cm:
            validate_vocab("a😀b", source="test input")
        self.assertIn("😀", str(cm.exception))

    def test_extra_symbols(self):
        self.assertEqual(EXTRA_SYMBOLS, {"ʕ": 7, "ħ": 8})
        # and they validate clean:
        validate_vocab("ʕaħ")


@unittest.skipUnless(
    os.path.exists(PARQUET), f"corpus parquet not found at {PARQUET}"
)
class TestCorpus(unittest.TestCase):
    def test_200_rows_zero_unknown_symbols(self):
        pd = pytest_import_pandas(self)
        df = pd.read_parquet(PARQUET)
        rows = (
            df["Arabic"].dropna().sample(200, random_state=42).tolist()
        )
        g2p = EgyptianG2P(espeak_root=ESPEAK_ROOT)
        pairs = g2p.process_batch(rows)
        self.assertEqual(len(pairs), 200)
        for (display, ph), raw in zip(pairs, rows):
            # validate_vocab already ran inside clean_phonemes; reaching here
            # with no exception IS the assertion. Spot-check the invariant:
            self.assertNotRegex(ph, r"[0-9\u0660-\u0669]")
        print(f"\n[g2p summary] {g2p.summary()}")


def pytest_import_pandas(testcase):
    try:
        import pandas

        return pandas
    except ImportError:
        testcase.skipTest("pandas not installed")


if __name__ == "__main__":
    unittest.main()
