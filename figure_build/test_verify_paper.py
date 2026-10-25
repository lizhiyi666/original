"""Regression tests for the manuscript's narrowly authorized metadata edit."""
from contextlib import redirect_stdout
from io import StringIO
import re
import unittest

from verify_paper import BASELINE, git, main_float_labels, verify_locked_metadata, verify_page_layout


class LockedMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = git("show", f"{BASELINE}:paper-www2027/main.tex")
        cls.authorized = cls.original.replace("eliminates residual", "reduces residual", 1)

    def check(self, text):
        with redirect_stdout(StringIO()):
            verify_locked_metadata(self.original, text)

    def test_authorized_correction_passes(self):
        self.check(self.authorized)

    def test_original_overclaim_is_rejected(self):
        with self.assertRaises(AssertionError):
            self.check(self.original)

    def test_other_abstract_word_is_rejected(self):
        with self.assertRaises(AssertionError):
            self.check(self.authorized.replace("largely", "completely", 1))

    def test_extra_abstract_sentence_is_rejected(self):
        with self.assertRaises(AssertionError):
            self.check(self.authorized.replace(r"\end{abstract}", r"Extra claim.\end{abstract}", 1))

    def test_title_is_locked(self):
        with self.assertRaises(AssertionError):
            self.check(self.authorized.replace("Training-Free Partial-Order Control", "New Title", 1))

    def test_keywords_are_locked(self):
        with self.assertRaises(AssertionError):
            self.check(self.authorized.replace("synthetic human mobility,", "different keyword,", 1))

    def test_author_order_is_locked(self):
        authors = re.findall(r"\\author\{[^}]+\}", self.authorized)
        self.assertEqual(len(authors), 2)
        swapped = self.authorized.replace(authors[0], "AUTHOR_SWAP", 1)
        swapped = swapped.replace(authors[1], authors[0], 1).replace("AUTHOR_SWAP", authors[1], 1)
        with self.assertRaises(AssertionError):
            self.check(swapped)

    def test_missing_abstract_is_rejected(self):
        text = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", "", self.authorized, flags=re.S)
        with self.assertRaises(AssertionError):
            self.check(text)


class PageLayoutTests(unittest.TestCase):
    def aux(self, body=8, refs=9, appendix=11, figure=2):
        return "\n".join(
            rf"\newlabel{{{label}}}{{{{1}}{{{page}}}{{Heading}}{{anchor}}{{}}}}"
            for label, page in (("end:main", body), ("start:references", refs),
                                ("app:algo", appendix), ("fig:example", figure)))

    def test_exact_eight_page_layout(self):
        self.assertEqual(verify_page_layout(self.aux(), 12, {"fig:example"}), (8, 9, 11))

    def test_short_main_text_rejected(self):
        with self.assertRaisesRegex(AssertionError, "expected p8"):
            verify_page_layout(self.aux(body=7), 12, {"fig:example"})

    def test_late_main_text_rejected(self):
        with self.assertRaises(AssertionError):
            verify_page_layout(self.aux(body=9), 12, {"fig:example"})

    def test_float_spilling_into_references_rejected(self):
        with self.assertRaisesRegex(AssertionError, "float outside"):
            verify_page_layout(self.aux(figure=9), 12, {"fig:example"})

    def test_early_references_rejected(self):
        with self.assertRaisesRegex(AssertionError, "expected p9"):
            verify_page_layout(self.aux(refs=8), 12, {"fig:example"})

    def test_appendix_order_rejected(self):
        with self.assertRaisesRegex(AssertionError, "follow references"):
            verify_page_layout(self.aux(appendix=9), 12, {"fig:example"})

    def test_thirteenth_page_rejected(self):
        with self.assertRaisesRegex(AssertionError, "maximum 12"):
            verify_page_layout(self.aux(), 13, {"fig:example"})

    def test_missing_float_label_rejected(self):
        with self.assertRaisesRegex(AssertionError, "Missing page label"):
            verify_page_layout(self.aux(), 12, {"fig:missing"})

    def test_main_include_tree_excludes_appendix_floats(self):
        self.assertEqual(main_float_labels(), {
            "fig:postswap-pcdg", "fig:framework", "fig:tradeoff",
            "tab:method", "tab:method-jsd", "tab:ablation-jsd", "tab:ablation",
            "alg:pcdg"})


if __name__ == "__main__":
    unittest.main()
