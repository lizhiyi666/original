"""Regression tests for the manuscript's narrowly authorized metadata edit."""
from contextlib import redirect_stdout
from io import StringIO
import re
import unittest

from verify_paper import BASELINE, git, verify_locked_metadata


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


if __name__ == "__main__":
    unittest.main()
