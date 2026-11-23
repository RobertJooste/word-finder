from io import BytesIO
import unittest
from zipfile import ZIP_DEFLATED, ZipFile

from word_finder import (
    Chapter,
    TimedWord,
    align_chapters,
    distribute_caption_words,
    extract_chapters,
    search_words,
)


class WordFinderTests(unittest.TestCase):
    def test_phrase_search_ignores_case_and_punctuation(self) -> None:
        words = [
            TimedWord("It", 0.0, 0.2),
            TimedWord("was,", 0.2, 0.5),
            TimedWord("the", 0.5, 0.7),
            TimedWord("best", 0.7, 1.0),
        ]

        self.assertEqual(search_words(words, "WAS THE"), [(1, 0.2)])

    def test_caption_word_times_stay_inside_caption_interval(self) -> None:
        words = distribute_caption_words([{"text": "one two", "start": 2, "duration": 4}])

        self.assertEqual([word.text for word in words], ["one", "two"])
        self.assertEqual(words[0].start, 2)
        self.assertAlmostEqual(words[-1].end, 6)
        self.assertLessEqual(words[0].end, words[1].start)

    def test_text_book_splits_chapters(self) -> None:
        chapters = extract_chapters(
            b"Chapter One\nA beginning.\n\nChapter Two\nA new beginning.",
            "book.txt",
        )

        self.assertEqual([chapter.title for chapter in chapters], ["Chapter One", "Chapter Two"])
        self.assertEqual(chapters[1].text, "A new beginning.")

    def test_epub_uses_spine_order_and_extracts_text(self) -> None:
        data = BytesIO()
        with ZipFile(data, "w", ZIP_DEFLATED) as archive:
            archive.writestr(
                "META-INF/container.xml",
                '<container><rootfiles><rootfile full-path="OPS/book.opf"/></rootfiles></container>',
            )
            archive.writestr(
                "OPS/book.opf",
                '<package xmlns="http://www.idpf.org/2007/opf">'
                '<manifest><item id="c1" href="text/chapter.xhtml"/></manifest>'
                '<spine><itemref idref="c1"/></spine></package>',
            )
            archive.writestr(
                "OPS/text/chapter.xhtml",
                "<html><head><title>Chapter Title</title></head>"
                "<body><h1>Chapter One</h1><p>Readable chapter text.</p></body></html>",
            )

        chapters = extract_chapters(data.getvalue(), "book.epub")

        self.assertEqual(len(chapters), 1)
        self.assertEqual(chapters[0].title, "Chapter One")
        self.assertIn("Readable chapter text.", chapters[0].text)

    def test_chapter_alignment_finds_matching_phrase_timestamp(self) -> None:
        passage = "the old house stood beside the river and the moon was rising"
        words = [
            TimedWord(token, index * 0.5, index * 0.5 + 0.4)
            for index, token in enumerate(("noise before " + passage).split())
        ]

        matches = align_chapters([Chapter("Chapter One", passage)], words)

        self.assertEqual(len(matches), 1)
        self.assertGreaterEqual(matches[0].start, 0.5)
        self.assertEqual(matches[0].title, "Chapter One")
        self.assertGreater(matches[0].matched_phrases, 0)


if __name__ == "__main__":
    unittest.main()
