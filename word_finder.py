from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from html.parser import HTMLParser
import posixpath
import re
from typing import Iterable
from xml.etree import ElementTree
from zipfile import ZipFile
from io import BytesIO


@dataclass(frozen=True)
class TimedWord:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Chapter:
    title: str
    text: str


@dataclass(frozen=True)
class ChapterMatch:
    title: str
    start: float
    excerpt: str
    matched_phrases: int


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:['’][a-z0-9]+)*", text.casefold())


def search_words(words: list[TimedWord], query: str) -> list[tuple[int, float]]:
    """Return each exact phrase match as a word index and timestamp."""
    needle = tokenize(query)
    if not needle:
        return []

    tokens = [tokenize(word.text) for word in words]
    normalized = [token[0] if token else "" for token in tokens]
    matches: list[tuple[int, float]] = []
    width = len(needle)
    for index in range(len(normalized) - width + 1):
        if normalized[index : index + width] == needle:
            matches.append((index, words[index].start))
    return matches


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.headings: dict[str, list[str]] = {"h1": [], "h2": [], "title": []}
        self._heading_stack: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._ignored_depth += 1
        if tag in self.headings:
            self._heading_stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1
        if tag in self._heading_stack:
            self._heading_stack.reverse()
            self._heading_stack.remove(tag)
            self._heading_stack.reverse()

    def handle_data(self, data: str) -> None:
        cleaned = " ".join(data.split())
        if cleaned and not self._ignored_depth:
            self.parts.append(cleaned)
            for heading in self._heading_stack:
                self.headings[heading].append(cleaned)


def _chapter_from_html(content: bytes, fallback_title: str) -> Chapter:
    parser = _TextExtractor()
    parser.feed(content.decode("utf-8", errors="replace"))
    title = next(
        (" ".join(parser.headings[tag]).strip() for tag in ("h1", "h2", "title") if parser.headings[tag]),
        fallback_title,
    )
    return Chapter(title=title, text=" ".join(parser.parts))


def _resolve_epub_path(base: str, href: str) -> str:
    path = posixpath.normpath(posixpath.join(posixpath.dirname(base), href.split("#", 1)[0]))
    return path.lstrip("/")


def _read_epub(data: bytes) -> list[Chapter]:
    with ZipFile(BytesIO(data)) as archive:
        container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
        rootfile = container.find(".//{*}rootfile")
        if rootfile is None or not rootfile.attrib.get("full-path"):
            raise ValueError("This EPUB is missing its package document.")

        package_path = rootfile.attrib["full-path"]
        package = ElementTree.fromstring(archive.read(package_path))
        manifest = {
            item.attrib["id"]: item.attrib["href"]
            for item in package.findall(".//{*}manifest/{*}item")
            if item.attrib.get("id") and item.attrib.get("href")
        }
        chapters: list[Chapter] = []
        for itemref in package.findall(".//{*}spine/{*}itemref"):
            href = manifest.get(itemref.attrib.get("idref", ""))
            if not href:
                continue
            path = _resolve_epub_path(package_path, href)
            try:
                content = archive.read(path)
            except KeyError:
                continue
            chapter = _chapter_from_html(content, posixpath.basename(path))
            if tokenize(chapter.text):
                chapters.append(chapter)
        if not chapters:
            raise ValueError("No readable chapters were found in this EPUB.")
        return chapters


_CHAPTER_HEADING = re.compile(
    r"(?im)^\s*((?:chapter|part|book)\s+(?:[\divxlc]+|one|two|three|four|five|six|seven|eight|nine|ten)\b.*|"
    r"(?:prologue|epilogue|introduction|preface|afterword)\s*)$"
)


def _read_text_book(text: str) -> list[Chapter]:
    lines = text.splitlines()
    starts = [
        (index, match.group(1).strip())
        for index, line in enumerate(lines)
        if (match := _CHAPTER_HEADING.match(line))
    ]
    if not starts:
        return [Chapter("Book", text.strip())] if tokenize(text) else []

    chapters: list[Chapter] = []
    if any(line.strip() for line in lines[: starts[0][0]]):
        chapters.append(Chapter("Opening", "\n".join(lines[: starts[0][0]]).strip()))
    for position, (line_index, title) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        chapters.append(Chapter(title, "\n".join(lines[line_index + 1 : end]).strip()))
    return [chapter for chapter in chapters if tokenize(chapter.text)]


def extract_chapters(data: bytes, filename: str) -> list[Chapter]:
    if filename.casefold().endswith(".epub"):
        return _read_epub(data)
    text = data.decode("utf-8-sig", errors="replace")
    chapters = _read_text_book(text)
    if not chapters:
        raise ValueError("The book file does not contain readable text.")
    return chapters


def align_chapters(
    chapters: Iterable[Chapter],
    words: list[TimedWord],
    phrase_size: int = 7,
    lookahead: int = 300,
) -> list[ChapterMatch]:
    """Estimate chapter starts from exact phrase overlaps with the transcript."""
    transcript = [normalized[0] if (normalized := tokenize(word.text)) else "" for word in words]
    if len(transcript) < phrase_size:
        return []

    index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for position in range(len(transcript) - phrase_size + 1):
        phrase = tuple(transcript[position : position + phrase_size])
        if all(phrase):
            index[phrase].append(position)

    results: list[ChapterMatch] = []
    for chapter in chapters:
        book_tokens = tokenize(chapter.text)[:lookahead]
        votes: Counter[int] = Counter()
        positions: dict[int, list[int]] = defaultdict(list)
        excerpts: dict[int, str] = {}
        for book_position in range(len(book_tokens) - phrase_size + 1):
            phrase = tuple(book_tokens[book_position : book_position + phrase_size])
            for audio_position in index.get(phrase, ()):
                bucket = audio_position // 30
                votes[bucket] += 1
                positions[bucket].append(audio_position)
                excerpts.setdefault(bucket, " ".join(book_tokens[book_position : book_position + phrase_size]))
        if votes:
            best_bucket = min(votes, key=lambda bucket: (-votes[bucket], bucket))
            start_index = min(positions[best_bucket])
            if start_index < len(words):
                results.append(
                    ChapterMatch(
                        title=chapter.title,
                        start=words[start_index].start,
                        excerpt=excerpts[best_bucket],
                        matched_phrases=votes[best_bucket],
                    )
                )
    return results


def distribute_caption_words(captions: Iterable[dict]) -> list[TimedWord]:
    """Approximate word times by spreading each caption's duration over its words."""
    words: list[TimedWord] = []
    for caption in captions:
        text = str(caption.get("text", "")).replace("\n", " ")
        tokens = tokenize(text)
        if not tokens:
            continue
        start = float(caption.get("start", 0))
        end = start + max(0.0, float(caption.get("duration", 0)))
        weights = [max(1, len(token)) for token in tokens]
        total_weight = sum(weights)
        cursor = start
        for token, weight in zip(tokens, weights):
            next_cursor = cursor + (end - start) * weight / total_weight
            words.append(TimedWord(token, cursor, next_cursor))
            cursor = next_cursor
    return words
