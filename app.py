from __future__ import annotations

import re
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import streamlit as st
from faster_whisper import WhisperModel
from youtube_transcript_api import YouTubeTranscriptApi

from word_finder import (
    TimedWord,
    align_chapters,
    distribute_caption_words,
    extract_chapters,
    search_words,
    tokenize,
)


st.set_page_config(page_title="Word Finder", page_icon="🔎", layout="wide")
st.markdown(
    """
    <style>
    .block-container { max-width: 1050px; padding-top: 2.4rem; }
    [data-testid="stMetric"] { background: #f3f5f8; padding: 1rem; border-radius: 12px; }
    .small-note { color: #64748b; font-size: .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model(model_size: str) -> WhisperModel:
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def youtube_video_id(url: str) -> str:
    parsed = urlparse(url.strip())
    host = parsed.netloc.casefold().split(":")[0]
    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/")[0]
    elif host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            parts = parsed.path.strip("/").split("/")
            video_id = parts[1] if len(parts) > 1 and parts[0] in {"embed", "shorts", "live"} else ""
    else:
        video_id = ""
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Enter a valid YouTube video URL.")
    return video_id


def transcribe_upload(audio_file, model_size: str, language: str) -> list[TimedWord]:
    suffix = Path(audio_file.name).suffix or ".audio"
    with tempfile.NamedTemporaryFile(suffix=suffix) as temporary:
        temporary.write(audio_file.getvalue())
        temporary.flush()
        model = load_model(model_size)
        segments, _info = model.transcribe(
            temporary.name,
            language=language or None,
            word_timestamps=True,
            vad_filter=True,
            beam_size=5,
        )
        timed_words: list[TimedWord] = []
        for segment in segments:
            segment_words = getattr(segment, "words", None) or []
            if segment_words:
                timed_words.extend(
                    TimedWord(word.word.strip(), float(word.start), float(word.end))
                    for word in segment_words
                    if word.word.strip()
                )
                continue
            fallback_tokens = tokenize(segment.text)
            duration = max(0.0, float(segment.end) - float(segment.start))
            for index, token in enumerate(fallback_tokens):
                start = segment.start + duration * index / max(1, len(fallback_tokens))
                end = segment.start + duration * (index + 1) / max(1, len(fallback_tokens))
                timed_words.append(TimedWord(token, start, end))
        return timed_words


def format_time(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds_part:02d}" if hours else f"{minutes}:{seconds_part:02d}"


def set_transcript(words: list[TimedWord], source: str, source_kind: str) -> None:
    st.session_state["timed_words"] = words
    st.session_state["source_label"] = source
    st.session_state["source_kind"] = source_kind
    st.session_state["book_chapters"] = []
    st.session_state["chapter_matches"] = []
    st.session_state["seek_to"] = 0.0


st.title("Word Finder")
st.caption("Find a word, passage, or chapter in an audiobook or a captioned YouTube video.")

audio_tab, youtube_tab = st.tabs(["Audio file", "YouTube captions"])

with audio_tab:
    st.write("Audio is transcribed on this computer. The first run downloads the selected speech model.")
    audio_file = st.file_uploader(
        "Choose an audio file",
        type=["mp3", "m4a", "wav", "flac", "ogg", "aac", "wma"],
        key="audio_upload",
    )
    audio_options, audio_action = st.columns([2, 1])
    with audio_options:
        model_size = st.selectbox(
            "Speech model",
            ["base", "small", "tiny"],
            index=0,
            help="Base is a reasonable speed/accuracy balance. Small may be more accurate but slower.",
        )
        language = st.text_input(
            "Language code (optional)",
            placeholder="Auto-detect, or e.g. en",
            help="Use a Whisper language code such as en, af, fr, or es. Leave blank to auto-detect.",
        )
    with audio_action:
        st.write("")
        st.write("")
        transcribe_clicked = st.button(
            "Transcribe audio",
            type="primary",
            disabled=audio_file is None,
            use_container_width=True,
        )
    if transcribe_clicked and audio_file is not None:
        try:
            with st.spinner("Transcribing audio. Longer recordings can take a while on a CPU..."):
                words = transcribe_upload(audio_file, model_size, language.strip())
            if not words:
                st.error("No speech was detected in that audio file.")
            else:
                st.session_state["audio_bytes"] = audio_file.getvalue()
                audio_mime_types = {
                    ".aac": "audio/aac",
                    ".flac": "audio/flac",
                    ".m4a": "audio/mp4",
                    ".mp3": "audio/mpeg",
                    ".ogg": "audio/ogg",
                    ".wav": "audio/wav",
                    ".wma": "audio/x-ms-wma",
                }
                suffix = Path(audio_file.name).suffix.casefold()
                st.session_state["audio_format"] = audio_mime_types.get(suffix, "audio/wav")
                set_transcript(words, audio_file.name, "audio")
                st.success(f"Transcribed {len(words):,} words.")
        except Exception as error:
            st.error(f"Could not transcribe this audio: {error}")

with youtube_tab:
    st.write("Uses captions provided for the video; it does not download the video or audio.")
    youtube_url = st.text_input("YouTube video URL", placeholder="https://www.youtube.com/watch?v=...")
    caption_language = st.text_input("Caption language", value="en", help="Language code, for example en or af.")
    if st.button("Load captions", type="primary", key="load_captions"):
        try:
            video_id = youtube_video_id(youtube_url)
            transcript = YouTubeTranscriptApi().fetch(
                video_id,
                languages=[caption_language.strip() or "en"],
            )
            captions = transcript.to_raw_data()
            words = distribute_caption_words(captions)
            if not words:
                st.error("The selected captions did not contain searchable words.")
            else:
                set_transcript(words, f"YouTube video {video_id}", "youtube")
                st.session_state["youtube_url"] = f"https://www.youtube.com/watch?v={video_id}"
                st.success(f"Loaded {len(words):,} caption words.")
        except Exception as error:
            st.error(
                "Could not load captions. The video may not have captions in that language, "
                f"or YouTube may be temporarily blocking caption access. Details: {error}"
            )

if "timed_words" in st.session_state:
    words = st.session_state["timed_words"]
    st.divider()
    st.subheader(st.session_state["source_label"])
    if st.session_state["source_kind"] == "audio":
        st.audio(
            st.session_state["audio_bytes"],
            format=f"audio/{st.session_state['audio_format']}",
            start_time=int(st.session_state.get("seek_to", 0.0) * 1000),
        )
    else:
        current_url = st.session_state["youtube_url"]
        st.video(current_url, start_time=int(st.session_state.get("seek_to", 0.0)))
        st.caption("YouTube times are estimated within each caption line; caption text is not word-timed.")

    st.subheader("Search the transcript")
    query = st.text_input(
        "Word or phrase",
        placeholder="Type a word or paste a short passage from your e-book",
        key="search_query",
    )
    if query.strip():
        hits = search_words(words, query)
        if hits:
            st.caption(f"{len(hits)} exact match{'es' if len(hits) != 1 else ''}")
            for number, (word_index, seconds) in enumerate(hits[:100], start=1):
                left, right = st.columns([5, 1])
                with left:
                    st.write(f"**{format_time(seconds)}**")
                with right:
                    if st.session_state["source_kind"] == "youtube":
                        link = f"{st.session_state['youtube_url']}&t={int(seconds)}s"
                        st.link_button("Open here", link, key=f"youtube_hit_{number}")
                    elif st.button("Play here", key=f"audio_hit_{number}"):
                        st.session_state["seek_to"] = seconds
                        st.rerun()
        else:
            st.info("No exact match found. Try a shorter phrase or check that the audio uses the same edition.")

    st.divider()
    st.subheader("Find chapter locations from a book")
    st.write(
        "Upload an EPUB or plain-text book. Word Finder looks for matching phrases near each chapter's "
        "beginning; results are approximate and work best when the audiobook uses the same edition."
    )
    book_file = st.file_uploader("Choose an EPUB or TXT book", type=["epub", "txt"], key="book_upload")
    if st.button("Match chapters", disabled=book_file is None, key="match_chapters"):
        try:
            chapters = extract_chapters(book_file.getvalue(), book_file.name)
            matches = align_chapters(chapters, words)
            st.session_state["book_chapters"] = chapters
            st.session_state["chapter_matches"] = matches
            if not matches:
                st.warning(
                    "No chapter passages matched the transcript. The book may be a different edition, "
                    "or the recording may be abridged. Try searching a passage manually above."
                )
        except Exception as error:
            st.error(f"Could not read this book file: {error}")

    chapter_matches = st.session_state.get("chapter_matches", [])
    if chapter_matches:
        st.caption("Estimated chapter positions (matched text is shown as a cross-check).")
        for number, match in enumerate(chapter_matches, start=1):
            left, middle, right = st.columns([3, 3, 1])
            with left:
                st.write(f"**{match.title}** · {format_time(match.start)}")
            with middle:
                st.caption(f'“{match.excerpt}…” · {match.matched_phrases} phrase matches')
            with right:
                if st.session_state["source_kind"] == "youtube":
                    link = f"{st.session_state['youtube_url']}&t={int(match.start)}s"
                    st.link_button("Open here", link, key=f"youtube_chapter_{number}")
                elif st.button("Play here", key=f"audio_chapter_{number}"):
                    st.session_state["seek_to"] = match.start
                    st.rerun()

st.divider()
st.markdown(
    '<p class="small-note">Audio transcription runs locally. YouTube caption lookup requires an internet '
    "connection. This app stores no account or transcript data on a server.</p>",
    unsafe_allow_html=True,
)
