# Word Finder

Word Finder helps you get back to a place in an audiobook from a word, passage, or chapter in your e-book. It runs as a small web app on your computer.

## Run it

Use Python 3.10–3.14 (Python 3.13 was verified). Python 3.15 is not currently supported by the required speech-model dependency wheels.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

If you use a different supported Python version, replace `python3.13` with that interpreter's command.
Open the local address Streamlit prints (usually <http://localhost:8501>).

## Use it

1. **Audio file:** Choose an audio file and select **Transcribe audio**. The first run downloads the Whisper speech model you select. Transcription runs locally and can take a while, especially for long recordings.
2. **YouTube captions:** Paste a video URL and choose the caption language. This uses captions YouTube makes available; it does not download the video or audio. Some videos or networks may block caption access. If you already installed an earlier version of Word Finder, update dependencies with `python -m pip install -r requirements.txt` and restart the app.
3. Search for a word or paste a short phrase from your e-book. Audio matches can be played from their timestamp; YouTube matches open the video at that time.
4. Optionally upload an **EPUB** or **TXT** book and choose **Match chapters**. Chapter positions are estimates based on matching text near each chapter's beginning. They work best when the audio and e-book use the same edition and wording.

Audio, book files, and transcripts are processed for the current app session and are not uploaded to an app server. YouTube caption lookup requires internet access. This app does not download YouTube media.

## Notes

- Search is case-insensitive and exact after basic punctuation normalization. If a phrase is not found, try fewer words.
- YouTube caption timestamps are approximate within each caption line; uploaded audio uses Whisper word-level timestamps.
- The default `base` model balances speed and accuracy. `small` may improve recognition at the cost of more processing time; `tiny` is faster but less accurate.
- For chapter alignment, mismatched editions, abridged recordings, or heavily edited text can prevent matches. The app shows the matched phrase to help you judge each estimate.

## Tests

```bash
python -m unittest
```
