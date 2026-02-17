# Spoken-to-Signed Translation (Text, Audio, JSON, Video, Unity)

This repository extends the original **gloss-based spoken-to-signed pipeline** with a practical API and Unity integration:

- **Text → Gloss → Pose → JSON**
- **Text → Gloss → Pose → MP4 (landmarks)**
- **Audio → Transcription → Translation → Pose → UDP to Unity**
- **UDP text overlay in Unity**

![Visualization of our pipeline](assets/pipeline.jpg)

---

## What’s Inside

- **FastAPI backend** (`backend_api.py`) with endpoints for:
  - audio transcription
  - JSON landmarks generation
  - MP4 landmarks rendering
  - UDP streaming to Unity
- **Unity scripts** for:
  - UDP landmarks receiver + text overlay
  - Record button (red/green), microphone capture, and audio upload
- **Dummy lexicon** (`assets/dummy_lexicon`) for quick testing

---

## Quick Start (Backend)

```bash
# create/activate venv
python -m venv venv
source venv/bin/activate

# install deps
pip install -r requirements.txt

# run API
uvicorn backend_api:app --host 127.0.0.1 --port 5000
```

API will be available at:
- Swagger UI: `http://127.0.0.1:5000/docs`

---

## Endpoints

### `POST /transcribe`
Upload audio (`wav`) and return transcription. Also generates landmarks and streams them to Unity over UDP.

- Sends landmarks to `UDP_REPLAY_PORT` (default **5053**)
- Sends text to `UDP_TEXT_PORT` (default **5054**)

### `POST /text-to-landmarks-json`
Send text, get landmarks JSON in response.

### `POST /text-to-landmarks-video`
Send text, get MP4 landmarks video.

### `POST /text-to-landmarks-udp`
Send text, stream landmarks via UDP to Unity (and send text).

---

## Translation + Language Detection

For text endpoints, language is auto‑detected and translated to German before pose generation:

- `langdetect` detects input language
- `deep-translator` translates to **DE**

You can disable or override with payload fields:

```json
{
  "translate": true,
  "auto_detect_language": true,
  "source_language": "auto",
  "target_language": "de"
}
```

---

## Unity Integration

### UDP Receiver (Landmarks + Text)
Script: `UdpReceiver.cs`

- Receives landmarks on port **5053**
- Receives text on port **5054**
- Requires a `TextMeshPro` field to display text

### Record & Send (Microphone + Button)
Script: `RecordAndSendAudio.cs`

- Red button = recording
- Green button = idle
- Sends audio to `http://127.0.0.1:5000/transcribe`
- Updates transcript text

---

## Environment Variables

```bash
UDP_IP=127.0.0.1
UDP_PORT=5052
UDP_REPLAY_PORT=5053
UDP_TEXT_PORT=5054
```

---

## Notes

- Dummy lexicon is minimal and only covers a few German words.
- If you see lookup errors for unknown words, use the provided lexicon or install a larger one.
- UDP landmarks match MediaPipe layout (pose/face/hands).

---

## Original Project (Upstream)

This repo is built on top of the **ZurichNLP spoken-to-signed** pipeline:
- Paper: https://arxiv.org/abs/2305.17714
- Demo: https://sign.mt

---

## License

See `LICENSE`.
