# STT.ai Python SDK

Official Python client for the [STT.ai](https://stt.ai) Speech-to-Text API.

Transcribe audio and video files with state-of-the-art Whisper models, speaker diarization, real-time streaming, and AI-powered summarization.

## Installation

```bash
pip install stt-ai
```

## Quick Start

```python
from stt_ai import STTClient

client = STTClient("your-api-key")

# Transcribe a file
result = client.transcribe("meeting.mp3")
print(result["text"])

# Transcribe with speaker diarization
result = client.transcribe("interview.wav", diarize=True, speakers=2)
for segment in result["segments"]:
    print(f"[{segment['speaker']}] {segment['text']}")
```

## Authentication

Get your API key from [stt.ai/account](https://stt.ai/account).

Pass it directly or set the `STT_API_KEY` environment variable:

```bash
export STT_API_KEY="your-api-key"
```

```python
client = STTClient()  # reads from STT_API_KEY
```

## API Reference

### Transcribe a File

```python
result = client.transcribe(
    "audio.mp3",
    model="large-v3-turbo",  # Model to use
    language="auto",          # Language code or "auto"
    diarize=True,             # Enable speaker diarization
    speakers=0,               # Number of speakers (0 = auto-detect)
    response_format="json",   # "json", "text", "srt", "vtt", "verbose_json"
)
```

### Transcribe from URL

```python
result = client.transcribe_url(
    "https://example.com/podcast.mp3",
    model="large-v3-turbo",
    language="en",
)
```

### Summarize Text

```python
summary = client.summarize(
    result["text"],
    style="brief",  # "brief", "detailed", "bullets", "action_items"
)
print(summary["summary"])
```

### List Available Models

```python
models = client.models()
for model in models["models"]:
    print(f"{model['id']}: {model['description']}")
```

### List Supported Languages

```python
languages = client.languages()
for lang in languages["languages"]:
    print(f"{lang['code']}: {lang['name']}")
```

### Health Check

```python
status = client.health()
print(status)
```

### Real-Time Streaming

Stream audio for live transcription over WebSocket:

```python
def on_transcript(data):
    print(data["text"], end="\r")

session = client.stream(on_transcript, model="large-v3-turbo", language="en")

# Send audio chunks (PCM 16-bit, 16kHz, mono)
with open("audio.raw", "rb") as f:
    while chunk := f.read(4096):
        session.send(chunk)

# Get the final result
result = session.finish()
print("\nFinal:", result["text"])
```

## Error Handling

The SDK raises specific exceptions for different error types:

```python
from stt_ai import STTClient, AuthError, RateLimitError, CreditError, STTError

client = STTClient("your-api-key")

try:
    result = client.transcribe("audio.mp3")
except AuthError:
    print("Invalid API key")
except CreditError:
    print("Insufficient credits - top up at stt.ai/pricing")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after}s")
except STTError as e:
    print(f"API error ({e.status_code}): {e.message}")
```

## Configuration

```python
client = STTClient(
    api_key="your-api-key",
    base_url="https://api.stt.ai",  # Custom API endpoint
    timeout=300,                      # Request timeout in seconds
)
```

## Requirements

- Python 3.8+
- `requests`
- `websocket-client` (for streaming)

## License

MIT - see [LICENSE](LICENSE) for details.
