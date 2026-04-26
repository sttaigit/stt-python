"""STT.ai API client."""

import os
import json
import tempfile
import mimetypes
from urllib.parse import urlparse

import requests

from sttai.exceptions import STTError, AuthError, RateLimitError, CreditError


class STTClient:
    """Client for the STT.ai Speech-to-Text API.

    Args:
        api_key: Your STT.ai API key. If not provided, reads from the
            ``STT_API_KEY`` environment variable.
        base_url: Base URL for the API. Defaults to ``https://api.stt.ai``.
        timeout: Request timeout in seconds. Defaults to 300 (5 minutes)
            to accommodate long audio files.

    Raises:
        AuthError: If no API key is provided or found in the environment.

    Example::

        from sttai import STTClient

        client = STTClient("your-api-key")
        result = client.transcribe("meeting.mp3")
        print(result["text"])
    """

    def __init__(self, api_key=None, base_url="https://api.stt.ai",
                 web_base_url="https://stt.ai", timeout=300):
        self.api_key = api_key or os.environ.get("STT_API_KEY")
        if not self.api_key:
            raise AuthError(
                "API key is required. Pass it as an argument or set the "
                "STT_API_KEY environment variable."
            )
        self.base_url = base_url.rstrip("/")
        # Transcript-management endpoints (list, get, chat, etc.) live on the
        # Django web app at stt.ai/api/v1/, not the GPU API at api.stt.ai.
        self.web_base_url = web_base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": "Bearer {}".format(self.api_key),
            "User-Agent": "stt-ai-python/0.1.0",
        })

    def _handle_response(self, response):
        """Process an API response and raise appropriate exceptions on error.

        Returns:
            Parsed JSON response as a dictionary.

        Raises:
            AuthError: On 401/403 responses.
            RateLimitError: On 429 responses.
            CreditError: On 402 responses.
            STTError: On any other error response.
        """
        if response.ok:
            return response.json()

        status = response.status_code
        try:
            body = response.json()
            message = body.get("error", body.get("detail", response.text))
        except (ValueError, KeyError):
            message = response.text or "Unknown error"

        if status in (401, 403):
            raise AuthError(message, status_code=status, response=response)
        elif status == 402:
            raise CreditError(message, status_code=status, response=response)
        elif status == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    retry_after = int(retry_after)
                except (ValueError, TypeError):
                    retry_after = None
            raise RateLimitError(
                message,
                status_code=status,
                response=response,
                retry_after=retry_after,
            )
        else:
            raise STTError(message, status_code=status, response=response)

    def transcribe(
        self,
        file_path,
        model="large-v3-turbo",
        language="auto",
        diarize=True,
        speakers=0,
        response_format="json",
    ):
        """Transcribe an audio or video file.

        Args:
            file_path: Path to the audio/video file to transcribe.
            model: Whisper model to use. Defaults to ``"large-v3-turbo"``.
            language: Language code (e.g. ``"en"``, ``"es"``) or ``"auto"``
                for automatic detection.
            diarize: Whether to enable speaker diarization.
            speakers: Expected number of speakers. ``0`` for automatic detection.
            response_format: Output format — ``"json"``, ``"text"``, ``"srt"``,
                ``"vtt"``, or ``"verbose_json"``.

        Returns:
            dict: Transcription result. The structure depends on
            ``response_format``.

        Raises:
            STTError: If the file cannot be read or the API returns an error.
            FileNotFoundError: If ``file_path`` does not exist.
        """
        file_path = os.path.expanduser(file_path)
        if not os.path.isfile(file_path):
            raise FileNotFoundError("File not found: {}".format(file_path))

        content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"

        data = {
            "model": model,
            "language": language,
            "diarize": str(diarize).lower(),
            "speakers": str(speakers),
            "response_format": response_format,
        }

        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, content_type)}
            response = self._session.post(
                "{}/v1/transcribe".format(self.base_url),
                data=data,
                files=files,
                timeout=self.timeout,
            )

        return self._handle_response(response)

    def transcribe_url(
        self,
        url,
        model="large-v3-turbo",
        language="auto",
        diarize=True,
        speakers=0,
        response_format="json",
    ):
        """Download an audio/video file from a URL and transcribe it.

        The file is downloaded to a temporary location, transcribed, and then
        deleted.

        Args:
            url: URL of the audio/video file to download and transcribe.
            model: Whisper model to use.
            language: Language code or ``"auto"``.
            diarize: Whether to enable speaker diarization.
            speakers: Expected number of speakers.
            response_format: Output format.

        Returns:
            dict: Transcription result.

        Raises:
            STTError: If the download or transcription fails.
        """
        parsed = urlparse(url)
        ext = os.path.splitext(parsed.path)[1] or ".tmp"

        try:
            dl = requests.get(url, stream=True, timeout=self.timeout)
            dl.raise_for_status()
        except requests.RequestException as e:
            raise STTError("Failed to download file from URL: {}".format(e))

        tmp = None
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            for chunk in dl.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp.close()

            return self.transcribe(
                tmp.name,
                model=model,
                language=language,
                diarize=diarize,
                speakers=speakers,
                response_format=response_format,
            )
        finally:
            if tmp and os.path.exists(tmp.name):
                os.unlink(tmp.name)

    def summarize(self, text, style="brief"):
        """Summarize transcribed text.

        Args:
            text: The text to summarize.
            style: Summary style — ``"brief"``, ``"detailed"``, ``"bullets"``,
                or ``"action_items"``.

        Returns:
            dict: Summary result with ``"summary"`` key.
        """
        payload = {"text": text, "style": style}
        response = self._session.post(
            "{}/v1/summarize".format(self.base_url),
            json=payload,
            timeout=self.timeout,
        )
        return self._handle_response(response)

    def models(self):
        """List available transcription models.

        Returns:
            dict: Dictionary containing available models and their details.
        """
        response = self._session.get(
            "{}/v1/models".format(self.base_url),
            timeout=self.timeout,
        )
        return self._handle_response(response)

    def languages(self):
        """List supported languages.

        Returns:
            dict: Dictionary containing supported languages.
        """
        response = self._session.get(
            "{}/v1/languages".format(self.base_url),
            timeout=self.timeout,
        )
        return self._handle_response(response)

    def health(self):
        """Check API health status.

        This endpoint does not require authentication.

        Returns:
            dict: Health status of the API.
        """
        response = requests.get(
            "{}/health".format(self.base_url),
            timeout=30,
        )
        return self._handle_response(response)

    # ------------------------------------------------------------------
    # Transcript management — live on the Django web app at stt.ai/api/v1/
    # ------------------------------------------------------------------

    def list_transcripts(self, limit=20, offset=0, status=None):
        """List your transcripts.

        Args:
            limit: Max number of transcripts to return (default 20).
            offset: Number of transcripts to skip (for pagination).
            status: Filter by status — ``"completed"``, ``"processing"``,
                ``"failed"``, etc. ``None`` returns all.

        Returns:
            dict: ``{"results": [...], "count": int, "next": str|None,
            "previous": str|None}``.
        """
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        response = self._session.get(
            "{}/api/v1/transcripts/".format(self.web_base_url),
            params=params,
            timeout=self.timeout,
        )
        return self._handle_response(response)

    def get_transcript(self, transcript_id):
        """Fetch a single transcript with its segments and metadata.

        Args:
            transcript_id: Numeric transcript ID OR slug (the URL fragment).

        Returns:
            dict: Full transcript including ``segments``, ``speakers``,
            ``summary`` (if generated), and metadata.
        """
        response = self._session.get(
            "{}/api/v1/transcripts/{}/".format(self.web_base_url, transcript_id),
            timeout=self.timeout,
        )
        return self._handle_response(response)

    def export_transcript(self, transcript_id, fmt="txt"):
        """Download a transcript in a specific format.

        Args:
            transcript_id: Numeric ID or slug.
            fmt: Output format — ``"txt"``, ``"srt"``, ``"vtt"``, ``"json"``,
                ``"csv"``, ``"docx"``, ``"pdf"``.

        Returns:
            For text formats (txt/srt/vtt/csv/json): the raw response text.
            For binary formats (docx/pdf): the raw bytes.
        """
        response = self._session.get(
            "{}/api/v1/transcripts/{}/export/{}/".format(
                self.web_base_url, transcript_id, fmt
            ),
            timeout=self.timeout,
        )
        if not response.ok:
            return self._handle_response(response)
        if fmt in ("docx", "pdf"):
            return response.content
        return response.text

    def transcript_summarize(self, transcript_id, force=False):
        """Get or generate an AI summary for an existing transcript.

        Args:
            transcript_id: Numeric ID or slug.
            force: If ``True``, regenerate even if cached. Otherwise returns
                the cached summary when available.

        Returns:
            dict: ``{"summary": str, "topics": [...], ...}``.
        """
        if force:
            response = self._session.post(
                "{}/api/v1/transcripts/{}/summarize/".format(
                    self.web_base_url, transcript_id
                ),
                timeout=self.timeout,
            )
        else:
            response = self._session.get(
                "{}/api/v1/transcripts/{}/summarize/".format(
                    self.web_base_url, transcript_id
                ),
                timeout=self.timeout,
            )
        return self._handle_response(response)

    def transcript_analyze(self, transcript_id, kinds=None):
        """Run AI analysis on a transcript: sentiment, topics, entities,
        action items, questions, PII redaction.

        Args:
            transcript_id: Numeric ID or slug.
            kinds: List of analysis kinds to run, or ``None`` for all.
                E.g. ``["sentiment", "action_items"]``.

        Returns:
            dict: Analysis results keyed by kind.
        """
        payload = {}
        if kinds:
            payload["kinds"] = kinds
        response = self._session.post(
            "{}/api/v1/transcripts/{}/analyze/".format(
                self.web_base_url, transcript_id
            ),
            json=payload,
            timeout=self.timeout,
        )
        return self._handle_response(response)

    def transcript_generate(self, transcript_id, kind, prompt=None):
        """Generate content from a transcript: blog post, social media,
        meeting notes, study guide, flashcards, quiz.

        Args:
            transcript_id: Numeric ID or slug.
            kind: One of ``"blog_post"``, ``"social_twitter"``,
                ``"social_linkedin"``, ``"meeting_notes"``, ``"study_guide"``,
                ``"flashcards"``, ``"quiz"``, ``"newsletter"``,
                ``"podcast_show_notes"``.
            prompt: Optional custom instruction to steer generation.

        Returns:
            dict: ``{"content": str, ...}``.
        """
        payload = {"kind": kind}
        if prompt:
            payload["prompt"] = prompt
        response = self._session.post(
            "{}/api/v1/transcripts/{}/generate/".format(
                self.web_base_url, transcript_id
            ),
            json=payload,
            timeout=self.timeout,
        )
        return self._handle_response(response)

    def transcript_chat(self, transcript_id, message, session_id=None):
        """Ask a question about a transcript via RAG. Cites source segments.

        Args:
            transcript_id: Numeric ID or slug.
            message: The user's question.
            session_id: Optional session ID to maintain conversation context
                across multiple chat() calls.

        Returns:
            dict: ``{"answer": str, "sources": [{"segment_id": ...,
            "text": ...}, ...], "session_id": str}``.
        """
        payload = {"message": message}
        if session_id:
            payload["session_id"] = session_id
        response = self._session.post(
            "{}/api/v1/transcripts/{}/chat/".format(
                self.web_base_url, transcript_id
            ),
            json=payload,
            timeout=self.timeout,
        )
        return self._handle_response(response)

    def stream(self, callback, model="large-v3-turbo", language="auto"):
        """Stream audio for real-time transcription via WebSocket.

        Opens a WebSocket connection to the STT.ai streaming endpoint.
        Returns a ``StreamSession`` object that you can use to send audio
        chunks and receive partial transcriptions.

        Args:
            callback: A callable that receives each partial transcription
                result as a dictionary. Called on every message from the server.
            model: Whisper model to use.
            language: Language code or ``"auto"``.

        Returns:
            StreamSession: A session object with ``send()``, ``finish()``,
            and ``close()`` methods.

        Example::

            def on_transcript(data):
                print(data["text"])

            session = client.stream(on_transcript)
            with open("audio.raw", "rb") as f:
                while chunk := f.read(4096):
                    session.send(chunk)
            result = session.finish()
        """
        return StreamSession(
            base_url=self.base_url,
            api_key=self.api_key,
            callback=callback,
            model=model,
            language=language,
        )


class StreamSession:
    """A WebSocket streaming session for real-time transcription.

    Do not instantiate directly; use :meth:`STTClient.stream` instead.
    """

    def __init__(self, base_url, api_key, callback, model, language):
        try:
            import websocket
        except ImportError:
            raise ImportError(
                "websocket-client is required for streaming. "
                "Install it with: pip install websocket-client"
            )

        self._callback = callback
        self._final_result = None
        self._error = None

        ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = "{}/v1/stream?model={}&language={}&token={}".format(
            ws_url, model, language, api_key
        )

        self._ws = websocket.WebSocketApp(
            ws_url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        import threading
        self._thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._thread.start()

        # Wait briefly for connection to establish
        import time
        time.sleep(0.5)

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
        except (ValueError, TypeError):
            data = {"text": message}

        if data.get("final"):
            self._final_result = data

        self._callback(data)

    def _on_error(self, ws, error):
        self._error = error

    def _on_close(self, ws, close_status_code, close_msg):
        pass

    def send(self, audio_bytes):
        """Send a chunk of audio data.

        Args:
            audio_bytes: Raw audio bytes (PCM 16-bit, 16kHz, mono recommended).

        Raises:
            STTError: If the connection has been closed or errored.
        """
        if self._error:
            raise STTError("WebSocket error: {}".format(self._error))
        self._ws.send(audio_bytes, opcode=0x2)  # binary frame

    def finish(self, timeout=30):
        """Signal end of audio and wait for the final transcription.

        Args:
            timeout: Maximum seconds to wait for the final result.

        Returns:
            dict: The final transcription result.

        Raises:
            STTError: If a timeout or error occurs.
        """
        # Send empty message to signal end of stream
        try:
            self._ws.send(b"", opcode=0x2)
        except Exception:
            pass

        self._thread.join(timeout=timeout)

        if self._error:
            raise STTError("WebSocket error: {}".format(self._error))

        return self._final_result

    def close(self):
        """Close the WebSocket connection immediately."""
        try:
            self._ws.close()
        except Exception:
            pass
        self._thread.join(timeout=5)
