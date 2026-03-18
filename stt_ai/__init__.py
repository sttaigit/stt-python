"""STT.ai Python SDK - Official client for the STT.ai Speech-to-Text API."""

from stt_ai.client import STTClient
from stt_ai.exceptions import STTError, RateLimitError, AuthError, CreditError

__version__ = "0.1.0"
__all__ = ["STTClient", "STTError", "RateLimitError", "AuthError", "CreditError"]
