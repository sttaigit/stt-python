"""Exceptions for the STT.ai Python SDK."""


class STTError(Exception):
    """Base exception for STT.ai API errors.

    Attributes:
        message: Human-readable error message.
        status_code: HTTP status code from the API, if available.
        response: Raw response object, if available.
    """

    def __init__(self, message, status_code=None, response=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response = response

    def __str__(self):
        if self.status_code:
            return "STTError({}): {}".format(self.status_code, self.message)
        return "STTError: {}".format(self.message)


class AuthError(STTError):
    """Raised when authentication fails (invalid or missing API key)."""

    def __str__(self):
        return "AuthError: {}".format(self.message)


class RateLimitError(STTError):
    """Raised when the API rate limit is exceeded.

    Attributes:
        retry_after: Seconds to wait before retrying, if provided by the API.
    """

    def __init__(self, message, status_code=None, response=None, retry_after=None):
        super().__init__(message, status_code, response)
        self.retry_after = retry_after

    def __str__(self):
        msg = "RateLimitError: {}".format(self.message)
        if self.retry_after:
            msg += " (retry after {}s)".format(self.retry_after)
        return msg


class CreditError(STTError):
    """Raised when the account has insufficient credits."""

    def __str__(self):
        return "CreditError: {}".format(self.message)
