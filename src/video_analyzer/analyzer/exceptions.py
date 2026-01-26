class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    pass


class QuotaExceeded(Exception):
    """Raised when daily quota is exceeded."""

    pass


class GeminiAPIError(Exception):
    """Raised when Gemini API returns an error."""

    pass


class GroqAPIError(Exception):
    """Raised when Groq API returns an error."""

    pass
