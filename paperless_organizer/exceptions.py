"""Custom exceptions for Paperless-NGX Organizer."""

from __future__ import annotations


class OrganizerError(Exception):
    """Base exception for all organizer errors."""


class ConfigError(OrganizerError):
    """Configuration is invalid or missing."""


class PaperlessAPIError(OrganizerError):
    """Paperless-NGX API returned an error."""

    def __init__(self, message: str, status_code: int | None = None, detail: str = ""):
        self.status_code = status_code
        self.detail = detail
        super().__init__(message)


class LLMError(OrganizerError):
    """LLM endpoint returned an error or unexpected response."""


class LLMTimeoutError(LLMError):
    """LLM request timed out."""
