"""
Custom exceptions for the Legal Advisory System.

This module defines specific exception types for different error scenarios
in the application, making error handling more precise and user-friendly.
"""


class APIKeyMissingError(Exception):
    """Raised when OpenAI API key is missing from configuration."""
    pass


class APIKeyInvalidError(Exception):
    """Raised when OpenAI API key is invalid or expired."""
    pass


class QdrantConnectionError(Exception):
    """Raised when connection to Qdrant database fails."""
    pass


class QdrantOperationError(Exception):
    """Raised when a Qdrant operation (create, delete, upsert, search) fails."""
    pass


class EmbeddingModelError(Exception):
    """Raised when embedding model fails to load or encode."""
    pass


class PDFProcessingError(Exception):
    """Raised when a PDF file cannot be processed (corrupted, password-protected, etc.)."""
    pass


class DataImportError(Exception):
    """Raised when data import process fails (no documents loaded, splitting errors, etc.)."""
    pass


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    pass


class AllProvidersFailedError(LLMProviderError):
    """Raised when all configured LLM providers fail."""
    pass
