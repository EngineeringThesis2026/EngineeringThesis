"""
LLM Provider abstraction with fallback support.

This module provides a unified interface for different LLM providers (OpenAI, Anthropic)
with automatic fallback when the primary provider fails.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import openai
import anthropic
import time

from exceptions import LLMProviderError, AllProvidersFailedError, APIKeyMissingError
from logger import logger


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, name: str, model: str, temperature: float = 0, max_tokens: Optional[int] = None):
        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

        # Validation tracking
        self._validation_status = None  # 'valid', 'invalid', 'quota_exceeded', 'permission_denied', 'unknown'
        self._validation_timestamp = None
        self._validation_message = None

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the LLM client. Should be called before use."""
        pass

    @abstractmethod
    def validate_key(self) -> Tuple[bool, str]:
        """
        Validate API key before initialization.

        Returns:
            Tuple of (is_valid, status_message)
            - is_valid: True if key is valid and usable
            - status_message: Description of validation result
        """
        pass

    def is_validation_cached(self, cache_duration_minutes: int = 10) -> bool:
        """
        Check if validation result is still fresh.

        Args:
            cache_duration_minutes: How long to cache validation results (default 10 minutes)

        Returns:
            True if validation result exists and is not expired
        """
        if self._validation_timestamp is None:
            return False

        elapsed = time.time() - self._validation_timestamp
        return elapsed < (cache_duration_minutes * 60)

    @abstractmethod
    def invoke(self, messages: List[Dict[str, str]]) -> Any:
        """
        Invoke the LLM with messages.

        Args:
            messages: List of message dicts or LangChain message objects

        Returns:
            Response object with .content attribute
        """
        pass

    def is_available(self) -> bool:
        """
        Check if this provider is available (has valid API key and initialized client).

        Returns:
            True if client is initialized AND validation passed (or not yet validated)
        """
        # Client must exist
        if self._client is None:
            return False

        # If validation was performed, it must have passed
        if self._validation_status is not None:
            return self._validation_status == 'valid'

        # If not validated yet, assume available (will be validated on first use)
        return True


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider."""

    def __init__(self, model: str = 'gpt-4o-mini', temperature: float = 0,
                 max_tokens: Optional[int] = None, max_retries: int = 2):
        super().__init__(name="OpenAI", model=model, temperature=temperature, max_tokens=max_tokens)
        self.max_retries = max_retries

    def validate_key(self) -> Tuple[bool, str]:
        """Validate OpenAI API key using models.list() endpoint (free)."""
        try:
            # Check if API key exists in secrets
            if "OPENAI_API_KEY" not in st.secrets:
                self._validation_status = 'invalid'
                self._validation_timestamp = time.time()
                self._validation_message = "API key not found in secrets"
                return False, "API key not found in secrets"

            api_key = st.secrets["OPENAI_API_KEY"]

            # Check if key is non-empty
            if not api_key or not isinstance(api_key, str) or api_key.strip() == "":
                self._validation_status = 'invalid'
                self._validation_timestamp = time.time()
                self._validation_message = "API key is empty or invalid"
                return False, "API key is empty or invalid"

            # Test the key with models.list() endpoint (free, fast)
            logger.info(f"Validating {self.name} API key...")
            test_client = openai.OpenAI(api_key=api_key, timeout=5.0)
            models = test_client.models.list()

            # Success!
            self._validation_status = 'valid'
            self._validation_timestamp = time.time()
            self._validation_message = "API key validated successfully"
            logger.info(f"✅ {self.name} API key is valid")
            return True, "API key validated successfully"

        except openai.AuthenticationError as e:
            # 401 - Invalid or expired key
            self._validation_status = 'invalid'
            self._validation_timestamp = time.time()
            msg = "Invalid or expired API key"
            self._validation_message = msg
            logger.error(f"❌ {self.name} validation failed (401): {e}")
            return False, msg

        except openai.PermissionDeniedError as e:
            # 403 - Key lacks permissions
            self._validation_status = 'permission_denied'
            self._validation_timestamp = time.time()
            msg = "API key lacks required permissions"
            self._validation_message = msg
            logger.error(f"❌ {self.name} validation failed (403): {e}")
            return False, msg

        except openai.RateLimitError as e:
            # 429 - Rate limit or billing issue (key exists but can't be used)
            self._validation_status = 'quota_exceeded'
            self._validation_timestamp = time.time()

            error_message = str(e).lower()
            if 'quota' in error_message or 'billing' in error_message or 'insufficient' in error_message:
                msg = "API key valid but has quota/billing issues"
            else:
                msg = "API key valid but rate limited"

            self._validation_message = msg
            logger.warning(f"⚠️ {self.name} validation (429): {msg}")
            return False, msg

        except openai.APIConnectionError as e:
            # Network/connection issues
            self._validation_status = 'unknown'
            self._validation_timestamp = time.time()
            msg = f"Connection error: {str(e)[:100]}"
            self._validation_message = msg
            logger.error(f"❌ {self.name} validation connection error: {e}")
            return False, msg

        except openai.Timeout as e:
            # Request timeout
            self._validation_status = 'unknown'
            self._validation_timestamp = time.time()
            msg = "Validation request timed out"
            self._validation_message = msg
            logger.error(f"❌ {self.name} validation timeout: {e}")
            return False, msg

        except Exception as e:
            # Unexpected error
            self._validation_status = 'unknown'
            self._validation_timestamp = time.time()
            msg = f"Unexpected error: {type(e).__name__}: {str(e)[:100]}"
            self._validation_message = msg
            logger.error(f"❌ {self.name} validation unexpected error: {e}")
            return False, msg

    def initialize(self) -> None:
        """Initialize OpenAI client with validation."""
        # Validate API key first if not cached
        if not self.is_validation_cached():
            is_valid, message = self.validate_key()
            if not is_valid:
                logger.warning(f"{self.name} initialization skipped: {message}")
                return
        elif self._validation_status != 'valid':
            # Cached validation failed
            logger.warning(f"{self.name} initialization skipped: cached validation failed")
            return

        try:
            api_key = st.secrets["OPENAI_API_KEY"]

            # Create client
            self._client = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                max_retries=self.max_retries,
                api_key=api_key,
            )

            logger.info(f"OpenAI provider initialized with model: {self.model}")

        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI provider: {e}")
            self._client = None

    def invoke(self, messages: List[Any]) -> Any:
        """Invoke OpenAI LLM."""
        if not self._client:
            raise LLMProviderError(f"{self.name} provider not initialized")

        try:
            response = self._client.invoke(messages)
            logger.info(f"Successfully invoked {self.name}")
            return response

        except openai.RateLimitError as e:
            logger.error(f"{self.name} rate limit exceeded: {e}")
            raise LLMProviderError(f"{self.name} rate limit exceeded")

        except openai.APIError as e:
            logger.error(f"{self.name} API error: {e}")
            raise LLMProviderError(f"{self.name} API error: {e}")

        except openai.Timeout as e:
            logger.error(f"{self.name} timeout: {e}")
            raise LLMProviderError(f"{self.name} request timeout")

        except openai.AuthenticationError as e:
            logger.error(f"{self.name} authentication error: {e}")
            raise LLMProviderError(f"{self.name} authentication failed")

        except Exception as e:
            logger.error(f"{self.name} unexpected error: {type(e).__name__}: {e}")
            raise LLMProviderError(f"{self.name} error: {e}")


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider."""

    def __init__(self, model: str = 'claude-3-haiku-20240307', temperature: float = 0,
                 max_tokens: Optional[int] = None, max_retries: int = 2):
        super().__init__(name="Anthropic Claude", model=model, temperature=temperature, max_tokens=max_tokens)
        self.max_retries = max_retries

    def validate_key(self) -> Tuple[bool, str]:
        """Validate Anthropic API key with minimal request (costs ~$0.00001)."""
        try:
            # Check if API key exists in secrets
            if "ANTHROPIC_API_KEY" not in st.secrets:
                self._validation_status = 'invalid'
                self._validation_timestamp = time.time()
                self._validation_message = "API key not found in secrets"
                return False, "API key not found in secrets"

            api_key = st.secrets["ANTHROPIC_API_KEY"]

            # Check if key is non-empty
            if not api_key or not isinstance(api_key, str) or api_key.strip() == "":
                self._validation_status = 'invalid'
                self._validation_timestamp = time.time()
                self._validation_message = "API key is empty or invalid"
                return False, "API key is empty or invalid"

            # Test with minimal request (costs ~0.00001 USD)
            logger.info(f"Validating {self.name} API key...")
            test_client = anthropic.Anthropic(api_key=api_key, timeout=5.0)
            response = test_client.messages.create(
                model="claude-3-haiku-20240307",  # Cheapest model
                max_tokens=1,  # Minimal output
                messages=[{"role": "user", "content": "Hi"}]
            )

            # Success!
            self._validation_status = 'valid'
            self._validation_timestamp = time.time()
            self._validation_message = "API key validated successfully"
            logger.info(f"✅ {self.name} API key is valid")
            return True, "API key validated successfully"

        except anthropic.AuthenticationError as e:
            # 401 - Invalid or expired key
            self._validation_status = 'invalid'
            self._validation_timestamp = time.time()
            msg = "Invalid or expired API key"
            self._validation_message = msg
            logger.error(f"❌ {self.name} validation failed (401): {e}")
            return False, msg

        except anthropic.PermissionDeniedError as e:
            # 403 - Key lacks permissions
            self._validation_status = 'permission_denied'
            self._validation_timestamp = time.time()
            msg = "API key lacks required permissions"
            self._validation_message = msg
            logger.error(f"❌ {self.name} validation failed (403): {e}")
            return False, msg

        except anthropic.RateLimitError as e:
            # 429 - Rate limit exceeded (key exists but can't be used)
            self._validation_status = 'quota_exceeded'
            self._validation_timestamp = time.time()

            error_message = str(e).lower()
            if 'billing' in error_message or 'quota' in error_message:
                msg = "API key valid but has quota/billing issues"
            else:
                msg = "API key valid but rate limited"

            self._validation_message = msg
            logger.warning(f"⚠️ {self.name} validation (429): {msg}")
            return False, msg

        except anthropic.APIConnectionError as e:
            # Network/connection issues
            self._validation_status = 'unknown'
            self._validation_timestamp = time.time()
            msg = f"Connection error: {str(e)[:100]}"
            self._validation_message = msg
            logger.error(f"❌ {self.name} validation connection error: {e}")
            return False, msg

        except anthropic.APIStatusError as e:
            # Generic API error
            self._validation_status = 'unknown'
            self._validation_timestamp = time.time()
            msg = f"API error (status {e.status_code}): {str(e)[:100]}"
            self._validation_message = msg
            logger.error(f"❌ {self.name} validation API error: {e}")
            return False, msg

        except Exception as e:
            # Unexpected error
            self._validation_status = 'unknown'
            self._validation_timestamp = time.time()
            msg = f"Unexpected error: {type(e).__name__}: {str(e)[:100]}"
            self._validation_message = msg
            logger.error(f"❌ {self.name} validation unexpected error: {e}")
            return False, msg

    def initialize(self) -> None:
        """Initialize Anthropic client with validation."""
        # Validate API key first if not cached
        if not self.is_validation_cached():
            is_valid, message = self.validate_key()
            if not is_valid:
                logger.warning(f"{self.name} initialization skipped: {message}")
                return
        elif self._validation_status != 'valid':
            # Cached validation failed
            logger.warning(f"{self.name} initialization skipped: cached validation failed")
            return

        try:
            api_key = st.secrets["ANTHROPIC_API_KEY"]

            # Create client
            self._client = ChatAnthropic(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens or 1024,  # Anthropic requires max_tokens
                max_retries=self.max_retries,
                api_key=api_key,
            )

            logger.info(f"Anthropic provider initialized with model: {self.model}")

        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic provider: {e}")
            self._client = None

    def invoke(self, messages: List[Any]) -> Any:
        """Invoke Anthropic LLM."""
        if not self._client:
            raise LLMProviderError(f"{self.name} provider not initialized")

        try:
            response = self._client.invoke(messages)
            logger.info(f"Successfully invoked {self.name}")
            return response

        except Exception as e:
            logger.error(f"{self.name} error: {type(e).__name__}: {e}")
            raise LLMProviderError(f"{self.name} error: {e}")


class LLMManager:
    """
    Manages multiple LLM providers with automatic fallback.

    Tries providers in order until one succeeds. Mimics ChatOpenAI interface
    for compatibility with existing code.
    """

    def __init__(self, providers: List[BaseLLMProvider]):
        """
        Initialize LLM manager with list of providers.

        Args:
            providers: List of provider instances in fallback order
        """
        self.providers = providers
        self._current_provider = None

        # Initialize all providers
        for provider in self.providers:
            provider.initialize()

        # Find first available provider
        available_providers = [p for p in self.providers if p.is_available()]

        if not available_providers:
            logger.error("No LLM providers are available!")
            raise AllProvidersFailedError("No LLM providers configured with valid API keys")

        self._current_provider = available_providers[0]
        logger.info(f"LLMManager initialized with {len(available_providers)} available provider(s)")

        # Initialize session state for UI
        try:
            st.session_state['current_provider'] = self._current_provider.name
        except Exception as e:
            logger.debug(f"Could not initialize session state: {e}")

    def invoke(self, messages: List[Any]) -> Any:
        """
        Invoke LLM with automatic fallback to next provider on failure.

        Args:
            messages: List of message dicts or LangChain message objects

        Returns:
            Response object with .content attribute

        Raises:
            AllProvidersFailedError: If all providers fail
        """
        errors = []
        available_count = len([p for p in self.providers if p.is_available()])

        logger.info(f"LLMManager.invoke() called with {available_count} available provider(s)")

        for idx, provider in enumerate(self.providers, 1):
            if not provider.is_available():
                logger.debug(f"Skipping unavailable provider: {provider.name}")
                continue

            try:
                logger.info(f"[Attempt {idx}/{available_count}] Trying provider: {provider.name}")
                response = provider.invoke(messages)

                # Success!
                if self._current_provider != provider:
                    logger.info(f"✅ Switched to provider: {provider.name}")
                    self._current_provider = provider
                    # Update session state for UI
                    try:
                        st.session_state['current_provider'] = provider.name
                    except Exception as e:
                        logger.debug(f"Could not update session state: {e}")
                else:
                    logger.info(f"✅ Response received from {provider.name}")

                return response

            except LLMProviderError as e:
                logger.warning(f"❌ Provider {provider.name} failed: {e}")
                errors.append(f"{provider.name}: {e}")

                # Check if there are more providers to try
                remaining = available_count - idx
                if remaining > 0:
                    logger.info(f"🔄 Attempting fallback to next provider ({remaining} remaining)...")
                continue

        # All providers failed
        error_summary = " | ".join(errors)
        logger.error(f"🚫 All {available_count} provider(s) failed: {error_summary}")
        raise AllProvidersFailedError(f"All LLM providers failed: {error_summary}")

    @property
    def temperature(self) -> float:
        """Get temperature of current provider."""
        return self._current_provider.temperature if self._current_provider else 0

    @property
    def max_tokens(self) -> Optional[int]:
        """Get max_tokens of current provider."""
        return self._current_provider.max_tokens if self._current_provider else None

    def get_current_provider_name(self) -> str:
        """Get name of currently active provider."""
        return self._current_provider.name if self._current_provider else "None"

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return [p.name for p in self.providers if p.is_available()]

    def get_provider_statuses(self) -> List[Dict[str, Any]]:
        """
        Get detailed status information for all providers.

        Returns:
            List of dicts with keys:
            - name: Provider name
            - available: Whether provider has valid client initialized
            - active: Whether this is the currently active provider
            - validation_status: 'valid', 'invalid', 'quota_exceeded', 'permission_denied', 'unknown', or None
            - validation_message: Detailed message about validation result
        """
        statuses = []
        current_name = self._current_provider.name if self._current_provider else None

        for provider in self.providers:
            statuses.append({
                'name': provider.name,
                'available': provider.is_available(),
                'active': provider.name == current_name,
                'validation_status': provider._validation_status,
                'validation_message': provider._validation_message
            })

        return statuses
