import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Optional
import openai

from exceptions import APIKeyMissingError, APIKeyInvalidError, AllProvidersFailedError
from logger import logger
from llm_providers import LLMManager, OpenAIProvider, AnthropicProvider

def create_llm(model='gpt-4o-mini', temperature=0, max_tokens=None, timeout=None, max_retries=2):
    """
    Create and return an LLMManager with automatic fallback support.

    Primary provider: OpenAI (gpt-4o-mini)
    Fallback provider: Anthropic Claude (claude-3-haiku)

    Args:
        model (str): The name of the OpenAI model to use.
        temperature (float): The temperature setting for the model.
        max_tokens (int, optional): The maximum number of tokens for the response.
        timeout (int, optional): The timeout setting for the model (not used for fallback).
        max_retries (int): The maximum number of retries for API calls.

    Returns:
        LLMManager: An LLM manager instance with fallback support.

    Raises:
        AllProvidersFailedError: If no providers are configured with valid API keys.
    """
    try:
        # Create providers in fallback order
        providers = [
            # Primary: OpenAI
            OpenAIProvider(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries
            ),
            # Fallback: Anthropic Claude (Haiku - fast and cheap)
            AnthropicProvider(
                model='claude-3-haiku-20240307',
                temperature=temperature,
                max_tokens=max_tokens or 1024,
                max_retries=max_retries
            )
        ]

        # Create LLM manager with fallback support
        llm_manager = LLMManager(providers=providers)

        logger.info(f"LLM Manager created with {len(llm_manager.get_available_providers())} available provider(s)")
        logger.info(f"Available providers: {', '.join(llm_manager.get_available_providers())}")

        return llm_manager

    except AllProvidersFailedError as e:
        logger.error(f"Failed to initialize any LLM provider: {e}")
        raise APIKeyMissingError(
            "No LLM providers available. Please configure at least one API key:\n"
            "- OPENAI_API_KEY for OpenAI (primary)\n"
            "- ANTHROPIC_API_KEY for Anthropic Claude (fallback)"
        )

    except Exception as e:
        logger.error(f"Unexpected error creating LLM manager: {type(e).__name__}: {e}")
        raise APIKeyInvalidError(f"Failed to create LLM manager: {e}")

# # Przykład użycia system message z ChatPromptTemplate
# system_prompt = SystemMessagePromptTemplate.from_template(
#     "Jesteś pomocnym i profesjonalnym asystentem AI specjalizującym się w doradztwie prawnym. Odpowiadaj wyłącznie na pytania związane z prawem, dostarczając dokładne i zwięzłe informacje."
# )
# human_prompt = HumanMessagePromptTemplate.from_template("{user_input}")

# chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
# print(chat_prompt.format_messages(user_input="Tutaj trafia przykładowe pytanie użytkownika."))

def create_chat_prompt_template(system_template: str, human_template: str) -> Optional[ChatPromptTemplate]:
    """
    Create and return a ChatPromptTemplate with specified system and human templates.
    Args:
        system_template (str): The template for the system message.
        human_template (str): The template for the human message.
    Returns:
        ChatPromptTemplate: An instance of ChatPromptTemplate, or None if error occurs.
    Raises:
        ValueError: If template format is invalid or template variables are missing.
    """
    try:
        system_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

        logger.info("Successfully created chat prompt template")
        return chat_prompt

    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"Error creating chat prompt template: {type(e).__name__}: {e}")
        raise ValueError(f"Failed to create chat prompt template: {e}")

    except Exception as e:
        logger.error(f"Unexpected error creating prompt template: {type(e).__name__}: {e}")
        raise ValueError(f"Unexpected error in prompt template creation: {e}")




