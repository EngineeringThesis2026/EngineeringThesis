import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Optional
import openai

from app.exceptions import APIKeyMissingError, APIKeyInvalidError
from app.logger import logger

def create_llm(model='gpt-4o-mini',temperature=0, max_tokens=None,timeout=None,max_retries=2) -> Optional[ChatOpenAI]:
    """
    Create and return a ChatOpenAI language model instance with specified parameters.
    Args:
        model (str): The name of the OpenAI model to use.
        temperature (float): The temperature setting for the model.
        max_tokens (int, optional): The maximum number of tokens for the response.
        timeout (int, optional): The timeout setting for the model.
        max_retries (int): The maximum number of retries for API calls.
    Returns:
        ChatOpenAI: An instance of the ChatOpenAI language model, or None if error occurs.
    Raises:
        APIKeyMissingError: If OpenAI API key is not found in secrets.
        APIKeyInvalidError: If OpenAI API key is invalid or expired.
    """
    try:
        # Check if API key exists in secrets
        if "OPENAI_API_KEY" not in st.secrets:
            logger.error("OpenAI API key not found in Streamlit secrets")
            raise APIKeyMissingError("OpenAI API key not found in .streamlit/secrets.toml")

        api_key = st.secrets["OPENAI_API_KEY"]

        # Validate API key is not empty
        if not api_key or not isinstance(api_key, str) or api_key.strip() == "":
            logger.error("OpenAI API key is empty or invalid format")
            raise APIKeyMissingError("OpenAI API key is empty or invalid")

        # Create LLM instance
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            api_key=api_key,
        )

        logger.info(f"Successfully created ChatOpenAI instance with model: {model}")
        return llm

    except KeyError as e:
        logger.error(f"KeyError accessing secrets: {e}")
        raise APIKeyMissingError(f"OpenAI API key not found in secrets: {e}")

    except openai.AuthenticationError as e:
        logger.error(f"OpenAI authentication error: {e}")
        raise APIKeyInvalidError(f"Invalid or expired OpenAI API key: {e}")

    except (ValueError, TypeError) as e:
        logger.error(f"Invalid parameters for ChatOpenAI: {e}")
        raise APIKeyInvalidError(f"Invalid configuration for OpenAI model: {e}")

    except Exception as e:
        logger.error(f"Unexpected error creating LLM: {type(e).__name__}: {e}")
        raise APIKeyInvalidError(f"Failed to create OpenAI model: {e}")

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




