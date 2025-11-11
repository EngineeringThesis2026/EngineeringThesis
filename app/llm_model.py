import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_llm(model='gpt-4o-mini',temperature=0, max_tokens=None,timeout=None,max_retries=2)->ChatOpenAI:
    """
    Create and return a ChatOpenAI language model instance with specified parameters.
    Args:
        model (str): The name of the OpenAI model to use.
        temperature (float): The temperature setting for the model.
        max_tokens (int, optional): The maximum number of tokens for the response.
        timeout (int, optional): The timeout setting for the model.
        max_retries (int): The maximum number of retries for API calls.
    Returns:
        ChatOpenAI: An instance of the ChatOpenAI language model.
    """
    # Validate OpenAI API key exists in secrets
    try:
        if "OPENAI_API_KEY" not in st.secrets:
            raise KeyError("OPENAI_API_KEY not found in Streamlit secrets")

        api_key = st.secrets["OPENAI_API_KEY"]

        if not api_key or api_key.strip() == "":
            raise ValueError("OPENAI_API_KEY is empty in Streamlit secrets")

        # Basic format validation (OpenAI keys start with 'sk-')
        if not api_key.startswith("sk-"):
            print("WARNING: OpenAI API key does not start with 'sk-' - may be invalid")

        print("INFO: OpenAI API key validated successfully")
    except KeyError as e:
        print(f"CRITICAL ERROR: {e}")
        print("Please ensure .streamlit/secrets.toml exists and contains OPENAI_API_KEY")
        raise RuntimeError(f"Missing OpenAI API key in configuration: {e}") from e
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to validate OpenAI API key: {e}")
        raise RuntimeError(f"Cannot validate OpenAI API key: {e}") from e

    # Create LLM instance
    try:
        print(f"INFO: Creating ChatOpenAI instance with model '{model}'")
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            api_key=api_key,
        )
        print("INFO: ChatOpenAI instance created successfully")
        return llm
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to create ChatOpenAI instance: {e}")
        raise RuntimeError(f"Cannot create LLM: {e}") from e

# # Przykład użycia system message z ChatPromptTemplate
# system_prompt = SystemMessagePromptTemplate.from_template(
#     "Jesteś pomocnym i profesjonalnym asystentem AI specjalizującym się w doradztwie prawnym. Odpowiadaj wyłącznie na pytania związane z prawem, dostarczając dokładne i zwięzłe informacje."
# )
# human_prompt = HumanMessagePromptTemplate.from_template("{user_input}")

# chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
# print(chat_prompt.format_messages(user_input="Tutaj trafia przykładowe pytanie użytkownika."))

def create_chat_prompt_template(system_template: str, human_template: str) -> ChatPromptTemplate:
    """
    Create and return a ChatPromptTemplate with specified system and human templates.
    Args:
        system_template (str): The template for the system message.
        human_template (str): The template for the human message.
    Returns:
        ChatPromptTemplate: An instance of ChatPromptTemplate.
    """
    system_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    return chat_prompt




