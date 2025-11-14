import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_llm(model='gpt-4o-mini',temperature=0, max_tokens=None,timeout=None,max_retries=2, api_key=None)->ChatOpenAI:
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
    if api_key is None:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = api_key
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        api_key=api_key,
    )
    return llm

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




