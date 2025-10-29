import streamlit as st
from langchain_openai import ChatOpenAI

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
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        api_key=st.secrets["OPENAI_API_KEY"],
    )
    return llm
