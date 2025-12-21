import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


class LLMModel:
    """
    A class to encapsulate the creation of a ChatOpenAI language model and chat prompt templates.
    """

    def __init__(self, api_key: str | None = None):
        if api_key is None:
            self.api_key = st.secrets["OPENAI_API_KEY"]
        else:
            self.api_key = api_key

    def create_llm(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0,
        max_tokens: int | None = None,
        timeout: int | None = None,
        max_retries: int = 2,
    ) -> ChatOpenAI:
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
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            api_key=self.api_key,
        )

    @staticmethod
    def create_chat_prompt_template(
        system_template: str, human_template: str
    ) -> ChatPromptTemplate:
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
