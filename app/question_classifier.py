"""
Question Classifier for Legal Advisory System.

This module provides functionality to classify whether a user's question
is related to legal topics, preventing the system from answering off-topic questions.
"""

from logger import logger
from typing import Any

# Example legal questions to show users when their question is rejected
LEGAL_QUESTION_EXAMPLES = [
    "Jakie mam prawa jako najemca mieszkania?",
    "Jak napisać testament zgodnie z polskim prawem?",
    "Czy mogę rozwiązać umowę o pracę bez wypowiedzenia?",
    "Jakie są konsekwencje prawne niepłacenia alimentów?",
    "Co to jest przedawnienie roszczeń?",
    "Jak wygląda procedura rozwodowa w Polsce?",
    "Jakie prawa przysługują konsumentowi przy zakupie wadliwego towaru?"
]


def is_legal_question(question: str, llm_manager: Any) -> bool:
    """
    Classify whether a question is related to legal topics.

    Uses the provided LLM manager (with fallback support) to determine if the question is about law.
    If classifier fails, defaults to True (fail-open) to not block users.

    Args:
        question: The user's question to classify
        llm_manager: LLMManager instance to use for classification

    Returns:
        bool: True if question is legal-related, False otherwise
              Returns True if classifier fails (fail-open)
    """
    try:
        # Fail-open: if llm_manager not available, allow question through
        if llm_manager is None:
            logger.warning("LLM manager not available, allowing question through")
            return True

        # Prepare classification prompt
        system_prompt = """Jesteś klasyfikatorem pytań prawnych dla systemu doradztwa prawnego.

Twoim zadaniem jest określić czy pytanie użytkownika dotyczy prawa polskiego lub kwestii prawnych.

**Odpowiedz TYLKO słowem "TAK" lub "NIE".**

Przykłady pytań prawnych (TAK):
- pytania o przepisy prawa, kodeksy, ustawy
- pytania o prawa i obowiązki obywateli, pracowników, konsumentów
- pytania o procedury prawne (rozwód, testament, umowa)
- pytania o roszczenia, pozwy, sprawy sądowe
- pytania o prawo karne, cywilne, rodzinne, pracy, konsumenckie
- pytania o odpowiedzialność prawną
- pytania o dokumenty prawne (umowy, akty notarialne)

Przykłady pytań NIE-prawnych (NIE):
- ogólne pytania życiowe nie związane z prawem (pogoda, gotowanie, sport, rozrywka)
- nauki ścisłe i matematyka
- porady zdrowotne i medyczne
- technologia i programowanie (chyba że o prawie IT)
- historia, geografia (chyba że historia prawa)
- pytania filozoficzne nie związane z prawem
"""

        user_prompt = f"Pytanie użytkownika: {question}"

        # Invoke classifier using LLM manager (with automatic fallback)
        # Note: We use the same LLM manager but it will use minimal tokens for classification
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = llm_manager.invoke(messages)
        answer = response.content.strip().upper()

        # Parse response
        is_legal = "TAK" in answer
        logger.info(f"Question classified as legal: {is_legal} (provider: {llm_manager.get_current_provider_name()}) | Question: {question[:50]}...")

        return is_legal

    except Exception as e:
        # Fail-open: on any error, allow question through
        logger.warning(f"Classifier error: {type(e).__name__}: {e}, allowing question through")
        return True


def get_rejection_message() -> str:
    """
    Get the message to show users when their question is rejected as non-legal.

    Returns:
        str: Formatted rejection message with example legal questions
    """
    examples_text = "\n".join(f"• {example}" for example in LEGAL_QUESTION_EXAMPLES)

    message = f"""Przepraszam, ale specjalizuję się wyłącznie w doradztwie prawnym i nie mogę odpowiedzieć na pytania spoza tej dziedziny.

📋 **Oto przykładowe pytania prawne, na które mogę odpowiedzieć:**

{examples_text}

Czy masz pytanie dotyczące prawa polskiego?"""

    return message
