import streamlit as st
import datetime
import openai
import llm_model
import vector_database
import process_data
import question_classifier

from exceptions import (
    APIKeyMissingError,
    APIKeyInvalidError,
    QdrantConnectionError,
    QdrantOperationError,
    EmbeddingModelError,
    DataImportError,
    AllProvidersFailedError,
    LLMProviderError,
)
from logger import logger

def build_history():
    return "\n".join(f"{m['role']}: {m['content']}" for m in st.session_state["messages"])

def transform_to_conversation_text()->str:
    """Loop through the history of all messages in a session and save all the contents to a string variable, then return it."""
    conversation_text = "\n\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages if msg['role'] != 'system'])
    return conversation_text

# Setting up the Streamlit page configuration
st.set_page_config(page_title="Interaktywny system doradczy do konsultacji prawnych oparty na sztucznej inteligencji")

st.title("Konsultacje prawne AI.")

# Checking the time when the conversation starts
if "conversation_data" not in st.session_state:
    st.session_state['conversation_data'] = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')

if 'data_imported' not in st.session_state:
    st.session_state['data_imported'] = False

# Initialize question classification metrics
if 'legal_questions_count' not in st.session_state:
    st.session_state['legal_questions_count'] = 0

if 'rejected_questions_count' not in st.session_state:
    st.session_state['rejected_questions_count'] = 0

if st.session_state['data_imported'] == False:
    st.write("Importowanie danych prawnych...")

    try:
        # Get data folder path
        data_folder_path = process_data.get_data_folder_path()

        # Load and process PDFs
        all_data_from_pdfs = process_data.load_data_from_pdf(file_path=data_folder_path)

        # Split documents into chunks
        all_splits = process_data.split_docks_into_chunks(documents=all_data_from_pdfs)

        # Get embedding model
        embedding_model = process_data.get_embedding_model()

        # Create embeddings with metadata
        embeddings = process_data.create_embeddings_with_metadata(
            sentences=all_splits,
            embedding_model=embedding_model
        )

        # Check if embeddings were created
        if not embeddings:
            raise DataImportError("No embeddings were generated")

        # Get vector size for Qdrant collection
        vector_size = len(embeddings[0]['embedding'])

        # Get Qdrant client
        qdrant_client = vector_database.get_qdrant_client()

        # Create collection if needed
        vector_database.create_collection_if_not_exists(
            vector_database_client=qdrant_client,
            collection_name=vector_database._collection_name,
            vector_size=vector_size
        )

        # Create points and upload to Qdrant
        points = vector_database.create_points_from_embeddings(embeddings=embeddings)
        vector_database.upload_to_qdrant(
            vector_database_client=qdrant_client,
            collection_name=vector_database._collection_name,
            points=points
        )

        st.success("✅ Pomyślnie załadowano dane prawne!")
        logger.info("Data import completed successfully")
        st.session_state['data_imported'] = True

    except DataImportError as e:
        error_msg = f"❌ Błąd importu danych: {e}"
        st.error(error_msg)
        logger.error(f"Data import error: {e}")
        st.info("📋 Instrukcje:\n- Upewnij się, że pliki PDF znajdują się w folderze `app/data/`\n- Sprawdź, czy pliki nie są uszkodzone lub zabezpieczone hasłem")
        st.stop()

    except QdrantConnectionError as e:
        error_msg = f"❌ Błąd połączenia z bazą danych Qdrant: {e}"
        st.error(error_msg)
        logger.error(f"Qdrant connection error during import: {e}")
        st.info(f"📋 Instrukcje:\n- Upewnij się, że Qdrant jest uruchomiony\n- Sprawdź URL: {vector_database._qdrant_url}\n- Docker: `docker-compose up`\n- Lokalnie: `docker run -p 6333:6333 qdrant/qdrant`")
        st.stop()

    except QdrantOperationError as e:
        error_msg = f"❌ Błąd operacji na bazie danych: {e}"
        st.error(error_msg)
        logger.error(f"Qdrant operation error during import: {e}")
        st.stop()

    except EmbeddingModelError as e:
        error_msg = f"❌ Błąd modelu embeddingów: {e}"
        st.error(error_msg)
        logger.error(f"Embedding model error during import: {e}")
        st.info("📋 Instrukcje:\n- Sprawdź połączenie internetowe (model wymaga pobrania)\n- Spróbuj ponownie uruchomić aplikację")
        st.stop()

    except (IndexError, KeyError) as e:
        error_msg = f"❌ Błąd struktury danych: {e}"
        st.error(error_msg)
        logger.error(f"Data structure error during import: {e}")
        st.info("📋 To może oznaczać, że nie udało się przetworzyć żadnych dokumentów PDF")
        st.stop()

    except Exception as e:
        error_msg = f"❌ Nieoczekiwany błąd podczas importu danych: {type(e).__name__}: {e}"
        st.error(error_msg)
        logger.error(f"Unexpected error during data import: {type(e).__name__}: {e}")
        st.stop()

# user settings in sidebar
st.sidebar.write(f'Ustawienia')

model_tokens = st.sidebar.slider('Maksymalna ilość tokenów', min_value=500, max_value=2500, value=1500)  # this is a widget

model_temperature = st.sidebar.slider('Kreatywność modelu', max_value=100)

# MAIN LOGIC

# Creating LLM instance with user-defined settings
try:
    with st.spinner("🔑 Sprawdzanie kluczy API..."):
        llm = llm_model.create_llm(
            temperature=model_temperature / 100,
            max_tokens=model_tokens
        )

    # Display LLM configuration and active provider
    st.sidebar.write(f"llm temp: {llm.temperature}, llm max tokens: {llm.max_tokens}")  # For debugging

    # Display active provider and statuses
    st.sidebar.markdown("---")
    st.sidebar.write("**🤖 Modele AI:**")

    # Get current provider from session state (updated in real-time)
    current_provider = st.session_state.get('current_provider', llm.get_current_provider_name())

    # Get detailed statuses for all providers
    provider_statuses = llm.get_provider_statuses()

    for status in provider_statuses:
        name = status['name']
        available = status['available']
        active = status['active']
        validation_status = status.get('validation_status')
        validation_message = status.get('validation_message', '')

        # Determine icon, label, and style based on validation status
        if validation_status == 'valid':
            if active:
                icon = "✅"
                label = f"{name} (aktywny)"
                st.sidebar.success(f"{icon} {label}")
            else:
                icon = "✓"
                label = f"{name} (dostępny)"
                st.sidebar.info(f"{icon} {label}")

        elif validation_status == 'invalid':
            icon = "❌"
            label = f"{name} (nieprawidłowy klucz)"
            st.sidebar.error(f"{icon} {label}")
            if validation_message:
                st.sidebar.caption(f"↳ {validation_message}")

        elif validation_status == 'quota_exceeded':
            icon = "⚠️"
            label = f"{name} (quota exceeded)"
            st.sidebar.warning(f"{icon} {label}")
            if validation_message:
                st.sidebar.caption(f"↳ {validation_message}")

        elif validation_status == 'permission_denied':
            icon = "❌"
            label = f"{name} (brak uprawnień)"
            st.sidebar.error(f"{icon} {label}")
            if validation_message:
                st.sidebar.caption(f"↳ {validation_message}")

        elif validation_status == 'unknown':
            icon = "⚠️"
            label = f"{name} (błąd walidacji)"
            st.sidebar.warning(f"{icon} {label}")
            if validation_message:
                st.sidebar.caption(f"↳ {validation_message}")

        else:
            # validation_status is None - not yet validated
            if available:
                icon = "?"
                label = f"{name} (nie walidowano)"
                st.sidebar.info(f"{icon} {label}")
            else:
                icon = "❌"
                label = f"{name} (niedostępny)"
                st.sidebar.warning(f"{icon} {label}")

    # Display classification metrics
    st.sidebar.markdown("---")
    st.sidebar.write("**📊 Statystyki pytań:**")
    st.sidebar.write(f"✅ Pytań prawnych: {st.session_state['legal_questions_count']}")
    st.sidebar.write(f"❌ Odrzuconych: {st.session_state['rejected_questions_count']}")

    logger.info("LLM instance created successfully")

except (APIKeyMissingError, AllProvidersFailedError) as e:
    error_msg = f"❌ Brak kluczy API: {e}"
    st.error(error_msg)
    logger.error(f"No LLM providers available: {e}")
    st.info("📋 Instrukcje:\n1. Skopiuj plik `.streamlit/secrets.toml.example` jako `.streamlit/secrets.toml`\n2. Dodaj przynajmniej jeden klucz API:\n```\nOPENAI_API_KEY = \"sk-...\"  # Primary\nANTHROPIC_API_KEY = \"sk-ant-...\"  # Fallback (optional)\n```\n3. Uruchom aplikację ponownie")
    st.stop()

except APIKeyInvalidError as e:
    error_msg = f"❌ Nieprawidłowy klucz API: {e}"
    st.error(error_msg)
    logger.error(f"API key invalid: {e}")
    st.info("📋 Instrukcje:\n- Sprawdź poprawność kluczy API w `.streamlit/secrets.toml`\n- Upewnij się, że klucze nie wygasły\n- OpenAI: https://platform.openai.com/api-keys\n- Anthropic: https://console.anthropic.com/settings/keys")
    st.stop()

except Exception as e:
    error_msg = f"❌ Nieoczekiwany błąd tworzenia modelu LLM: {type(e).__name__}: {e}"
    st.error(error_msg)
    logger.error(f"Unexpected error creating LLM: {type(e).__name__}: {e}")
    st.stop()

# Creating Chat Prompt Template
try:
    prompt_template = llm_model.create_chat_prompt_template(
        system_template="Jesteś pomocnym i profesjonalnym asystentem AI specjalizującym się w doradztwie prawnym. Odpowiadaj wyłącznie na pytania związane z prawem, dostarczając dokładne i zwięzłe informacje.",
        human_template="Historia rozmowy: {history}\nPytanie użytkownika: {question}\n\nKontekst:\n{context}"
    )
    logger.info("Prompt template created successfully")

except ValueError as e:
    error_msg = f"❌ Błąd tworzenia szablonu promptu: {e}"
    st.error(error_msg)
    logger.error(f"Prompt template error: {e}")
    st.stop()

# Creating Retriever
try:
    retriever = vector_database.create_retriever()
    logger.info("Retriever created successfully")

except QdrantConnectionError as e:
    error_msg = f"❌ Błąd połączenia z bazą wektorową: {e}"
    st.error(error_msg)
    logger.error(f"Retriever creation - Qdrant connection error: {e}")
    st.info("📋 Upewnij się, że import danych zakończył się pomyślnie")
    st.stop()

except QdrantOperationError as e:
    error_msg = f"❌ Błąd tworzenia retrievera: {e}"
    st.error(error_msg)
    logger.error(f"Retriever creation error: {e}")
    st.stop()

except Exception as e:
    error_msg = f"❌ Nieoczekiwany błąd tworzenia retrievera: {type(e).__name__}: {e}"
    st.error(error_msg)
    logger.error(f"Unexpected error creating retriever: {type(e).__name__}: {e}")
    st.stop()

# RAG chain
def combine_docs(docs):
        """Combine documents' page_content into a single string"""
        return "\n\n".join([f"{d.metadata}\n{d.page_content}" for d in docs])


def rag_chain_fn(question: str, retriever=retriever, llm=llm, prompt_template=prompt_template):
    """
    RAG: retrieves documents, combines context, and invokes LLM.
    Includes question classification to reject non-legal questions.
    Returns the LLM response or an error message string if something fails.
    """
    try:
        # STEP 1: Classify question (is it legal-related?)
        is_legal = question_classifier.is_legal_question(question, llm_manager=llm)

        if not is_legal:
            # Question is not legal - reject with helpful message
            logger.info(f"Non-legal question rejected: {question[:50]}...")
            st.session_state['rejected_questions_count'] += 1

            # Return rejection message with examples
            rejection_message = question_classifier.get_rejection_message()

            class RejectionResponse:
                def __init__(self, content):
                    self.content = content

            return RejectionResponse(rejection_message)

        # Question is legal - proceed with RAG
        logger.info(f"Legal question accepted: {question[:50]}...")
        st.session_state['legal_questions_count'] += 1

        # STEP 2: Retrieve relevant documents (normal RAG flow)
        history = build_history()
        docs = retriever.invoke(question)
        context = combine_docs(docs)
        print(context)  # For debugging

        # Format messages with prompt template
        messages = prompt_template.format_messages(question=question, context=context, history=history)

        # Invoke LLM
        response = llm.invoke(messages)
        logger.info("RAG chain executed successfully")
        return response

    except AllProvidersFailedError as e:
        # All configured LLM providers failed
        error_msg = "🚫 Wszystkie dostawcy AI są niedostępne."
        logger.error(f"All providers failed in RAG chain: {e}")

        # Get list of available providers to show user which ones were tried
        available = llm.get_available_providers()
        tried_info = f"\n\n🔄 Próbowano: {', '.join(available) if available else 'brak dostępnych providerów'}"

        help_msg = "\n\n💡 Możliwe przyczyny:\n" \
                   "• Rate limit (zbyt wiele zapytań)\n" \
                   "• Brak środków na koncie API\n" \
                   "• Problemy z połączeniem internetowym\n" \
                   "• Nieprawidłowe klucze API\n\n" \
                   "Sprawdź logi aplikacji po więcej szczegółów."

        class ErrorResponse:
            def __init__(self, content):
                self.content = content
        return ErrorResponse(f"❌ {error_msg}{tried_info}{help_msg}")

    except LLMProviderError as e:
        # Single provider error (shouldn't happen with LLMManager, but included for safety)
        error_msg = f"Błąd dostawcy AI: {e}"
        logger.error(f"LLM provider error in RAG chain: {e}")

        class ErrorResponse:
            def __init__(self, content):
                self.content = content
        return ErrorResponse(f"❌ {error_msg}")

    # Legacy OpenAI error handling (kept for direct OpenAI usage without LLMManager)
    except openai.RateLimitError as e:
        error_msg = "Przekroczono limit zapytań do API OpenAI."
        logger.error(f"OpenAI rate limit: {e}")

        # Check if fallback is available
        available = llm.get_available_providers()
        if len(available) > 1:
            error_msg += f"\n\n🔄 Dostępne alternatywy: {', '.join(available)}"

        class ErrorResponse:
            def __init__(self, content):
                self.content = content
        return ErrorResponse(f"❌ {error_msg}\n\nSpróbuj ponownie za chwilę.")

    except openai.APIError as e:
        error_msg = f"Błąd API OpenAI: {e}"
        logger.error(f"OpenAI API error: {e}")

        class ErrorResponse:
            def __init__(self, content):
                self.content = content
        return ErrorResponse(f"❌ {error_msg}")

    except openai.Timeout as e:
        error_msg = "Przekroczono czas oczekiwania na odpowiedź od OpenAI."
        logger.error(f"OpenAI timeout: {e}")

        class ErrorResponse:
            def __init__(self, content):
                self.content = content
        return ErrorResponse(f"❌ {error_msg}\n\nSpróbuj ponownie.")

    except (ConnectionError, QdrantConnectionError) as e:
        error_msg = f"Błąd połączenia z bazą danych: {e}"
        logger.error(f"Connection error in RAG chain: {e}")
        class ErrorResponse:
            def __init__(self, content):
                self.content = content
        return ErrorResponse(f"❌ {error_msg}")

    except ValueError as e:
        error_msg = f"Błąd przetwarzania zapytania: {e}"
        logger.error(f"ValueError in RAG chain: {e}")
        class ErrorResponse:
            def __init__(self, content):
                self.content = content
        return ErrorResponse(f"❌ {error_msg}")

    except Exception as e:
        error_msg = f"Nieoczekiwany błąd: {type(e).__name__}: {e}"
        logger.error(f"Unexpected error in RAG chain: {type(e).__name__}: {e}")
        class ErrorResponse:
            def __init__(self, content):
                self.content = content
        return ErrorResponse(f"❌ {error_msg}")


# INIT SESSION HISTORY
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# DISPLAY HISTORY
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# USER INPUT
user_input = st.chat_input("Wpisz swoje pytanie prawne tutaj:")

if user_input:
    # Show user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Get model response
    try:
        response = rag_chain_fn(user_input)

        # Save + display
        st.session_state["messages"].append({"role": "assistant", "content": response.content})
        with st.chat_message("assistant"):
            st.write(response.content)

    except Exception as e:
        # This is a fallback in case rag_chain_fn doesn't catch an error
        error_msg = f"❌ Wystąpił błąd podczas przetwarzania Twojego pytania: {type(e).__name__}: {e}"
        logger.error(f"Uncaught error in chat loop: {type(e).__name__}: {e}")
        st.session_state["messages"].append({"role": "assistant", "content": error_msg})
        with st.chat_message("assistant"):
            st.error(error_msg)

# Download conversation button
if len(st.session_state["messages"]) > 1:
    try:
        conversation_text = transform_to_conversation_text()
        st.download_button(
            label='Pobierz rozmowę',
            data=conversation_text,
            file_name=f"zapis_rozmowy_{st.session_state['conversation_data']}.txt"
        )
    except (AttributeError, TypeError, KeyError) as e:
        logger.warning(f"Error creating download button: {type(e).__name__}: {e}")
        st.warning("⚠️ Nie można przygotować pliku do pobrania. Spróbuj ponownie.")
    except Exception as e:
        logger.error(f"Unexpected error in download button: {type(e).__name__}: {e}")
        st.warning("⚠️ Wystąpił błąd podczas przygotowywania pliku do pobrania.")

