import streamlit as st
import datetime
import llm_model
import vector_database
import process_data
import prompt_texts
import upload_pdf_file


collections_names_dict = {
  "rulings_collection_name": "rulings",
  "civil_code_collection_name": "civil_code",
  "labor_code_collection_name": "labor_code"
}

def build_history():
    return "\n".join(
        f"{m['role']}: {m['content']}" for m in st.session_state["messages"]
    )


def transform_to_conversation_text() -> str:
    """Loop through the history of all messages in a session and save all the contents to a string variable, then return it."""
    conversation_text = "\n\n".join(
        [
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in st.session_state.messages
            if msg["role"] != "system"
        ]
    )
    return conversation_text


# Setting up the Streamlit page configuration
st.set_page_config(
    page_title="Interaktywny system doradczy do konsultacji prawnych oparty na sztucznej inteligencji"
)

st.title("Konsultacje prawne AI.")

# Checking the time when the conversation starts
if "conversation_data" not in st.session_state:
    st.session_state["conversation_data"] = datetime.datetime.now().strftime(
        "%Y-%m-%d-%H:%M"
    )

if "data_imported" not in st.session_state:
    st.session_state["data_imported"] = False

if "user_input_openai_api_key" not in st.session_state:
    st.session_state["user_input_openai_api_key"] = None

if "user_uploaded_pdf_text" not in st.session_state:
    st.session_state["user_uploaded_pdf_text"] = None

if st.session_state["data_imported"] == False:
    with st.spinner("Importowanie danych... Proszę czekać."):
        # sleep(5)  # Simulating a delay for data import
        for collection_name in collections_names_dict.values():
            data_folder_path = process_data.get_data_folder_path(collection_name)

            all_data_from_pdfs = process_data.load_data_from_pdf(file_path=data_folder_path)
            if all_data_from_pdfs is None:
                st.error('Brak plików PDF w folderze danych. Proszę dodać pliki i ponowić próbę.')
                st.error(f'Przejdź do folderu: {data_folder_path} i dodaj materiały prawne w postaci plików PDF.')
                st.stop()
            
            # SPLITTING DATA
            all_splits = process_data.split_docks_into_chunks(documents=all_data_from_pdfs)

            # CREATING EMBEDDINGS
            embeddings = process_data.create_embeddings_with_metadata(sentences=all_splits,embedding_model=process_data._embedding_model)

            # VECTOR DATABASE OPERATIONS
            vector_size = len(embeddings[0]['embedding'])

            vector_database.create_collection_if_not_exists(vector_database_client=vector_database._vector_database_client,
                                                            collection_name=collection_name, vector_size=vector_size)
            
            points = vector_database.create_points_from_embeddings(embeddings=embeddings)
            vector_database.upload_to_qdrant(vector_database_client=vector_database._vector_database_client,
                                                collection_name=collection_name, points=points)
        st.sidebar.success('Dane zostały pomyślnie zaimportowane i przetworzone.')
        st.session_state['data_imported'] = True

# user settings in sidebar
st.sidebar.title(f"Ustawienia")

model_tokens = st.sidebar.slider(
    "Maksymalna ilość tokenów", min_value=500, max_value=2500, value=1500
)  # this is a widget

model_temperature = st.sidebar.slider("Kreatywność modelu", max_value=100)

# User st.radio for choosing data collections for RAG
chosen_collection_r_button = st.sidebar.radio(
    "Wybierz skąd chcesz zaczerpnąc danych do odpowiedzi:",
    ["Kodeks cywilny", "Kodeks pracy", "Automatyczny wybór"],
    captions=[
        "Najlepszy do spraw cywilnych.",
        "Najlepszy do spraw zawodowych.",
        "Pozwól modelowi zdecydować samodzielnie.",
    ],index=2
)

# For debugging 
# if chosen_collection_r_button == "Kodeks cywilny":
#     st.sidebar.write("Wybrałeś Kodeks cywilny.")
# elif chosen_collection_r_button == "Kodeks pracy":
#     st.sidebar.write("Wybrałeś Kodeks pracy.")
# else:
#     st.sidebar.write("Wybrałeś Automatyczny wybór kolekcji przez model.")

# MAIN LOGIC 
# Creating LLM instance with user-defined settings
api_key_available = st.session_state['user_input_openai_api_key'] or st.secrets.get('OPENAI_API_KEY')

if not api_key_available:
    st.error(
        "Brak klucza API OpenAI. Proszę dodać OPENAI_API_KEY do sekretów aplikacji lub wprowadzić tymczasowo poniżej."
    )
    user_input_key = st.text_input("Wpisz swój klucz API OpenAI:", type="password")
    if user_input_key:
        st.session_state["user_input_openai_api_key"] = user_input_key
        api_key_available = user_input_key  # aktualizacja dostępnego klucza

if api_key_available:
    # Create LLM instance for question classification
    llm_question_classifier = llm_model.create_llm(
        api_key=api_key_available,
    )
    # Create main LLM instance
    llm = llm_model.create_llm(
        api_key=api_key_available,
        temperature=model_temperature / 100,
        max_tokens=model_tokens,
    )
    # Create LLM instance for choosing collection
    llm_collection_selector = llm_model.create_llm(
        api_key=api_key_available,
    )
    st.sidebar.success('Połączenie z modelem LLM.')

if st.session_state["data_imported"]:
    st.sidebar.success("Dostęp do danych prawnych jest gotowy.")

# st.sidebar.write(f"llm temp: {llm.temperature}, llm max tokens: {llm.max_tokens}")  # For debugging

# Chat Prompt Template for question classification
question_classification_prompt = llm_model.create_chat_prompt_template(
    system_template=prompt_texts.text_for_system_template_question_classification_prompt,
    human_template="Historia konwersacji: {history}\nPytanie użytkownika: {question}\n Kontekst dodany przez użytkownika: {user_uploaded_pdf_text}",
)

# Creating Chat Prompt Template for main LLM functionality with RAG
prompt_template = llm_model.create_chat_prompt_template(
    system_template="Jesteś pomocnym i profesjonalnym asystentem AI specjalizującym się w doradztwie prawnym. Odpowiadaj wyłącznie na pytania związane z prawem, dostarczając dokładne i zwięzłe informacje. Na wiadomość uytkownika z podziękowaniem odpowiadaj miło wyrażając chęć dalszej pomocy.",
    human_template="Historia konwersacji: {history}\nPytanie użytkownika: {question}\n\nKontekst z bazy danych:\n{context}\n Kontekst dodany przez użytkownika: {user_uploaded_pdf_text}"
)

# Creating Chat Prompt Template for collection selection
collection_selection_prompt = llm_model.create_chat_prompt_template(
    system_template=prompt_texts.text_for_system_template_collection_selector,
    human_template="Historia konwersacji: {history}\nPytanie użytkownika: {question}\n Kontekst dodany przez użytkownika: {user_uploaded_pdf_text}"
)

# Creating retrievers for each collection
rulings_retriever = vector_database.create_retriever(collections_names_dict["rulings_collection_name"])
civil_code_retriever = vector_database.create_retriever(collections_names_dict["civil_code_collection_name"])
labor_code_retriever = vector_database.create_retriever(collections_names_dict["labor_code_collection_name"])


# RAG chain
def combine_docs(docs):
    """Combine documents' page_content into a single string"""
    return "\n\n".join([f"{d.metadata}\n{d.page_content}" for d in docs])


def rag_chain_fn(question: str,
                retriever=rulings_retriever,
                llm=llm, 
                prompt_template=prompt_template,
                llm_question_classifier=llm_question_classifier,
                question_classification_prompt=question_classification_prompt,
                ):
        """RAG: retrieves documents, combines context, and invokes LLM"""
        
        # Build conversation history
        conv_history = build_history()

        # Get user-uploaded PDF text if available
        if st.session_state['user_uploaded_pdf_text'] is not None:
            user_uploaded_pdf_text = st.session_state['user_uploaded_pdf_text']
        else:
            user_uploaded_pdf_text = "brak dodatkowego kontekstu od użytkownika"

        # QUESTION CLASSIFICATION: if the question is legal go to RAG, else answer directly.
        msg_to_classify = question_classification_prompt.format_messages(question=question, history=conv_history, user_uploaded_pdf_text=user_uploaded_pdf_text)
        classification_response = llm_question_classifier.invoke(msg_to_classify)
        classification_answer = classification_response.content.strip().upper()
        print(f"Classification answer: {classification_answer}")  # For debugging
        if classification_answer == "NIE":
            messages = prompt_template.format_messages(question=question, context='brak kontekstu', history=conv_history, user_uploaded_pdf_text=user_uploaded_pdf_text)
            st.write(messages)  # For debugging
            return llm.invoke(messages)
        else:
            # Check witch data collection to use based on user choice
            if chosen_collection_r_button == "Kodeks cywilny":
                retriever = civil_code_retriever

            elif chosen_collection_r_button == "Kodeks pracy":
                retriever = labor_code_retriever

            elif chosen_collection_r_button == "Automatyczny wybór":
                # Ask the llm model to choose the collection
                msg_to_classify_collection = collection_selection_prompt.format_messages(question=question, history=conv_history, user_uploaded_pdf_text=user_uploaded_pdf_text)
                collection_selection_response = llm_collection_selector.invoke(msg_to_classify_collection)
                collection_selection_answer = collection_selection_response.content.strip().upper()
                print(f"=========Collection selection answer=========:\n{collection_selection_answer}")  # For debugging
                st.sidebar.write(f"Wybór kolekcji przez model: {collection_selection_answer}")

                if collection_selection_answer == "KODEKS_CYWILNY":
                    retriever = civil_code_retriever
                    # Adding bonus informations from vector db rulings for civil code cases.
                    rulings_docs = rulings_retriever.invoke(question)
                    ruling_context = combine_docs(rulings_docs)
                    # informations from civil_code vector db
                    docs = retriever.invoke(question)
                    civil_code_context = combine_docs(docs)
                    context = civil_code_context + " " + ruling_context
                    messages = prompt_template.format_messages(question=question,
                                                                context=context,
                                                                history=conv_history,
                                                                user_uploaded_pdf_text=user_uploaded_pdf_text)
                    st.write(messages)  # For debugging
                    return llm.invoke(messages)
                elif collection_selection_answer == "KODEKS_PRACY":
                    retriever = labor_code_retriever
                    docs = retriever.invoke(question)  # retrieve relevant documents
                    context = combine_docs(docs)
                    print('=='*20)
                    print(context)       # combine docks page_content's into a string
                    messages = prompt_template.format_messages(question=question,
                                                                context=context,
                                                                history=conv_history,
                                                                user_uploaded_pdf_text=user_uploaded_pdf_text)
                    st.write(messages)  # For debugging
                    return llm.invoke(messages)

                else:
                    st.write("""Model wykrył ze tematem rozmowy jest inna kategoria prawa niz prowao cywilne lub pracy.
                            W tym wypadku model dokonuje odpowiedzi bez dodatkowej bazy wiedzy.""")    # TODO: CHANGE NEDED -> 
                    messages = prompt_template.format_messages(question=question,
                                                                context="brak kontekstu",
                                                                history=conv_history,
                                                                user_uploaded_pdf_text=user_uploaded_pdf_text)
                    st.write(messages)  # For debugging
                    return llm.invoke(messages)

            docs = retriever.invoke(question)  # retrieve relevant documents
            print(docs)
            context = combine_docs(docs)
            print('=='*20)
            print(context)       # combine docks page_content's into a string
            # print(context)  # For debugging
            messages = prompt_template.format_messages(question=question,
                                                        context=context,
                                                        history=conv_history,
                                                        user_uploaded_pdf_text=user_uploaded_pdf_text)
            st.write(messages)  # For debugging
            return llm.invoke(messages)


# INIT SESSION HISTORY
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# DISPLAY HISTORY
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# USER INPUT
user_input = st.chat_input("Wpisz swoje pytanie prawne tutaj:")


uploaded_file = st.file_uploader("Załącz plik PDF", type="pdf")

# PDF UPLOAD HANDLING
if uploaded_file is not None:
    extracted_text_from_pdf = upload_pdf_file.extract_text_from_pdf(uploaded_file)
    st.session_state["user_uploaded_pdf_text"] = extracted_text_from_pdf
    st.sidebar.success("Dołączono dodatkowy kontekst z pliku PDF.")


if user_input:
    # Show user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("Generowanie odpowiedzi..."):
        # Get model response
        response = rag_chain_fn(user_input)

        # Save + display
        st.session_state["messages"].append(
            {"role": "assistant", "content": response.content}
        )
        with st.chat_message("assistant"):
            st.write(response.content)


print("END OF THE ITERATION")
uploaded_file = None
extracted_text_from_pdf = None
st.session_state["user_uploaded_pdf_text"] = None

# Download conversation button
if len(st.session_state["messages"]) > 1:
    st.download_button(
        label="Pobierz rozmowę",
        data=transform_to_conversation_text(),
        file_name=f"zapis_rozmowy_{st.session_state['conversation_data']}.txt",
    )
