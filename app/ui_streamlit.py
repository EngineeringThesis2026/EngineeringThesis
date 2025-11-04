import streamlit as st
import datetime
import llm_model
import vector_database
import process_data

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

if st.session_state['data_imported'] == False:
    st.write("Data has not been imported yet. Importing now...")
    
    data_folder_path = process_data.get_data_folder_path()
    all_data_from_pdfs = process_data.load_data_from_pdf(file_path=data_folder_path)

    # SPLITTING DATA
    all_splits = process_data.split_docks_into_chunks(documents=all_data_from_pdfs)

    # CREATING EMBEDDINGS
    embeddings = process_data.create_embeddings_with_metadata(sentences=all_splits,embedding_model=process_data._embedding_model)

    # VECTOR DATABASE OPERATIONS
    vector_size = len(embeddings[0]['embedding'])

    vector_database.create_collection_if_not_exists(vector_database_client=vector_database._vector_database_client,
                                                    collection_name=vector_database._collection_name, vector_size=vector_size)
    
    points = vector_database.create_points_from_embeddings(embeddings=embeddings)
    vector_database.upload_to_qdrant(vector_database_client=vector_database._vector_database_client,
                                        collection_name=vector_database._collection_name, points=points)

    st.write("Data import completed.")
    st.session_state['data_imported'] = True

# user settings in sidebar
st.sidebar.write(f'Ustawienia')

model_tokens = st.sidebar.slider('Maksymalna ilość tokenów', min_value=500, max_value=2500, value=1500)  # this is a widget

model_temperature = st.sidebar.slider('Kreatywność modelu', max_value=100)

# MAIN LOGIC 

# Creating LLM instance with user-defined settings
llm = llm_model.create_llm(
    temperature=model_temperature / 100,
    max_tokens=model_tokens
)
st.sidebar.write(f"llm temp: {llm.temperature}, llm max tokens: {llm.max_tokens}")  # For debugging

# Creating Chat Prompt Template
prompt_template = llm_model.create_chat_prompt_template(
    system_template="Jesteś pomocnym i profesjonalnym asystentem AI specjalizującym się w doradztwie prawnym. Odpowiadaj wyłącznie na pytania związane z prawem, dostarczając dokładne i zwięzłe informacje.",
    human_template="Historia rozmowy: {history}\nPytanie użytkownika: {question}\n\nKontekst:\n{context}"
)

retriever = vector_database.create_retriever()

# RAG chain
def combine_docs(docs):
        """Combine documents' page_content into a single string"""
        return "\n\n".join([f"{d.metadata}\n{d.page_content}" for d in docs])


def rag_chain_fn(question: str, retriever=retriever, llm=llm, prompt_template=prompt_template):
        """RAG: retrieves documents, combines context, and invokes LLM"""
        history = build_history()
        docs = retriever.invoke(question)  # retrieve relevant documents
        context = combine_docs(docs)       # combine docks page_content's into a string
        print(context)  # For debugging
        messages = prompt_template.format_messages(question=question, context=context, history=history)
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

if user_input:
    # Show user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Get model response
    response = rag_chain_fn(user_input)

    # Save + display
    st.session_state["messages"].append({"role": "assistant", "content": response.content})
    with st.chat_message("assistant"):
        st.write(response.content)

# Download conversation button
if len(st.session_state["messages"]) > 1:
    st.download_button(label='Pobierz rozmowę',data=transform_to_conversation_text(),
                    file_name=f"zapis_rozmowy_{st.session_state['conversation_data']}.txt")

