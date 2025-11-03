import streamlit as st
import datetime
import llm_model
import vector_database
# import main

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
    human_template="Pytanie użytkownika: {question}\n\nKontekst:\n{context}"
)

retriever = vector_database.create_retriever()

# RAG chain
def combine_docs(docs):
        """Combine documents' page_content into a single string"""
        return "\n\n".join([f"{d.metadata}\n{d.page_content}" for d in docs])


def rag_chain_fn(question: str):
        """RAG: retrieves documents, combines context, and invokes LLM"""
        docs = retriever.invoke(question)  # retrieve relevant documents
        context = combine_docs(docs)       # combine docks page_content's into a string
        print(context)  # For debugging
        messages = prompt_template.format_messages(question=question, context=context)
        return llm.invoke(messages)



# ==============================
# STREAMLIT CHAT UI
# ==============================

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

