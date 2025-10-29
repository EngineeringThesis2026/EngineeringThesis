# Importing necessary libraries
from openai import OpenAI
import streamlit as st
import datetime

def transform_to_conversation_text()->str:
    """Loop through the history of all messages in a session and save all the contents to a string variable, then return it."""
    conversation_text = "\n\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages if msg['role'] != 'system'])
    return conversation_text

OPENAI_MODEL = "gpt-4o-mini"

# USER SETTINGS in sidebar
st.sidebar.write(f'Ustawienia')
# for example we can use slider for max tokens used in prompt or temperature of model
model_tokens = st.sidebar.slider('Maksymalna ilość tokenów', min_value=500, max_value=2500)  # this is a widget
st.sidebar.write(f'Maksymalna ilość tokenów ustawiona na: {model_tokens}') # For debugging

model_temperature = st.sidebar.slider('Kreatywność modelu', max_value=100)
st.sidebar.write(f'Kreatywność modelu ustawiona na: {model_tokens}') # For debugging 

# MAIN LOGIC

# Setting up the Streamlit page configuration
st.set_page_config(page_title="Interaktywny system doradczy do konsultacji prawnych oparty na sztucznej inteligencji")
st.title("Konsultacje prawne AI.")

# Initializing the OpenAI client using the API key from Streamlit's secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Setting up the OpenAI model in session state if it is not already defined
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = OPENAI_MODEL

# Checking the time when the conversation starts
if "conversation_data" not in st.session_state:
    st.session_state['conversation_data'] = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')

# Initializing the 'messages' list 
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system",
                                  "content": "Jesteś asystentem prawnym, który ma na celu pomaganie w zrozumieniu zagadnień prawnych oraz udzielanie informacji na temat przepisów i procedur. Staraj się przekazywać proste w odbiorze informacje. WAZNE: Jezeli pytanie nie jest związane z prawem, odpowiedz ze twoim przeznaczeniem jest odpowiadanie na pytania z zakresu prawa."}]


# Looping through the 'messages' list to display each message except system messages
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
# Input field for the user to send a new message, := checks if input is not empty.
if prompt := st.chat_input("Twoja odpowiedź."):
    # Appending the user's input to the 'messages' list in session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display the user's message in a chat bubble
    with st.chat_message("user"):
        st.markdown(prompt)
   
    # Assistant's response
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True, # This line enables streaming for real-time response
        )
        # Display the assistant's response as it streams
        response = st.write_stream(stream)
     # Append the assistant's full response to the 'messages' list
    st.session_state.messages.append({"role": "assistant", "content": response})

# Download conversation button
st.download_button(label='Pobierz rozmowę',data=transform_to_conversation_text(),
                   file_name=f"zapis_rozmowy_{st.session_state['conversation_data']}.txt")



