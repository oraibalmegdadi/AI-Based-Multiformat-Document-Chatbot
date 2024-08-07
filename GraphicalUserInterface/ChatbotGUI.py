import streamlit as st  # Streamlit library for building interactive web apps
from htmlTemplates import css, bot_template, user_template  # HTML template libraries for formatting
import os  # Library for interacting with the operating system
from shutil import copytree  # Library for recursively copying a directory and its contents
import sys  # Library for system-specific parameters and functions
from langchain.vectorstores import FAISS  # FAISS library for vector storage and similarity search

# Add the path to the parent directory to access ChatbotFunctions.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ChatbotFunctions import (
    get_document_text,
    get_text_chunks,
    get_vectorstore,
    get_conversation_chain,
    save_state,
    load_state,
   # clear_conversation,
    check_service_availability
)

def handle_userinput(user_question):
    """
    Handles the user's input question, generates a response using the conversation chain,
    and updates the chat history and display.

    Parameters:
    user_question: The question input by the user.
    """
    if st.session_state.conversation is None:
        st.warning("This is a ChatBot designed to answer questions from a set of documents. Please upload documents or load a conversation with previous documents.")
        return

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def add_custom_static_path():
    """
    Adds a custom static path for serving images within the Streamlit app.

    This function copies the custom avatars directory to Streamlit's internal static file directory
    to ensure that the images are served correctly.
    """
    # Streamlit's static directory path
    static_path = os.path.join(st.__path__[0], 'static')

    # Custom avatars directory
    custom_static_dir = 'avatars'

    # Target avatars directory within Streamlit static directory
    target_dir = os.path.join(static_path, 'avatars')

    # Copy the custom avatars directory to the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        copytree(custom_static_dir, target_dir)

add_custom_static_path()

def main():
    """
    Main function to set up the Streamlit interface and handle user interactions.
    """
    st.set_page_config(page_title="Chat with multiple PDFs and TXTs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "chat_display" not in st.session_state:
        st.session_state.chat_display = []

    st.header("Chat with Multiple PDFs and TXTs Locally :books:")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    # Display the chat history in the main area
    for message in st.session_state.chat_display:
        st.write(message, unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader("Upload your PDFs and TXTs here and click on 'Process'", accept_multiple_files=True, type=['pdf', 'txt'])
        if st.button("Process"):
            with st.spinner("Processing"):
                # Convert uploaded files to a list of their paths
                doc_paths = []
                for doc in docs:
                    doc_path = f"temp_{doc.name}"
                    with open(doc_path, "wb") as f:
                        f.write(doc.getbuffer())
                    doc_paths.append(doc_path)
                    
                raw_text, processed_docs = get_document_text(doc_paths)
                if isinstance(raw_text, str) and raw_text.strip():
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                else:
                    st.error("No valid text extracted from the documents.")

        save_filename = st.text_input("Save filename", value="conversation_1.pkl")
        save_button_disabled = st.session_state.conversation is None or not st.session_state.chat_history

        if st.button("Save", disabled=save_button_disabled, key="save_button"):
            if not save_button_disabled:
                save_state(st.session_state.conversation, st.session_state.chat_history, "", [], save_filename)
                st.success("State saved successfully.")

        load_filename = st.text_input("Load filename", value="conversation_1.pkl")
        if st.button("Load"):
            state = load_state(load_filename)
            if state:
                st.session_state.conversation = state['conversation']
                st.session_state.chat_history = state['chat_history']

                # Populate chat_display directly and trigger update
                st.session_state.chat_display = []
                for i, message in enumerate(st.session_state.chat_history):
                    if i % 2 == 0:
                        st.session_state.chat_display.append(user_template.replace("{{MSG}}", message.content))
                    else:
                        st.session_state.chat_display.append(bot_template.replace("{{MSG}}", message.content))

                st.success("State loaded successfully.")
                st.experimental_rerun()
            else:
                st.error("No saved state found.")

#        if st.button("Clear"):
 #           clear_conversation()
  #          st.experimental_rerun()

if __name__ == '__main__':
    main()
