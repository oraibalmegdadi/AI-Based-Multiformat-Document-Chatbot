import streamlit as st
from htmlTemplates import css, bot_template, user_template
import os
from shutil import copytree
import sys
from langchain_community.vectorstores import FAISS
from ChatbotFunctions import (
    get_document_text,
    get_text_chunks,
    get_vectorstore,
    get_conversation_chain,
    save_state,
    load_state,
    check_service_availability
)

def handle_basic_questions(user_question):
    """
    Handles predefined basic questions and provides predefined responses.

    Parameters:
    user_question: The question input by the user.

    Returns:
    str: Response to the basic question, or None if it's not a basic question.
    """
    basic_responses = {
        "how are you": "I'm just a chatbot, but I'm here to help you with your documents!",
        "how is your health": "I'm just a bunch of code, so I don't have health, but I'm always ready to assist you!",
        "what can you do": "I can help you understand and answer questions about the documents you upload.",
        "how can you assist me": "Upload your PDFs or text files, and I'll assist you by answering any questions you have about the content.",
        "explain me what you can do": "I analyze the documents you provide and generate answers to your questions based on their content.",
        "who created you": "I was created by a team of developers to assist with document-based queries.",
        "what is your name": "I'm a chatbot designed to help with your documents. You can call me DocBot.",
        "what languages do you speak": "I primarily understand and respond in English.",
        "how do you work": "I analyze uploaded documents and generate responses to questions based on their content.",
        "can you help me with something else": "My main function is to assist with questions about documents you upload.",
        "what is the weather today": "I'm focused on document-related tasks, but you can check the weather using a weather app or website.",
        "can you learn": "I don't learn from interactions, but I can process and respond to questions based on the documents provided.",
        "what technology do you use": "I utilize natural language processing and machine learning technologies to analyze documents and generate responses.",
        "are you human": "No, I'm a chatbot, a software application designed to simulate human conversation.",
        "can you read books": "Yes, I can process and answer questions about the text within books if they are uploaded as PDFs or text files."
    }

    question_lower = user_question.lower()
    for key_phrase, response in basic_responses.items():
        if key_phrase in question_lower:
            return response
    return None

def handle_userinput():
    user_question = st.session_state.user_input
    basic_response = handle_basic_questions(user_question)
    if basic_response:
        st.session_state.chat_display.append(user_template.replace("{{MSG}}", user_question))
        st.session_state.chat_display.append(bot_template.replace("{{MSG}}", basic_response))
    else:
        if st.session_state.conversation is None:
            st.warning("This is a ChatBot designed to answer questions from a set of documents. Please upload documents or load a conversation with previous documents.")
            return

        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.session_state.chat_display.append(user_template.replace("{{MSG}}", message.content))
            else:
                st.session_state.chat_display.append(bot_template.replace("{{MSG}}", message.content))

    st.session_state.user_input = ''  # Clear the input field after submission

def add_custom_static_path():
    static_path = os.path.join(st.__path__[0], 'static')
    custom_static_dir = 'avatars'
    target_dir = os.path.join(static_path, 'avatars')

    if not os.path.exists(target_dir):
        copytree(custom_static_dir, target_dir)

add_custom_static_path()

def main():
    st.set_page_config(page_title="Chat with multiple PDFs and TXTs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "chat_display" not in st.session_state:
        st.session_state.chat_display = []

    if "user_input" not in st.session_state:
        st.session_state.user_input = ''

    st.header("Chat with Multiple PDFs and TXTs Locally :books:")

    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_display:
            st.write(message, unsafe_allow_html=True)

    # User input at the bottom
    st.text_input("Ask a question about your documents:", key="user_input", on_change=handle_userinput)

    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader("Upload your PDFs and TXTs here and click on 'Process'", accept_multiple_files=True, type=['pdf', 'txt'])
        
        # Preserve the chat history and other states on file upload
        if docs:
            st.session_state.new_docs = docs

        if st.button("Process"):
            with st.spinner("Processing"):
                if "new_docs" in st.session_state:
                    doc_paths = []
                    for doc in st.session_state.new_docs:
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

                st.session_state.chat_display = []
                for i, message in enumerate(st.session_state.chat_history):
                    if i % 2 == 0:
                        st.session_state.chat_display.append(user_template.replace("{{MSG}}", message.content))
                    else:
                        st.session_state.chat_display.append(bot_template.replace("{{MSG}}", message.content))

                st.success("State loaded successfully.")
                st.experimental_rerun()

if __name__ == '__main__':
    main()
