import streamlit as st  # Streamlit library for building interactive web apps
from langchain_community.document_loaders import WebBaseLoader  # Library for loading documents from the web
from langchain_community import embeddings  # Library for handling embeddings
from langchain_community.llms import Ollama  # Ollama Language Model System (LLMS) library
from langchain_community.embeddings import OllamaEmbeddings  # Embeddings related to Ollama model
from langchain_core.runnables import RunnablePassthrough  # RunnablePassthrough from Langchain core
from langchain_core.output_parsers import StrOutputParser  # Output parser for string output
from langchain_core.prompts import ChatPromptTemplate  # Chat prompt template from Langchain core

from dotenv import load_dotenv  # Library for loading environment variables from a .env file
from PyPDF2 import PdfReader  # Library for reading PDF files
from langchain.text_splitter import CharacterTextSplitter  # Library for splitting text into characters
from langchain.vectorstores import FAISS  # FAISS library for vector storage and similarity search
import InstructorEmbedding  # Custom instructor embedding module
from langchain.memory import ConversationBufferMemory  # Library for conversation buffer memory
from langchain.chains import ConversationalRetrievalChain  # Library for building conversational retrieval chains
from htmlTemplates import css, bot_template, user_template  # HTML template libraries for formatting
import json  # Library for handling JSON data
import requests  # Library for making HTTP requests
import gzip  # Library for working with gzip-compressed files
import pickle  # Library for serializing and deserializing Python objects
import os  # Library for interacting with the operating system


def get_document_text(docs):
    """
    Extracts text from a list of uploaded documents.
    Supports PDF and plain text files.

    Parameters:
    docs: List of uploaded documents.

    Returns:
    text: Extracted text from all documents combined.
    """
    text = ""
    for doc in docs:
        try:
            if doc.type == "application/pdf":
                pdf_reader = PdfReader(doc)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            elif doc.type == "text/plain":
                text += doc.read().decode("utf-8")
            else:
                print(f"Unsupported file type: {doc.type}")
        except Exception as e:
            print(f"Error processing {doc.name}: {e}")
    return text


def get_text_chunks(text):
    """
    Splits the extracted text into smaller chunks.

    Parameters:
    text: The text to be split into chunks.

    Returns:
    chunks: List of text chunks, each chunk is 1000 characters with 200 characters overlap.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,  # Specifies the maximum size of each chunk, set to 1000 characters.
        chunk_overlap=200, # Determines the number of characters that each chunk overlaps with the next one, set to 200 characters.
        length_function=len    # Specifies the function to calculate the length of the text, here it's the built-in len() function.
    )
    chunks = text_splitter.split_text(text)
    
    with open('textchunksMistral.txt', 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            f.write(f'chunk_{i}: {chunk}\n')
    
    return chunks     # Return the list of text chunks, each chunk is 1000 characters


def get_vectorstore(text_chunks):
    """
    Converts text chunks into vector embeddings and stores them in a FAISS vector store.

    Parameters:
    text_chunks: List of text chunks to be converted to vectors.

    Returns:
    vectorstore: FAISS vector store containing the text embeddings.
    """
    model_name='mistral'
    embed=embeddings.ollama.OllamaEmbeddings(model=model_name)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embed)
    
    with open('embeddingsMistral.txt', 'w') as f:
        for i in range(vectorstore.index.ntotal):
            embedding = vectorstore.index.reconstruct(i)
            emb_str = ','.join(map(str, embedding))
            f.write(f'embedding_{i}: {emb_str}\n')
    
    return vectorstore


def get_conversation_chain(vectorstore):
    """
    Creates a conversational retrieval chain from the given vector store.

    Parameters:
    vectorstore: FAISS vector store containing the text embeddings.

    Returns:
    conversation_chain: Conversational retrieval chain for interacting with the user.
    """
    llm=Ollama(model="mistral")
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


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


def save_state(filename):
    """
    Saves the current conversation state to a file.

    Parameters:
    filename: The name of the file to save the state.
    """
    with open(filename, 'wb') as f:
        pickle.dump({
            'conversation': st.session_state.conversation,
            'chat_history': st.session_state.chat_history
        }, f)
    st.success("State saved successfully.")
    

def load_state(filename):
    """
    Loads a conversation state from a file and updates the session state.

    Parameters:
    filename: The name of the file to load the state from.
    """
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            state = pickle.load(f)
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
        
        # Force rerun the main script to update the main area display
        st.experimental_rerun()
    else:
        st.error("No saved state found.")


def clear_conversation():
    """
    Clears the current conversation, chat history, and chat display.
    """
    st.session_state.conversation = None
    st.session_state.chat_history = []
    st.session_state.chat_display = []
    st.experimental_rerun()


def main():
    """
    Main function to set up the Streamlit interface and handle user interactions.
    """
    st.set_page_config(page_title="Chat with multiple PDFs and TXTs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "chat_display" not in st.session_state:
        st.session_state.chat_display = []

    st.header("Chat with Multiple PDFs and TXTs Locally (Mistral) :books:")

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
                raw_text = get_document_text(docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

        save_filename = st.text_input("Save filename", value="conversation_state1.pkl")
        if st.button("Save"):
            save_state(save_filename)

        load_filename = st.text_input("Load filename", value="conversation_state1.pkl")
        if st.button("Load"):
            load_state(load_filename)

        if st.button("Clear"):
            clear_conversation()

if __name__ == '__main__':
    main()
