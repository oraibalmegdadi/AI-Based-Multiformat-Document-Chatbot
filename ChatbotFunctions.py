import os  # Library for interacting with the operating system
import pickle  # Library for serializing and deserializing Python objects
import json  # Library for handling JSON data
import gzip  # Library for working with gzip-compressed files
import requests  # Library for making HTTP requests
from requests.exceptions import ConnectionError  # Exception handling for HTTP request errors
from dotenv import load_dotenv  # Library for loading environment variables from a .env file
from PyPDF2 import PdfReader  # Library for reading PDF files
from langchain.text_splitter import CharacterTextSplitter  # Library for splitting text into characters
from langchain_community.vectorstores import FAISS  # FAISS library for vector storage and similarity search
from langchain_community.embeddings import OllamaEmbeddings  # Embeddings related to Ollama model
from langchain_community.llms import Ollama  # Ollama Language Model System (LLMS) library
from langchain.memory import ConversationBufferMemory  # Library for conversation buffer memory
from langchain.chains import ConversationalRetrievalChain  # Library for building conversational retrieval chains
from langchain.document_loaders import WebBaseLoader  # Library for loading documents from the web
from langchain_core.runnables import RunnablePassthrough  # RunnablePassthrough from Langchain core
from langchain_core.output_parsers import StrOutputParser  # Output parser for string output
from langchain_core.prompts import ChatPromptTemplate  # Chat prompt template from Langchain core


# Configuration
#model_name = "qwen:0.5b"
model_name = "phi3:mini"
temp_state_file = "temp_session.pkl"

def check_service_availability(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            #print("Service is available.")
            return True
        else:
            print(f"Service returned status code {response.status_code}.")
            return False
    except ConnectionError:
        print("Service is not available. Please start the service and try again.")
        return False
    """
def get_document_text(docs, existing_text=""):
    text = existing_text
    processed_docs = []
    for doc in docs:
        try:
            if doc.endswith(".pdf"):
                with open(doc, "rb") as f:
                    pdf_reader = PdfReader(f)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    processed_docs.append(doc)
            elif doc.endswith(".txt"):
                with open(doc, "r", encoding="utf-8") as f:
                    text += f.read()
                    processed_docs.append(doc)
            else:
                print(f"Unsupported file type: {doc}")
        except FileNotFoundError:
            print(f"Cannot process {doc}: File not found.")
        except Exception as e:
            print(f"Error processing {doc}: {e}")
    return text, processed_docs
    """

def get_document_text(docs: list, existing_text="") -> str:
    """
    Extracts text from a list of document file paths and appends it to existing_text.

    Parameters:
    docs: List of document file paths.
    existing_text: Text to append the extracted text to.

    Returns:
    text: Extracted text from all documents combined.
    processed_docs: List of successfully processed document file paths.
    """
    text = existing_text
    processed_docs = []
    for doc in docs:
        try:
            if doc.endswith(".pdf"):
                with open(doc, "rb") as f:
                    pdf_reader = PdfReader(f)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text
                    processed_docs.append(doc)
            elif doc.endswith(".txt"):
                with open(doc, "r", encoding="utf-8") as f:
                    text += f.read()
                    processed_docs.append(doc)
            else:
                print(f"Unsupported file type: {doc}")
        except FileNotFoundError:
            print(f"Cannot process {doc}: File not found.")
        except Exception as e:
            print(f"Error processing {doc}: {e}")
    return text, processed_docs


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    with open('textchunks.txt', 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            f.write(f'chunk_{i}: {chunk}\n')
    
    return chunks

def get_vectorstore(text_chunks):
    """
    Converts text chunks into vector embeddings and stores them in a FAISS vector store.

    Parameters:
    text_chunks: List of text chunks to be converted to vectors.

    Returns:
    vectorstore: FAISS vector store containing the text embeddings.
    """
    embed = OllamaEmbeddings(model=model_name)
    if not text_chunks:
        raise ValueError("No text chunks to create embeddings.")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embed)
    
    with open('embeddings.txt', 'w') as f:
        for i in range(vectorstore.index.ntotal):
            embedding = vectorstore.index.reconstruct(i)
            emb_str = ','.join(map(str, embedding))
            f.write(f'embedding_{i}: {emb_str}\n')
    
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = Ollama(model=model_name)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(conversation, user_question):
    response = conversation.invoke({'question': user_question})
    chat_history = response['chat_history']
    return chat_history[-1].content if chat_history else "I couldn't find an answer to your question."

def save_state(conversation, chat_history, raw_text, filenames, filename):
    with open(filename, 'wb') as f:
        pickle.dump({
            'conversation': conversation,
            'chat_history': chat_history,
            'raw_text': raw_text,
            'filenames': filenames
        }, f)
   # print("State saved successfully.")

def load_state(filename, question_mode=False, loading_session=False):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            state = pickle.load(f)
            return state
    else:
        print("No saved state found.")
        return None

def process_documents(docs, existing_text=""):
    raw_text, processed_docs = get_document_text(docs, existing_text)
    text_chunks = get_text_chunks(raw_text)
    if not text_chunks:
        raise ValueError("No valid text extracted from the documents.")
    vectorstore = get_vectorstore(text_chunks)
    return get_conversation_chain(vectorstore), raw_text, processed_docs
