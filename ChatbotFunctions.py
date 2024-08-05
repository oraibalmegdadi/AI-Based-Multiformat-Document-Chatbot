import os
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import requests
from requests.exceptions import ConnectionError

# Configuration
model_name = "qwen:0.5b"
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
