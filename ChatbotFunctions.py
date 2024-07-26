import pickle
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Configuration
model_name = "qwen:0.5b"  # Model name for embeddings and LLM
temp_state_file = "temp_session.pkl"  # Temporary state file for saving sessions

def get_document_text(docs, existing_text=""):
    """
    Reads and extracts text from a list of documents (PDFs and TXTs).
    
    Args:
        docs (list): List of document filenames.
        existing_text (str): Existing text to append new text to.
    
    Returns:
        str: Combined text from all documents.
    """
    text = existing_text
    for doc in docs:
        try:
            if doc.endswith(".pdf"):
                with open(doc, "rb") as f:
                    pdf_reader = PdfReader(f)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
            elif doc.endswith(".txt"):
                with open(doc, "r", encoding="utf-8") as f:
                    text += f.read()
            else:
                print(f"Unsupported file type: {doc}")
        except Exception as e:
            print(f"Error processing {doc}: {e}")
    return text

def get_text_chunks(text):
    """
    Splits the given text into smaller chunks for processing.
    
    Args:
        text (str): Text to be split into chunks.
    
    Returns:
        list: List of text chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Save chunks to a file for reference
    with open('textchunksMistral.txt', 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            f.write(f'chunk_{i}: {chunk}\n')
    
    return chunks

def get_vectorstore(text_chunks):
    """
    Creates a vector store from text chunks using embeddings.
    
    Args:
        text_chunks (list): List of text chunks.
    
    Returns:
        FAISS: Vector store containing the embeddings.
    """
    embed = OllamaEmbeddings(model=model_name)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embed)
    
    # Save embeddings to a file for reference
    with open('embeddingsMistral.txt', 'w') as f:
        for i in range(vectorstore.index.ntotal):
            embedding = vectorstore.index.reconstruct(i)
            emb_str = ','.join(map(str, embedding))
            f.write(f'embedding_{i}: {emb_str}\n')
    
    return vectorstore

def get_conversation_chain(vectorstore):
    """
    Initializes a conversational chain for handling queries.
    
    Args:
        vectorstore (FAISS): Vector store to use for retrieval.
    
    Returns:
        ConversationalRetrievalChain: Initialized conversation chain.
    """
    llm = Ollama(model=model_name)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(conversation, user_question):
    """
    Processes a user's question using the conversation chain.
    
    Args:
        conversation (ConversationalRetrievalChain): Conversation chain.
        user_question (str): User's question.
    """
    response = conversation.invoke({'question': user_question})
    chat_history = response['chat_history']
    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            print(f"User: {message.content}")
        else:
            print(f"Bot: {message.content}")

def save_state(conversation, chat_history, raw_text, filenames, filename):
    """
    Saves the current state to a file.
    
    Args:
        conversation (ConversationalRetrievalChain): Conversation chain.
        chat_history (list): List of chat history messages.
        raw_text (str): Combined raw text of documents.
        filenames (list): List of document filenames.
        filename (str): Filename to save the state to.
    """
    with open(filename, 'wb') as f:
        pickle.dump({
            'conversation': conversation,
            'chat_history': chat_history,
            'raw_text': raw_text,
            'filenames': filenames
        }, f)
    print("State saved successfully.")

def load_state(filename, question_mode=False, loading_session=False):
    """
    Loads a saved state from a file.
    
    Args:
        filename (str): Filename to load the state from.
        question_mode (bool): Whether in question mode.
        loading_session (bool): Whether loading a full session.
    
    Returns:
        dict: Loaded state data or None if not found.
    """
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            state = pickle.load(f)
            if not question_mode:
                if loading_session:
                    print(f"Pickle file '{filename}' loaded successfully and contains analysis of the following documents:")
                else:
                    print(f"New documents added successfully. Current session contains analysis of the following documents:")
                filenames = state.get('filenames', [])
                if not filenames:
                    print("- (no filenames available)")
                else:
                    for doc in filenames:
                        print(f"- {doc}")
                print("Please start asking questions to get answers based on these documents. Remember, you can also add more documents to this session.")
            return state
    else:
        print("No saved state found.")
        return None

def process_documents(docs, existing_text=""):
    """
    Processes documents and returns a conversation chain, raw text, and filenames.
    
    Args:
        docs (list): List of document filenames.
        existing_text (str): Existing text to append new text to.
    
    Returns:
        tuple: Conversation chain, raw text, and filenames.
    """
    raw_text = get_document_text(docs, existing_text)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    return get_conversation_chain(vectorstore), raw_text, docs
