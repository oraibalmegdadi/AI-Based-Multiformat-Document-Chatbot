import argparse
import pickle
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

model_name = "mistral"
temp_state_file = "temp_session.pkl"

def get_document_text(docs, existing_text=""):
    """
    Extracts text from a list of uploaded documents.
    Supports PDF and plain text files.

    Parameters:
    docs: List of uploaded documents.
    existing_text: Text from previously uploaded documents.

    Returns:
    text: Extracted text from all documents combined.
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
    Splits the extracted text into smaller chunks.

    Parameters:
    text: The text to be split into chunks.

    Returns:
    chunks: List of text chunks, each chunk is 1000 characters with 200 characters overlap.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
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
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embed)
    return vectorstore

def get_conversation_chain(vectorstore):
    """
    Creates a conversational retrieval chain from the given vector store.

    Parameters:
    vectorstore: FAISS vector store containing the text embeddings.

    Returns:
    conversation_chain: Conversational retrieval chain for interacting with the user.
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
    Handles the user's input question, generates a response using the conversation chain,
    and updates the chat history and display.

    Parameters:
    user_question: The question input by the user.
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
    Saves the current conversation state to a file.

    Parameters:
    conversation: The conversation chain object.
    chat_history: The chat history.
    raw_text: The combined raw text from all documents.
    filenames: List of filenames of processed documents.
    filename: The name of the file to save the state.
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
    Loads a conversation state from a file and returns the state.

    Parameters:
    filename: The name of the file to load the state from.
    question_mode: Boolean flag to determine if the state is being loaded for asking a question.
    loading_session: Boolean flag to differentiate between loading a session and adding new documents.

    Returns:
    state: The loaded state.
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
    Process the uploaded documents to extract text, split into chunks,
    and create a vector store.

    Parameters:
    docs: List of uploaded documents.
    existing_text: Text from previously uploaded documents.

    Returns:
    conversation_chain: Updated conversational retrieval chain.
    raw_text: Combined text of all processed documents.
    filenames: List of filenames of processed documents.
    """
    raw_text = get_document_text(docs, existing_text)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    return get_conversation_chain(vectorstore), raw_text, docs

def main():
    """
    Main function to handle command-line arguments and orchestrate the workflow,
    including loading, saving, processing documents, and handling questions.
    """
    parser = argparse.ArgumentParser(description="Chat with multiple PDFs and TXTs")
    parser.add_argument("--docs", nargs="+", help="List of documents to process")
    parser.add_argument("--add", nargs="+", help="List of documents to add to the existing session")
    parser.add_argument("--question", help="Question to ask about the documents")
    parser.add_argument("--save", help="Filename to save the state")
    parser.add_argument("--load", help="Filename to load the state from")
    parser.add_argument("--clear", help="Clear the current conversation", action='store_true')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Initialize variables to manage the conversation and state
    conversation = None  # Holds the conversation chain object
    chat_history = []    # List to store the chat history
    raw_text = ""        # Stores the raw text extracted from documents
    filenames = []       # List to store the filenames of processed documents
    processed = False    # Flag to indicate if any documents were processed

    # Load a saved session if specified
    if args.load:
        state = load_state(args.load, question_mode=bool(args.question), loading_session=True)
        if state:
            conversation = state['conversation']
            chat_history = state['chat_history']
            raw_text = state['raw_text']
            filenames = state.get('filenames', [])

    # Load the temporary session file if it exists and no specific load file is provided
    if os.path.exists(temp_state_file) and not args.load:
        state = load_state(temp_state_file, question_mode=bool(args.question))
        if state:
            conversation = state['conversation']
            chat_history = state['chat_history']
            raw_text = state['raw_text']
            filenames = state.get('filenames', [])

    # Process new documents if specified
    if args.docs:
        conversation, raw_text, filenames = process_documents(args.docs)
        save_state(conversation, chat_history, raw_text, filenames, temp_state_file)
        processed = True

    # Add documents to the existing session if specified
    if args.add:
        conversation, new_raw_text, new_filenames = process_documents(args.add, raw_text)
        raw_text = new_raw_text
        filenames.extend(new_filenames)
        save_state(conversation, chat_history, raw_text, filenames, temp_state_file)
        processed = True

    # Clear the current conversation if specified
    if args.clear:
        conversation = None
        chat_history = []
        raw_text = ""
        filenames = []
        if os.path.exists(temp_state_file):
            os.remove(temp_state_file)

    # Handle user question if specified
    if args.question:
        if conversation:
            handle_userinput(conversation, args.question)
            # Save the state after handling the question to keep the chat history
            save_state(conversation, chat_history, raw_text, filenames, temp_state_file)
        else:
            print("No conversation loaded or processed. Please load a state or process documents first.")

    # Save the current state if specified
    if args.save:
        if conversation:
            save_state(conversation, chat_history, raw_text, filenames, args.save)
        else:
            print("No conversation to save. Please load a state or process documents first.")

if __name__ == "__main__":
    main()
