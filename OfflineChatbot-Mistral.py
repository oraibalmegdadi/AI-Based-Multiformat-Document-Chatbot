import streamlit as st  # Used for creating interactive web applications for ML and data science.
from langchain_community.document_loaders import WebBaseLoader  # Provides functionality for loading documents from the web.
from langchain_community import embeddings  # Used for embedding text data.
from langchain_community.llms import Ollama  # Implements large language models.
from langchain_community.embeddings import OllamaEmbeddings  # Implements embeddings using Ollama models.
from langchain_core.runnables import RunnablePassthrough  # Provides a runnable for passing input/output.
from langchain_core.output_parsers import StrOutputParser  # Used for parsing string output.
from langchain_core.prompts import ChatPromptTemplate  # Implements chat prompt templates.
from dotenv import load_dotenv  # Required to load environment variables from .env files.
from PyPDF2 import PdfReader  # Used for reading PDF files.
from langchain.text_splitter import CharacterTextSplitter  # Used for splitting text into chunks.
from langchain.vectorstores import FAISS  # Provides functionality for similarity search and clustering of dense vectors.
import InstructorEmbedding  # Used for instructor embeddings.
from langchain.memory import ConversationBufferMemory  # Implements conversation memory functionality.
from langchain.chains import ConversationalRetrievalChain  # Implements conversational retrieval chains.
from htmlTemplates import css, bot_template, user_template  # Provides HTML templates for UI.
import json  # Used for working with JSON data.
import requests  # Used for making HTTP requests.
import gzip  # Used for reading and writing gzip files.



def get_document_text(docs):
    text = ""
    for doc in docs:
        if doc.type == "application/pdf":
            pdf_reader = PdfReader(doc)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif doc.type == "text/plain":
            text += doc.read().decode("utf-8")
        elif doc.type == "application/gzip" or doc.name.endswith(".txt.gz"):
            with gzip.open(doc, 'rt', encoding='utf-8') as f:
                text += f.read()
    return text


def get_text_chunks(text):
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
    
    return chunks     # Return the list of text chunks, each chunk is 1000 char

def get_vectorstore(text_chunks):
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    #embeddings=Ollama(model="mistral")
    #model_name = 'mistralai/Mistral-7B-Instruct-v0.2'
    #embeddings.ollama.pull(model_name)
    
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
    # llm = ChatOpenAI()
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
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
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

            
def main():
    st.set_page_config(page_title="Chat with multiple PDFs, TXTs, and TXT.GZs",page_icon=":books:")
    
    st.write(css, unsafe_allow_html=True)

    # Check if the 'conversation' key is not already in the session state
    if "conversation" not in st.session_state:
        # If not, initialize it to None
        # 'conversation' will store the ongoing conversation between the user and the chatbot
        # It allows maintaining context and continuity in the conversation
        st.session_state.conversation = None

    # Check if the 'chat_history' key is not already in the session state
    if "chat_history" not in st.session_state:
        # If not, initialize it to None
        # 'chat_history' will store the history of the conversation between the user and the chatbot
        # It allows users to review previous interactions or track the progress of the conversation
        st.session_state.chat_history = None
    
    #st.write(css, unsafe_allow_html=True)
    st.header("Chat with Multiple PDFs, TXTs, and TXT.GZs Locally (Mistral) :books:")
    #st.text_input("Ask a question about your documents:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)
  
    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader("Upload your PDFs, TXTs, and TXT.GZs here and click on 'Process'", accept_multiple_files=True, type=['pdf', 'txt', 'gz'])
        if st.button("Process"):
            with st.spinner("Processing"):
                # get document text
                raw_text = get_document_text(docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
    

    
if __name__ == '__main__':
    main()
