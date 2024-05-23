import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
#from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv # require to Connect with the API token Key 
from PyPDF2 import PdfReader # used for reading the pdf files in get_pdf_text
from langchain.text_splitter import CharacterTextSplitter #used to divid the text (from all the documents) to chunks
from langchain.vectorstores import FAISS
import InstructorEmbedding
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import json 
import requests




def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
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
    st.set_page_config(page_title="Chat with multiple PDFs",page_icon=":books:")
    
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
    st.header("Chat with Multiple PDFs Locally (Mistral) :books:")
    #st.text_input("Ask a question about your documents:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)
  
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs=st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                #st.write(text_chunks)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                #st.write(vectorstore)


                
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
    

    
if __name__ == '__main__':
    main()