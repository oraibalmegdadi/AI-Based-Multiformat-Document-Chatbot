#!/usr/bin/env python

#import streamlit as st  # Streamlit library for building interactive web apps
import langchain_community
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

#FIXME
from langchain_core._api.deprecation import suppress_langchain_deprecation_warning

with suppress_langchain_deprecation_warning():
    from langchain.vectorstores import FAISS  # FAISS library for vector storage and similarity search
import InstructorEmbedding  # Custom instructor embedding module

import langchain
from langchain.memory import ConversationBufferMemory  # Library for conversation buffer memory
from langchain.chains import ConversationalRetrievalChain  # Library for building conversational retrieval chains
#from htmlTemplates import css, bot_template, user_template  # HTML template libraries for formatting
import json  # Library for handling JSON data
import requests  # Library for making HTTP requests
import gzip  # Library for working with gzip-compressed files
import pickle  # Library for serializing and deserializing Python objects
import os  # Library for interacting with the operating system
from shutil import copytree  # Library for recursively copying a directory and its contents
import time
import readline

#TODO:
#readline.set_completer(complete)
#  ....["/ls", "/quit", "/execute", "/help", "/read", "/load", "/save"])
readline.parse_and_bind('tab: complete')
# %%

HELP_CMDs= """Available internal commands:
              /read FileName -- reads a pdf or txt into RAG
              /load /save /clean  -- manages the current context
              /execute -- executes commands/dialog from a file
              /ls -- shows folders
              /help /quit /history -- controls the RAG CLI"""

model_name = "llama3" #Other LLMs model can be choosen from https://ollama.com/library, example:
#model_name="mistral" #Other LLMs model can be choosen from https://ollama.com/library 

global log_file

log_file = "log"  # Store current embeddings and text chunks in a log
#log_file = None  # don't store embeddings and text chunks


# %%%


def get_document_text(docs:list)->str:
    """
    Extracts text from a list of uploaded documents.
    Supports PDF and plain text files.

    Parameters:
    docs: List of document file names.

    Returns:
    text: Extracted text from all documents combined.
    """
    text = ""
    for doc in docs:
            #try:
            if doc[-3:] == "pdf":
                pdf_reader = PdfReader(doc)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            elif doc[-3:] == "txt":
                text += doc.read().decode("utf-8")
            else:
                print(f"Unsupported file type: {doc}")
            #except Exception as e:
            #   print(f"Error processing {doc.name}: {e}")
    return text


# if __name__ == '__main__':
#     txt = get_document_text(source_docs)
#     print("Files {source_docs} in memory (txt)")

#%%%
def get_text_chunks(text: str) -> list:
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
   
    if log_file is not None:
        with open(log_file + 'Textchunks.txt', 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                f.write(f'chunk_{i}: {chunk}\n')
    
    return chunks     # Return the list of text chunks, each chunk is 1000 characters

# if __name__ == '__main__':
#     lc = get_text_chunks(txt)
#     print(len(lc), " text chuncks")

#%%%

def get_vectorstore(text_chunks : list) -> \
    langchain_community.vectorstores.faiss.FAISS:
    """
    Converts text chunks into vector embeddings and stores them in a FAISS vector store.

    Parameters:
    text_chunks: List of text chunks to be converted to vectors.

    Returns:
    vectorstore: FAISS vector store containing the text embeddings.
    """
    
    embed=embeddings.ollama.OllamaEmbeddings(model=model_name)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embed)
    
    if log_file is not None:
        with open(log_file + 'Embeddings.txt', 'w') as f:
            for i in range(vectorstore.index.ntotal):
                embedding = vectorstore.index.reconstruct(i)
                emb_str = ','.join(map(str, embedding))
                f.write(f'embedding_{i}: {emb_str}\n')
    
    return vectorstore

# if __name__ == '__main__':
#     vc = get_vectorstore(lc)
#     print("vc vectorstore loaded")

# %%%

def get_conversation_chain(vectorstore :
                           langchain_community.vectorstores.faiss.FAISS) ->  \
      langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain :
    """
    Creates a conversational retrieval chain from the given vector store.

    Parameters:
    vectorstore: FAISS vector store containing the text embeddings.

    Returns:
    conversation_chain: Conversational retrieval chain for interacting with the user.
    """
    llm=Ollama(model=model_name)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


# if __name__ == '__main__':
#     print("Chat with Multiple PDFs and TXTs Locally ("+model_name+") ")
    
#     global conversation 
#     conversation = get_conversation_chain(vc)
    
#     global chat_history 
    
#     chat_history =  None

#%%

# old: def handle_userinput(user_question):

def prompt_llm(user_question):
    """
    Handles the user's input question, generates a response using the conversation chain,
    and updates the chat history and display.

    Parameters:
    user_question: The question input by the user.
    """
    global conversation 
    global chat_history 

    if conversation is None:
        init_conversation()
        
        
    response = conversation.invoke(input = user_question) #conversation({'question': user_question})
    chat_history = response['chat_history']
    print_chat_history()


def print_chat_history() -> None:
        """ """
                    
        global chat_history 
        
        print(Style.BRIGHT)
        for i, message in enumerate(chat_history):
            if i % 2 == 0:
                print(Fore.RED+"USER:", Fore.GREEN+message.content)
            else:
                print(Fore.RED+"BOT:", Fore.WHITE+ message.content)
        print(Style.RESET_ALL)
            
# %%

def save_state(filename):
    """
    Saves the current conversation state to a file.

    Parameters:
    filename: The name of the file to save the state.
    """
    global conversation 
    global chat_history 


    with open(filename, 'wb') as f:
        pickle.dump({
            'conversation': conversation,
            'chat_history': chat_history
        }, f)
    print("State saved successfully.")
    
            
# %%

def load_state(filename):
    """
    Loads a conversation state from a file and updates the session state.

    Parameters:
    filename: The name of the file to load the state from.
    """
    
    global conversation 
    global chat_history 
    
    
    if chat_history is None:
            clear_conversation()

    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            state = pickle.load(f)
            conversation = state['conversation']
            chat_history = state['chat_history']
        
        # Populate chat_display directly and trigger update
        print_chat_history()
    else:
        print("No saved state found.")

            
# %%

def clear_conversation():
    """
    Clears the current conversation, chat history, and chat display.
    """
        
    global conversation 
    global chat_history 
    
    conversation = None
    chat_history = []
    #st.session_state.chat_display = []
    #st.experimental_rerun()



# %%
# Read several files
#            
# %%

def cmd_read(files : str):
    """
    Main function to set up the Streamlit interface and handle user interactions.
    """
    
    print("str", str)
    global conversation 
    global chat_history 
    
    docs = files.split(" ")
    print(docs)    
    raw_text = get_document_text(docs)
    print("read: "+' ' .join(docs)+" ("+str( len(docs))+" docs, "+
          str(len(raw_text))+" chars).")
    text_chunks = get_text_chunks(raw_text)
    print("using ",len(text_chunks), "chunks.")
    
    vectorstore = get_vectorstore(text_chunks)
    print("Vectorized")
    conversation = get_conversation_chain(vectorstore)
    print("conversation prepared.")
    prompt_llm("what is this about?")
 

# %%
# EXECUTE COMMANDS FROM FILE
#
def cmd_execute(file_cmds: str) -> None:
    """ """
    #   process infile and outfile
    inf = open_file(file_cmds, 'r')

    # The main loop
    cmd = '#'
    while cmd != '/quit':
        if cmd[0] != '#':
            process_one_cmd(cmd)
        cmd = read_command(inf)
    # Final cleanup
    close_file(inf)
    
    
# %%
# AUXILIAR FUNCTIONS
#


def open_file(nome: str, modo: str):  # retorna um descitor de ficheiro ou None
    """ Abre um ficheiro no modo indicado """
    if nome == "":
        return None
    return open(nome, modo)


def close_file(f):  # f é um descritor de ficheiro ou None
    """ Fecha o ficheiro indicado """
    if f is not None:
        f.close()


def strip_list(x: list) -> list:
    """
    Strip all the strings in the list
    """
    return [i.strip() for i in x]


def upper_command(s: str) -> str:
    """
    Converts the first word in 's' to uppercase.
    """
    words = strip_list(s.split(' ', maxsplit=1))
    words[0] = words[0].upper()
    cmd = ' '.join(words)
    return cmd


from colorama import init as colorama_init
from colorama import Fore
from colorama import Style




def read_command(f) -> str:
    """
    Reads a command.
    retuns the command word in uppercase followed by the given arguments.
    """
    if f is None:
        # reads from keyboard
        cmd = ''
        print(Style.BRIGHT)
        while cmd == '':
            print(Fore.RED+'RAG.CLI>', end="")
            print(Fore.GREEN," ", end="")
            cmd = input("").strip()
        print(Style.RESET_ALL)
    else:
        # reads from file 'f'
        cmd = f.readline()
        if cmd == '':
            cmd = '/quit'
    return cmd.strip()


# %%
# PROCESS ONE COMMAND
#
def process_one_cmd(user_input: str) -> None:
    """
    Processa um comando (com os seus argumentos).
    O COMANDO está em MAIUSCULAS,
    Os argumentos podem ou não estar em maiúsculas.
    Ignora os comandos inválidos.
    """
    global log_file 
    
    if len(user_input) > 0 and user_input[0] != '/':
        prompt_llm(user_input)
        return
    
    user_input += ' '
    # print("O comand que introduziu foi:", comando, "OF:", outfile)
    cmd = strip_list(user_input.split(' ', 1))
    #print("O comand que introduziu foi:", cmd) #, file=sys.stderr)

    if cmd[0] == "/read":
        cmd_read(cmd[1])
    elif cmd[0] == "/load":
        load_state(cmd[1])
    elif cmd[0] == "/save":
        save_state( cmd[1])
    elif cmd[0] == "/clean":
        clear_conversation()
    elif cmd[0] == "/history":
        print_chat_history()
    elif cmd[0] == "/ls":
        col = 0
        try:
            if cmd[1] == "":
                l = os.listdir()
            else:
                l = os.listdir(cmd[1])
        except Exception as SomeException:
            print(f"{SomeException}")
            l = []

        for f in l:
            print(f"{f:25}",end='\t')
            if(col > 70):
                col = 0
                print()
            col += max(33, len(f))
    elif cmd[0] == "/sleep":
        print("taking a break for "+cmd[1]+"s...")
        time.sleep(int(cmd[1]))
    elif cmd[0] == "/log":
        print("log changed to "+cmd[1])
        log_file = cmd[1]

    elif cmd[0] == "/execute":
        cmd_execute(cmd[1])


    elif cmd[0] == "/quit":
        pass

    elif cmd[0] == "/help":
        print(HELP_CMDs)
    
    else:
        print('Unknown command:', cmd[0])
    return None


# %%% INIT CONVERSATION (if not available)


def force_init_conversation() -> None:
    
    global conversation
    global chat_history
    
    sample_text = [\
"This program is an an implementation of an offline Retrieval-Augmented "+ \
"Generation (RAG) model for a Document Chatbot, providing Conversational "+ \
"AI capabilities for multiformat PDFs and TXTs. The model combines cutting-edge " + \
"technologies and tools to provide a robust, interactive, and efficient document "+
"querying system. Users can upload multiple document formats, including PDFs, "+
"and text files, and interact with the documents through a conversational "+
"AI interface. Moreover, The application allows to save the conversation "+
"and has the ability to load it later and continue the conversation without "+
"a need to upload the documents again. Additionally, the model extracts text "+
"chunks and generates embeddings for each document, which can be useful for "+
"further analysis or processing tasks.",
HELP_CMDs]
    vectorstore = get_vectorstore(sample_text)
    conversation = get_conversation_chain(vectorstore)
    
    chat_history = []
    
def init_conversation() -> None:
     
     if os.path.exists("start_conversation.pkl"):
         load_state("start_conversation.pkl")
     else:
         force_init_conversation()
         save_state("start_conversation.pkl")

# %%
# MAIN FUNTION TO PROCESS ALL COMMANDS
#

def process_cmds(inf=None) -> None:
    """
     Processes a sequence of commands, producing corresponding outputs.
     db_file: name of the database where to place/read the information.
     inf: (please ignore, call this function with only the first argument)
    """

    # The main loop
    cmd = '#'
    while cmd != '/quit':
        if cmd[0] != '#':
            process_one_cmd(cmd)
        cmd = read_command(inf)

    print("BYE!")


def main():
    
    # session_state = globals()
    
    # if "conversation" not in session_state:
    #     conversation = None

    # if "chat_history" not in session_state:
    #     chat_history = None
    colorama_init()

    clear_conversation()
    
    print("# Chat with multiple PDFs and TXTs CLI v0.1")
    print("Chat with Multiple PDFs and TXTs Locally ("+model_name+") \n\n")
    process_cmds()


if __name__ == '__main__':
    main()
