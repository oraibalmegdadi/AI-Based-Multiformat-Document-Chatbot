# Repository Description

This repository contains an implementation of an offline Retrieval-Augmented Generation (RAG) model for a Document Chatbot, providing Conversational AI capabilities for multiformat PDFs and TXTs. The model combines cutting-edge technologies and tools to provide a robust, interactive, and efficient document querying system. Users can upload multiple document formats, including PDFs, and plain text files, and interact with the documents through a conversational AI interface. Moreover, The application allow to save the conversation and  has the ability to load it later and continue the conversation without a need to upload the documents again. Additionally, the model extracts text chunks and generates embeddings for each document, which can be useful for further analysis or processing tasks.
## Key Technologies and Tools

- **[Ollama](https://ollama.com/)**: Used for large language models and embeddings.
- **[Langchain](https://www.langchain.com/)**: Provides the framework for building applications with language models.
- **[Mistral](https://mistral.ai/)**: A state-of-the-art language model used for generating responses.
- **[FAISS](https://ai.meta.com/tools/faiss/)**: A library for efficient similarity search and clustering of dense vectors.
- **[Streamlit](https://streamlit.io/)**: An open-source app framework for Machine Learning and Data Science teams to create interactive web applications.


## Main Model Description

This model utilizes the Ollama software tool to run large language models (LLMs) locally on a computer, enabling offline language processing tasks. Ollama provides access to a diverse set of pre-trained LLM models. Langchain complements Ollama by acting as a framework, allowing developers to integrate these locally-run LLMs into applications. This makes the LLMs easier to use and build upon. Langchain goes a step further by enabling the development of RAG systems, which use various tools to process information, resulting in more refined and informative responses from the language model.


## Extra Features

- **Multi-format Document Support:** Upload and process PDFs and TXTs files.
- **Text Chunking:** Efficiently splits large texts into manageable chunks for better processing.
- **Vector Store Creation:** Uses FAISS to create a vector store of document embeddings.
- **Conversational Interface:** Enables users to interact with the documents through natural language queries.
- **Memory Buffer:** Maintains context across conversations for a seamless user experience.


## Using the Chatbot

1. **Install and Run Ollama** from [https://ollama.com/download](https://ollama.com/download).

   **Using Python Terminal:**

2. **Clone the repository**:
    ```sh
    git clone https://github.com/oraibalmegdadi/AI-Based-Multiformat-Document-Chatbot
    ```

3. **Download requirements**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Run Ollama with the selected LLM**:
	The default LLM is "Mistral". You can choose other LLMs from the supported libraries on the Ollama website: [Ollama Library](https://ollama.com/library).
    ```sh
    ollama pull mistral
    ```

5. **Run the application**:
    ```sh
    streamlit run OfflineChatbot-Mistral.py
    ```

6. **Launching the Application**:
    Upon launching the application, the server interface will be displayed on the internet browser, as shown in the below image. You have the option to upload your own documents or utilize the two documents provided located in the example folder. These documents, generated by ChatGPT, contain valuable information about dams.

7. **Processing Documents**:
    Once you click the "Process" button, the text will undergo chunking and embedding processes, resulting in two separate text files. Examples of these resulting files can be found in the example folder for reference.

8. **Start the Conversation**:
    After the application finishes processing, you will be able to start the conversation.


![langchain](RAG.png)



## Explanation of Main Functions

The image illustrates the overall steps of the model. It begins with uploading multiple PDF files, followed by processing them before engaging in conversation, as outlined below:

1. Call the `get_pdf_text(pdf_docs)` function to read and concatenate text from each page of the uploaded PDF documents.
   - **Main method** for parsing: PyPDF2 library.
   - **Output**: raw_text

2. Call the `get_text_chunks(raw_text)` function to split the extracted text into manageable chunks.
   - **Main method**: langchain library.
   - **CharacterTextSplitter Parameters**: chunk size: 1000 characters, overlap: 200 characters.
   - **Output**: text_chunks

3. Call the `get_vectorstore(text_chunks)` function to generate embeddings for the text chunks and store them in a vector database.
   - **Main method**:
     - **Embeddings**: our model use Mistral LLM, other supported models could be found at [Ollama Library](https://ollama.com/library)
     - **FAISS**: A library for efficient similarity search  A library that allows developers to quickly search for embeddings of multimedia documents that are similar to each other. [More info](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)
   - **Convert the embeddings vectorspace to txt file (embeddings.txt)**

4. Call the `get_conversation_chain(vectorstore)` function to create a conversational chain with the generated vectorstore.
   - **This operation relies on the langchain library for creating conversational chains.**

5. **Save State Function**
   - Call the `save_state(filename)` function to save the current state of the conversation.
   - **Parameters**: 
     - `filename`: The name of the file to save the state.
   - **Main method**: 
     - Uses the `pickle` library to serialize and save the current conversation state (conversation and chat history) to a file.
   - **Output**: State saved successfully message.

6. **Load State Function**
   - Call the `load_state(filename)` function to load a previously saved conversation state.
   - **Parameters**: 
     - `filename`: The name of the file to load the state from.
   - **Main method**:
     - Uses the `pickle` library to deserialize and load the saved conversation state (conversation and chat history) from a file.
     - Populates the chat display directly and triggers an update to reflect the loaded state.
   - **Output**: State loaded successfully message or error if no saved state is found.

7. **Clear Conversation Function**
   - Call the `clear_conversation()` function to clear the current conversation, chat history, and chat display.
   - **Main method**:
     - Resets the session state variables `conversation`, `chat_history`, and `chat_display` to their initial values.
     - Triggers an update to reflect the cleared state.
   - **Output**: Conversation cleared and the interface is reset.


## Useful Tutorials: 
1. Playlist by @alejandro_ao:  https://www.youtube.com/watch?v=LBNpyjcbv0o&list=PLMVV8yyL2GN_n41v1ESBvDHwMbYYhlAh1

2. Playlist by @datasciencebasics: https://www.youtube.com/watch?v=0iBV-eM418Y&list=PLz-qytj7eIWX-bpcRtvkixvo9fuejVr8y

