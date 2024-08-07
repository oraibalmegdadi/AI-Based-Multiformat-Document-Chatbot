# Repository Description

This repository features an implementation of an offline Retrieval-Augmented Generation (RAG) model for a Document Chatbot, offering advanced Conversational AI capabilities for various document formats, including PDFs and TXTs. The model integrates state-of-the-art technologies to deliver a robust, interactive, and efficient document querying system.

Users can upload multiple document formats and engage with them through either a Command Line Interface (CLI) or a Graphical User Interface (GUI). Additionally, the application allows users to save conversations and reload them later, enabling continued interaction without needing to re-upload the documents.

The model also extracts text chunks and generates embeddings for each document, facilitating further analysis or processing tasks.

## Key Technologies and Tools

- **[Ollama](https://ollama.com/)**: Used for large language models and embeddings.
- **[Langchain](https://www.langchain.com/)**: Provides the framework for building applications with language models.
- **[large language models (LLMs)](https://ollama.com/library)**: Ollama supports multiple LLMs from various companies, each available in different parameter sizes to meet diverse project performance, computational requirements, and application suitability. Each model size is designed to address specific needs, balancing resource usage and performance to offer flexibility depending on the deployment context. Generally, the larger the model, the more powerful it is, but it also requires more computational resources to run efficiently. For a complete list of supported models, visit [Ollama's model library](https://ollama.com/library). You can change the LLM used by modifying the **"model_name"** parameter in **ChatbotFunctions.py**.
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

## Using the Offline Chatbot

**Install and Run Ollama** from [https://ollama.com/download](https://ollama.com/download).

Our repository offers two versions of the chatbot:

- **Command Line Interface (CLI)**: 
For a comprehensive guide on setting up and using the chatbot via the command line prompt, including the full code, installation requirements, detailed steps, and usage examples, please refer to the [CommandLineInterface](https://github.com/oraibalmegdadi/AI-Based-Multiformat-Document-Chatbot/tree/main/CommandLineInterface) folder. This resource will provide you with everything you need to get started with the CLI version of the chatbot.

- **Graphical User Interface (GUI)**: 
For a comprehensive guide on setting up and using the chatbot via the command line prompt, including the full code, installation requirements, detailed steps, and usage examples, please refer to the [GraphicalUserInterface](https://github.com/oraibalmegdadi/AI-Based-Multiformat-Document-Chatbot/blob/main/GraphicalUserInterface) folder. This resource will provide you with everything you need to get started with the CLI version of the chatbot.



## Useful Tutorials: 
1. Playlist by @alejandro_ao:  https://www.youtube.com/watch?v=LBNpyjcbv0o&list=PLMVV8yyL2GN_n41v1ESBvDHwMbYYhlAh1

2. Playlist by @datasciencebasics: https://www.youtube.com/watch?v=0iBV-eM418Y&list=PLz-qytj7eIWX-bpcRtvkixvo9fuejVr8y

## Avatars: 

Avatars are created for free from   [https://openart.ai/home](https://openart.ai/home)  
