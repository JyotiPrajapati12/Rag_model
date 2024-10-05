# 📝 Gemini LLM PDF Question Answering System 
This repository implements a question-answering system using Google Gemini LLM with FAISS-based document retrieval from PDF files. The web interface is built with Streamlit to allow users to upload and process PDF documents and ask questions. The system uses the Google Gemini LLM and LangChain to generate responses from both the language model and relevant PDF content.
## Features
1. Load and process PDF documents.
2. Split large text content into manageable chunks using RecursiveCharacterTextSplitter.
3. Create a FAISS vector store for efficient document retrieval.
4. Ask questions and get responses from both the Gemini LLM and the processed PDF content.
5. Streamlit-powered UI with interactive components to load PDFs and query the system.
## Tech Stack
1. Python: Backend logic.
2. Streamlit: Frontend for building the interactive UI.
3. LangChain: Provides the integration with the Google Gemini LLM and document retrieval.
4. FAISS: Vector search engine to store and search document embeddings.
5. Google Gemini LLM: For answering general questions and generating responses.
6. PyPDF2: To extract text from PDF files.


### Install the required dependencies using `pip install -r requirements.txt`.
