# 📝 Gemini LLM PDF Question Answering System 
This repository implements a question-answering system using Google Gemini LLM with FAISS-based document retrieval from PDF files. The web interface is built with Streamlit to allow users to upload and process PDF documents and ask questions. The system uses the Google Gemini LLM and LangChain to generate responses from both the language model and relevant PDF content.
# Features
Load and process PDF documents.
Split large text content into manageable chunks using RecursiveCharacterTextSplitter.
Create a FAISS vector store for efficient document retrieval.
Ask questions and get responses from both the Gemini LLM and the processed PDF content.
Streamlit-powered UI with interactive components to load PDFs and query the system.
# Tech Stack
Python: Backend logic.
Streamlit: Frontend for building the interactive UI.
LangChain: Provides the integration with the Google Gemini LLM and document retrieval.
FAISS: Vector search engine to store and search document embeddings.
Google Gemini LLM: For answering general questions and generating responses.
PyPDF2: To extract text from PDF files.
