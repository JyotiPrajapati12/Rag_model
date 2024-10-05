import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load PDF and extract text
def get_pdf_text(pdf_folder):
    text = ""
    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, filename)
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

# Function to split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create a conversational chain with Gemini LLM
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
   
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                    temperature=0.6)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process user question and respond using Gemini model
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response

# Add function to directly ask the question from Gemini model
def get_gemini_response(question):
    model = genai.GenerativeModel("gemini-pro")
    chat = model.start_chat(history=[])
    response = chat.send_message(question, stream=True)
    
    # Process and display the response
    return ''.join([chunk.text for chunk in response])

# Main Streamlit application
def main():
    # Specify the folder where the PDFs are stored
    pdf_folder = "./Data" 
    st.header("💁💁💁💁💁")

    # Button to load and process PDFs
    if st.button("Load and Process PDFs"):
        with st.spinner("Loading and processing PDF documents..."):
            raw_text = get_pdf_text(pdf_folder)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.session_state.processed = True  # Mark as processed
            st.success("PDF documents processed successfully.")
    else:
        if 'processed' in st.session_state:
            st.success("PDF documents have already been processed.")
        else:
            st.warning("No PDFs processed yet. Click the button to load PDFs.")

    user_question = st.text_input("Ask a Question")

    if user_question:
        # Call the function to ask the question from the Gemini LLM
        gemini_answer = get_gemini_response(user_question)

        # Process with the PDF documents if available
        if 'processed' in st.session_state:
            response = user_input(user_question)
            left_column, right_column = st.columns(2)

            # Add scrollable content to the left column
            with left_column:
                st.subheader("Response from Gemini")
                st.markdown(
                    f"""
                    <div style="height: 400px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px;">
                    {gemini_answer}
                    </div>
                    """, unsafe_allow_html=True)

            # Add scrollable content to the right column
            with right_column:
                st.subheader("Response from PDFs")
                st.markdown(
                    f"""
                    <div style="height: 400px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px;">
                    Reply: {response["output_text"]}
                    </div>""", unsafe_allow_html=True)
        else:
            st.warning("Please load and process PDFs first.")

   

if __name__ == "__main__":
    main()
