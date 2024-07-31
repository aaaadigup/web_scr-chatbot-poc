import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import fitz  # PyMuPDF for PDF parsing
import time

load_dotenv()

# Load the GROQ and OpenAI API Key 
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model Document Q&A")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
"""
Generate questions based on the provided context.
Please provide a list of questions that can be asked based on the context.
<context>
{context}
</context>
"""
)

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata else {}

def extract_text_from_pdf(file):
    text = ""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def vector_embedding(extracted_text):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Load extracted text as a document
        st.session_state.docs = [Document(page_content=extracted_text)]

        # Chunk creation
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

        # Vector embeddings
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if st.button("Process PDF") and uploaded_file:
    extracted_text = extract_text_from_pdf(uploaded_file)
    vector_embedding(extracted_text)
    st.write("Vector Store DB Is Ready")

if uploaded_file and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Generate questions from documents
    context_text = " ".join([doc.page_content for doc in st.session_state.final_documents])
    question_generation_prompt = {
        'context': context_text,
        'input': 'Generate questions based on the provided context.'
    }
    
    start = time.process_time()
    response = retrieval_chain.invoke(question_generation_prompt)
    st.write("Response time:", time.process_time() - start)
    
    st.subheader("Generated Questions:")

    # Display questions in a horizontal format
    questions = response['answer'].split('\n')
    formatted_questions = " | ".join([f"{i+1}. {question.strip()}" for i, question in enumerate(questions) if question])
    st.markdown(f"**{formatted_questions}**")
