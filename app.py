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
from bs4 import BeautifulSoup
import requests
import time

load_dotenv()

# Load the GROQ and OpenAI API Key 
groq_api_key = os.getenv('GROQ_API_KEY')

st.title("Gemma Model Document Q&A")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Questions: {input}
    """
)

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata else {}

def scrape_website(url):
    # Fetch the content from URL
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')

    # Extract text from the webpage
    text = soup.get_text()

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

# URL input
url = st.text_input("Enter the URL of a website to scrape")

if st.button("Scrape and Embed") and url:
    extracted_text = scrape_website(url)
    vector_embedding(extracted_text)
    st.write("Vector Store DB Is Ready")

prompt1 = st.text_input("Enter Your Question From Documents")

if prompt1 and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.write("Response time:", time.process_time() - start)

    st.subheader("Generated Answer:")
    st.write(response['answer'])

    # With a Streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
