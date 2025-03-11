import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS  # Vectorstore
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load the Groq and Google API Key
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model Document Q&A")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

# Define the prompt
prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Question: {input}
"""
)


# Function to create vector embeddings
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.write("Vector store DB is Ready ✅")


# User input
prompt1 = st.text_input("What do you want to ask from the document?")

# Ensure vector embeddings are created before querying
if st.button("Document embedding"):
    vector_embedding()

if prompt1:
    if "vectors" not in st.session_state:
        st.warning("Please click 'Document embedding' first to initialize the vector store.")
    else:
        # Create chains for retrieval and response generation
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retriever_chain = create_retrieval_chain(retriever, document_chain)

        # Process user query
        start = time.process_time()
        response = retriever_chain.invoke({'input': prompt1})
        st.write(f"Response time: {time.process_time() - start:.2f} seconds")
        st.write(response['answer'])

        # Display relevant document excerpts
        with st.expander("Document similarity search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("-----------------------------------")
