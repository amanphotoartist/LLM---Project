import os
import pickle
import time
import streamlit as st
from dotenv import load_dotenv

from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA


# Initialize HuggingFace QA model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

st.title("🧠 AskTheArticle – Intelligent Article Query Tool")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_huggingface.pkl"
main_placeholder = st.empty()

if process_url_clicked:
    # Load and split data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Loading data from URLs... 📥")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    docs = text_splitter.split_documents(data)
    main_placeholder.text("Splitting documents... ✂️")

    # Create embeddings using HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Creating vector store with embeddings... 🧠")

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

    main_placeholder.success("Data processing completed ✅")

# Ask a question
query = main_placeholder.text_input("Ask a question about the articles:")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        retriever = vectorstore.as_retriever()
        relevant_docs = retriever.get_relevant_documents(query)
        context = " ".join([doc.page_content for doc in relevant_docs[:3]])

        result = qa_pipeline(question=query, context=context)

        st.header("Answer")
        st.write(result["answer"])

        st.subheader("Context Used")
        st.write(context)
