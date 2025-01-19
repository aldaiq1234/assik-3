import streamlit as st
import requests
import chromadb
import numpy as np
from PyPDF2 import PdfReader  
from docx import Document    

chromadb.api.client.SharedSystemClient.clear_system_cache()
client = chromadb.Client()

collection = client.get_or_create_collection(name="constitution_data")

def get_ollama_response(prompt):
    url = "http://localhost:11434/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }
    data = {
        "model": "llama3.2:latest",
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error Ollama API: {response.status_code}. Response: {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama API: {e}"

def generate_random_embedding(text, embedding_size=384):
    np.random.seed(len(text))
    return np.random.rand(embedding_size).tolist()

def add_to_chromadb(text):
    embeddings = generate_random_embedding(text)
    existing_ids = collection.get().get("ids", [])
    document_id = f"doc-{len(existing_ids) + 1}"

    collection.add(
        ids=[document_id],
        embeddings=[embeddings],
        documents=[text]
    )
    print(f"Text added to ChromaDB with ID: {document_id}")

def search_in_chromadb(query):
    try:
        query_embeddings = generate_random_embedding(query)
        results = collection.query(
            query_embeddings=query_embeddings, 
            n_results=5
        )
        if results["documents"]:
            return results["documents"]
        else:
            return ["No matches found."]
    except Exception as e:
        return [f"Error searching in ChromaDB: {e}"]

def process_uploaded_file(file):
    content = ""
    if file.name.endswith(".txt"):
        content = file.read().decode("utf-8")
    elif file.name.endswith(".pdf"):
        reader = PdfReader(file)
        for page in reader.pages:
            content += page.extract_text()
    elif file.name.endswith(".docx"):
        doc = Document(file)
        for paragraph in doc.paragraphs:
            content += paragraph.text + "\n"
    else:
        st.warning(f"Unsupported file format: {file.name}")
    return content

st.title("Constitutional AI Assistant")

uploaded_files = st.file_uploader("Upload Constitution files", accept_multiple_files=True, type=["txt", "pdf", "docx"])
if uploaded_files:
    for file in uploaded_files:
        file_content = process_uploaded_file(file)
        if file_content:
            st.write(f"Content of {file.name}:")
            st.write(file_content[:500])
            add_to_chromadb(file_content)
            st.success(f"File {file.name} added to ChromaDB!")

user_input = st.text_input("Ask a question about the Constitution:")
if user_input:
    search_results = search_in_chromadb(user_input)
    st.write("Relevant context from ChromaDB:")
    for result in search_results:
        st.write(result)

    ollama_response = get_ollama_response(user_input)
    st.write(f"AI Response: {ollama_response}")

search_query = st.text_input("Search history in ChromaDB:", "")
if search_query:
    search_results = search_in_chromadb(search_query)
    st.write("Search results:")
    for result in search_results:
        st.write(result)
