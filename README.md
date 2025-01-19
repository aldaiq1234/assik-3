# Ollama Chatbot with File Uploads

## Description

This project is a chatbot that allows users to upload documents and ask questions based on their content. The bot provides answers using the context of the uploaded files and integrates with **Ollama API** to generate responses.

## Features

- Upload files in `.txt`, `.pdf`, and `.docx` formats (either one or multiple files at a time).
- Extract text from uploaded files.
- Save the content of the files to the **ChromaDB** database for context-based search.
- Answer user questions related to the content of uploaded documents.
- Provide a search functionality for query history.


1. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/macOS
   venv\Scripts\activate     # For Windows
