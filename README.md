# Interactive AI Assistant

## Project Overview
LlamaChat is an interactive Retrieval-Augmented Generation (RAG) chatbot designed to provide contextual answers based on uploaded documents, particularly focusing on the **Constitution of Kazakhstan**. This AI-powered assistant leverages advanced language models to process legal texts, making it an invaluable tool for understanding constitutional law, answering detailed questions, and assisting with legal research. Users can upload .txt or .pdf files containing the Constitution of Kazakhstan or similar legal documents, and the chatbot will provide answers based on the content.

Built using **Streamlit** for the frontend, **Ollama** for model embeddings, **ChromaDB** for vector storage, and **MongoDB** for document and chat history storage, LlamaChat offers a seamless and intuitive interface for real-time, document-based question answering.

---

## Key Features
1. **Document Upload**:
   - Upload the **Constitution of Kazakhstan** or other legal documents in `.txt` or `.pdf` format.
   - The chatbot processes the content, generates embeddings, and stores them for efficient retrieval.

2. **Contextual Question Answering**:
   - Ask questions about the Constitution and receive precise, context-aware answers directly from the document.
   - Example questions:
     - *"What are the key principles of the Republic of Kazakhstan?"*
     - *"What does Article 2 of the Constitution state about sovereignty?"*
     - *"How does the Constitution define the responsibilities of the President?"*

3. **Real-time Interaction**:
   - Get instant, AI-generated responses with relevant sections of the Constitution referenced in real-time.

4. **Persistent Data Storage**:
   - Document embeddings are stored in **ChromaDB**, ensuring fast and scalable retrieval for future queries.
   - Chat history and uploaded documents are stored in **MongoDB** for persistent storage and easy retrieval.

5. **Multi-Model Support**:
   - Use **Ollama** with models like `llama3.2` for generating embeddings and responses.
   - Supports **SentenceTransformers** for embedding generation.

6. **Interactive Chat Interface**:
   - A user-friendly interface built with **Streamlit** for seamless interaction.

7. **Advanced Query Parsing**:
   - The chatbot can identify specific articles or sections mentioned in the user's query and provide precise answers.

8. **Text Splitting for Large Documents**:
   - Large documents like the Constitution are split into smaller chunks for efficient processing and retrieval.

---

## Technologies Used
- **Streamlit**: For building the user-friendly web interface.
- **Ollama**: For creating high-quality embeddings and generating model responses.
- **ChromaDB**: To store document embeddings persistently and facilitate efficient document retrieval.
- **MongoDB**: For storing uploaded documents and chat history.
- **SentenceTransformers**: For document embedding and vector-based similarity search.
- **PyPDF2**: For extracting text from uploaded PDF files.
- **BeautifulSoup**: For web scraping and extracting text from online sources (e.g., the Constitution of Kazakhstan).

---

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.7+
- pip (Python package manager)

### Steps to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/aldaiq1234/assik-3-4
   cd LlamaChat
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
4. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the application:
   ```bash
   streamlit run app.py
   ```
6. Open your browser and go to the URL shown in the terminal, usually `http://localhost:8501`.

---

## Usage

### 1. **Upload the Constitution of Kazakhstan**
   - Upload the full text of the **Constitution of Kazakhstan** (or other legal documents) in `.txt` or `.pdf` format.
   - The chatbot will process the content and store it for querying.


---

### 2. **Ask Questions**
   - Type any questions related to the Constitution or legal provisions. For example:
     - *"What are the key principles of the Republic of Kazakhstan?"*
     - *"What does Article 2 of the Constitution state about sovereignty?"*
     - *"How does the Constitution define the responsibilities of the President?"*
   - The chatbot will reference the uploaded Constitution to provide specific and accurate answers based on the relevant sections or articles.


---

### 3. **Retrieve Contextual Information**
   - The chatbot will highlight relevant sections or articles from the Constitution to support its answers.


---

### 4. **View Chat History**
   - All chat interactions are stored in **MongoDB** and can be retrieved for future reference.


---

### 5. **Advanced Features**
   - **Text Splitting**: Large documents are split into smaller chunks for efficient processing.
   - **Specific Article Retrieval**: The chatbot can identify and retrieve specific articles mentioned in the query.
   - **Multi-Model Support**: Choose from different models like `llama3.2` for generating responses.



---

## Focus on the Constitution of Kazakhstan
LlamaChat is particularly tailored for exploring and understanding the **Constitution of Kazakhstan**. The chatbot can:
- Analyze and interpret articles, clauses, and provisions of the Constitution.
- Provide detailed responses about the legal and historical context of specific articles.
- Assist legal professionals, students, and educators in answering questions related to constitutional law in Kazakhstan.
- Offer insights into Kazakhstan’s governance, legal structure, and key principles such as sovereignty, independence, and the rule of law.

---

## Configuration
You can customize the following settings:
- **Model**: Choose from available AI models, such as Llama 3.2, to optimize the chatbot’s ability to process and answer constitutional queries.
- **Storage**: Adjust the database path if necessary (SQLite by default via ChromaDB).

---

## Contribution
We welcome contributions to enhance this project! To contribute:
1. **Fork the Repository**: Create a copy of the repository to work on your changes.
2. **Create a Feature Branch**: Use a descriptive branch name for your changes.
   ```bash
   git checkout -b feature/your-feature
   ```
3. **Commit Your Changes**: Provide clear, concise commit messages.
   ```bash
   git commit -m "Added feature to analyze constitutional law"
   ```
4. **Push Changes**:
   ```bash
   git push origin feature/your-feature
   ```

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments
Special thanks to the following tools that made this project possible:
- **Streamlit** for the easy-to-use web interface.
- **ChromaDB** for persistent and scalable data storage.
- **Ollama** for providing powerful AI embeddings and language model support.
- **SentenceTransformers** for efficient document embedding and similarity searching.
- **MongoDB** for persistent storage of documents and chat history.

---



---
