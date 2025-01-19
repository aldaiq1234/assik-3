# Constitutional AI Assistant

## Overview
The Constitutional AI Assistant is an interactive application that allows users to:
- Upload documents related to the Constitution of the Republic of Kazakhstan.
- Search and retrieve relevant context from the uploaded files using ChromaDB.
- Ask questions about the Constitution and receive responses powered by the Ollama AI model.

## Features
- **File Upload:** Supports `.txt`, `.pdf`, and `.docx` file formats.
- **ChromaDB Integration:** Adds uploaded documents to a vector database for efficient querying.
- **AI-Powered Responses:** Uses the Ollama API to provide contextual answers to user questions.
- **Interactive Interface:** Built with Streamlit for an easy-to-use web interface.

## Installation

### Prerequisites
- Python 3.8 or later
- Necessary Python libraries:
  - `streamlit`
  - `requests`
  - `chromadb`
  - `numpy`
  - `PyPDF2`
  - `python-docx`

### Steps
1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure Ollama API is running locally on port `11434`.
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Upload Files**
   - Click on the "Upload Constitution files" button.
   - Select files in `.txt`, `.pdf`, or `.docx` format.

2. **Ask Questions**
   - Enter your question about the Constitution in the text input field.
   - View relevant context retrieved from ChromaDB.
   - See AI-powered responses to your query.

3. **Search History**
   - Use the "Search history in ChromaDB" input field to query the database for specific terms.

## File Structure
```
project-folder/
├── src/                # Source code directory
├── test/               # Test cases directory
├── requirements.txt    # List of dependencies
├── README.md           # Project documentation
├── LICENSE             # License file
└── app.py              # Main application file
```

## Example
### Uploading Files
1. Upload the English version of the Constitution of the Republic of Kazakhstan.
2. Verify that the text content is displayed and stored in ChromaDB.
   ![image](https://github.com/user-attachments/assets/dcf5ee39-1167-44c8-8038-bf79992134c8)


### Asking Questions
- Example input: "What are the powers of the President?"
- Output:
  - Relevant excerpts from the Constitution.
  - AI-generated answer explaining the President's powers.



