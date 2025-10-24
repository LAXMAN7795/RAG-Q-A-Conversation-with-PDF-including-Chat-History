# Conversational RAG with PDF Uploads and Chat History

This project is a **Conversational Retrieval-Augmented Generation (RAG) system** built with **Streamlit** and **LangChain**, allowing users to upload PDFs and interactively chat with their contents while maintaining conversation history.

---

## Features

- Upload multiple PDF documents for analysis.
- Split documents into chunks and create embeddings using **HuggingFace embeddings** (`all-MiniLM-L6-v2`).
- Use **Groq LLM** (`llama-3.3-70b-versatile`) for generating responses.
- Maintain **chat history** for context-aware responses.
- Rewrite user questions based on previous chat for better context understanding.
- Streamlit web interface for interactive Q&A.

---

## Installation

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd <your-repo-directory>
Create a virtual environment

bash
Copy code
python -m venv venv
Activate the virtual environment

Windows:

bash
Copy code
venv\Scripts\activate
macOS/Linux:

bash
Copy code
source venv/bin/activate
Install dependencies

bash
Copy code
pip install -r requirements.txt
Example requirements.txt:

arduino
Copy code
langchain
langchain-core
langchain-community
langchain-chroma
langchain-huggingface
langchain-groq
chromadb
sentence-transformers
pypdf
langchain-text-splitters
python-dotenv
streamlit
fastapi
uvicorn
langserve
Setup
Set your HuggingFace API key in a .env file:

ini
Copy code
HUGGINGFACE_API_KEY=your_huggingface_api_key
Run the app:

bash
Copy code
streamlit run app.py
Enter your Groq API key in the Streamlit interface when prompted.

Usage
Enter a session ID (default: default_session) to track chat history.

Upload one or more PDF files.

Ask questions in the input box.

The assistant will provide answers based on the uploaded PDFs and maintain context-aware chat history.

File Structure
bash
Copy code
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (HuggingFace API key)
├── README.md              # Project documentation
└── venv/                  # Virtual environment (ignored in git)
Notes
Ensure you have a valid Groq API key and HuggingFace API key.

Only PDF files are supported for upload.

Chat history is stored per session ID, enabling multiple independent conversations.

License
This project is open-source and available under the MIT License.

Author
Laxman Sannu Gouda
laxman.sg0104@gmail.com
