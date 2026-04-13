# 📄 Conversational RAG with PDF Upload & Chat History

🚀 **Live Demo:**
👉 https://rag-q-a-conversation-with-pdf-including-chat-history-kk8pehn3p.streamlit.app/

---

## 📌 Overview

This project is a **Conversational RAG (Retrieval-Augmented Generation) system** that allows users to:

* 📄 Upload one or more PDF documents
* 💬 Ask questions about the content
* 🧠 Maintain chat history (context-aware responses)
* ⚡ Get fast answers using Groq LLM

It combines **LangChain, ChromaDB, and Groq** to build an intelligent document-based chatbot.

---

## ✨ Features

* 📂 Multi-PDF upload support
* 🔍 Context-aware question answering
* 🧠 Chat history memory
* ⚡ Fast inference using Groq (LLaMA 3.3)
* 📊 Text chunking & embeddings
* 🗂️ Vector database (ChromaDB)
* 🌐 Deployed on Streamlit Cloud

---

## 🏗️ Tech Stack

* **Frontend:** Streamlit
* **LLM:** Groq (LLaMA 3.3 70B)
* **Framework:** LangChain
* **Vector DB:** ChromaDB
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **PDF Loader:** PyPDF

---

## ⚙️ How It Works

1. 📄 User uploads PDF(s)
2. ✂️ Text is split into chunks
3. 🔢 Embeddings are generated
4. 🗂️ Stored in Chroma vector database
5. 🔍 Relevant chunks retrieved for queries
6. 🤖 LLM generates context-aware answers
7. 🧠 Chat history improves follow-up questions

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Create environment (recommended)

```bash
conda create -n rag_env python=3.10
conda activate rag_env
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create `.env` (for local use):

```env
GROQ_API_KEY=your_api_key_here
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 🌐 Deployment

This app is deployed using **Streamlit Community Cloud**.

To deploy:

1. Push code to GitHub
2. Go to Streamlit Cloud
3. Connect repository
4. Add secret:

   ```toml
   GROQ_API_KEY="your_key_here"
   ```
5. Deploy 🚀

---

## 📚 Learnings

* Implemented RAG architecture
* Integrated LLM with vector search
* Built memory-aware chatbot
* Handled deployment challenges (dependencies, protobuf, etc.)

---

## 🔮 Future Improvements

* 💬 ChatGPT-style UI
* 💾 Persistent vector database
* 📂 Support for multiple file formats (CSV, DOCX)
* 🔐 User authentication
* 🌍 Multi-user session handling

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork and improve the project.

---

## 📧 Contact

If you have any questions or suggestions, feel free to reach out.

---

## ⭐ Show Your Support

If you like this project, please ⭐ the repo!

---
