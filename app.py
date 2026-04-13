import streamlit as st
import tempfile

from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- UI ----------------
st.set_page_config(page_title="RAG PDF Chatbot", page_icon="📄")
st.title("📄 Conversational RAG with PDF Uploads")
st.write("Upload PDF(s) and ask questions.")

# ---------------- API KEY (from Streamlit Secrets) ----------------
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("⚠️ GROQ_API_KEY not found. Please add it in Streamlit Secrets.")
    st.stop()

# ---------------- LLM ----------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)

# ---------------- Embeddings ----------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---------------- Session ----------------
session_id = st.text_input("Session ID:", value="default_session")

if "store" not in st.session_state:
    st.session_state.store = {}

# ---------------- File Upload ----------------
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    documents = []

    for uploaded_file in uploaded_files:
        # Cloud-safe temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name

        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        documents.extend(docs)

    # ---------------- Split ----------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500
    )
    splits = text_splitter.split_documents(documents)

    # ---------------- Vector DB ----------------
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever()

    # ---------------- Prompts ----------------
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given chat history and latest question, rewrite it as a standalone question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer using only the provided context. "
         "If not found, say you don't know.\n\nContext:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )

    # ---------------- Memory ----------------
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # ---------------- Chat ----------------
    user_input = st.text_input("Ask a question:")

    if user_input:
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )

        st.success(response["answer"])

        # Optional: show chat history
        st.write("### Chat History")
        st.write(st.session_state.store[session_id].messages)

else:
    st.warning("Please upload at least one PDF")
