# RAG Q&A Conversation with PDF including chat history
import streamlit as st
from langchain.chains.history_aware_retriever import create_history_aware_retriever 
from langchain.chains.retrieval import create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['HUGGINGFACE_API_KEY']=os.getenv('HUGGINGFACE_API_KEY')
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# set up streamlit app
st.title("Conversational RAG with PDF Uploads and chat history.")
st.write("Upload pdf and chat with their contents.")

# set up groq API key
groq_api_key = st.text_input("Enter your groq api key:",type="password")

# check if the api key is provided
if groq_api_key:
    llm = ChatGroq(groq_api_key=groq_api_key,model_name='llama-3.3-70b-versatile')

    # chat interface
    session_id = st.text_input("Session_ID:",value="default_session")

    # statefully manage chat history
    if 'store' not in st.session_state:
        st.session_state.store = {}
    
    uploaded_files = st.file_uploader("Choose a pdf file",type='pdf',accept_multiple_files=True)

    # process uploaded pdfs
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        # split and create embeddins for docs
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        splitter = text_splitter.split_documents(documents)
        vectorestore = Chroma.from_documents(embedding=embeddings,documents=splitter)
        retriever = vectorestore.as_retriever()

        # prompts
        contextualize_q_system_prompt = (
            "Given the chat history and the latest user question, "
            "which might reference context in the chat history, "
            "formulate a standalone question that can be understood "
            "without the chat history. Do not answer it; only rewrite it if needed."
        )

        contextual_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # creating history aware retriever
        history_aware_retriever = create_history_aware_retriever(llm,retriever,contextual_q_prompt)

        # answer questions prompts
        qa_system_prompt = (
            "You are a helpful assistant. Use the provided context to answer the question.\n\n"
            "If the answer is not in the context, say you don't know.\n\n"
            "Context:\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        qestion_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,qestion_answer_chain)

        def get_session_state(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,get_session_state,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Enter your qestion:")
        if user_input:
            session_history = get_session_state(session_id)
            response = conversational_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable":{"session_id":session_id}
                },
            )
            st.write(st.session_state.store)
            st.success(f"Assistant:{response['answer']}")
            st.write("Chat History:",session_history.messages)
    else:
        st.warning("Please upload PDF")

else:
    st.warning("Please enter your groq api key")

        

