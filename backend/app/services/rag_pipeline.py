# backend/app/services/rag_pipeline.py
import os
from typing import Dict, List, Tuple

# Document loading / splitting / embeddings / vectorstore / chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

# Globals / paths
VECTOR_DB_PATH = "vector_db/chroma_index"
os.makedirs(os.path.dirname(VECTOR_DB_PATH) or ".", exist_ok=True)

# Initialize embedding + LLM
# embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
# embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
)

# llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

# Global state
_db = None
_retriever = None
_rag_chain = None

# Simple per-session chat histories
# key: session_id -> list of (question, answer)
chat_histories: Dict[str, List[Tuple[str, str]]] = {}

def ingest_pdf(pdf_path: str):
    """
    Load PDF, split into chunks, embed, and save/update Chroma vector store.
    """
    global _db, _retriever, _rag_chain

    # 1. Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # 3. Build / update Chroma DB
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=VECTOR_DB_PATH
    )
    db.persist()

    # 4. Build retriever
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # --- Conversational Retrieval replacement ---
    # Prompt to rewrite user queries using history
    contextualize_q_prompt = ChatPromptTemplate.from_template(
        "Given the chat history and the latest user question, "
        "reformulate the question into a standalone query.\n\n"
        "Chat history:\n{chat_history}\n\nQuestion: {input}\n\nStandalone query:"
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Prompt for answering from documents
    qa_prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the following context:\n{context}\n\nQuestion: {input}"
    )
    document_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    # Replace globals
    _db = db
    _retriever = retriever
    _rag_chain = rag_chain


def answer_query(question: str, session_id: str = "default"):
    global _rag_chain, chat_histories
    if _rag_chain is None:
        return {"answer": "Please upload a PDF first.", "sources": []}

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    chain_history = chat_histories[session_id].copy()

    # invoke with chat history + input
    result = _rag_chain.invoke({
        "input": question,
        "chat_history": chain_history
    })

    # result contains "answer" + "context"
    answer = result.get("answer", "")
    # Extract sources if present
    # sources = [doc.metadata.get("page", "Unknown") for doc in result.get("context", [])]

    seen = set()
    sources = []
    for doc in result.get("context", []):
        src = f"{doc.metadata.get('source', 'Unknown')} (page {doc.metadata.get('page', '?')})"
        if src not in seen:
            sources.append(src)
            seen.add(src)


    chat_histories[session_id].append((question, answer))

    return {"answer": answer, "sources": sources}


def get_history_for_session(session_id: str):
    return chat_histories.get(session_id, [])
