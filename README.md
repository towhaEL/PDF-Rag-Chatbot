# 📚 PDF RAG Chatbot

A web-based chatbot that allows you to ask questions about your PDF documents. It uses **retrieval-augmented generation (RAG)** to extract relevant chunks from PDFs and answer questions using **OpenAI embeddings**. Built with **Streamlit** for the frontend and **FastAPI + LangChain** for the backend.

---

## 🖼 Screenshot

<img width="1919" height="925" alt="2Capture" src="https://github.com/user-attachments/assets/2ffe63c6-1e57-413d-abed-845d50432c23" />

> Example of the chatbot UI showing uploaded PDFs, user queries, and retrieved sources with page numbers.

## 🧰 Features

* Upload one or multiple PDFs to build a knowledge base.
* Ask questions about the PDFs and get context-aware answers.
* See **source documents and page numbers** for transparency.
* Maintains **per-session chat history**.
* Supports **Chroma** vector stores.
* Modern RAG pipeline using **LangChain**.

---

## ⚡ Tech Stack

* **Backend**: Python, FastAPI, LangChain, OpenAI Embeddings
* **Vector Store**: Chroma
* **Frontend**: Streamlit
* **PDF parsing**: `PyPDFLoader` from `langchain_community`
* **Environment management**: `.env` file for API keys

---

## 📦 Installation

1. Clone the repo:

```bash
git clone https://github.com/yourusername/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Add environment variables in `.env`:

```env
OPENAI_API_KEY=your_openai_api_key
```

---

## 🚀 Running the Project

### 1. Start the backend (FastAPI)

```bash
uvicorn backend.main:app --reload
```

* Backend will run at `http://localhost:8000`.

### 2. Start the frontend (Streamlit)

```bash
streamlit run frontend/app.py
```

* Frontend will run at `http://localhost:8501`.

---

## 🗂 Project Structure

```
pdf-rag-chatbot/
├─ backend/
│  ├─ app/
│  │  ├─ services/
│  │  │  └─ rag_pipeline.py    # RAG pipeline logic
│  │  └─ main.py               # FastAPI endpoints
├─ frontend/
│  └─ app.py                   # Streamlit UI
├─ requirements.txt
├─ .env.example
└─ README.md
```

---

## 📄 Usage

1. Upload PDFs using the sidebar.
2. Ask questions in the chat input.
3. Toggle “Show sources” to see filenames and page numbers of retrieved chunks.
4. Clear chat with the **Clear Chat** button.

---

## 🧠 How it Works

1. **PDF Ingestion**: PDFs are loaded and split into chunks using `RecursiveCharacterTextSplitter`.
2. **Embeddings**: Chunks are embedded using OpenAI embeddings.
3. **Vector Store**: Chroma stores the embeddings for retrieval.
4. **Querying**:

   * User question + chat history → LLM rewrites question as a standalone query.
   * Retriever fetches relevant chunks.
   * LLM answers based on retrieved chunks.
5. **Chat History**: Maintains conversation context per session.

---

## ✅ Notes

* Duplicate chunks from the same page are automatically deduplicated when displaying sources.
* Page numbers, total pages, and source filenames are available for transparency.
* Supports multiple sessions using unique session IDs.

---

## 📌 Future Improvements

* Add multi-file search across all uploaded PDFs.
* Highlight the exact text in PDF where the answer was found.
* Support multimodal PDFs (images + text) using Google Gemini.

---

## 📄 License

MIT License © \[Your Name]

---