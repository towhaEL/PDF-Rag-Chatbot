import streamlit as st
import requests
import uuid
import base64

API_URL = "http://localhost:8000"

# --- Page config ---
st.set_page_config(page_title="📚 PDF RAG Chatbot", layout="wide")

# --- Session ID ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "history" not in st.session_state:
    st.session_state.history = []

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = {}

# --- Sidebar ---
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("Upload one or more PDFs to build your knowledge base.")

uploaded_files = st.sidebar.file_uploader(
    "📂 Upload PDFs", type="pdf", accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
        headers = {"X-Session-Id": st.session_state.session_id}
        with st.spinner(f"Processing {uploaded_file.name}..."):
            resp = requests.post(f"{API_URL}/upload", files=files, headers=headers)
        if resp.status_code == 200:
            st.sidebar.success(f"{uploaded_file.name} ✅")

            st.session_state.uploaded_docs[uploaded_file.name] = uploaded_file.read()
        else:
            st.sidebar.error(f"❌ {uploaded_file.name}: {resp.text}")

if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.history = []
    st.success("Chat history cleared.")

show_sources = st.sidebar.checkbox("🔎 Show sources", value=False)

# --- Main App ---
st.title("📚 PDF RAG Chatbot")
st.caption("Ask questions about your uploaded PDFs. The chatbot will retrieve relevant chunks and answer using context.")

# Chat input
st.markdown("---")
user_input = st.chat_input("Ask me something about the PDFs...")

if user_input:
    payload = {"question": user_input, "session_id": st.session_state.session_id}
    with st.spinner("🤔 Thinking..."):
        resp = requests.post(f"{API_URL}/ask", json=payload)

    if resp.status_code == 200:
        answer = resp.json().get("answer", "")
        sources = resp.json().get("sources", [])
        st.session_state.history.append({"role": "user", "content": user_input})
        st.session_state.history.append({"role": "assistant", "content": answer, "sources": sources})
    else:
        st.error(f"Backend error: {resp.status_code} {resp.text}")

# --- Display Chat ---
for msg in st.session_state.history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
            if show_sources and msg.get("sources"):
                with st.expander("📄 Sources"):
                    for src in msg["sources"]:
                        st.markdown(f"- {src}")
