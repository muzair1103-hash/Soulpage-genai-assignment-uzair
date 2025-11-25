import streamlit as st  # type:ignore
import requests  # type:ignore

BASE_URL = "http://localhost:8000"

st.title("Minimal Knowledge Base UI")

st.header("1. Upload Files")
knowledge_name = st.text_input("Knowledge Name", "default_knowledge")
user_id = st.text_input("User ID", "user123")
upload_files = st.file_uploader(
    "Upload files", type=["pdf", "txt", "md", "docx"], accept_multiple_files=True
)

if st.button("Upload"):
    if not upload_files:
        st.error("Please select at least one file.")
    else:
        files = [("files", (f.name, f, f.type)) for f in upload_files]
        params = {"knowledge_name": knowledge_name, "user_id": user_id}
        res = requests.post(f"{BASE_URL}/rag/upload", params=params, files=files)
        st.write(res.json())

st.header("2. Index Files")
if st.button("Index Files"):
    params = {"knowledge_name": knowledge_name, "user_id": user_id}
    res = requests.post(f"{BASE_URL}/rag/index-file", params=params)
    st.write(res.json())

st.header("3. Ask the Knowledge Base")
query = st.text_area("Enter your question")
if st.button("Ask"):
    if query.strip():
        params = {"knowledge_name": knowledge_name, "user_id": user_id, "query": query}
        res = requests.post(f"{BASE_URL}/rag/ask", params=params)
        st.write(res.text)
    else:
        st.error("Please enter a query.")
