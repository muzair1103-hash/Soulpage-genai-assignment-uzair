from fastapi import UploadFile
import os
from typing import Any
import fitz  # type:ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from logger import logger
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables.config import RunnableConfig
from prompts import REFOMRULATE_PROMPT
from memory import Memory
from main_graph import build_graph, State
from config import MODEL_NAME, TEMPERATURE, BASE_URL, API_KEY

KNOWLEDGE_RAG_DIR = "knowledges"
EMBED_MODEL = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={"trust_remote_code": True, "device": "cuda"},
    encode_kwargs={
        "batch_size": 4,
        "convert_to_tensor": False,
    },
)


def get_docs_path(knowledge_name: str, user_id: str) -> str:
    return os.path.join(
        KNOWLEDGE_RAG_DIR,
        user_id,
        knowledge_name,
        "docs",
    )


def get_vs_path(knowledge_name: str, user_id: str) -> str:
    return os.path.join(
        KNOWLEDGE_RAG_DIR,
        user_id,
        knowledge_name,
        "vectorstore",
    )


async def save_uploaded_file(
    knowledge_name: str, file: UploadFile, user_id: str
) -> tuple[str, bool]:
    filename = file.filename
    if not filename:
        return "", False

    docs_path = get_docs_path(knowledge_name=knowledge_name, user_id=user_id)
    os.makedirs(docs_path, exist_ok=True)

    try:
        file_location = os.path.join(docs_path, filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())
        return filename, True
    except Exception as e:
        logger.error(f"Failed to upload {filename}: {e}")
        return filename, False


async def index_all_pdfs(knowledge_name: str, user_id: str) -> dict[str, Any]:
    docs_path = get_docs_path(knowledge_name=knowledge_name, user_id=user_id)

    all_files = os.listdir(docs_path)
    pdf_files = [f for f in all_files if f.lower().endswith(".pdf")]

    if not pdf_files:
        return {"success": False, "message": "No PDF files found."}

    all_pages = []

    for pdf in pdf_files:
        pdf_full_path = os.path.join(docs_path, pdf)

        try:
            doc = fitz.open(pdf_full_path)
        except Exception as e:
            logger.error(f"Error opening {pdf}: {e}")
            continue

        for i in range(len(doc)):
            text = doc[i].get_text()
            all_pages.append(
                Document(page_content=text, metadata={"page": i + 1, "file_name": pdf})
            )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    docs = splitter.split_documents(all_pages)

    faiss_index = FAISS.from_documents(docs, EMBED_MODEL)

    vs_path = get_vs_path(knowledge_name=knowledge_name, user_id=user_id)
    faiss_index.save_local(vs_path)

    logger.info("FAISS index built and saved.")

    return {"success": True, "message": "All files indexed."}


async def reformulate_question(state: State):
    if not state.messages:
        return {"question": state.question}
    model = ChatOpenAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        base_url=BASE_URL,
        api_key=API_KEY,  # type:ignore
    )

    if model is None:
        raise ValueError("Failed to initialize language model.")
    reformulated_question = await model.ainvoke(
        REFOMRULATE_PROMPT.format(question=state.question, conversation=state.messages)
    )
    return {"question": str(reformulated_question.content)}


async def get_sratchpad_from_messages(messages: list[AnyMessage]) -> str:
    scratchpad = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            scratchpad += f"\nUser(Human) Asked : {msg.content}\n"
        elif isinstance(msg, ToolMessage):
            scratchpad += f"\nTool Result of {msg.name} tool: {msg.content}"
        elif isinstance(msg, AIMessage):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                scratchpad += f"\nTool Called : {msg.tool_calls[0]['name']}\n"
            else:
                scratchpad += f"\nAssistant(AI) Message : {msg.content}\n"
    return scratchpad


async def ask(knowledge_name: str, user_id: str, query: str) -> str:

    memory = await Memory.initialize_memory()
    rag_graph: CompiledStateGraph = build_graph()
    if rag_graph.checkpointer is None:
        rag_graph.checkpointer = memory

    config = RunnableConfig(
        configurable={
            "thread_id": f"{user_id}_{knowledge_name}",
        }
    )
    response = await rag_graph.ainvoke(
        {
            "knowledge_name": knowledge_name,
            "question": query,
            "docs": [],
            "user_id": user_id,
        },
        config=config,
    )

    return response["answer"]
