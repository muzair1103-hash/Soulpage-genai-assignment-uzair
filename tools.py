from logger import logger
from langchain_community.vectorstores import FAISS
from typing import Annotated
from langchain_core.tools import InjectedToolArg, tool
from langchain_community.docstore.document import Document
from tavily import TavilyClient  # type:ignore
from langchain_openai import ChatOpenAI
from prompts import SEARCH_PROMPT, SUMMARIZER_PROMPT
from langchain_core.messages import ToolMessage
from langchain_core.runnables.config import RunnableConfig
import os

import fitz  # type:ignore

tavily_client = TavilyClient(api_key="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")


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    async def __call__(self, inputs):
        # if messages := inputs.get("messages", []):
        if messages := inputs.messages:
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = await self.tools_by_name[tool_call["name"]].ainvoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=tool_result,
                    # content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


@tool("retrieve_tool")
async def retrieve(
    question: str,
    knowledge_name: Annotated[str, InjectedToolArg],
    user_id: Annotated[str, InjectedToolArg],
) -> list[Document]:
    """
    This tool takes question as input and returns the relevant documents.
    Args:
        question: str
        knowledge_name: str
        user_id: str
    Returns:
        list[Document]
    """
    from backend import get_vs_path, EMBED_MODEL

    logger.info("Retrieve Tool Triggered")
    faiss_path = get_vs_path(knowledge_name=knowledge_name, user_id=user_id)
    vectorstore = FAISS.load_local(
        faiss_path, EMBED_MODEL, allow_dangerous_deserialization=True
    )
    results = vectorstore.similarity_search(question, k=5)

    return results


@tool("query_tool")
async def query(
    question: str,
    knowledge_name: Annotated[str, InjectedToolArg],
    user_id: Annotated[str, InjectedToolArg],
    docs: Annotated[list[Document], InjectedToolArg],
) -> str:
    """
    This tool takes question as input and returns the answer solely based on the retrieved docs.
    Args:
        question: str
        knowledge_name: str
        user_id: str
        docs: list[Document]
    Returns:
        String
    """
    from ask_graph import build_ask_graph

    logger.info("Query Tool Triggered")
    ask_graph = build_ask_graph()
    config = RunnableConfig(
        configurable={
            "thread_id": f"{user_id}_{knowledge_name}_ask",
        }
    )
    response = await ask_graph.ainvoke(
        {
            "knowledge_name": knowledge_name,
            "question": question,
            "user_id": user_id,
            "docs": docs,
        },
        config=config,
    )

    return response["answer"]


@tool("search_tool")
async def search(
    question: str,
) -> str:
    """
    This tool takes question as input and returns the answer by searching the web.
    Args:
        question: str
    Returns:
        String
    """
    logger.info("Search Tool Triggered")
    search_results = tavily_client.search(question)
    model = ChatOpenAI(
        model="qwen2.5:7b",
        temperature=0,
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # type:ignore
    )
    if model is None:
        raise RuntimeError("Model not available ")
    response = await model.ainvoke(
        SEARCH_PROMPT.format(question=question, results=search_results)
    )
    return str(response.content)


@tool("summarizer_tool")
async def summarizer(
    question: str,
    knowledge_name: Annotated[str, InjectedToolArg],
    user_id: Annotated[str, InjectedToolArg],
) -> str:
    """
    This tool takes question as input and returns the answer..
    Args:
        question: str
    Returns:
        String
    """
    from backend import get_docs_path

    logger.info("Summarizer Tool Triggered")
    model = ChatOpenAI(
        model="qwen2.5:7b",
        temperature=0,
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # type:ignore
    )
    if model is None:
        raise RuntimeError("Model not available ")
    docs_path = get_docs_path(knowledge_name=knowledge_name, user_id=user_id)

    all_files = os.listdir(docs_path)
    pdf_files = [f for f in all_files if f.lower().endswith(".pdf")]

    if not pdf_files:
        return str("No PDF files found.")

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

    response = await model.ainvoke(
        SUMMARIZER_PROMPT.format(question=question, docs=all_pages)
    )
    return str(response.content)


@tool("doc_related_tool")
async def doc_related(
    question: str, docs: Annotated[list[Document], InjectedToolArg]
) -> str:
    """
    This tool takes question as input and returns the answer solely based on the retrieved documents.
    Args:
        question: str
        docs: list[Document]
    Returns:
        String
    """
    logger.info("Doc related Tool Triggered")
    model = ChatOpenAI(
        model="qwen2.5:7b",
        temperature=0,
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # type:ignore
    )
    if model is None:
        raise RuntimeError("Model not available ")
    response = await model.ainvoke(
        SEARCH_PROMPT.format(question=question, results=docs)
    )
    return str(response.content)


tool_retrieve = BasicToolNode([retrieve])
tool_search = BasicToolNode([search])
tool_query = BasicToolNode([query])
tool_summary = BasicToolNode([summarizer])
tool_doc_related = BasicToolNode([doc_related])
