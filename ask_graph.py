from typing import Annotated

from enum import StrEnum
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

from langchain_community.docstore.document import Document
from logger import logger
from pydantic import BaseModel
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import END, StateGraph
from tools import (
    summarizer,
    doc_related,
    tool_summary,
    tool_doc_related,
)
from prompts import ASK_AGENT_PROMPT
from backend import get_sratchpad_from_messages


class AskState(BaseModel):
    knowledge_name: str
    user_id: str
    question: str
    docs: list[Document] | None = None
    answer: str | None = None
    messages: Annotated[list[AnyMessage], add_messages] = []
    agent_response: AIMessage | None = None


class AskSteps(StrEnum):
    DOC_RELATED_QUERY = "doc_related_query"
    SUMMARIZER = "summarizer"
    ASK_AGENT = "ask_agent"
    POST_PROCESSOR = "post_processor"


async def post_processor_ask(state: AskState):
    last_message = state.messages[-1]
    if isinstance(last_message, ToolMessage):
        answer = last_message.content

        return {
            "answer": answer,
            "messages": [
                HumanMessage(content=state.question),
                AIMessage(content=answer),
            ],
        }
    else:
        return


async def ask_agent(state: AskState):
    model = ChatOpenAI(
        model="qwen2.5:7b",
        temperature=0,
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # type:ignore
    )
    if model is None:
        raise RuntimeError("Model not available ")

    tools = [summarizer, doc_related]

    model = model.bind_tools(tools)  # type:ignore
    scratchpad = await get_sratchpad_from_messages(state.messages)

    prompt = ASK_AGENT_PROMPT.format(
        question=state.question, scratchpad=scratchpad, docs=state.docs
    )
    response = await model.ainvoke(prompt)
    return {
        "agent_response": response,
        "messages": [response],
        "answer": response.content,
        # "agent_number_of_calls": state.agent_number_of_calls + 1,  # type:ignore
    }


def build_ask_graph() -> CompiledStateGraph:
    workflow = StateGraph(AskState)

    workflow.add_node(AskSteps.SUMMARIZER, tool_summary)
    workflow.add_node(AskSteps.DOC_RELATED_QUERY, tool_doc_related)
    workflow.add_node(AskSteps.ASK_AGENT, ask_agent)

    workflow.add_node(AskSteps.POST_PROCESSOR, post_processor_ask)
    workflow.set_entry_point(AskSteps.ASK_AGENT)

    def determine_router(state: AskState):
        last_message = state.agent_response
        if isinstance(last_message, AIMessage) and last_message.content != "":
            logger.info("(Ask) Agent response is a string, returning answer directly")
            return END

        if (
            hasattr(last_message, "tool_calls")
            and last_message.tool_calls  # type:ignore
        ):
            if (
                last_message.tool_calls[0]["name"]  # type:ignore
                == summarizer.name
            ):
                logger.info("Summary tool triggered")
                last_message.tool_calls[0]["args"][  # type:ignore
                    "knowledge_name"
                ] = state.knowledge_name
                last_message.tool_calls[0]["args"][  # type:ignore
                    "user_id"
                ] = state.user_id
                return AskSteps.SUMMARIZER  # tool
            elif (
                last_message.tool_calls[0]["name"]  # type:ignore
                == doc_related.name
            ):
                logger.info("Doc related tool triggered")
                last_message.tool_calls[0]["args"][  # type:ignore
                    "docs"
                ] = state.docs
                return AskSteps.DOC_RELATED_QUERY  # tool

    workflow.add_conditional_edges(AskSteps.ASK_AGENT, determine_router)
    workflow.add_edge(AskSteps.DOC_RELATED_QUERY, AskSteps.POST_PROCESSOR)
    workflow.add_edge(AskSteps.SUMMARIZER, AskSteps.POST_PROCESSOR)
    workflow.add_edge(AskSteps.POST_PROCESSOR, AskSteps.ASK_AGENT)

    return workflow.compile()
