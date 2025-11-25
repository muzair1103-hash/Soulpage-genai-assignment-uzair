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
    tool_retrieve,
    tool_search,
    retrieve,
    search,
    query,
    tool_query,
)
from prompts import AGENT_PROMPT
from settings import settings


class State(BaseModel):
    knowledge_name: str
    user_id: str
    question: str
    docs: list[Document]
    answer: str | None = None
    messages: Annotated[list[AnyMessage], add_messages] = []
    agent_response: AIMessage | None = None


class Steps(StrEnum):
    REFOMRULATE = "reformulate_question"
    RETRIEVE = "retrieve"
    QUERY = "query"
    WEB_SEARCH = "web_search"
    AGENT_NODE = "agent_node"
    POST_PROCESSOR = "post_processor"
    RETRIEVE_POST_PROCESSOR = "retrieve_post_processor"


async def retrieve_post_processor(state: State):
    last_message = state.messages[-1]
    if isinstance(last_message, ToolMessage):
        texts = last_message.content
        docs = [Document(page_content=text) for text in texts]  # type:ignore

        return {
            "docs": docs,
        }

    else:
        return


async def post_processor_main(state: State):
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


async def agent_node(state: State):
    from backend import get_sratchpad_from_messages

    model = ChatOpenAI(
        model=settings.models_settings.MODEL_NAME,
        temperature=settings.models_settings.TEMPERATURE,
        base_url=settings.models_settings.BASE_URL,
        api_key=settings.models_settings.API_KEY,  # type:ignore
    )
    if model is None:
        raise RuntimeError("Model not available ")

    tools = [retrieve, search, query]

    model = model.bind_tools(tools)  # type:ignore
    scratchpad = await get_sratchpad_from_messages(state.messages)

    prompt = AGENT_PROMPT.format(
        question=state.question,
        scratchpad=scratchpad,
        docs=state.docs if len(state.docs) > 0 else [],
    )
    response = await model.ainvoke(prompt)
    return {
        "agent_response": response,
        "messages": [response],
        "answer": response.content,
        # "agent_number_of_calls": state.agent_number_of_calls + 1,  # type:ignore
    }


def build_graph() -> CompiledStateGraph:
    from backend import reformulate_question

    workflow = StateGraph(State)

    workflow.add_node(Steps.REFOMRULATE, reformulate_question)
    workflow.add_node(Steps.AGENT_NODE, agent_node)
    workflow.add_node(Steps.QUERY, tool_query)
    workflow.add_node(Steps.RETRIEVE, tool_retrieve)
    workflow.add_node(Steps.WEB_SEARCH, tool_search)
    workflow.add_node(Steps.POST_PROCESSOR, post_processor_main)
    workflow.add_node(Steps.RETRIEVE_POST_PROCESSOR, retrieve_post_processor)

    workflow.set_entry_point(Steps.REFOMRULATE)
    workflow.add_edge(Steps.REFOMRULATE, Steps.AGENT_NODE)

    def determine_router(state: State):
        last_message = state.agent_response
        if isinstance(last_message, AIMessage) and last_message.content != "":
            logger.info("Agent response is a string, returning answer directly")
            return END

        if (
            hasattr(last_message, "tool_calls")
            and last_message.tool_calls  # type:ignore
        ):
            if (
                last_message.tool_calls[0]["name"]  # type:ignore
                == search.name
            ):
                logger.info("Search tool triggered")
                return Steps.WEB_SEARCH  # tool
            elif (
                last_message.tool_calls[0]["name"]  # type:ignore
                == retrieve.name
            ):
                logger.info("Retrieve tool triggered")
                last_message.tool_calls[0]["args"][  # type:ignore
                    "knowledge_name"
                ] = state.knowledge_name
                last_message.tool_calls[0]["args"][  # type:ignore
                    "user_id"
                ] = state.user_id

                return Steps.RETRIEVE  # tool
            elif (
                last_message.tool_calls[0]["name"]  # type:ignore
                == query.name
            ):
                logger.info("Query tool triggered")
                last_message.tool_calls[0]["args"][  # type:ignore
                    "knowledge_name"
                ] = state.knowledge_name
                last_message.tool_calls[0]["args"][  # type:ignore
                    "user_id"
                ] = state.user_id
                last_message.tool_calls[0]["args"][  # type:ignore
                    "docs"
                ] = state.docs

                return Steps.QUERY  # tool

    workflow.add_conditional_edges(Steps.AGENT_NODE, determine_router)
    workflow.add_edge(Steps.WEB_SEARCH, Steps.POST_PROCESSOR)
    workflow.add_edge(Steps.QUERY, Steps.POST_PROCESSOR)
    workflow.add_edge(Steps.POST_PROCESSOR, Steps.AGENT_NODE)
    workflow.add_edge(Steps.RETRIEVE, Steps.RETRIEVE_POST_PROCESSOR)
    workflow.add_edge(Steps.RETRIEVE_POST_PROCESSOR, Steps.AGENT_NODE)
    return workflow.compile()
