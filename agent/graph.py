from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import END, START, StateGraph

from agent.intent import classify_intent
from agent.lead import (
    extract_field_from_message,
    get_next_missing_field,
    get_question_for_field,
    mock_lead_capture,
)
from agent.rag import RAGPipeline
from agent.state import AgentState

logger = logging.getLogger(__name__)

def _build_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        api_key=os.environ["GOOGLE_API_KEY"],
    )


def _build_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        api_key=os.environ["GOOGLE_API_KEY"],
    )

def make_classify_intent_node(llm: ChatGoogleGenerativeAI):
    def classify_intent_node(state: AgentState) -> dict[str, Any]:
        if state.get("intent") == "high_intent" and not state.get("lead_captured"):
            logger.info("Sticking to high_intent for lead qualification...")
            return {"intent": "high_intent"}

        last_human_message = _get_last_human_message(state)
        intent = classify_intent(last_human_message, llm)
        logger.info("Intent classified: %s", intent)
        return {"intent": intent}

    return classify_intent_node


def make_greeting_node(llm: ChatGoogleGenerativeAI):
    _GREETING_SYSTEM = (
        "You are AutoStream's friendly AI assistant. "
        "The user has just greeted you. Respond warmly, introduce yourself briefly, "
        "and invite them to ask about AutoStream's video editing features or pricing. "
        "Keep it under 3 sentences. Do NOT ask for personal information."
    )

    def greeting_node(state: AgentState) -> dict[str, Any]:
        last_message = _get_last_human_message(state)
        response = llm.invoke(
            [
                {"role": "system", "content": _GREETING_SYSTEM},
                {"role": "user", "content": last_message},
            ]
        )
        ai_message = AIMessage(content=response.content.strip())
        return {"messages": [ai_message]}

    return greeting_node


def make_product_query_node(rag_pipeline: RAGPipeline):
    def product_query_node(state: AgentState) -> dict[str, Any]:
        last_message = _get_last_human_message(state)
        answer = rag_pipeline.retrieve_and_answer(last_message)
        ai_message = AIMessage(content=answer)
        return {"messages": [ai_message]}

    return product_query_node


def make_high_intent_node(llm: ChatGoogleGenerativeAI):
    _TRANSITION_MSG = (
        "That's great to hear! I'd love to get you set up with AutoStream's Pro plan. "
        "I just need a few quick details to get your account started."
    )

    def high_intent_node(state: AgentState) -> dict[str, Any]:
        last_message = _get_last_human_message(state)
        state_updates: dict[str, Any] = {}

        missing_field = get_next_missing_field(state)

        if missing_field:
            extracted_value = extract_field_from_message(missing_field, last_message, llm)
            if extracted_value:
                state_updates[missing_field] = extracted_value
                logger.info("Collected field '%s' = '%s'", missing_field, extracted_value)

        merged = {**state, **state_updates}

        next_missing = get_next_missing_field(merged)

        if next_missing:
            is_first_high_intent = (
                state.get("name") is None
                and state.get("email") is None
                and state.get("platform") is None
                and not state_updates  
            )

            if is_first_high_intent:
                reply = f"{_TRANSITION_MSG}\n\n{get_question_for_field('name')}"
            else:
                reply = get_question_for_field(next_missing)

        elif not merged.get("lead_captured", False):
            confirmation = mock_lead_capture(
                name=merged["name"],
                email=merged["email"],
                platform=merged["platform"],
            )
            state_updates["lead_captured"] = True
            reply = confirmation

        else:
            reply = (
                "We've already got your details on file! "
                "Is there anything else I can help you with?"
            )

        state_updates["messages"] = [AIMessage(content=reply)]
        return state_updates

    return high_intent_node

def _route_on_intent(state: AgentState) -> str:
    intent = state.get("intent", "product_query")
    return intent  

def _get_last_human_message(state: AgentState) -> str:
    """Extract the text of the most recent HumanMessage from state."""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.content
    return ""

def build_graph() -> Any:
    logger.info("Building AutoStream agent graph...")

    llm = _build_llm()
    embeddings = _build_embeddings()
    rag_pipeline = RAGPipeline(llm=llm, embeddings=embeddings)

    classify_node = make_classify_intent_node(llm)
    greeting_node = make_greeting_node(llm)
    product_node = make_product_query_node(rag_pipeline)
    high_intent_node = make_high_intent_node(llm)

    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_node)
    graph.add_node("greeting", greeting_node)
    graph.add_node("product_query", product_node)
    graph.add_node("high_intent", high_intent_node)

    graph.add_edge(START, "classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        _route_on_intent,
        {
            "greeting": "greeting",
            "product_query": "product_query",
            "high_intent": "high_intent",
        },
    )

    graph.add_edge("greeting", END)
    graph.add_edge("product_query", END)
    graph.add_edge("high_intent", END)

    compiled = graph.compile()
    logger.info("Graph compiled successfully.")
    return compiled
