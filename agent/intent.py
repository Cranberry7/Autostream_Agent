from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an intent classifier for AutoStream, a SaaS video editing platform.

Classify the user's message into EXACTLY ONE of the following intents:
  - greeting        : A casual greeting or opening message (hi, hello, hey, etc.)
  - product_query   : A question about features, pricing, plans, policies, or capabilities.
  - high_intent     : A clear signal of purchase intent, trial request, sign-up desire,
                      or readiness to get started / buy / try a specific plan.

Rules:
  1. Respond with ONLY the intent label. No punctuation, no explanation, no extra words.
  2. If the message contains both a question AND purchase intent, classify as high_intent.
  3. If unsure between greeting and product_query, use product_query.
  4. If unsure between product_query and high_intent, use high_intent.
"""

_VALID_INTENTS = {"greeting", "product_query", "high_intent"}


def classify_intent(user_message: str, llm: ChatGoogleGenerativeAI) -> str:
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    raw_label = response.content.strip().lower()

    if raw_label not in _VALID_INTENTS:
        logger.warning(
            "Unexpected intent label '%s' from LLM — defaulting to 'product_query'.",
            raw_label,
        )
        return "product_query"

    logger.debug("Intent classified: %s", raw_label)
    return raw_label
