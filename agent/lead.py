from __future__ import annotations

import logging
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.state import AgentState

logger = logging.getLogger(__name__)

_FIELD_QUESTIONS = {
    "name": "Could you please share your name?",
    "email": "Thanks! What's the best email address to reach you at?",
    "platform": (
        "Great! Which platform(s) do you primarily create content on? "
        "(e.g., YouTube, TikTok, Instagram, LinkedIn)"
    ),
}

_FIELD_ORDER = ["name", "email", "platform"]

_EXTRACT_SYSTEM_PROMPT = """\
You are a data extraction assistant. Extract the {field} from the user's message.

Rules:
  - Return ONLY the extracted value, nothing else.
  - If the message does not contain a clear {field}, return the string: UNKNOWN
  - Do NOT guess or fabricate a value.

Examples for field=name:
  User: "My name is Sarah Chen" → Sarah Chen
  User: "call me Alex" → Alex
  User: "sure" → UNKNOWN

Examples for field=email:
  User: "it's john@gmail.com" → john@gmail.com
  User: "john at gmail dot com" → john@gmail.com
  User: "yeah okay" → UNKNOWN

Examples for field=platform:
  User: "I'm on YouTube mainly" → YouTube
  User: "YouTube and TikTok" → YouTube, TikTok
  User: "maybe" → UNKNOWN
"""


def extract_field_from_message(
    field: str, message: str, llm: ChatGoogleGenerativeAI
) -> Optional[str]:
    messages = [
        SystemMessage(content=_EXTRACT_SYSTEM_PROMPT.format(field=field)),
        HumanMessage(content=message),
    ]
    response = llm.invoke(messages)
    value = response.content.strip()

    if value.upper() == "UNKNOWN" or not value:
        logger.debug("Field '%s' not found in message.", field)
        return None

    logger.debug("Extracted field '%s' = '%s'", field, value)
    return value

def get_next_missing_field(state: AgentState) -> Optional[str]:
    for field in _FIELD_ORDER:
        if state.get(field) is None:
            return field
    return None


def get_question_for_field(field: str) -> str:
    return _FIELD_QUESTIONS.get(field, f"Could you share your {field}?")

def mock_lead_capture(name: str, email: str, platform: str) -> str:
    print(f"\n[TOOL] Lead captured successfully: {name}, {email}, {platform}\n")
    logger.info("mock_lead_capture executed: name=%s email=%s platform=%s", name, email, platform)

    return (
        f"You're all set, {name}! We've captured your details and a member of our "
        f"team will reach out to your {platform} inbox shortly. "
        f"In the meantime, your free Pro trial is being activated for {email}. "
        f"Welcome to AutoStream!"
    )
