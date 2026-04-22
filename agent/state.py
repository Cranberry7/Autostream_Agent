from __future__ import annotations

from typing import Annotated, Optional
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]
    lead_captured: bool
