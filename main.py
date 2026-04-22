from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,          
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

if not os.environ.get("GOOGLE_API_KEY"):
    print(
        "\n[ERROR] GOOGLE_API_KEY is not set.\n"
        "  → Create a .env file with: GOOGLE_API_KEY=AIz...\n"
        "  → Or export it: set GOOGLE_API_KEY=AIz...  (Windows)\n"
    )
    sys.exit(1)

from agent.graph import build_graph
from agent.state import AgentState

def _initial_state() -> AgentState:
    return AgentState(
        messages=[],
        intent="",
        name=None,
        email=None,
        platform=None,
        lead_captured=False,
    )


def _print_state_summary(state: AgentState) -> None:
    print(
        f"\n  [State] intent={state['intent']!r}  "
        f"name={state.get('name')!r}  "
        f"email={state.get('email')!r}  "
        f"platform={state.get('platform')!r}  "
        f"lead_captured={state.get('lead_captured', False)}"
    )


def _get_last_ai_message(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage):
            return message.content
    return "(no response)"

def main() -> None:
    print("\n" + "=" * 60)
    print("  AutoStream AI Agent — Social-to-Lead Workflow")
    print("  Powered by LangGraph + Gemini 2.5 Flash + FAISS RAG")
    print("=" * 60)
    print("  Type your message and press Enter.")
    print("  Type 'quit' or 'exit' to end the session.\n")

    print("  [Startup] Building agent graph and FAISS index...")
    graph = build_graph()
    print("  [Startup] Ready!\n")

    state = _initial_state()
    turn = 0

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSession ended. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "bye"}:
            print("\nAgent: Thanks for chatting! Reach us at support@autostream.io anytime. 👋")
            break

        turn += 1

        state["messages"].append(HumanMessage(content=user_input))

        state = graph.invoke(state)

        response = _get_last_ai_message(state)
        print(f"\nAgent: {response}\n")

        _print_state_summary(state)
        print()

        if state.get("lead_captured"):
            print("-" * 60)
            print("Lead successfully captured. Agent session complete.")
            print("-" * 60)
            break


if __name__ == "__main__":
    main()
