"""Berisi fungsi helper yang digunakan untuk memakai agent dengan mudah"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool


def run_react_loop(
    llm_with_tools,
    messages: list[BaseMessage],
    tools: list[BaseTool],
    max_tool_rounds: int = 6,
) -> tuple[list[BaseMessage], AIMessage]:
    """Execute a ReAct (Reason + Act) loop.

    Keeps calling the LLM until it stops issuing tool calls or
    `max_tool_rounds` is exhausted.  Returns (all_new_messages, final_ai_msg).
    """
    tool_map = {t.name: t for t in tools}
    new_messages: list[BaseMessage] = []

    current = list(messages)
    final_response: AIMessage = AIMessage(content="")

    for _ in range(max_tool_rounds + 1):
        response: AIMessage = llm_with_tools.invoke(current)
        new_messages.append(response)
        current.append(response)
        final_response = response

        if not getattr(response, "tool_calls", None):
            break

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            try:
                result = tool_map[tool_name].invoke(tool_args)
            except Exception as exc:
                result = f"[Tool error] {exc}"

            tool_msg = ToolMessage(content=str(result), tool_call_id=tc["id"])
            new_messages.append(tool_msg)
            current.append(tool_msg)

    return new_messages, final_response


def extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from LLM output.

    Tries markdown code-fenced JSON first, then bare JSON, then returns
    {"raw": text} as a last resort.
    """
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return {"raw": text}


def format_constraints(bc) -> str:
    if bc is None:
        return "No constraints provided."
    return (
        f"Sector/Domain: {bc.sector_and_domain}\n"
        f"Target Audience: {bc.audience}\n"
        f"Business Idea: {bc.initial_prompt}"
    )
