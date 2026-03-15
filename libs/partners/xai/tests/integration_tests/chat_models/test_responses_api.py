"""Test Responses API usage for xAI."""

from typing import Literal

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessageChunk
from langchain_core.tools import tool
from pydantic import BaseModel

from langchain_xai import ChatXAI

MODEL_NAME = "grok-4-fast-reasoning"


def _check_response(response: AIMessage | None) -> None:
    assert isinstance(response, AIMessage)
    assert response.content
    assert response.usage_metadata
    assert response.response_metadata.get("model_provider") == "xai"


@pytest.mark.parametrize("output_version", ["", "v1"])
def test_reasoning(output_version: Literal["", "v1"]) -> None:
    """Test reasoning with Responses API."""
    if output_version:
        llm = ChatXAI(
            model="grok-3-mini",
            reasoning_effort="low",
            temperature=0,
            output_version=output_version,
        )
    else:
        llm = ChatXAI(
            model="grok-3-mini",
            reasoning_effort="low",
            temperature=0,
        )
    response = llm.invoke("What is 3^3?")
    _check_response(response)
    assert response.additional_kwargs.get("reasoning_content")

    full: BaseMessageChunk | None = None
    for chunk in llm.stream("What is 3^3?"):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.additional_kwargs.get("reasoning_content")


def test_web_search() -> None:
    llm = ChatXAI(model=MODEL_NAME, temperature=0).bind_tools([{"type": "web_search"}])
    response = llm.invoke("Look up the current time in Boston, MA.")
    _check_response(response)
    content_types = {block["type"] for block in getattr(response, "content_blocks", [])}
    assert (
        "non_standard" in content_types
        or "server_tool_call" in content_types
        or "tool_call" in content_types
    )
    assert "text" in content_types


@pytest.mark.parametrize("output_version", ["", "v1"])
def test_function_calling(output_version: Literal["", "v1"]) -> None:
    @tool
    def multiply(x: int, y: int) -> int:
        """return x * y"""
        return x * y

    llm = ChatXAI(model=MODEL_NAME, output_version=output_version)
    bound_llm = llm.bind_tools([multiply])
    ai_msg = bound_llm.invoke("whats 5 * 4")
    assert isinstance(ai_msg, AIMessage)
    assert len(ai_msg.tool_calls) == 1
    assert ai_msg.tool_calls[0]["name"] == "multiply"


class Foo(BaseModel):
    response: str


def test_parsed_pydantic_schema() -> None:
    llm = ChatXAI(model="grok-3-mini", use_responses_api=True)
    response = llm.invoke("how are ya", response_format=Foo)
    assert isinstance(response, AIMessage)
    parsed = response.additional_kwargs.get("parsed")
    assert parsed is not None
    _check_response(response)
