from typing import Any

from deepagents import create_deep_agent
from deepagents_docker import DockerSandbox
from langchain.agents.middleware.types import (
    AgentState,
    InputAgentState,
    OutputAgentState,
)
from langgraph.graph.state import CompiledStateGraph

from models import model

with DockerSandbox(image="python:3.14") as backend:

    agent: CompiledStateGraph[
        AgentState[Any], Any, InputAgentState, OutputAgentState[Any]
    ] = create_deep_agent(
        name="docker_sandbox_agent",
        model=model,
        backend=backend,
        system_prompt=(
            "You are a coding assistant. When asked to run code, write the script "
            "to a file first, then execute it. Show both the script as well as the "
            "output in your final answer."
        ),
    )

    result: dict[str, Any] = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a Python script that prints the first 15 Fibonacci numbers, "
                    "save it to fib.py, and run it."
                ),
            }
        ]
    })
    print(result["messages"][-1].text)
