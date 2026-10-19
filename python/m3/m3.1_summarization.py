"""
python/m3/m3.1_summarization.py

Demonstrates how the built-in SummarizationMiddleware compresses conversation
history when the context window fills up.

By default the trigger is 85% of the model's real context window (200k tokens
for Claude Haiku 4.5), which is impractical to hit in a demo. This example
overrides model.profile["max_input_tokens"] to a small value so summarization
fires after a few turns.

Run:
    cd python && uv run ./m3/m3.1_summarization.py
"""

import asyncio

from uuid import uuid4
from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from models import model

# Shrink the reported context window so summarization triggers at ~595 tokens
# (85% of 700) instead of the real threshold. Must use model object, not string.
if model.profile:
    model.profile = {**model.profile, "max_input_tokens": 700}
else:
    model.profile = {"max_input_tokens": 700}

# Build the agent. MemorySaver gives each thread its own in-memory checkpoint,
# which is what lets us read back the full (untrimmed) message history after
# each turn. The one-sentence system prompt keeps the demo's output compact.
agent = create_deep_agent(
    name="m3.1_summarization_agent",
    model=model,
    checkpointer=MemorySaver(),
    system_prompt="You are a helpful assistant. Keep every response to one sentence.",
)

# A single thread id is reused across every turn so the whole demo is one
# conversation; the uuid keeps reruns isolated from each other.
THREAD = {"configurable": {"thread_id": f"demo-{uuid4()}"}}

async def turn(message: str) -> str:
    """Send one user message and return just the agent's final reply text."""
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=message)]},
        config=THREAD,
    )
    # The last message is the agent's response to this turn.
    return result["messages"][-1].text


async def show_state() -> None:
    """Inspect the checkpoint after a turn and report how much the model saw.

    The stored history is the FULL raw transcript (it is never trimmed). When
    summarization has fired, the model no longer sees the old messages directly
    -- it sees a summary plus the messages after the recorded cutoff instead.
    """
    state = await agent.aget_state(THREAD)
    messages = state.values.get("messages", [])
    event = state.values.get("_summarization_event")
    print(f"  stored : {len(messages)} message(s) (raw history, never trimmed)")
    if event:
        cutoff = event.get("cutoff_index", "?")
        print(f"  model saw : summary + messages[{cutoff}:]  [SUMMARIZED]")
        print(f"  Summary event: {event.get('summary_message').content}")


async def main() -> None:
    # Five turns: the first and third each drop a personal detail, and the final
    # turn asks the agent to recall them -- a good moment for summarization to
    # have already compressed the earlier history.
    turns = [
        "My name is Alex. I work at Acme Corp.",
        "What is 2 + 2?",
        "I have been building a distributed cache for three months.",
        "What is the capital of France?",
        "What do you remember about me?",
    ]

    # Drive the conversation one turn at a time, printing the exchange and then
    # the stored-vs-saw state so you can watch summarization take effect.
    for i, message in enumerate(turns, 1):
        print(f"\n{'─' * 50}")
        print(f"Turn {i}  User:  {message}")
        response = await turn(message)
        print(f"Turn {i}  Agent: {response}")
        await show_state()


if __name__ == "__main__":
    asyncio.run(main())
