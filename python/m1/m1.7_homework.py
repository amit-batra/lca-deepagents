# python/m1/m1.7_homework.py
"""M1.7 Homework: Design Your Own Multi-Thread Scenario.

THE IDEA
The lab used one topic (favorite colour) across two threads to show that
thread_a remembers it and thread_b doesn't. This homework asks you to
design your own multi-turn scenario and prove three things:
  1. that state persists within a thread across separate invoke() calls,
  2. that a different thread_id starts with no memory of it, and
  3. that this persistence lives in the MemorySaver instance rather than
     the thread_id string itself, by asking a brand-new agent with its
     own fresh MemorySaver the same question on thread_a's thread_id.

WHAT YOU FILL IN
  TODO 1: pick your own topic/fact for the agent to remember, and set up
    two or more of your own thread configs (different thread_ids).
  TODO 2: run the turns that demonstrate persistence (same thread
    remembers), isolation (a different thread doesn't), and checkpointer
    scope (a fresh MemorySaver on the same thread_id doesn't either).

RUN
  cd python
  uv run ./m1/m1.7_homework.py
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver

from models import model

agent = create_deep_agent(
    name="hw_threads_checkpointers_1",
    model=model,
    checkpointer=MemorySaver(),
)


# ════════════════════════════════════════════════════════════════════════
# TODO 1: Pick your own topic and set up two or more thread configs.
#
# Requirements:
#   - Use a topic/fact of your own choosing (not favorite colour, the
#     lab's example).
#   - Define at least two thread configs with different thread_ids, e.g.
#     thread_a = {"configurable": {"thread_id": "my-thread-a"}}
# ════════════════════════════════════════════════════════════════════════

thread_a = {"configurable": {"thread_id": "my-thread-a"}} 
thread_b = {"configurable": {"thread_id": "my-thread-b"}}


# ════════════════════════════════════════════════════════════════════════
# TODO 2: Run the turns that demonstrate persistence, isolation, and
# checkpointer scope.
#
# Requirements:
#   - In thread_a, send at least two turns: one that gives the agent your
#     fact, and a later one that asks it back. Print both responses.
#   - In thread_b (a different thread_id), ask the same follow-up
#     question with no prior context, and print the response. It should
#     NOT know the fact from thread_a.
#   - Build a SECOND agent with its own fresh MemorySaver(), and invoke it
#     on thread_a's thread_id, asking the same follow-up question. Print
#     the response and a line explaining why it doesn't know the fact
#     even though the thread_id matches: memory lives in the MemorySaver
#     instance, not in the thread_id string alone.
#
# Example shape for the second agent (delete this and write your own):
#   fresh_agent = create_deep_agent(model=model, checkpointer=MemorySaver())
#   result = fresh_agent.invoke(
#       {"messages": [{"role": "user", "content": "..."}]},
#       config=thread_a,
#   )
# ════════════════════════════════════════════════════════════════════════

def run_scenario():
    # Thread A, turn 1: Give the agent a fact to remember.
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Remember that my favorite animal is a dolphin."}]},
        config=thread_a,
    )
    print("Thread A, turn 1:")
    print(result["messages"][-1].text)

    # Thread A, turn 2: Ask the agent to recall the fact.
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What does my favorite animal like to eat?"}]},
        config=thread_a,
    )
    print("\nThread A, turn 2:")
    print(result["messages"][-1].text)

    # Thread B, turn 1: Ask the same question in a different thread.
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What does my favorite animal like to eat?"}]},
        config=thread_b,
    )
    print("\nThread B, turn 1:")
    print(result["messages"][-1].text)

    # Fresh agent with its own MemorySaver, asking the same question in thread_a.
    fresh_agent = create_deep_agent(
      name="hw_threads_checkpointers_2",
      model=model,
      checkpointer=MemorySaver()
    )
    result = fresh_agent.invoke(
      {"messages": [{"role": "user", "content": "What does my favorite animal like to eat?"}]},
      config=thread_a,
    )
    print("\nFresh Agent on Thread A:")
    print(result["messages"][-1].text)
    print("This fresh agent does not know the fact because it has its own MemorySaver instance.")

run_scenario()
