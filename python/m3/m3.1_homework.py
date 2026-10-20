# python/m3/m3.1_homework.py
"""M3.1 Homework: Trigger Chained Summarization.

THE IDEA
In the lesson, you watched SummarizationMiddleware compress a demo
conversation ONCE, past the 85% threshold. But summarization doesn't just
fire once and stop: on a long enough conversation it fires again, and again
each time it re-summarizes the previous summary plus whatever's new since,
while the FULL evicted history keeps piling up in a single backend file at
/conversation_history/{thread_id}.md.

This homework asks you to build a conversation on a topic you pick that's
long enough to trigger summarization AT LEAST TWICE, then confirm two things:
the model can still recall a detail from your very first turn (after being
compacted multiple times), and the conversation history file on the backend
actually accumulated multiple "Summarized at ..." sections
instead of losing earlier ones.

WHAT YOU FILL IN
  TODO 1: write your own list of user turns (at least 8) about a topic YOU
    pick. Put an important detail in one of your first two turns, then keep
    talking about the topic for several more turns, and end with a turn
    that asks the agent to recall that early detail.
  TODO 2: choose model.profile["max_input_tokens"] so summarization fires
    at least twice before your last turn, not just once. Tune it by trial
    and error, same as the lesson did with 700. Keep it in the low
    thousands; much lower and the final recall fails.

RUN
  cd python
  uv run ./m3/m3.1_homework.py
"""

import asyncio, uuid

from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from models import model


# ════════════════════════════════════════════════════════════════════════
# STEP 1: Write your own multi-turn scenario.
#
# Requirements:
#   - At least 8 user turns, all about ONE topic of your choosing.
#   - One of your first two turns should state a concrete detail (a number,
#     a name, a decision).
#   - Your last turn should ask the agent to recall that detail, after
#     several more turns of unrelated follow-up on the same topic.
#
# Example shape (delete this and write your own):
#   return [
#       "I'm planning a two-week trip to ...",
#       "My total budget is ...",
#       ...,
#       ...,
#       ...,
#       "Quick recap: what was my total budget?",
#   ]
# ════════════════════════════════════════════════════════════════════════

def build_turns() -> list[str]:
    """Return the ordered list of user messages that drive the conversation.

    The scenario below is about one topic (the user's software career) and
    plants a concrete early detail -- "27 years" of experience, plus a list of
    areas of mastery -- in the first few turns. After several more turns of
    related follow-up, the final turn asks the agent to recall that early
    detail, which forces it to rely on a *summary* rather than raw history if
    summarization has already fired.

    Each element is one turn's user message; the loop in ``main`` sends them
    one at a time and reuses the same thread between sends.
    """
    return [
        "You know I have been working in the software industry for 27 years now.",
        "In these 27 years I have worked in diverse areas, and have attained mastery in most of them.",
        (
            "Some of the areas in which I have achieved mastery are: system software, operating systems, "
            "application software, desktop applications, cloud services and agentic systems."
        ),
        "I love working both in the capacity of a software architect, as well as a software developer.",
        "Had it not been for software development, I would have felt lost in my life.",
        "This industry evolves so fast that one has to constantly keep updating our skills.",
        "You never know what's going to be the next biggest skill in demand.",
        "How much work experience do I have overall and in what area(s)?"
    ]

# ════════════════════════════════════════════════════════════════════════
# STEP 2: Choose the summarization threshold.
#
# Lower model.profile["max_input_tokens"] to a value that makes
# SummarizationMiddleware fire (at 85% of that number) AT LEAST TWICE
# across your conversation from STEP 1, not just once. The lesson used 700
# for a 5-turn demo that fired once; your number depends on how many turns
# you wrote and how long they are.
#
# Keep it in the low thousands; the filled copy uses 3000 for ten turns. Set
# it much lower and summarization fires on nearly every turn, each summary
# gets compacted away before the next one lands, and the final recall fails.
# ════════════════════════════════════════════════════════════════════════

MAX_INPUT_TOKENS: int = 1000

# Report a small context window to the agent so summarization triggers early.
# The imported model object is shared, so the new profile replaces the old one;
# merge the existing profile (rather than discarding it) when one is present.
if model.profile:
    model.profile = {**model.profile, "max_input_tokens": MAX_INPUT_TOKENS}
else:
    model.profile = {"max_input_tokens": MAX_INPUT_TOKENS}

# Build the agent. MemorySaver keeps each thread's full, untrimmed message
# history in memory, which is what lets show_state read it back after a turn.
# The one-sentence system prompt keeps the demo output compact.
agent = create_deep_agent(
    name="m3.1_homework_agent",
    model=model,
    checkpointer=MemorySaver(),
    system_prompt="You are a helpful assistant. Keep every response to one sentence.",
)

# A single thread id is reused across every turn so the whole run is one
# conversation; the uuid keeps each rerun isolated from the previous one.
THREAD = {"configurable": {"thread_id": f"m3.1_homework_{uuid.uuid4()}"}}

# Where SummarizationMiddleware accumulates the evicted (compacted) history on
# the backend -- one Markdown file per thread, one "## Summarized at ..." section
# per firing. We read it at the end to confirm multiple sections piled up.
HISTORY_PATH = f"/conversation_history/{THREAD['configurable']['thread_id']}.md"

async def turn(message: str) -> str:
    """Send one user message and return just the agent's final reply text.

    Reuses THREAD so every call continues the same conversation. The agent
    appends to the checkpoint and returns the full message list; the last
    element is the agent's response to this turn.
    """
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=message)]},
        config=THREAD,
    )
    return result["messages"][-1].text


async def show_state(seen_cutoffs: set) -> bool:
    """Print state after a turn. Returns True if this turn produced a NEW
    summarization event (as opposed to still living under a previous one)."""
    state = await agent.aget_state(THREAD)
    messages = state.values.get("messages", [])
    event = state.values.get("_summarization_event")
    print(f"  stored : {len(messages)} message(s) (raw history, never trimmed)")

    if not event:
        return False

    is_new_event = False
    cutoff = event.get("cutoff_index", "?")
    if cutoff not in seen_cutoffs:
        seen_cutoffs.add(cutoff)
        is_new_event = True

    tag = "  <-- NEW EVENT" if is_new_event else ""
    print(f"  model saw : summary + messages[{cutoff}:]  [SUMMARIZED]{tag}")
    return is_new_event


async def main() -> None:
    """Drive the conversation and verify summarization fired at least twice.

    Two things must be true at the end (the homework's acceptance criteria):
      1. The model can still recall an early detail after being compacted
         several times -- that happens naturally because the final turn asks
         for it and summarization has moved the raw history out of view.
      2. The backend history file accumulated multiple "## Summarized at ..."
         sections, i.e. earlier summaries were kept rather than overwritten.
    """
    turns = build_turns()
    # Track which summarization cutoffs we've already seen so we can count
    # distinct firing *events*, not just turns that still carry a stale event.
    seen_cutoffs: set = set()
    event_count = 0
    for i, message in enumerate(turns, 1):
        print(f"\n{'─' * 50}")
        print(f"Turn {i}  User:  {message}")
        response = await turn(message)
        print(f"Turn {i}  Agent: {response}")
        # show_state reports the stored-vs-saw split and flags a NEW event.
        if await show_state(seen_cutoffs):
            event_count += 1

    print(f"\nSummarization fired {event_count} time(s) across {len(turns)} turns.")
    if event_count < 2:
        print(
            "That's fewer than 2. Lower MAX_INPUT_TOKENS, or add more turns/detail, "
            "so it fires again before the conversation ends."
        )

    # Second acceptance check: the evicted history must have piled up as
    # multiple "## Summarized at ..." sections in the per-thread history file.
    state = await agent.aget_state(THREAD)
    history_file = state.values.get("files", {}).get(HISTORY_PATH)
    if history_file:
        content = history_file["content"] if isinstance(history_file, dict) else history_file
        sections = content.count("## Summarized at")
        print(f"\n--- {HISTORY_PATH} ({sections} section(s)) ---")
        print(content)


if __name__ == "__main__":
    asyncio.run(main())