# python/m4/m4.2_homework.py
"""M4.2 Homework: Give Each Subagent Its Own Scoped Scratch Folder.

THE IDEA
The lab's genre-researcher subagents each wrote raw search notes to their own
assigned /research/<genre>/ folder, kept out of the editor's context by
FilesystemPermission scoping: researchers could write under /research/**, the
editor could not.

This homework asks you to build a small team of 2 subagent types for
a domain YOU pick (e.g., trip planning, home renovation), each with
its own private, permission-scoped folder under /scratch/<name>/ to
stash raw notes in before answering. The harness below wires up the scratch
folder, the permissions, and the write-before-answering instruction for you;
you just decide who your two subagents are.

WHAT YOU FILL IN
  TODO 1: for each of the two entries in SUBAGENT_SPECS, fill in "name",
    "description" (when the main agent should call it), and "role_prompt"
    (who this subagent is and what its job is). Everything else -- the
    scratch folder, the permissions, the instruction to save raw notes
    before answering -- is handled for you.
  TODO 2: write the main agent's system prompt, telling it which subagent
    to call for which part of the job, and a user request that should
    trigger delegation to BOTH subagents.

RUN
  cd python
  uv run ./m4/m4.2_homework.py
"""

import os
from typing import Any
from uuid import uuid4

from deepagents import FilesystemPermission, create_deep_agent
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph
from tavily import TavilyClient

from models import model

# The root directory where subagents can store their raw notes.
SCRATCH_ROOT: str = "/scratch"

# Initialize the Tavily client for web searching.
tavily_api_key: str = os.environ.get("TAVILY_API_KEY", "")
if not tavily_api_key:
    raise RuntimeError(
        "TAVILY_API_KEY is required for running this script. "
        "Set it in your environment and then re-run this script."
    )

_tavily: TavilyClient = TavilyClient(api_key=tavily_api_key)

@tool
def internet_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search the web for a given query and return the top results.

    Subagents use this to gather the raw facts they need before writing
    their notes. Keep queries specific and focused on a single aspect of
    the question.
    """
    return _tavily.search(query, max_results=max_results)


# Returns the specific file path where a subagent should store its notes.
def scratch_path(subagent_name: str) -> str:
    return f"{SCRATCH_ROOT}/{subagent_name}/notes.md"


def scratch_permissions(subagent_name: str) -> list:
    """Return the permissions that scope a subagent to its own scratch folder.

    This reuses the lab's first-match-wins allow-then-deny pattern (the same
    one used for research_permissions/editor_permissions): allow access inside
    the subagent's folder, then deny writes everywhere else so each subagent's
    notes stay isolated.
    """
    return [
        # Allow reading and writing within the subagent's specific directory.
        FilesystemPermission(
            operations=["read", "write"],
            paths=[f"{SCRATCH_ROOT}/{subagent_name}/**"], mode="allow"
        ),
        # Deny writing anywhere else to keep scratch folders isolated.
        FilesystemPermission(
            operations=["write"],
            paths=["/**"], mode="deny"
        ),
    ]


# Builds the instruction that makes a subagent save its raw notes to its own
# private scratch file before it returns its final answer.
def scratch_instruction(subagent_name: str) -> str:
    return (
        f'Before you answer, call write_file on "{scratch_path(subagent_name)}" '
        "with your raw notes or reasoning. Then give your final answer using "
        "only the polished result -- do not repeat those raw notes in your reply."
    )


def build_subagents(specs: list[dict]) -> list[dict]:
    """
    Constructs the subagent configurations by combining the user's specs
    with the mandatory scratch folder permissions and instructions.
    """
    team = []
    for spec in specs:
        name = spec["name"]
        team.append(
            {
                "name": name,
                "tools": [internet_search],
                "description": spec["description"],
                "system_prompt": spec["role_prompt"] + "\n\n" + scratch_instruction(name),
                "permissions": scratch_permissions(name),
            }
        )
    return team


# ════════════════════════════════════════════════════════════════════════
# STEP 1: Fill in your two subagents.
#
# For each entry: "name" is the handle the main agent calls it by, kebab-
# case (e.g. "flight-finder"). "description" is how the main agent decides
# which one to use. "role_prompt" is that subagent's own job description --
# don't mention scratch files or write_file here, that's added for you.
# ════════════════════════════════════════════════════════════════════════

SUBAGENT_SPECS: list[dict[str, str]] = [
    {
        "name": "months-suggestion-agent",
        "description": (
            "This agent determines what months of the year are best "
            "suited to visit the chosen destination and returns that "
            "information in its response."
        ),
        "role_prompt": (
            "Your job is to search the Internet to determine what "
            "months of the year are best suited to visit the "
            "specified place. If the Internet search returns conflicting "
            "results, choose the best of 5 as your response."
        ),
    },
    {
        "name": "visa-requirement-agent",
        "description": (
            "This agent determines whether a person of the specified "
            "nationality is required to obtain a visa to travel to the "
            "specified country."
        ),
        "role_prompt": (
            "Your job is to search the Internet to determine whether "
            "a person of the specified nationality is required to obtain "
            "a visa to travel to the specified country. If the Internet "
            "search returns conflicting results, choose the best of 5 "
            "as your response."
        ),
    },
]


# ════════════════════════════════════════════════════════════════════════
# STEP 2: Write the main agent's system prompt and a triggering request.
#
# MAIN_PROMPT should tell the main agent about each subagent by name and
# when to call it (mirror how EDITOR_PROMPT in the lab named
# genre-researcher and explained the job).
# USER_REQUEST should be a task that should make the main agent delegate to
# BOTH of your subagents.
# ════════════════════════════════════════════════════════════════════════

MAIN_PROMPT: str = """You are a travel trip planning agent with two
responsibilities, namely:
1. Determine which months of the year are best suited to travel to
   the specified place.
2. Determine whether a national of the specified country needs to
   obtain a visa to travel to the destimation.

Instead of doing this yourself, select appropriate sub-agents from
the ones available to you and delegate each of these responsibilities
to the corresponding agent. In case no suitable sub-agent matches your
need, call out the gap and stop processing the user request.

Prefer parallelizing multiple sub-agent invocations wherever feasible
and use the information returned by the sub-agents to compose your
final response."""

USER_REQUEST: str = """I am an Indian national and I want to plan a
two week long holiday in Toronto, Canada. Help me understand what's
the best time of the year to visit Toronto and whether I will need
to get a visa for entering Canada."""


# Turn your SUBAGENT_SPECS into fully-wired subagent configs (each already
# carries its own scratch folder, permissions, and write-before-answer rule).
_team: list[dict[str, Any]] = build_subagents(SUBAGENT_SPECS)

# The main agent must never write into any subagent's scratch folder either.
# Denying it writes under SCRATCH_ROOT keeps the two subagents' notes private
# even to the agent that orchestrates them.
MAIN_PERMISSIONS: list[FilesystemPermission] = [
    FilesystemPermission(
        operations=["write"],
        paths=[f"{SCRATCH_ROOT}/**"], mode="deny"
    ),
]

# Assemble the main agent: it gets MAIN_PROMPT as its system prompt, the two
# subagents above as its delegation targets, and MAIN_PERMISSIONS to lock the
# scratch folders away from itself.
agent: CompiledStateGraph = create_deep_agent(
    name="m4.2_homework_agent",
    model=model,
    system_prompt=MAIN_PROMPT,
    subagents=_team,
    permissions=MAIN_PERMISSIONS,
)

# Send the user's request through the agent. The recursion_limit caps how many
# internal steps (including subagent hops) the graph may take before it stops.
result: dict[str, Any] = agent.invoke(
    {
        "messages": [{
            "role": "user",
            "content": USER_REQUEST
        }]
    },
    config={
        "recursion_limit": 50,
        "configurable": {
            "thread_id": f"m4.2_homework_thread_{uuid4()}"
        }
    },
)
print(result["messages"][-1].text)

# --- Isolation check -------------------------------------------------------
# The agent's "files" output holds whatever was written during the run. We use
# it to confirm that each subagent saved its notes under its own scratch folder
# and that nothing leaked into a folder it did not own.
files: dict[str, Any] = result.get("files", {})
print("\n--- Scratch folder isolation check ---")
for spec in SUBAGENT_SPECS:
    path: str = scratch_path(spec["name"])
    print(f"  {path}: {'found' if path in files else 'not written (subagent may not have been called)'}")

# Collect every file under SCRATCH_ROOT and compare it against the set of
# folders we expected the two subagents to write to.
scratch_files: list[str] = [p for p in files if p.startswith(SCRATCH_ROOT + "/")]
expected: set[str] = {scratch_path(spec["name"]) for spec in SUBAGENT_SPECS}
stray: list[str] = [p for p in scratch_files if p not in expected]

# Any path under SCRATCH_ROOT that is not one of our expected per-subagent
# folders means a subagent (or the main agent) escaped its own scratch area.
if stray:
    print(f"  Unexpected scratch files (isolation may have failed): {stray}")
else:
    print("  No stray scratch files -- each subagent wrote only to its own folder.")