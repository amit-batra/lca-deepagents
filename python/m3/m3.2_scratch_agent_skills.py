"""M3.2 — scratch agent wired with skills and two independent sales threads.

Builds a deep agent that loads reusable "skills" from the ``/skills`` directory
(via a virtual-mode filesystem backend rooted at this folder) and then exercises
it on two separate conversation threads:

1. Qualifying a sales lead.
2. Drafting a cold outreach message.

Each scenario uses its own ``thread_id`` so the two conversations stay isolated
and neither one's history leaks into the other.
"""

from pathlib import Path
from uuid import uuid4

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend

from models import model

m3_dir = Path(__file__).parent
# The backend lets the agent read/write "skills" under a virtual filesystem
# rooted at this module's folder (virtual_mode keeps paths sandboxed).
backend = FilesystemBackend(root_dir=str(m3_dir), virtual_mode=True)

agent = create_deep_agent(
    name="m3.2_scratch_agent_skills",
    model=model,
    backend=backend,
    # "/skills" points at a virtual path the agent can load reusable skills from.
    skills=["/skills"],
    system_prompt="You are a sales assistant.",
)


def run_scenario(thread_id: str, user_message: str) -> None:
    """Run the agent on a single user message and print its final reply.

    A fresh ``thread_id`` keeps each conversation isolated, so the agent's
    history from one scenario never bleeds into the next.
    """
    thread = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_message,
                }
            ]
        },
        config=thread,
    )
    print(result["messages"][-1].text)


# Scenario 1: qualify a specific lead with rich context.
run_scenario(
    thread_id=f"qualify-lead-{uuid4()}",
    user_message=(
        "Qualify this lead: Acme Corp, 200-person logistics company. I spoke with "
        "Sarah Chen, VP of Sales: she's the decision maker. They have $45k budgeted "
        "for CRM this year. Main pain: deals are slipping through the cracks due to "
        "poor pipeline visibility. They want a solution live by the end of Q3."
    ),
)

# A visual separator between the two scenario outputs.
print(f"\n{'─' * 50}\n")

# Scenario 2: draft a cold outreach on a separate, isolated thread.
run_scenario(
    thread_id=f"draft-pitch-{uuid4()}",
    user_message="Write a cold outreach message for a prospect at a mid-size logistics company.",
)