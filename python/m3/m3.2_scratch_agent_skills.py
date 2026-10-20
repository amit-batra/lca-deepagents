from pathlib import Path
from uuid import uuid4

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend

from models import model

m3_dir = Path(__file__).parent
backend = FilesystemBackend(root_dir=str(m3_dir), virtual_mode=True)

agent = create_deep_agent(
    name="m3.2_scratch_agent_skills",
    model=model,
    backend=backend,
    skills=["/skills"],
    system_prompt="You are a sales assistant.",
)

thread_qualify_sales_lead = {"configurable": {"thread_id": f"qualify-lead-{uuid4()}"}}
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": (
                "Qualify this lead: Acme Corp, 200-person logistics company. I spoke with "
                "Sarah Chen, VP of Sales: she's the decision maker. They have $45k budgeted "
                "for CRM this year. Main pain: deals are slipping through the cracks due to "
                "poor pipeline visibility. They want a solution live by the end of Q3."
            )
        }
    ]
}, config=thread_qualify_sales_lead)
print(result["messages"][-1].text)

print(f"\n{'─' * 50}\n")

thread_draft_sales_pitch = {"configurable": {"thread_id": f"draft-pitch-{uuid4()}"}}
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Write a cold outreach message for a prospect at a mid-size logistics company."
        }
    ]
}, config=thread_draft_sales_pitch)
print(result["messages"][-1].text)