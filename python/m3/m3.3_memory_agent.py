from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.backends.utils import create_file_data
from langgraph.store.memory import InMemoryStore
from uuid import uuid4

from models import model

store = InMemoryStore()
memory_path = "/memories/AGENTS.md"
store_memory_path = "/AGENTS.md"
demo_context = {"workspace_id": "acme", "user_id": "u_alex"}


def namespace_from_context(context):
    return (
        "memory",
        context["workspace_id"],
        context["user_id"],
    )


def memory_namespace(runtime):
    return namespace_from_context(runtime.context)


store.put(
    namespace_from_context(demo_context),
    store_memory_path,
    create_file_data("""\
# Project Guidelines

## Code Style
- All functions must have type annotations
- Use f-strings for string formatting
- Maximum line length is 88 characters
- Use `pathlib.Path` for file operations, not `os.path`

## Workflow
- Run tests with: `uv run pytest`
- The CI pipeline runs on every push to `main`
- Open a draft PR early so reviewers can follow along
"""),
)

agent = create_deep_agent(
    name="m3.3_memory_agent",
    model=model,
    backend=CompositeBackend(
        default=StateBackend(),
        routes={"/memories/": StoreBackend(namespace=memory_namespace)},
    ),
    store=store,
    memory=[memory_path],
    system_prompt="You are a helpful coding assistant for this project.",
)

thread_id = f"m3.3_memory_agent-thread-{uuid4()}"
config = {"configurable": {"thread_id": thread_id}}

# First invoke: agent answers using memory content
result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What tool should I use for file paths in this project?"}
        ]
    },
    config=config,
    context=demo_context,
)
print("--- Question 1 ---")
print(result["messages"][-1].text)

# Second invoke: agent writes to memory
result2 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Remember: the team switched to ruff for linting.",
            }
        ]
    },
    config=config,
    context=demo_context,
)
print("\n--- Question 2 ---")
print(result2["messages"][-1].text)

print("\n--- AGENTS.md after write ---")
stored_memory = store.get(namespace_from_context(demo_context), store_memory_path)
print(stored_memory.value["content"])
