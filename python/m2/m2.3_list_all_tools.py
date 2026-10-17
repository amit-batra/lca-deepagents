from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend

from models import model

# Instantiate the Deep Agent with a LocalShellBackend
agent = create_deep_agent(
    model=model,
    backend=LocalShellBackend()
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": (
                "List down all tools that you have access to, with their names, "
                "decsriptions and the input parameters. Also call out whether "
                "each input parameter is required or optional."
            )
        }
    ]
})

# This is expected to contain the `execute` tool
print(result["messages"][-1].content)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": (
                "Use the `execute` tool to list all files in the working directory."
            )
        }
    ]
})

print(result["messages"][-1].content)
