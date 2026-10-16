from deepagents import create_deep_agent

from models import model

agent = create_deep_agent(model=model)

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

print(result["messages"][-1].content)
