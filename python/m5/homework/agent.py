# python/m5/homework/agent.py
"""M5.2 Homework: Deploy Your Own Agent.

THE IDEA
The lab deployed a fairly bare-bones agent (no tools, no persona, just
create_deep_agent(model=model)) and you only ever talked to it through
Studio's chat panel. This homework has two parts: first, deploy an agent
with your personal touch; second, talk to it the way any other client
would, straight over the Agent Server API this lesson covers, instead of
through Studio.

WHAT YOU FILL IN
  TODO 1: write your own @tool-decorated function on a topic of your
    choosing. A plain Python dict lookup is enough, no external API or
    key required.
  TODO 2: write a system_prompt that gives the agent a persona of your
    choosing and tells it to call your tool before answering.
  Then open call_agent_api.py in this same folder for TODO 3, which talks
  to this deployed agent over HTTP instead of through Studio.

RUN
  cd python/m5/homework
  uv run langgraph dev
Then chat with your agent in the Studio window that opens, or see
call_agent_api.py to talk to it over the API instead.
"""

from langchain_core.tools import tool

from deepagents import create_deep_agent
from models import model

# STEP 1: Write a lookup_fact tool

facts: dict[str, str] = {
  "astronomy": "The moon is 384,000 km away from the earth.",
  "chemistry": "Isotopes have identical atomic numbers but differ in their atomic weights.",
  "biology": "An adult healthy human body has 206 bones.",
  "math": "The derivative of of e^x is e^x, and so is the integral.",
  "psychology": "Our minds can often trick us into believing something that may not be true.",
  "philosophy": "What is the purpose of our existence?",
  "history": "India got independence from the British in 1947.",
  "geology": "Mount Everest is more than 8 km in height.",
  "politics": "Power corrupts. Absolute power corrupts absolutely.",
}

@tool
def lookup_fact(topic: str) -> str:
  """Look up a fact for the given topic.

  Args:
    topic: The topic to look up, e.g. "astronomy" or "history".

  Returns:
    The fact associated with `topic`, or a message saying the topic is
    unknown if no fact is available for that topic.
  """
  fact: str | None = facts.get(topic.strip().lower())
  if fact:
    return fact
  else:
    return f"I don't know about {topic!r}."

# STEP 2: Define a system prompt for the agent

SYSTEM_PROMPT = """You are a facts expert. Use the `lookup_fact` tool
to look up facts on any topic that the user is interested in. Do not
generate any response from your own knowledge or from any other available
tools."""

# `langgraph.json` points at this module-level variable: "./agent.py:graph".
graph = create_deep_agent(model=model, tools=[lookup_fact], system_prompt=SYSTEM_PROMPT)
