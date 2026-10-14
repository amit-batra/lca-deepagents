# python/m1/m1.5_homework.py
"""M1.5 Homework: Build Your Own Custom Tool.

THE IDEA
The lab wired up one custom tool (read_sql) for one fixed topic (the
Chinook music database). This homework asks you to do the same thing for
a topic YOU pick: something you actually know or care about (a game, a
sport, a show, your favorite band's discography, local trivia, whatever).
There's no single correct topic or persona here, that's the point. Two
students doing this homework could end up with two completely different
tools and agents.

WHAT YOU FILL IN
  TODO 1: write your own custom tool with the @tool decorator. Pick any
    topic, store a small lookup (a dict is fine, no API needed) of facts
    about it, and return one back based on the argument the model passes.
  TODO 2: write a system prompt that gives the agent a persona of your
    choosing and tells it to use your tool before answering.

RUN
  cd python
  uv run ./m1/m1.5_homework.py
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core.tools import tool

from deepagents import create_deep_agent
from models import model


# ════════════════════════════════════════════════════════════════════════
# TODO 1: Define your own custom tool(s).
#
# Requirements:
#   - Keep the @tool decorator.
#   - Give it a real docstring: one sentence the model will read to decide
#     when to call this tool.
#   - Have it take at least one argument and return a string.
#   - The lookup data can just live in this file (a dict, a list, whatever
#     fits your topic). No external API or key needed.
#
# Example shape (delete this and write your own):
#   @tool
#   def lookup_something(query: str) -> str:
#       """One sentence describing what this returns and when to call it."""
#       ...
# ════════════════════════════════════════════════════════════════════════

@tool
def add_numbers(num1: float, num2: float) -> float:
    """Add two numbers and return the result."""
    return num1 + num2

@tool
def multiply_numbers(num1: float, num2: float) -> float:
    """Multiply two numbers and return the result."""
    return num1 * num2

@tool
def subtract_numbers(num1: float, num2: float) -> float:
    """Subtract the second number from the first and return the result."""
    return num1 - num2

@tool
def divide_numbers(num1: float, num2: float) -> float:
    """Divide the first number by the second and return the result."""
    if num2 == 0:
        return "Error: Division by zero is not allowed."
    return num1 / num2

# ════════════════════════════════════════════════════════════════════════
# TODO 2: Write a system prompt for your agent.
#
# Give it a persona (a name, a voice, a personality, anything you want)
# and tell it to call your_custom_tool (rename it if you like) before
# answering, the same way the lab's SYSTEM_PROMPT pointed the agent at
# read_sql.
# ════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are an expert at solving mathematical calculations. "
    "When asked to perform a calculation, use the available tools to do so. "
    "In your final answer, show the calculation steps and the result."
)

agent = create_deep_agent(
    model=model,
    name="Homework_Agent",
    tools=[add_numbers, multiply_numbers, subtract_numbers, divide_numbers],
    system_prompt=SYSTEM_PROMPT,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is (100 + 50) / (10 * (10 - 5))?"}]}
)

print(result["messages"][-1].text)
