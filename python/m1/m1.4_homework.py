# python/m1/m1.4_homework.py
"""M1.4 Homework: Scope the Agent to One Domain.

THE IDEA
Lab 1 had you swap personas (pirate, cowboy, Shakespeare) on top of the
butler system prompt, which only changes the agent's voice. This homework
uses `system_prompt` differently: instead of persona, write a constraint
that scopes the agent to a single domain of your choosing (cooking,
houseplants, retro video games, personal finance, etc.) and
instructs it to refuse or redirect anything outside that domain.

There's no single correct domain here, that's the point. What matters is
that the refusal actually holds, not just that the agent sounds like
something.

WHAT YOU FILL IN
  TODO 1: write your own SYSTEM_PROMPT string that scopes the agent to a
    single domain of your choosing and tells it to refuse or redirect
    anything outside that domain (no persona/voice requirement here,
    just the scope + refusal instruction).
  TODO 2: invoke the agent with two test prompts, one inside your domain
    and one clearly outside it, and print both responses so you can see
    whether the refusal actually held.

RUN
  cd python
  uv run ./m1/m1.4_homework.py
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from deepagents import create_deep_agent
from models import model


# ════════════════════════════════════════════════════════════════════════
# TODO 1: Write a system prompt that scopes the agent to one domain and
# tells it to refuse or redirect anything outside that domain.
#
# Requirements:
#   - Pick one domain (a subject, not a persona).
#   - State clearly what the agent should do when asked about something
#     outside that domain (e.g. say it can't help, redirect back to the
#     domain, ask a domain-relevant follow-up).
#
# Example shape (delete this and write your own):
#   SYSTEM_PROMPT = (
#       "You only answer questions about ... . If asked about anything "
#       "else, ... ."
#   )
# ════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are an expert in cooking. You only answer questions about "
    "cooking techniques, recipes, and food science. If asked "
    "about anything else, politely inform the user that you can only provide "
    "information on cooking and suggest they ask a question related to "
    "that domain."
)


agent = create_deep_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    name="Homework_Agent",
)


# ════════════════════════════════════════════════════════════════════════
# TODO 2: Run one in-domain prompt and one out-of-domain prompt through
# the agent and print both responses, so you can check whether the
# refusal actually held.
# ════════════════════════════════════════════════════════════════════════

def run_test_prompts():
    in_domain_prompt = "What are some tips for making a perfect soufflé?"
    out_of_domain_prompt = "Can you explain the theory of relativity?"

    in_domain_result = agent.invoke({"messages": [{"role": "user", "content": in_domain_prompt}]})
    out_of_domain_result = agent.invoke({"messages": [{"role": "user", "content": out_of_domain_prompt}]})

    print("In-Domain Prompt Response:")
    print(in_domain_result["messages"][-1].text)
    print("\nOut-of-Domain Prompt Response:")
    print(out_of_domain_result["messages"][-1].text)

run_test_prompts()
