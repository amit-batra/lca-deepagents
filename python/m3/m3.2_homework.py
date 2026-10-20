# python/m3/m3.2_homework.py
"""M3.2 Homework: Bundle a Reference File Into Your Skill.

THE IDEA
The lab's two skills (qualify-lead and draft-pitch) are each a single flat
SKILL.md file with everything inline. But the lesson also covered a third
stage of progressive disclosure: a skill can point to supporting files (a
reference doc, a template, a script) that live alongside SKILL.md and that
the agent only reads when it actually needs them, instead of stuffing
everything into the system prompt up front. This homework asks you to write
a skill for a topic or workflow YOU pick (not sales) that bundles a SECOND
file with details the agent needs but that aren't in SKILL.md itself, then
confirm from the trace that the agent actually called `read_file` on that
second file before answering, rather than guessing.

WHAT YOU FILL IN
  TODO 1: write your own SKILL.md content. It must instruct the agent to
    read a `reference.md` file (in the same skill directory) for specific
    details it needs, and must NOT restate those details inline. The `name`
    field in your frontmatter must exactly match SKILL_NAME below.
  TODO 2: write reference.md's content: the specific facts, numbers, or
    template your skill's instructions point to and depend on.
  TODO 3: write a system prompt and a user question that should activate
    your skill.

RUN
  cd python
  uv run ./m3/m3.2_homework.py
"""

import tempfile, uuid
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend

from models import model

# This name becomes the skill's directory name. It must exactly match the
# `name:` field you write in the frontmatter inside build_skill_md() below.
SKILL_NAME: str = "is-product-release-worthy"
REFERENCE_PATH: str = f"/skills/{SKILL_NAME}/reference.md"


# ════════════════════════════════════════════════════════════════════════
# STEP 1: Write your own SKILL.md content.
#
# Requirements:
#   - YAML frontmatter with `name` (must equal SKILL_NAME above) and
#     `description` (a specific sentence describing WHEN to use this skill).
#   - Steps that tell the agent to open `reference.md` (in this same skill
#     directory) for the specific details it needs to do the task well.
#   - Do NOT put those details in SKILL.md itself; if the agent could do
#     the task correctly without ever reading reference.md, this doesn't
#     exercise progressive disclosure.
#
# Example shape (delete this and write your own):
#   return """---
#   name: your-skill-name
#   description: Use when the user wants to ...
#   ---
#
#   # Your Skill Title
#
#   **Step 1: ...**: ...
#   **Step 2: ...**: before proceeding, read reference.md in this skill's
#     directory for the exact ... to use. Do not guess these.
#   """
# ════════════════════════════════════════════════════════════════════════

def build_skill_md() -> str:
  return f"""---
name: {SKILL_NAME}
description: Use this skill to determine whether the specified product build can be released to the public
---

# Determine the Release-Worthiness of a Product Build

Not every product build can released to the public. In order for a product build to be
considered release-worthy, it must meet certain requirements. You should declare the product
build release-worthy only if it meets ALL the requirements laid out.

If there's at least one (or more) requirement(s) that the product build does not meet, then
declare the product to be not ready for public release.

For both the scenarios (build meets or fails the release-worthiness criteria) list down the
specific criteria that the product build has passed and the ones that it has failed to meet.

The precise list of criteria that you need to evaluate for the given product build have been
listed in the file `reference.md` that you must read. Do not guess the criteria on your own and
do not invent any criteria not listed in the `reference.md` file.
"""

# ════════════════════════════════════════════════════════════════════════
# STEP 2: Write reference.md's content.
#
# This should contain the specific facts your SKILL.md pointed to and
# depends on: a rubric, a set of numbers, a template, a checklist. Specific
# enough that an answer produced without reading it would visibly differ
# from one produced with it.
# ════════════════════════════════════════════════════════════════════════

def build_reference_md() -> str:
  return """# Criteria for Determining the Release-Worthiness of a Product Build

1. The product build must pass with zero test failures.
2. The linter should not surface more than 5 warnings across the entire codebase.
3. No Jira issue with severity 1 or severity 2 should be open against this product build.
4. The file sizes in the release build must be smaller than the sizes of the corresponding files in the debug build.
5. The memory leak reporter must confirm zero memory leaks detected during the test run.
6. The performance numbers should not have degraded compared to the previous build.
7. The code coverage of the automated tests must be at least 85%.
8. The penetration testing must not reveal any unaddressed security vulnerabilities.
"""


# Write the skill to a scratch directory so it's discoverable through a
# FilesystemBackend, the same mechanism the lab uses for python/m3/skills/.
_tmp_root = Path(tempfile.mkdtemp(prefix=f"m3_2_homework_{uuid.uuid4()}"))
_skill_dir = _tmp_root / "skills" / SKILL_NAME
_skill_dir.mkdir(parents=True, exist_ok=True)
(_skill_dir / "SKILL.md").write_text(build_skill_md())
(_skill_dir / "reference.md").write_text(build_reference_md())

backend = FilesystemBackend(root_dir=str(_tmp_root), virtual_mode=True)
print(f"Skill files written to: {_skill_dir}")


# ════════════════════════════════════════════════════════════════════════
# STEP 3: Write a system prompt and a triggering question.
#
# SYSTEM_PROMPT: give the agent a persona of your choosing (a name, a
# voice, anything you want).
# USER_QUESTION: a question that should match your skill's `description`
# closely enough that the agent activates it.
# ════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a product build expert certifier."""
USER_QUESTION = """Today we have generated build versioned 1.3.5 of our flagship
software product 'Vistaris Moonshot`. We made the following observations about the
product build:
1. All the automated tests passed.
2. The linter surfaced 3 warnings across the entire codebase.
3. Only 3 Jira issues with severity 4 are open against this product build.
4. The files in the release build and in the debug build are sized 50 MB and 150 MB respectively.
5. Two memory leaks were detected by the leak reporter during the test run.
6. The performance numbers have slightly improved over the previous build.
7. The code coverage of the automated tests was 79%.
8. The penetration testing revealed 2 unaddressed security vulnerabilities.

Please determine whether we can release this product build to the public. If not, then why?
"""

agent = create_deep_agent(
    name="m3.2_homework_agent",
    model=model,
    backend=backend,
    skills=["/skills"],
    system_prompt=SYSTEM_PROMPT,
)

result = agent.invoke({"messages": [{"role": "user", "content": USER_QUESTION}]})
print(result["messages"][-1].text)

read_calls = [
    call
    for msg in result["messages"]
    for call in getattr(msg, "tool_calls", [])
    if call["name"] == "read_file"
]
reference_was_read = any(call["args"].get("file_path") == REFERENCE_PATH for call in read_calls)
print(f"\n--- Did the agent read {REFERENCE_PATH}? {reference_was_read} ---")
if not reference_was_read:
    print(
        "It didn't. Either SKILL.md isn't clearly telling it to, or the "
        "task is answerable without the details in reference.md."
    )
