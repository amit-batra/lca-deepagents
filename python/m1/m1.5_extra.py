# python/m1/m1.5_homework.py
"""M1.5 Extra (not a part of the course)

THE IDEA
This script validates that the built-in file editing tools that come
with the LangChain Deep Agents framework can ebe used to create, edit
and delete files.

HIGH-LEVEL IDEA
The high level sequence of the actions that this script performs is:
0. Validate that no file by the name "out.txt" exists in the current
   working directory. If it exists, delete the same and confirm the
   deletion was successful.
1. Ask the agent to create a new file named "out.txt" with the contents
   "This is a sample file created by the Agent".
2. Validate that the file was successfully created and it has the
   expected line of text.
3. Replace the the line of text with "This is a ssample file created by
   the Deep Agent Harness".
4. Confirm that the contents of the file were successfully updated.
5. Finally, delete the file. Confirm successful deletion and exit.

RUN
  cd python
  uv run ./m1/m1.5_extra.py
"""

import os
import warnings, asyncio

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core.tools import tool

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from models import model

SYSTEM_PROMPT = (
    "You are a file creation and editing agent that uses "
    "the available tools to create, edit and delete files "
    "on the filesystem."
)

OUT_FILE_PATH = "out.txt"
CURRENT_DIR = os.getcwd()

agent = create_deep_agent(
    model=model,
    name="File_Editing_Agent",
    system_prompt=SYSTEM_PROMPT,
    backend=FilesystemBackend(root_dir=CURRENT_DIR, virtual_mode=True),
)

async def execute_all_scenarios() -> None:
    # ════════════════════════════════════════════════════════════════════════
    # STEP 0: Validate that no file by the name "out.txt" exists in the current
    # working directory. If it exists, delete the same and confirm the deletion
    # was successful.
    # ════════════════════════════════════════════════════════════════════════
    if os.path.exists(OUT_FILE_PATH):
        print(f"File {OUT_FILE_PATH} already exists. Deleting it for a clean start.")
        os.remove(OUT_FILE_PATH)
        if not os.path.exists(OUT_FILE_PATH):
            print(f"Successfully deleted existing file: {OUT_FILE_PATH}")
        else:
            print(f"Failed to delete existing file: {OUT_FILE_PATH}")

    # ════════════════════════════════════════════════════════════════════════
    # STEP 1: Ask the agent to create a new file named "out.txt" with the
    # contents: "This is a sample file created by the Agent".
    # ════════════════════════════════════════════════════════════════════════
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Please create a new file named '{OUT_FILE_PATH}' "
                        "with the contents: 'This is a sample file created by the Agent'."
                    ),
                }
            ]
        }
    )
    print(f"Agent response: {result['messages'][-1].text}")

    # ════════════════════════════════════════════════════════════════════════
    # STEP 2: Validate that the file was successfully created and it has the
    # expected line of text.
    # ════════════════════════════════════════════════════════════════════════
    print(f"Checking if the file {OUT_FILE_PATH} was created with the expected content...")
    if os.path.exists(OUT_FILE_PATH):
        with open(OUT_FILE_PATH, "r") as f:
            content = f.read().strip()
            print(f"Content of {OUT_FILE_PATH}: {content}")
            if content == "This is a sample file created by the Agent":
                print(f"File {OUT_FILE_PATH} was successfully created with the expected content.")
            else:
                print(f"File {OUT_FILE_PATH} was created but the content is not as expected.")
    else:
        print(f"File {OUT_FILE_PATH} was not created as expected.")
        return  # Exit early if the file was not created

    # ════════════════════════════════════════════════════════════════════════
    # STEP 3: Replace the the line of text with "This is a ssample file created
    # by the Deep Agent Harness".
    # ════════════════════════════════════════════════════════════════════════
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Please replace the contents of '{OUT_FILE_PATH}' with: "
                        "'This is a sample file created by the Deep Agent Harness'."
                    ),
                }
            ]
        }
    )
    print(f"Agent response: {result['messages'][-1].text}")

    # ════════════════════════════════════════════════════════════════════════
    # STEP 4: Confirm that the contents of the file were successfully updated.
    # ════════════════════════════════════════════════════════════════════════
    print(f"Checking if the file {OUT_FILE_PATH} was updated with the expected content...")
    with open(OUT_FILE_PATH, "r") as f:
        updated_content = f.read().strip()
        print(f"Updated content of {OUT_FILE_PATH}: {updated_content}")
        if updated_content == "This is a sample file created by the Deep Agent Harness":
            print(f"File {OUT_FILE_PATH} was successfully updated with the expected content.")
        else:
            print(f"File {OUT_FILE_PATH} was updated but the content is not as expected.")

    # ════════════════════════════════════════════════════════════════════════
    # STEP 5: Finally, delete the file. Confirm successful deletion and exit.
    # ════════════════════════════════════════════════════════════════════════
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"Please delete the file '{OUT_FILE_PATH}'.",
                }
            ]
        }
    )
    print(f"Agent response: {result['messages'][-1].text}")

    print(f"Checking if the file {OUT_FILE_PATH} was deleted...")
    if not os.path.exists(OUT_FILE_PATH):
        print(f"File {OUT_FILE_PATH} was successfully deleted.")
    else:
        print(f"Failed to delete the file {OUT_FILE_PATH}.")

asyncio.run(execute_all_scenarios())