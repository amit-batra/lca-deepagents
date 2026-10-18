from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends.protocol import (
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents_docker import DockerSandbox
from langchain.agents.middleware.types import (
    AgentState,
    InputAgentState,
    OutputAgentState,
)
from langgraph.graph.state import CompiledStateGraph

from models import model

DB_PATH:str = Path(__file__).resolve().parent / "chinook.db"

# Create a Docker Sandbox with a 10-minutes timeout
with DockerSandbox(image="python:3.14", timeout=600) as backend:

    # Upload chinook.db to the Docker sandbox
    with open(DB_PATH, "rb") as f:
        upload_results: list[FileUploadResponse] = backend.upload_files([("/chinook.db", f.read())])

    # Ensure that the file upload was successful
    for upload_result in upload_results:
        if upload_result.error:
            raise RuntimeError(f"Failed to upload {upload_result.path}: {upload_result.error}")

    # Create a Deep Agent passing it a reference to the Docker Sandbox
    # along with the System Prompt.
    agent: CompiledStateGraph[
        AgentState[Any], Any, InputAgentState, OutputAgentState[Any]
    ] = create_deep_agent(
        name="sales_agent_docker",
        model=model,
        backend=backend,
        system_prompt=(
            "You are a sales data analyst with access to the Chinook music store database "
            "at chinook.db. Use `sqlite3` and `matplotlib` to answer questions with charts. "
            "Before creating and running any Python scripts in the sandbox, provision a "
            "Python virtual environment. Install any Python dependencies that you need "
            "within the virtual environment, and use the same virtual environment to run "
            "all Python scripts. Python `sqlite3` dependency is available as a part of "
            "stdlib, so do not install it. When asked to produce a chart, write a Python "
            "script, execute it, and confirm that the output file was created. Finally "
            "summarize all the steps that you performed, including listing the filenames "
            "and the contents of all the Python scripts that you wrote."
        ),
    )

    # Invoke the Deep Agent with User Instructions and print the final output
    result: dict[str, Any] = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": (
                    "Query the Chinook database from `chinook.db` to get total revenue "
                    "by genre. Create a clean donut chart showing each genre's share "
                    "of total sales revenue. Group any genres that individually "
                    "account for less than 3% of total revenue into a single 'Other' "
                    "slice. Label each slice with the genre name and percentage. "
                    "Use a visually distinct color palette, leave a white center hole, "
                    "and make sure no labels overlap with each other or with the title. "
                    "Add enough top padding so the title is fully visible. "
                    "Save the chart to `genre_revenue.png`."
                ),
            }
        ]
    })
    print(result["messages"][-1].content)

    # Download the file produced by the Deep Agent
    out_path:Path = Path(__file__).parent / "genre_revenue.png"
    download_results: list[FileDownloadResponse] = backend.download_files(paths=["/genre_revenue.png"])
    for download_result in download_results:
        if download_result.error:
            raise RuntimeError(f"Failed to download {out_path!r}")

        png_bytes: bytes | None = download_result.content
        if png_bytes is not None:
            out_path.write_bytes(png_bytes)
            print(f"Chart saved to {out_path!r}")
        else:
            print("The agent produced an empty file. Nothing saved!")

        break
