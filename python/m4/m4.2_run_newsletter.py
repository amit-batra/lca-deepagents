# python/m4/m4.2_run_newsletter.py
"""
Module to run the Newsletter Editor agent and capture its output to the local filesystem.

Architecture Note:
To maintain security, this script uses a 'Trusted Host' pattern. The agent executes
in a sandboxed 'StateBackend' (it has no direct disk access). This script acts
as the host that retrieves the files (the 'files' channel) from the agent's
state after it finishes and mirrors them to a local directory. This prevents
untrusted web-search content from directly writing to your machine.
"""

from pathlib import Path
from typing import Any, Union
from uuid import uuid4

from m4_2_newsletter_agent import agent

# Configuration
# The root directory where agent-produced files will be stored locally.
OUT_DIR: Path = Path(__file__).resolve().parent / "output"

def _content(fd: Union[dict[str, Any], str, list[str]]) -> str:
    """
    Extracts or normalizes the string content from a file data entry.

    Args:
        fd: A file data entry. Can be a raw string, a list of strings,
           or a dictionary containing a 'content' key.

    Returns:
        str: The extracted file content.
    """
    # Handle both FileData dicts (via 'content' key) and legacy raw strings/lists.
    body: Union[str, list[str]] = fd["content"] if isinstance(fd, dict) else fd
    return "\n".join(body) if isinstance(body, list) else body

def mirror_agent_files(files: dict[str, Any], out_dir: Path) -> None:
    """
    Validates and mirrors agent-produced files from memory to the local filesystem.

    Args:
        files: A dictionary mapping relative agent paths to their content.
        out_dir: The local directory where files should be written.

    Raises:
        SystemExit: If the required newsletter file is missing from the input.
    """
    # Validation: The core product (the newsletter) must be present.
    if "/output/newsletter.html" not in files:
        raise SystemExit("Error: The agent failed to produce the newsletter (/output/newsletter.html).")

    print(f"\nMirroring agent files to: {out_dir.resolve()}")
    out_root: Path = out_dir.resolve()

    for path in sorted(files.keys()):
        # --- Step A: Map internal paths to local filesystem ---
        # We treat '/output/' as the root of our local directory.
        # Other paths (like '/research/genre/...') preserve their folder structure.
        if path.startswith("/output/"):
            rel: str = path[len("/output/"):]
        else:
            rel: str = path.lstrip("/")

        dest: Path = (out_dir / rel).resolve()

        # --- Step B: Security Validation (Path Traversal Protection) ---
        # Since paths and contents come from the agent (which performs web searches),
        # we MUST verify the destination is within our 'out_root' to prevent
        # an agent from writing to sensitive system locations via '../' tricks.
        if dest != out_root and out_root not in dest.parents:
            print(f"  [SKIP] Security violation: {path} escapes output directory.")
            continue

        # --- Step C: Write to Disk ---
        # Ensure the directory exists and write the content as plain text.
        dest.parent.mkdir(parents=True, exist_ok=True)
        body: str = _content(files[path])
        dest.write_text(body, encoding="utf-8")
        print(f"  [OK] {path} -> {dest.relative_to(out_root)} ({len(body):,} chars)")

def main() -> None:
    """
    Executes the agent and processes its state to write files to disk.
    """
    # 1. SETUP
    # Ensure the local output directory exists before we start.
    OUT_DIR.mkdir(exist_ok=True)

    # 2. EXECUTION
    print("Invoking Newsletter Agent...")
    result: dict[str, Any] = agent.invoke(
        {"messages": [{"role": "user", "content": "Put together this week's newsletter."}]},
        config={
            "recursion_limit": 50,
            "configurable": {
                "thread_id": f"m4.2_newsletter_thread_{uuid4()}"
            }
        },
    )

    # 3. DISPLAY SUMMARY
    # Show the last message from the agent (its final coordination summary).
    print("\nAgent Summary:")
    print(result["messages"][-1].text)

    # 4. FILE EXTRACTION & MIRRORING
    # Retrieve the dictionary of files created during the agent's run.
    files: dict[str, Any] = result.get("files", {})
    mirror_agent_files(files, OUT_DIR)

    print(f"\nCompleted. Please view the results in: {OUT_DIR / 'newsletter.html'}")

if __name__ == "__main__":
    main()
