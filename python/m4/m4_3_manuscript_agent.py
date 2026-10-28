# python/m4/m4_3_manuscript_agent.py
"""Module 4 · Lesson 3: Dynamic subagents, "The Corrupted Manuscript" lab.

A handful of anachronistic sentences have been spliced into a combined,
public-domain corpus of the Iliad, Odyssey, and Aeneid (data/epic_corpus.txt;
see data/prepare_corpus.py for how it was built). The corpus is far too
large for one context window, so the main agent is given the interpreter and
a `book-scanner` subagent: it writes a workflow that reads the manuscript,
splits it on the normalized "=== EPIC BOOK N ===" headers, and dispatches one
scanner call per book, collecting whatever each one flags.

The corpus never enters the model's own context. Only `book-scanner`'s short,
distilled findings do: the interpreter holds the 2MB file, the model never
sees more than one book's worth of text (inside a subagent call) at a time.
"""

from pathlib import Path
from typing import Any

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_quickjs import CodeInterpreterMiddleware

from models import strong_model

# Directory containing the manuscript data files
DATA_DIR: Path = Path(__file__).resolve().parent / "data"

# --- The book-scanner subagent ----------------------------------------------
# System prompt for the subagent that scans individual books for anachronisms
SCANNER_PROMPT: str = """You are proofreading one book of an ancient Greek or Roman
epic (Iliad, Odyssey, or Aeneid) for anachronistic sentences that a mischievous
editor spliced in: sentences referencing anything that couldn't exist in the
Bronze Age or classical antiquity (phones, vending machines, wristwatches,
espresso machines, and the like).

You will be given the book's label and its full text.

Return ONLY a JSON array of the exact anachronistic sentences you find, quoted
verbatim from the text (no paraphrasing, no partial quotes). Return `[]` if you
find none. Most books have none; don't force a match."""

# Configuration for the book-scanner subagent
book_scanner: dict[str, Any] = {
    "name": "book-scanner",
    "description": (
        "Scan one book of the manuscript for anachronistic sentences. "
        "Delegate one book per call, passing that book's label and full text."
    ),
    "system_prompt": SCANNER_PROMPT,
    "model": strong_model,
}


# --- The main agent ----------------------------------------------------------
# System prompt for the main manuscript processing agent
MANUSCRIPT_PROMPT: str = """You have access to a manuscript at /epic_corpus.txt: a
combined public-domain translation of the Iliad, Odyssey, and Aeneid. A
mischievous editor spliced a handful of anachronistic sentences into it.

The manuscript is split into 60 books, each starting with a line formatted
exactly as "=== EPIC BOOK N ===" (e.g. "=== ILIAD BOOK 9 ===").

You need to process the first 6 books and ignore the remaining books. Ensure
that the 6th book only includes text up to the next marker (specified above)
and not the entire remaining manuscript.

Run a workflow that reads the manuscript, splits it into the first 6 books
and dispatches one book-scanner subagent call per book, batching 2 books
per Promise.all round (3 rounds total with 2 books per round) rather than
the batche size shown in your tool guidance. Never read a book's full text
into your own context; let the interpreter hold the file, and let each
subagent hold only its own book. Collect every subagent's findings
into one final report, grouped by "EPIC BOOK N", listing the anachronistic
sentence(s) found in that book (omit books with no findings)."""

# File system permissions for the agent - scanner only needs read access to the manuscript
manuscript_permissions: list[FilesystemPermission] = [
    FilesystemPermission(operations=["read"], paths=["/epic_corpus.txt"], mode="allow"),
    FilesystemPermission(operations=["read", "write"], paths=["/**"], mode="deny"),
]

# Create and configure the deep agent for manuscript processing
agent = create_deep_agent(
    name="m4.3_manuscript_agent",
    model=strong_model,
    middleware=[CodeInterpreterMiddleware(ptc=["read_file", "grep"])],
    system_prompt=MANUSCRIPT_PROMPT,
    subagents=[book_scanner],
    backend=FilesystemBackend(root_dir=DATA_DIR, virtual_mode=True),
    permissions=manuscript_permissions,
)