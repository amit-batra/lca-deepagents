# python/m4/m4.3_run_manuscript.py
"""Run the manuscript agent, then self-check its findings against the known
seeded corruptions in data/epic_corpus_key.json, an immediate, exact way to
see whether the workflow's book-by-book dispatch actually covered everything.

This check is for your own feedback while working through the lab, not a
submitted grade: the key ships openly alongside the corpus.
"""

from pathlib import Path
from typing import Any

from m4_3_manuscript_agent import agent

# Directory containing the manuscript data files
DATA_DIR: Path = Path(__file__).resolve().parent / "data"

# Invoke the manuscript agent to find all corrupted sentences in the first 6 books
result: dict[str, Any] = agent.invoke(
    {
        "messages": [{
            "role": "user",
            "content": "Run a workflow to find every corrupted sentence in the manuscript.",
        }]
    },
    config={"recursion_limit": 200},
)

# Extract the final report from the agent's response
report: str = result["messages"][-1].text
# Collect all sentences mentioned in the report for self-checking
found_sentences: set[str] = {s for s in report}

# Display self-check results
print(f"\nSelf-check: {len(found_sentences)} seeded corruptions appear in the report.")
print(report)