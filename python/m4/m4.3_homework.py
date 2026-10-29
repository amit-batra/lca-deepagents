# python/m4/m4.3_homework.py
"""M4.3 Homework: Write Your Own Dynamic Subagent Workflow.

THE IDEA
The lab gave the main agent a 2MB manuscript split into labeled books and
had it write a "workflow" that dispatched one book-scanner subagent per
book, so the full corpus never entered the main model's own context. This
homework asks you to do the same shape of thing on a scenario of your own
choosing: a synthetic corpus of your own, split into your own labeled
sections, and a subagent that scans each section for something other than
anachronisms.

A few starting points, if you want one:
  - A Sherlock Holmes story, split by chapter, scanned for clues the
    detective mentions but never actually explains.
  - The script of Bee Movie or Shrek, split by scene, scanned for lines
    that don't match the character who supposedly says them.
  - Your own corrupted classic, like the lab's, but seeded with a
    different kind of error: wrong units, swapped character names,
    continuity errors between chapters.

WHAT YOU FILL IN
  TODO 1: write your own corpus, a single string split into at least 5
    labeled sections using a consistent header format (like the lab's
    "=== EPIC BOOK N ===").
  TODO 2: write the section-scanner's system prompt (what should it flag in
    one section?) and the main agent's system prompt (telling it to run a
    workflow that splits your corpus and dispatches one scanner call per
    section).

RUN
  cd python
  uv run ./m4/m4.3_homework.py

NOTE
  This uses the code interpreter (langchain_quickjs), same as the lab. Make
  sure you've run `uv sync` from python/ so it's installed.
"""

from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_quickjs import CodeInterpreterMiddleware

from models import model, strong_model

DATA_DIR: Path = Path(__file__).resolve().parent / "homework_data"
DATA_DIR.mkdir(exist_ok=True)
CORPUS_PATH: Path = DATA_DIR / "my_corpus.txt"


# ════════════════════════════════════════════════════════════════════════
# STEP 1: Write your own corpus.
#
# Requirements:
#   - A single string with at least 5 labeled sections.
#   - Pick a consistent header format, e.g. "=== SECTION N ===" or
#     "=== TICKET N ===", and stick to it exactly: the main agent's prompt
#     (STEP 2) needs to describe the same format so it can split on it.
#   - Plant something worth finding in a few of the sections (an off-topic
#     sentence, a specific keyword, whatever your scanner in STEP 2 is
#     looking for) so there's something for the workflow to actually
#     surface.
#
# Example shape (delete this and write your own):
#   return """\
#   === SECTION 1 ===
#   ...
#
#   === SECTION 2 ===
#   ...
#   """
# ════════════════════════════════════════════════════════════════════════

def build_corpus() -> str:
  return """\
=== SECTION 1 ===

To Sherlock Holmes she is always the woman. I have seldom heard him mention
her under any other name. In his eyes she eclipses and predominates the whole
of her sex. It was not that he felt any emotion akin to love for Irene Adler.
All emotions, and that one particularly, were abhorrent to his cold, precise
but admirably balanced mind. He was, I take it, the most perfect reasoning
and observing machine that the world has seen, but as a lover he would have
placed himself in a false position. He never spoke of the softer passions,
save with a gibe and a sneer.

=== SECTION 2 ===

I had seen little of Holmes lately. My marriage had drifted us away from each
other. My own complete happiness, and the home-centred interests which rise
up around the man who first finds himself master of his own establishment,
were sufficient to absorb all my attention, while Holmes, who loathed every
form of society with his whole Bohemian soul, remained in our lodgings in
Baker Street, buried among his old books, and alternating from week to week
between cocaine and ambition, the drowsiness of the drug, and the fierce
energy of his own keen nature. He was still, as ever, deeply attracted by
the study of crime, and occupied his immense faculties and extraordinary
powers of observation in following out those clues, and clearing up those
mysteries which had been abandoned as hopeless by the official police.

=== SECTION 3 ===

He is dark, handsome, and dashing, never calls less than once a day, and
often twice. He is a Mr. Godfrey Norton, of the Inner Temple. See the
advantages of a cabman as a confidant. They had driven him home a dozen
times from Serpentine-mews, and knew all about him. When I had listened
to all they had to tell, I began to walk up and down near Briony Lodge
once more, and to think over my plan of campaign.

=== SECTION 4 ===

It was a quarter past six when we left Baker Street, and it still wanted
ten minutes to the hour when we found ourselves in Serpentine Avenue. It
was already dusk, and the lamps were just being lighted as we paced up
and down in front of Briony Lodge, waiting for the coming of its occupant.
The house was just such as I had pictured it from Sherlock Holmes’ succinct
description, but the locality appeared to be less private than I expected.

=== SECTION 5 ===

“You did, Doctor, but none the less you must come round to my view, for
otherwise I shall keep on piling fact upon fact on you until your reason
breaks down under them and acknowledges me to be right. Now, Mr. Jabez
Wilson here has been good enough to call upon me this morning, and to
begin a narrative which promises to be one of the most singular which I
have listened to for some time."""

CORPUS_PATH.write_text(build_corpus())


# ════════════════════════════════════════════════════════════════════════
# STEP 2: Write the scanner and main agent prompts.
#
# Return (scanner_prompt, main_prompt):
#   - scanner_prompt: what the section-scanner subagent should look for in
#     ONE section it's handed, and what it should return.
#   - main_prompt: tells the main agent about the corpus file, the header
#     format from TODO 1, and to run a WORKFLOW that splits the corpus and
#     dispatches one scanner call per section (the word "workflow" is what
#     triggers code-based dispatch, see the lesson).
# ════════════════════════════════════════════════════════════════════════

def build_prompts() -> tuple[str, str]:
  SCANNER_PROMPT: str ="""\
You are a text scanner who is an expert at finding all proper nouns in the
given text section. Go over the given text and make a note of every proper
noun that you come across. Your output should be in pure JSON format that
has an array of all proper nouns after removing duplicates. It should look
like this:
```json
[ "proper noun 1", "proper noun 2", ... "proper noun N" ]
```
If no proper nouns exist in the text, then return an empty JSON array instead.
"""

  MAIN_PROMPT: str ="""\
You are an orchestrator whose job is to divide the given work into independent
tasks and delegate the same to the sub-agents via a Javascript code block.

You will be given a long piece of text that is divided into several sections.
The indicator for every new section looks like "=== SECTION 1 ===" or
"=== SECTION 2 ===" and so on. You need to divide the given text into the
individual sections, then delegate the parsing of each section to a separate
sub-agent. Each sub-agent will return a JSON array of non-duplicate
proper nouns that it found. You need to do the following:

1. Individually print each sub-agent's list of proper nouns.
2. Combine the responses from all sub-agents into a single list of proper
   nouns, eliminate the duplicates and then print the final list.
"""

  return SCANNER_PROMPT, MAIN_PROMPT

SCANNER_PROMPT, MAIN_PROMPT = build_prompts()

section_scanner: dict[str, Any] = {
  "name": "section-scanner",
  "description": (
    "Scan the given text for proper nouns and return the same after "
    "de-duplication as a JSON array."
  ),
  "system_prompt": SCANNER_PROMPT,
  "model": model,
}

agent = create_deep_agent(
  name="m4.3_homework_agent",
  model=strong_model,
  middleware=[CodeInterpreterMiddleware()],
  system_prompt=MAIN_PROMPT,
  subagents=[section_scanner],
  backend=FilesystemBackend(root_dir=DATA_DIR, virtual_mode=True),
)

result = agent.invoke(
  {
    "messages": [
      {
        "role": "user",
        "content": (
          "Run a workflow to scan every section of my_corpus.txt "
          "and report what you find."
        )
      }
    ]
  },
  config={"recursion_limit": 100},
)
print(result["messages"][-1].text)