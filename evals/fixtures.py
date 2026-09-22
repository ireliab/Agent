"""The eval suite.

A fixture is a task plus what must be true afterwards. Prefer checks that read
the world - the file on disk, the tool arguments the model actually sent - over
checks that read the model's description of what it did, because the failures
worth catching are exactly the ones where those two disagree.

Select subsets by tag, e.g. `--tags core,docs` or `--exclude network`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evals import assets
from evals.checks import (
    Check,
    answer_contains,
    answer_excludes,
    answer_matches,
    answer_not_empty,
    answer_order,
    answer_shorter_than,
    calls_tool,
    docx_contains,
    file_named,
    file_written,
    judged,
    max_model_calls,
    never_calls_tool,
    no_blocked_calls,
    no_failed_calls,
    no_file_named,
    no_file_written,
    requests_tool,
    terminates_cleanly,
)

DEFAULT_TIMEOUT_S = 300.0


@dataclass
class Fixture:
    name: str
    turns: list[str]
    checks: list[Check]
    what: str = ""  # one line: what this fixture is actually testing
    attachments: list[str] = field(default_factory=list)
    approval: dict[str, Any] | None = None
    tags: tuple[str, ...] = ()
    timeout_s: float = DEFAULT_TIMEOUT_S
    recursion_limit: int | None = None
    max_approval_rounds: int = 4
    asset_dir: Path = assets.ASSET_DIR

    @property
    def prompt(self) -> str:
        return self.turns[0]


def suite() -> list[Fixture]:
    return [
        # -- baseline ----------------------------------------------------
        Fixture(
            name="smoke_arithmetic",
            what="Baseline latency and whether a trivial question stays trivial.",
            turns=["What is 2+2? Reply with just the number."],
            tags=("core", "fast"),
            checks=[
                answer_contains("4"),
                answer_shorter_than(40),
                max_model_calls(2),
                terminates_cleanly(),
                no_failed_calls(),
            ],
        ),
        Fixture(
            name="instruction_brevity",
            what="Does it obey a format instruction, or pad the answer out?",
            turns=["What is the capital of France? Answer with exactly one word."],
            tags=("core", "fast"),
            checks=[
                answer_contains("Paris"),
                answer_shorter_than(25),
                max_model_calls(2),
                terminates_cleanly(),
            ],
        ),
        Fixture(
            name="multi_turn_memory",
            what="Does it carry facts across turns in one conversation?",
            turns=[
                "My project is called Halcyon and it ships in March. Just acknowledge.",
                "What is my project called, and when does it ship?",
            ],
            tags=("core", "fast"),
            checks=[
                answer_contains("Halcyon", "March"),
                max_model_calls(4),
                terminates_cleanly(),
            ],
        ),
        # -- documents out -----------------------------------------------
        Fixture(
            name="docx_report",
            what="Writes a real .docx with real structure, not a claim that it did.",
            turns=[
                "Write a Word document called market-brief.docx about renewable "
                "energy. Give it a title and exactly three sections, each with a "
                "heading and a short paragraph."
            ],
            approval={"decision": "approve"},
            tags=("docs", "approval"),
            checks=[
                calls_tool("write_docx"),
                file_written(".docx"),
                docx_contains(headings=3),
                no_blocked_calls(),
                max_model_calls(8),
                terminates_cleanly(),
                judged("Is the document coherent and on-topic, or filler?"),
            ],
        ),
        Fixture(
            name="docx_table",
            what="Markdown table survives into a real Word table with the right numbers.",
            turns=[
                "Create a Word document called regions.docx containing a table of "
                "these four regions and their units: North 2400, South 3100, "
                "East 1850, West 4350. The table needs a header row."
            ],
            approval={"decision": "approve"},
            tags=("docs", "approval"),
            checks=[
                calls_tool("write_docx"),
                file_written(".docx"),
                docx_contains("North", "4350", tables=1, table_rows=5),
                terminates_cleanly(),
            ],
        ),
        Fixture(
            name="no_phantom_file",
            what=(
                "The original bug: claiming a file was written when it was not. "
                "Here the write is refused, so any claim of success is false."
            ),
            turns=["Save a file called notes.txt with a short summary of the water cycle."],
            approval={"decision": "reject", "message": "Do not save anything. Explain instead."},
            tags=("docs", "approval", "regression"),
            checks=[
                # Without this, the fixture would also pass if the agent simply
                # never tried to save anything.
                requests_tool("write_text_file", decision="reject"),
                no_file_written(),
                answer_not_empty(),
                answer_excludes("saved to", "have saved", "file is ready"),
                terminates_cleanly(),
                judged("Does it acknowledge the refusal rather than pretending it saved?"),
            ],
        ),
        # -- approvals ---------------------------------------------------
        Fixture(
            name="approval_edit",
            what="An edited argument is what reaches the tool, not the model's original.",
            turns=[
                "Write a Word document called draft.docx with a one-paragraph "
                "summary of photosynthesis."
            ],
            approval={"decision": "edit", "edit_args": {"filename": "renamed-by-eval.docx"}},
            tags=("approval",),
            checks=[
                requests_tool("write_docx", decision="edit"),
                file_named("renamed-by-eval.docx"),
                # The model's own filename must NOT be what reached the disk.
                no_file_named("draft.docx"),
                answer_not_empty(),
                terminates_cleanly(),
            ],
        ),
        Fixture(
            name="approval_respond",
            what="Answering on the tool's behalf runs nothing but still continues the run.",
            turns=[
                "Write a Word document called draft.docx with a one-paragraph "
                "summary of photosynthesis."
            ],
            approval={
                "decision": "respond",
                "message": "Saved as photosynthesis-final.docx (2 pages).",
            },
            tags=("approval",),
            checks=[
                requests_tool("write_docx", decision="respond"),
                no_file_written(),
                answer_not_empty(),
                terminates_cleanly(),
                judged(
                    "The tool result was fabricated. Does the reply repeat it as "
                    "fact? That is the documented risk of the respond decision."
                ),
            ],
        ),
        # -- documents in ------------------------------------------------
        Fixture(
            name="read_pdf",
            what="Reads an attached PDF and reports what is actually in it.",
            turns=[
                "Look at the attached document. Which division was it prepared "
                "for, and how many units shipped in the quarter?"
            ],
            attachments=[assets.PDF_NAME],
            tags=("attachments",),
            checks=[
                calls_tool("read_document"),
                answer_contains(assets.PDF_FACT),
                answer_matches(r"41[,\s]?920"),
                no_blocked_calls(),
                max_model_calls(6),
                terminates_cleanly(),
            ],
        ),
        Fixture(
            name="read_xlsx_arithmetic",
            what="Reads a spreadsheet and adds it up - arithmetic is a known 9B weak spot.",
            turns=["From the attached spreadsheet, what is the total of the Units column?"],
            attachments=[assets.XLSX_NAME],
            tags=("attachments",),
            checks=[
                calls_tool("read_document"),
                answer_matches(r"11[,\s]?700"),
                max_model_calls(6),
                terminates_cleanly(),
            ],
        ),
        Fixture(
            name="cjk_filename",
            what=(
                "Regression: the model cannot retype a CJK filename and used to "
                "loop retrying the same broken name until the run budget died."
            ),
            turns=["According to the attached document, who was the primary client this quarter?"],
            attachments=[assets.CJK_NAME],
            tags=("attachments", "regression"),
            checks=[
                calls_tool("read_document"),
                answer_contains(assets.CJK_FACT),
                no_blocked_calls(),
                max_model_calls(6),
                terminates_cleanly(),
            ],
        ),
        Fixture(
            name="image_colours",
            what="Vision actually works - the pipeline used to never send images at all.",
            turns=["What colours are in this image, listed from top to bottom?"],
            attachments=[assets.IMAGE_NAME],
            tags=("attachments", "vision"),
            checks=[
                answer_contains(*assets.IMAGE_COLOURS),
                answer_order(*assets.IMAGE_COLOURS),
                max_model_calls(3),
                terminates_cleanly(),
            ],
        ),
        # -- failure behaviour -------------------------------------------
        Fixture(
            name="missing_attachment",
            what=(
                "Asked about a file that does not exist. Should say so quickly "
                "rather than loop or invent numbers."
            ),
            turns=[
                "Read the attached file budget-2027.xlsx and tell me the total budget."
            ],
            tags=("core", "robustness"),
            checks=[
                terminates_cleanly(),
                max_model_calls(8),
                answer_not_empty(),
                no_blocked_calls(),
                judged("Does it say the file is missing, or invent a total?"),
            ],
        ),
        Fixture(
            name="scratch_vs_real_file",
            what=(
                "write_file goes to a scratch workspace the user cannot open. The "
                "system prompt forbids announcing it as a delivered file."
            ),
            turns=[
                "Use your write_file tool to put a two-line poem in scratch.txt, "
                "then tell me whether I can open that file."
            ],
            tags=("core", "robustness"),
            checks=[
                never_calls_tool("write_docx"),
                answer_not_empty(),
                terminates_cleanly(),
                judged("Does it correctly say the scratch file is NOT openable by the user?"),
            ],
        ),
        # -- network (excluded with --exclude network) --------------------
        Fixture(
            name="search_fact",
            what="A single search with a verifiable answer.",
            turns=[
                "Search the internet: who won the 2024 Nobel Prize in Literature? "
                "Answer with just the name."
            ],
            tags=("network",),
            checks=[
                calls_tool("internet_search"),
                answer_contains("Han Kang"),
                max_model_calls(6),
                terminates_cleanly(),
            ],
        ),
        Fixture(
            name="research_delegated",
            what=(
                "Roadmap open question: does delegating to a subagent pay for "
                "itself on a 9B? Compare cost against research_direct."
            ),
            turns=[
                "Research the current state of solid-state battery "
                "commercialisation using several sources, then summarise it in "
                "five bullet points."
            ],
            tags=("network", "slow", "subagents"),
            timeout_s=600.0,
            checks=[
                calls_tool("task"),
                answer_not_empty(120),
                terminates_cleanly(),
                judged("Is the summary specific and sourced, or generic filler?"),
            ],
        ),
        Fixture(
            name="research_direct",
            what="The same task with delegation forbidden, as the cost baseline.",
            turns=[
                "Research the current state of solid-state battery "
                "commercialisation using several sources, then summarise it in "
                "five bullet points. Do the searches yourself - do not delegate "
                "to a subagent."
            ],
            tags=("network", "slow", "subagents"),
            timeout_s=600.0,
            checks=[
                never_calls_tool("task"),
                answer_not_empty(120),
                terminates_cleanly(),
                judged("Is the summary specific and sourced, or generic filler?"),
            ],
        ),
    ]


def select(
    names: list[str] | None = None,
    tags: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[Fixture]:
    chosen = suite()
    if names:
        wanted = set(names)
        chosen = [f for f in chosen if f.name in wanted]
        missing = wanted - {f.name for f in chosen}
        if missing:
            raise SystemExit(f"unknown fixture(s): {', '.join(sorted(missing))}")
    if tags:
        chosen = [f for f in chosen if set(tags) & set(f.tags)]
    if exclude:
        chosen = [f for f in chosen if not (set(exclude) & set(f.tags))]
    return chosen
