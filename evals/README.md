# Eval harness

Answers one question: **did that change help, or did I get lucky once?**

On a 9B model a single run tells you almost nothing — the same prompt can
produce a clean answer and a tool-call loop ten minutes apart. So every fixture
runs several times and the report shows the spread, not one sample.

```bash
# see what is in the suite
python -m evals.run --list

# the offline suite, three times each
python -m evals.run --exclude network --reps 3

# one capability, many repetitions, in parallel
python -m evals.run --fixtures read_pdf,image_colours --reps 8 --concurrency 4

# everything, including the ones that need Tavily
python -m evals.run --reps 3 --concurrency 4
```

Results land in `.evals/<timestamp>/`: `results.jsonl` (one line per run),
`report.md`, and each run's own `outputs/` and `uploads/` directories so you can
open what the agent actually produced.

## What it measures

| | |
| --- | --- |
| Speed | Time to first token, total wall clock, time inside model calls vs tool calls vs graph overhead |
| Cost | Input and output tokens per run, output tokens/sec against real model time |
| Tool use | Which tools ran, which failed, which the repeated-call guard blocked, what arguments were actually sent |
| Approvals | What was requested, what decision it got, what reached the disk |
| Termination | Finished normally, hit a guardrail, timed out, or never stopped |
| Artifacts | The real `.docx` opened and inspected — headings, tables, row counts, text |

## Writing a fixture

A fixture is a task plus what must be true afterwards. It lives in
`fixtures.py`:

```python
Fixture(
    name="read_pdf",
    what="Reads an attached PDF and reports what is actually in it.",
    turns=["Which division was this prepared for?"],
    attachments=[assets.PDF_NAME],
    tags=("attachments",),
    checks=[
        calls_tool("read_document"),
        answer_contains(assets.PDF_FACT),
        no_blocked_calls(),
        terminates_cleanly(),
    ],
)
```

Prefer checks that read the world — the file on disk, the arguments the model
actually sent — over checks that read the model's account of what it did. The
failures worth catching are exactly the ones where those two disagree. The
first bug in this project was an agent insisting it had written a file that did
not exist; `docx_contains()` opens the document, and `no_file_written()` looks
at the directory.

Anything genuinely subjective goes in `judged()`. It is never scored — it
appears in the report with the artifact attached so you can judge it yourself.
That is deliberate: an unscored question is honest, a guessed score is not.

## Two things that will bite you

**`calls_tool` only sees calls that ran.** A tool stopped at the approval step
never reaches the tool wrapper, so it does not appear in `tool_calls` at all.
Use `requests_tool()` when what matters is that the agent *reached* for a tool.
Without it, `no_file_written()` also passes for a run where the agent never
tried — which is a pass for the wrong reason.

**Concurrency makes latency an upper bound.** `--concurrency 4` is much faster
against a batching server, but the runs queue against each other, so per-run
wall clock is inflated. Measure timing with `--concurrency 1`; use concurrency
for pass rates.

## Isolation

Each run gets its own `outputs/` and `uploads/` under the results directory,
via context variables that `asyncio` copies per task. Concurrent runs writing
the same filename do not collide, and **the harness never touches the real
`uploads/` or `outputs/`** — which matters, because an earlier version of this
project destroyed real uploaded files during testing.

## Testing the harness itself

```bash
python -m evals.selftest
```

Every check is run twice — against a result it should accept and one it should
reject — and has to get both right. A check that always returns `True` makes the
suite green and tells you nothing, and that failure mode is invisible without
this.

## No GPU? There is a stub

`fakemodel.py` is an OpenAI-compatible test double that plays the part of a
cooperative model: streamed text, streamed tool calls, usage chunks, and a log
of what it was sent. It is how the harness was verified without access to the
real endpoint, and how the image pipeline was proved to be sending images at
all.

```bash
python -m evals.fakemodel --port 8099
MODEL_BASE_URL=http://127.0.0.1:8099/v1 MODEL_NAME=fake python -m evals.run --exclude network
```

It is a test double, not a model. A green suite against the stub means the
harness works — it says nothing about the agent.
