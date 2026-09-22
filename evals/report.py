"""Turns a results file into something you can read and act on.

The per-check table is the part that matters. A fixture pass rate tells you
something broke; the check that failed tells you what. Both are reported per
fixture rather than pooled, because a suite average hides the one capability
that is completely broken behind nine that work.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _percentile(values: list[float], fraction: float) -> float | None:
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return None
    index = min(int(fraction * len(clean)), len(clean) - 1)
    return clean[index]


def _median(values: list[Any]) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.median(clean) if clean else None


def _fmt(value: Any, suffix: str = "", digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def _spread(values: list[float]) -> str:
    """Median and p90 together, because on a small model the tail is the story."""
    median, p90 = _median(values), _percentile(values, 0.9)
    if median is None:
        return "-"
    if p90 is not None and p90 > median * 1.15:
        return f"{median:.1f} / {p90:.1f}"
    return f"{median:.1f}"


def build(rows: list[dict], meta: dict[str, Any]) -> str:
    by_fixture: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_fixture[row["fixture"]].append(row)

    total = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    out: list[str] = []
    add = out.append

    add("# Agent eval report")
    add("")
    add(f"- **Model:** `{meta.get('model')}` at `{meta.get('base_url') or 'default endpoint'}`")
    add(f"- **Run:** {meta.get('started_at')} · {meta.get('wall_s', 0):.0f}s wall clock")
    add(f"- **Fixtures:** {len(by_fixture)} × {meta.get('reps')} reps = {total} runs")
    add(f"- **Passed:** {passed}/{total} ({100 * passed / total:.0f}%)" if total else "- No runs")
    if meta.get("concurrency", 1) > 1:
        add(
            f"- **Concurrency:** {meta['concurrency']} - wall-clock timings overlap, "
            f"so treat per-run latency as an upper bound"
        )
    add("")

    # -- headline table -------------------------------------------------
    add("## Results")
    add("")
    add("| Fixture | Pass | Model calls | Total s | TTFT s | In tok | Out tok | tok/s |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in sorted(by_fixture):
        runs = by_fixture[name]
        ok = sum(1 for r in runs if r["passed"])
        add(
            f"| {name} "
            f"| {ok}/{len(runs)} "
            f"| {_fmt(_median([r['model_calls'] for r in runs]), digits=0)} "
            f"| {_spread([r['latency'].get('total_s') for r in runs])} "
            f"| {_spread([r['latency'].get('ttft_s') for r in runs])} "
            f"| {_fmt(_median([r['tokens'].get('input') for r in runs]), digits=0)} "
            f"| {_fmt(_median([r['tokens'].get('output') for r in runs]), digits=0)} "
            f"| {_fmt(_median([r['tokens'].get('output_tps') for r in runs]))} |"
        )
    add("")
    add("Two numbers separated by `/` are median and p90; a single number means the")
    add("spread was tight. TTFT is time to the model's first streamed chunk.")
    add("")

    # -- what actually failed -------------------------------------------
    add("## Failing checks")
    add("")
    failures: list[str] = []
    for name in sorted(by_fixture):
        runs = by_fixture[name]
        tally: dict[str, list[int]] = defaultdict(list)
        details: dict[str, str] = {}
        for index, row in enumerate(runs):
            for check in row["checks"]:
                if check["passed"] is False:
                    tally[check["name"]].append(index)
                    details.setdefault(check["name"], check["detail"])
        for check_name, reps in sorted(tally.items()):
            failures.append(
                f"| {name} | `{check_name}` | {len(reps)}/{len(runs)} | {details[check_name][:110]} |"
            )
    if failures:
        add("| Fixture | Check | Failed | Detail from one failure |")
        add("| --- | --- | --- | --- |")
        out.extend(failures)
    else:
        add("None. Every scored check passed in every repetition.")
    add("")

    # -- how runs ended --------------------------------------------------
    terminations = Counter(r["termination"] for r in rows)
    if set(terminations) - {"normal"}:
        add("## How runs ended")
        add("")
        for how, count in terminations.most_common():
            add(f"- `{how}`: {count}")
        add("")
        add("`model_limit` / `tool_limit` mean a guardrail stopped the run, not that")
        add("the agent finished. `timeout` and `recursion` mean it never stopped at all.")
        add("")

    # -- loop and tool health --------------------------------------------
    blocked = Counter()
    errored = Counter()
    used = Counter()
    for row in rows:
        for call in row["tool_calls"]:
            used[call["name"]] += 1
            if call["status"] == "blocked":
                blocked[call["name"]] += 1
            elif call["status"] in {"error", "exception"}:
                errored[call["name"]] += 1
    if used:
        add("## Tool usage")
        add("")
        add("| Tool | Ran | Failed | Blocked as repeat |")
        add("| --- | --- | --- | --- |")
        for tool, count in used.most_common():
            add(f"| `{tool}` | {count} | {errored[tool]} | {blocked[tool]} |")
        add("")
        if sum(blocked.values()):
            add(
                "Blocked calls mean the repeated-call guard caught a loop. Without it "
                "those runs would have burned the whole budget retrying."
            )
            add("")

    decisions = Counter(
        (a["name"], a["decision"]) for row in rows for a in row.get("approvals") or []
    )
    if decisions:
        add("## Calls that went to the approval step")
        add("")
        add(
            "A call stopped here never reaches the tool, so it does not appear in the "
            "table above. This is where a refused write shows up."
        )
        add("")
        add("| Tool | Decision | Count |")
        add("| --- | --- | --- |")
        for (tool, decision), count in decisions.most_common():
            add(f"| `{tool}` | {decision} | {count} |")
        add("")

    # -- delegation cost --------------------------------------------------
    if "research_delegated" in by_fixture and "research_direct" in by_fixture:
        add("## Does delegating to a subagent pay for itself?")
        add("")
        add("| | Model calls | Total s | Input tokens |")
        add("| --- | --- | --- | --- |")
        for label, key in [("Delegated", "research_delegated"), ("Direct", "research_direct")]:
            runs = by_fixture[key]
            add(
                f"| {label} "
                f"| {_fmt(_median([r['model_calls'] for r in runs]), digits=0)} "
                f"| {_spread([r['latency'].get('total_s') for r in runs])} "
                f"| {_fmt(_median([r['tokens'].get('input') for r in runs]), digits=0)} |"
            )
        add("")

    # -- the subjective remainder -----------------------------------------
    judged_items = [
        (row, check)
        for row in rows
        for check in row["checks"]
        if check["passed"] is None
    ]
    if judged_items:
        add("## Needs your eye")
        add("")
        add("These are not scored. They are the questions a check cannot answer, ")
        add("shown with one sample answer each so you can judge them yourself.")
        add("")
        seen: set[str] = set()
        for row, check in judged_items:
            marker = f"{row['fixture']}::{check['name']}"
            if marker in seen:
                continue
            seen.add(marker)
            add(f"**{row['fixture']}** - {check['detail']}")
            add("")
            # Where the run produced a document, show the document. Judging it by
            # the model's closing remark judges the wrong thing: that remark says
            # the file is fine regardless of what is in it.
            artifact = (row.get("artifact_excerpt") or "").strip()
            if artifact:
                add(f"*{row['output_files'][0]}:*")
                add("")
                add("> " + artifact.replace("\n", "\n> "))
                add("")
                add("*and it said:*")
                add("")
            answer = (row["answer"] or "(no answer)").strip()
            excerpt = answer[:600] + (" ..." if len(answer) > 600 else "")
            add("> " + excerpt.replace("\n", "\n> "))
            add("")

    # -- harness failures --------------------------------------------------
    broken = [r for r in rows if not r["ok"]]
    if broken:
        add("## Runs that did not complete")
        add("")
        for row in broken:
            add(f"- `{row['fixture']}#{row['rep']}` - {row['termination']}: {row['error']}")
        add("")

    return "\n".join(out) + "\n"
