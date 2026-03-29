# Phase 0 Brainstorm Agent — Design Spec

**Date:** 2026-03-29
**Status:** Draft
**Author:** Claude + User

## Problem

Current pipeline starts at Phase 1 (Topic Specialist) with hardcoded values.
No mechanism to collect user requirements before running.
The `--interactive` flag exists but only prints a prompt without collecting input.

## Goal

Add a Phase 0 (Brainstorm Agent) at the front of the pipeline that:

1. Accepts user's raw input (one-liner, draft file, or question)
2. Extracts known key points and displays them
3. Asks user whether to confirm/add details
4. Walks through structured questions one at a time
5. Outputs a `UserRequirement` object for all downstream agents

## Architecture

### New File: `agents/brainstorm_agent.py`

### Data Structure

```python
@dataclass
class UserRequirement:
    raw_input: str                    # User's original input
    method_name: str                  # Method name
    method_summary: str               # Core method description (1-2 sentences)
    target_venue: str                 # Target conference/journal
    research_directions: list[str]    # Sub-directions to cover
    year_range: tuple[int, int]       # Literature year range
    max_papers: int                   # Max core papers
    draft_content: str                # Full draft text (if provided)
    strict_mode: bool                 # Reject papers without DOI
    length_preference: str            # "concise" / "standard" / "comprehensive"
    extra_requirements: list[str]     # Extra user requirements/questions
    confirmed: bool = False
```

### Interaction Flow (4 Steps)

**Step 1 — Receive Input**
- Accept raw text from CLI argument or stdin
- If `--draft` is provided, also load the draft file

**Step 2 — Extract & Display**
- Parse the raw input for known information (method name, venue, constraints)
- Display "I understand:" summary list

**Step 3 — Confirm or Skip**
- Ask: "Need to fill in more details? [y/n]"
- If n: use extracted info as-is, proceed to Phase 1
- If y: enter Step 4

**Step 4 — Structured Q&A** (one question at a time)
Each question shows a default value extracted from input; user presses Enter to skip.

| # | Question | Default |
|---|----------|---------|
| 1 | Paper title / method name | Extracted or "N/A" |
| 2 | Target venue | "ACM MM" |
| 3 | Core method description | Extracted or "N/A" |
| 4 | Sub-directions to cover | "passive, proactive" |
| 5a | Year range | "2022-2025" |
| 5b | Max core papers | "8" |
| 5c | Length preference | "concise" |
| 6 | Strict mode (reject no-DOI) | "y" |

### BrainstormAgent Class

```python
class BrainstormAgent:
    @classmethod
    def run(cls, raw_input: str, draft_content: str = None) -> UserRequirement:
        """Main entry point: parse input → display → confirm → Q&A → return UserRequirement"""

    @classmethod
    def _parse_raw_input(cls, text: str) -> dict:
        """Extract known fields from raw user input"""

    @classmethod
    def _display_understanding(cls, parsed: dict) -> None:
        """Print 'I understand:' summary"""

    @classmethod
    def _ask_yes_no(cls, prompt: str) -> bool:
        """Ask a y/n question, return bool"""

    @classmethod
    def _ask_question(cls, prompt: str, default: str = "") -> str:
        """Ask a question with default, return answer"""

    @classmethod
    def _run_qa(cls, partial: UserRequirement) -> UserRequirement:
        """Walk through Step 4 questions, fill in UserRequirement"""
```

## Integration Points

### `agents/orchestrator.py`

New method `run_phase0()`:
```python
def run_phase0(self, raw_input: str, draft_content: str = None) -> UserRequirement:
    self.requirement = BrainstormAgent.run(raw_input, draft_content)
    return self.requirement
```

Modify `run_phases_1_to_3()` to accept `UserRequirement`:
- `build_profile_from_requirement(req)` instead of `build_profile_from_draft(draft_content)`
- `req.target_venue` → `search_task.target_venues`
- `req.year_range` → `search_task.year_range`
- `req.strict_mode` → `Archivist.build_archive(strict_mode=...)`
- `req.max_papers` → `Archivist.select_core_papers(max_count=...)`

### `agents/topic_specialist.py`

New classmethod:
```python
@classmethod
def build_profile_from_requirement(cls, req: UserRequirement) -> ResearchProfile:
    """Build ResearchProfile from Phase 0 UserRequirement instead of hardcoded values"""
```

Keep `build_profile_from_draft()` for backward compatibility.

### `agents/synthesis_writer.py`

Add `length_preference` parameter to `write_related_work()`:
- "concise" → 2 subsections, minimal detail per method
- "standard" → 2-3 subsections, balanced detail
- "comprehensive" → 3+ subsections, full detail

### `run.py`

New CLI entry:
```
python run.py --brainstorm "帮我写 VeriPatent 相关工作"
python run.py --brainstorm --draft 相关工作.md
```

Modify `--interactive` to call Phase 0 instead of just printing a prompt.

### `config/settings.json`

Add brainstorm_agent config:
```json
{
  "brainstorm_agent": {
    "role": "需求定义官",
    "phase": 0,
    "input": "User's raw input (text or draft file)",
    "output": "UserRequirement (structured requirements for all downstream agents)"
  }
}
```

### `agents/__init__.py`

Export new classes:
```python
from agents.brainstorm_agent import BrainstormAgent, UserRequirement
```

## CLI Usage

```bash
# New: pure requirement mode (one-liner + Q&A)
python run.py --brainstorm
python run.py --brainstorm "帮我写 VeriPatent 相关工作，投 ACM MM，不要太长"

# New: draft + requirement confirmation
python run.py --brainstorm --draft 相关工作.md

# Modified: --interactive now calls Phase 0
python run.py --interactive

# Unchanged: skip confirmation, run directly
python run.py --draft 相关工作.md
python run.py --full
```

## Output Isolation: Per-Method Task Directories

Each method name gets its own output directory. Only final outputs are kept.

### Directory Structure

```
workspace/
├── VeriPatent/              ← method_name as directory
│   ├── requirements.json    ← Phase 0 UserRequirement (for reproducibility)
│   ├── related_work.tex     ← Phase 4 final draft
│   ├── audit_report.md      ← Phase 5 audit report
│   └── references.bib       ← BibTeX entries
├── AnotherMethod/
│   ├── requirements.json
│   ├── related_work.tex
│   ├── audit_report.md
│   └── references.bib
└── ...                      ← one directory per method
```

### Rules

1. Directory name = `method_name` (sanitized: no spaces/special chars)
2. Each run overwrites the previous output for that method (no versioning in filename)
3. Intermediate artifacts (search results, archive tables, core papers lists) are NOT saved to disk — they exist only in memory during pipeline execution
4. Only 4 files per task: `requirements.json`, `related_work.tex`, `audit_report.md`, `references.bib`

### Changes to Orchestrator

- `output_dir` is now `workspace/{method_name}/` instead of flat `workspace/`
- Remove all intermediate file writes (search_task_*.md, archive_*.md, core_papers_*.md, draft_*.tex, cross_reference_*.md, fact_table_*.md)
- Only write: requirements.json, related_work.tex, audit_report.md, references.bib
- Remove `run_cleanup()` standalone method — cleanup is no longer needed since intermediate files aren't saved

## File Changes Summary

| File | Action | Scope |
|------|--------|-------|
| `agents/brainstorm_agent.py` | **Create** | New file, ~120 lines |
| `agents/orchestrator.py` | Modify | Add `run_phase0()`, per-method output dirs, remove intermediate file writes |
| `agents/topic_specialist.py` | Modify | Add `build_profile_from_requirement()` |
| `agents/synthesis_writer.py` | Modify | Add `length_preference` parameter |
| `agents/cleanup_agent.py` | Modify | Simplify — no longer needed for intermediate cleanup |
| `run.py` | Modify | Add `--brainstorm` arg, modify `--interactive`, per-method output path |
| `config/settings.json` | Modify | Add `brainstorm_agent` config |
| `agents/__init__.py` | Modify | Add exports |
