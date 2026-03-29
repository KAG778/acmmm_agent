![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-Unknown-lightgrey.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

# Multi-Agent Related Work Writer

A unified pipeline system for automated citation verification, fact-checking, and Related Work generation that operates as a truth-guarded academic writing infrastructure combining LLM capabilities with rigorous verification protocols to prevent hallucinations and ensure citation accuracy.

An AI-powered academic writing assistant that verifies every citation, detects unsupported claims, and automatically fills gaps with real papers — zero hallucinations guaranteed. The system solves the critical problem of citation errors and hallucinations in academic writing by implementing a dual-layer verification protocol: existence check (validates DOI/arXiv formats before admitting papers to the knowledge base) and semantic reconciliation (compares draft claims against original abstract anchors sentence-by-sentence).

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Zero-Hallucination Guarantee via Abstract Anchoring**: The Archivist agent extracts 1-2 core sentences verbatim from original abstracts as "abstract anchors." All downstream writing and verification is constrained to use only these anchors as factual source material, ensuring complete accuracy.

- **LLM-Powered Gap Detection and Rewriting**: When the Fact-Checker detects unsupported claims (gaps), the LLMRewriter agent automatically rewrites those sections to incorporate verified citations, maintaining original formatting and LaTeX line-breaking styles.

- **Per-Task Output Isolation**: Each method/paper gets its own workspace directory with structured output (guide/ + output/), eliminating clutter and enabling reproducibility. Only final outputs are persisted; intermediate artifacts exist only in memory.

- **DOI/arXiv Strict Validation**: Papers without valid DOI (10.XXXX/...) or arXiv ID (YYMM.NNNNN) formats are rejected at the archivist stage. Citation keys like "E17-1010" are flagged as invalid DOIs, ensuring only legitimate papers enter the knowledge base.

- **Dual Operating Modes**: Draft Mode verifies and enhances existing LaTeX files by extracting citations, validating against knowledge base, detecting gaps and hallucinations, and rewriting unsupported claims. Brainstorm Mode generates Related Work from scratch through interactive Q&A, automated literature search, archive building, and narrative synthesis.

- **Stale Citation Detection**: Automatically flags papers older than 5-10 years and suggests newer alternatives may exist, helping maintain research currency and relevance.

## Quick Start

### Prerequisites

- **Python**: 3.10+ (code uses `list[str]`, `dict[str, int]` type annotations which require Python 3.10+)
- **Anthropic API Key**: Required for LLM rewriting functionality (Claude Sonnet 4)
- **Optional**: LaTeX distribution (texlive/miktex) for compiling generated `.tex` files

### Installation

```bash
# Install dependencies
pip install anthropic
```

The project has minimal dependencies — primarily the `anthropic` package for LLM API calls. All other functionality uses Python standard library (`re`, `json`, `dataclasses`, `typing`, `pathlib`, etc.).

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key for LLM rewriting functionality |

The Anthropic SDK automatically reads from `~/.anthropic/config.json`, or you can set via environment variable.

### Verification

```bash
# Check Python version (requires 3.10+)
python3 --version

# Verify Anthropic package installed
python3 -c "import anthropic; print(anthropic.__version__)"

# Test import of main module
python3 -c "from agents.orchestrator import PipelineOrchestrator; print('OK')"

# Run help command
python3 run.py --help
```

### Basic Usage

```bash
# Navigate to the project directory
cd /home/wangmeiyi/AuctionNet/Oasis+Mirofish+GEO/acmmm_agent

# Draft mode: Verify and enhance an existing LaTeX file
python3 run.py experiment_setup.tex
python3 run.py experiment_setup.tex --topic "实验设置" --auto-approve

# Brainstorm mode: Generate from scratch
python3 run.py --brainstorm "帮我写 VeriPatent 相关工作"
python3 run.py --brainstorm "帮我写 VeriPatent 相关工作" --venue "ACM MM"

# Specify custom output directory
python3 run.py -o ./output/
```

## Architecture

<details>
<summary>Click to expand architecture details</summary>

### System Overview

The system uses a 6-phase pipeline with specialized agents:

1. **Phase 0 (Brainstorm Agent)**: Collects user requirements
2. **Phase 1 (Topic Specialist)**: Analyzes content, generates search strategy
3. **Phase 2 (Search Agent)**: Finds relevant literature
4. **Phase 3 (Archivist)**: Builds structured literature archive
5. **Phase 4 (Synthesis Writer)**: Generates narrative draft
6. **Phase 5 (Fact Checker)**: Audits claims vs archive
7. **Phase 6 (Cleanup Agent)**: Removes invalid records, locks version

```mermaid
flowchart TB
    CLI[run.py CLI] --> ORCH[PipelineOrchestrator]

    ORCH -->|Draft Mode| CITEX[CitationExtractor]
    ORCH -->|Brainstorm Mode| BRAIN[BrainstormAgent]

    BRAIN --> |UserRequirement| REQ[Requirements JSON]
    CITEX --> |ExtractionResult| INP[PipelineInput]

    ORCH --> PH1[TopicSpecialist<br/>Phase 1]
    PH1 --> |SearchTaskDocument| PH2[SearchAgent<br/>Phase 2]

    PH2 --> |Literature[]| PH3[Archivist<br/>Phase 3]
    PH3 --> |ArchiveEntry[]| CORE[Core Papers Selection]

    CORE --> PH4[SynthesisWriter<br/>Phase 4]
    PH4 --> |Section[]| DRAFT[LaTeX Draft]

    DRAFT --> PH5[FactChecker<br/>Phase 5]
    PH5 --> |AuditReport| AUDIT[Audit Report]

    PH5 --> |VerificationReport| LLMR[LLMRewriter]
    LLMR --> |Rewritten Text| PH6[CleanupAgent<br/>Phase 6]

    PH6 --> OUT[Output Directory<br/>workspace/{topic}/]

    subgraph OUTPUTS
        OUT --> GUIDE[guide/README.md]
        OUT --> TEX[output/*.tex]
        OUT --> BIB[output/references.bib]
        OUT --> REPORT[output/Citation_Audit_Report.md]
        OUT --> CHANGELOG[output/change_log.md]
        OUT --> XREF[output/cross_reference.md]
    end

    style CLI fill:#e1f5ff
    style ORCH fill:#fff4e1
    style BRAIN fill:#f0e1ff
    style PH5 fill:#ffe1e1
    style OUT fill:#e1ffe1
```

### Module Responsibilities

| Module | Path | Description |
|---|---|---|
| **PipelineOrchestrator** | `agents/orchestrator.py` | Unified pipeline orchestration, state management, output formatting |
| **BrainstormAgent** | `agents/brainstorm_agent.py` | Parse user requirements, extract method name/summary/venue/year range |
| **TopicSpecialist** | `agents/topic_specialist.py` | Analyze draft or requirements, generate search strategy (keywords, venues, years) |
| **SearchAgent** | `agents/search_agent.py` | Execute web searches, match citation keys, return candidate literature |
| **Archivist** | `agents/archivist.py` | Validate metadata (DOI/arXiv), extract abstract anchors, select core papers |
| **SynthesisWriter** | `agents/synthesis_writer.py` | Generate narrative sections (evolutionary/taxonomic/comparative strategies) |
| **FactChecker** | `agents/fact_checker.py` | Existence check, semantic reconciliation, hallucination detection, gap detection |
| **LLMRewriter** | `agents/llm_rewriter.py` | Rewrite gap areas with LLM, insert verified citations |
| **CleanupAgent** | `agents/cleanup_agent.py` | Destroy invalid records, prune stale drafts, lock fact table version |
| **CitationExtractor** | `agents/citation_extractor.py` | Parse LaTeX citations, extract contexts, detect unsupported claims |

### Output Structure

Each task generates a structured workspace directory:

```
workspace/{topic_name}/
├── guide/
│   └── README.md                  # Usage guide for interpreting outputs
├── output/
│   ├── rewritten.tex              # LLM-rewritten version
│   ├── references.bib             # BibTeX with URLs
│   ├── Citation_Audit_Report.md   # Full audit with gap alerts
│   ├── change_log.md              # Change records
│   └── cross_reference.md         # Claim-to-anchor mapping table
└── requirements.json              # User requirements (brainstorm mode only)
```

</details>

## Usage

### Scenario 1: Researcher Polishing a Draft

I have a Related Work section with 12 citations. I need to verify they're all real and check if I'm missing any key papers.

```bash
python run.py related_work.tex
```

**Output**:
- Verification report showing 12/12 citations found, 0 hallucinations, 3 gaps detected
- LLM-rewritten .tex with gaps filled
- BibTeX file with URLs
- Cross-reference table mapping claims to abstract anchors

### Scenario 2: PhD Student Starting from Scratch

I need to write a Related Work section for my new method 'VeriPatent' targeting ACM MM, but I don't know where to start.

```bash
python run.py --brainstorm "帮我写 VeriPatent 相关工作，投 ACM MM，不要太长"
```

**Interactive Q&A**:
1. Method name: VeriPatent
2. Target venue: ACM MM
3. Method summary: [user provides description]
4. Year range: 2022-2025 (default)
5. Max papers: 8 (default)
6. Length: concise (default)

**Output**:
- `workspace/VeriPatent/related_work.tex`
- `workspace/VeriPatent/audit_report.md`
- `workspace/VeriPatent/references.bib`

### Scenario 3: Lab Head Maintaining Citation Standards

I need to ensure all students' papers have verified citations. I can't afford any citation errors or hallucinations in our group's publications.

```bash
# For each student's draft
python run.py student1/draft.tex --topic "MethodA" --auto-approve
python run.py student2/experiments.tex --topic "MethodB" --auto-approve
```

**Benefits**:
- Every citation validated against knowledge base
- Hallucinations flagged immediately
- Stale citations identified for review
- BibTeX files auto-generated with URLs
- Per-task isolation enables parallel processing

### Scenario 4: Academic Auditor Checking Submitted Manuscript

I'm reviewing a manuscript and need to verify all citations are legitimate and accurately represented.

```bash
python run.py manuscript.tex --auto-approve
```

**Output Analysis**:
- Citation_Audit_Report: Shows pass/fail for each citation
- Hallucination count: Must be 0 for acceptance
- Gap count: Identifies unsupported claims
- Cross-reference: Maps every claim to source material

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

<!-- License: Analysis not available — No LICENSE file was detected in the repository. Please consider adding a license file to clarify usage terms. -->

---

**Project Version**: 4.0.0
**Repository**: https://github.com/KAG778/acmmm_agent.git
**Generated**: 2026-03-29
