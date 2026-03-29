"""
Pipeline Orchestrator — Unified Academic Writing System (v5)
============================================================
统一流水线：单入口处理 draft 审核增强 和 brainstorm 从头生成。
输入：任意 LaTeX 文件（draft 模式）或需求描述（brainstorm 模式）
输出：workspace/{topic}/guide/ + output/ 子文件夹

Pipeline phases:
  Phase 0: Brainstorm            (仅 brainstorm 模式)
  Phase 1: TopicSpecialist       (分析内容 / 需求 → 搜索策略)
  Phase 2: SearchAgent           (主动搜索 + KB 匹配)
  Phase 3: Archivist             (建档 + 核心论文筛选)
  Phase 4: SynthesisWriter       (生成段落 / LLM 改写 gap)
  Phase 5: FactChecker           (审计 + 验证)
  Phase 6: CleanupAgent          (清理无效记录 + 版本锁定)
"""

import os
import re
from dataclasses import dataclass, field
from typing import Optional

from agents.topic_specialist import TopicSpecialist, ResearchProfile, SearchTaskDocument
from agents.search_agent import SearchAgent, Literature
from agents.archivist import Archivist, ArchiveEntry
from agents.synthesis_writer import SynthesisWriter, Section
from agents.fact_checker import (
    FactChecker, AuditReport, AuditIssue,
    VerificationReport, GapRecommendation,
)
from agents.brainstorm_agent import BrainstormAgent, UserRequirement
from agents.citation_extractor import (
    CitationExtractor, ExtractionResult,
    CitationRef, UnsupportedClaim,
)


# ======================================================================
# PipelineInput: 统一输入规范
# ======================================================================

@dataclass
class PipelineInput:
    """统一流水线输入 — 规范化 draft 和 brainstorm 两种模式。"""
    mode: str              # "draft" | "brainstorm"
    topic_name: str
    raw_text: str          # 原始 LaTeX 文本 / brainstorm 描述
    method_name: str = ""
    method_summary: str = ""
    target_venue: str = ""
    year_range: tuple = (2022, 2025)
    max_papers: int = 8
    length_preference: str = "concise"
    strict_mode: bool = True
    auto_approve: bool = False
    # 仅 draft 模式：引用提取结果
    extraction: Optional[ExtractionResult] = None


def sanitize_dirname(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name).strip("_")


def extraction_to_sections(extraction: ExtractionResult) -> list[Section]:
    section = Section(
        subsection_title="input",
        narrative_strategy="standard",
        content="",
        citations_used=extraction.cite_keys,
    )
    return [section]


class PipelineOrchestrator:

    def __init__(self, workspace_dir: str = "workspace"):
        self.workspace_dir = workspace_dir
        self.requirement: Optional[UserRequirement] = None
        self.profile: Optional[ResearchProfile] = None
        self.search_task: Optional[SearchTaskDocument] = None
        self.papers: list[Literature] = []
        self.archive: list[ArchiveEntry] = []
        self.core_papers: list[ArchiveEntry] = []
        self.sections: list[Section] = []
        self.audit_report: Optional[AuditReport] = None
        self.final_draft: str = ""
        self._verification_report: Optional[VerificationReport] = None

    # ==================================================================
    # v5 CORE: Unified Pipeline
    # ==================================================================

    @staticmethod
    def build_pipeline_input(
        *,
        input_text: str = None,
        file_path: str = None,
        raw_text: str = None,
        topic_name: str = "",
        method_name: str = "",
        method_summary: str = "",
        target_venue: str = "",
        year_range: tuple = None,
        max_papers: int = 8,
        length_preference: str = "concise",
        strict_mode: bool = True,
        auto_approve: bool = False,
    ) -> PipelineInput:
        """从文件（draft 模式）或原始文本（brainstorm 模式）构建统一输入。"""
        if file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            extraction = CitationExtractor.extract(content)
            resolved_topic = topic_name or extraction.topic_hint or "general"
            return PipelineInput(
                mode="draft",
                topic_name=resolved_topic,
                raw_text=content,
                extraction=extraction,
                strict_mode=strict_mode,
                auto_approve=auto_approve,
            )
        if input_text is not None:
            extraction = CitationExtractor.extract(input_text)
            resolved_topic = topic_name or extraction.topic_hint or "general"
            return PipelineInput(
                mode="draft",
                topic_name=resolved_topic,
                raw_text=input_text,
                extraction=extraction,
                strict_mode=strict_mode,
                auto_approve=auto_approve,
            )
        # brainstorm 模式
        return PipelineInput(
            mode="brainstorm",
            topic_name=topic_name or method_name or "general",
            raw_text=raw_text or "",
            method_name=method_name,
            method_summary=method_summary,
            target_venue=target_venue,
            year_range=year_range or (2022, 2025),
            max_papers=max_papers,
            length_preference=length_preference,
            strict_mode=strict_mode,
            auto_approve=auto_approve,
        )

    def run_pipeline(self, inp: PipelineInput) -> dict:
        """统一流水线：单入口，覆盖 draft 和 brainstorm 两种模式。"""
        topic_name = inp.topic_name
        task_dir = os.path.join(self.workspace_dir, sanitize_dirname(topic_name))
        os.makedirs(task_dir, exist_ok=True)

        # ===== Phase 0: Brainstorm (仅 brainstorm 模式) =====
        if inp.mode == "brainstorm":
            self._run_phase0_brainstorm(inp, task_dir)

        # ===== Phase 1: Topic Specialist =====
        self._run_phase1_topic_specialist(inp, task_dir)

        # ===== Phase 2: Search Agent =====
        self._run_phase2_search(inp)

        # ===== Phase 3: Archivist =====
        self._run_phase3_archivist(inp)

        # ===== Phase 4: Synthesis Writer =====
        self._run_phase4_synthesis(inp)

        # ===== Phase 5: Fact Checker =====
        self._run_phase5_fact_checker(inp, task_dir)

        # ===== Phase 6: Cleanup Agent =====
        self._run_phase6_cleanup(task_dir)

        # ===== Output =====
        return self._write_unified_output(inp, task_dir)

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    def _run_phase0_brainstorm(self, inp: PipelineInput, task_dir: str):
        print("\n" + "=" * 50)
        print("  PHASE 0: Brainstorm")
        print("=" * 50)

        self.requirement = BrainstormAgent.run(inp.raw_text)
        if self.requirement:
            inp.method_name = self.requirement.method_name or inp.method_name
            inp.method_summary = self.requirement.method_summary or inp.method_summary
            inp.target_venue = self.requirement.target_venue or inp.target_venue
            if self.requirement.year_range:
                inp.year_range = tuple(self.requirement.year_range)
            if self.requirement.max_papers:
                inp.max_papers = self.requirement.max_papers
            if self.requirement.length_preference:
                inp.length_preference = self.requirement.length_preference
            self.requirement.save(os.path.join(task_dir, "requirements.json"))

    def _run_phase1_topic_specialist(self, inp: PipelineInput, task_dir: str):
        print("\n" + "=" * 50)
        print("  PHASE 1: Topic Specialist")
        print("=" * 50)

        if inp.mode == "draft":
            self.profile = TopicSpecialist.build_profile_from_draft(inp.raw_text)
        elif self.requirement:
            self.profile = TopicSpecialist.build_profile_from_requirement(self.requirement)
        else:
            self.profile = ResearchProfile(
                method_name=inp.method_name,
                method_summary=inp.method_summary or inp.raw_text,
                target_venue=inp.target_venue,
            )

        self.search_task = TopicSpecialist.generate_search_task(self.profile)
        if inp.target_venue:
            self.search_task.target_venues = [
                v.strip() for v in inp.target_venue.split(",")
            ]
        if inp.year_range:
            self.search_task.year_range = tuple(inp.year_range)

    def _run_phase2_search(self, inp: PipelineInput):
        print("\n" + "=" * 50)
        print("  PHASE 2: Search Agent")
        print("=" * 50)

        # 主动搜索
        self.papers = SearchAgent.search(self.search_task)
        print(f"  Active search: {len(self.papers)} candidate papers.")

        # draft 模式：额外匹配已有引用
        if inp.mode == "draft" and inp.extraction:
            found_map = SearchAgent.search_by_keys(inp.extraction.cite_keys)
            existing_dois = {p.doi for p in self.papers}
            merged = 0
            for key, paper in found_map.items():
                if paper.doi not in existing_dois:
                    self.papers.append(paper)
                    existing_dois.add(paper.doi)
                    merged += 1
            print(f"  Key-match merge: +{merged} papers from existing citations.")

    def _run_phase3_archivist(self, inp: PipelineInput):
        print("\n" + "=" * 50)
        print("  PHASE 3: Archivist")
        print("=" * 50)

        method_summary = inp.method_summary or ""
        if self.requirement:
            method_summary = self.requirement.method_summary or method_summary

        self.archive, rejections = Archivist.build_archive(
            self.papers, user_method_summary=method_summary,
            strict_mode=inp.strict_mode,
        )
        if rejections:
            for r in rejections:
                print(f"    REJECTED: {r}")

        self.core_papers = Archivist.select_core_papers(
            self.archive, max_count=inp.max_papers,
        )
        print(f"  {len(self.core_papers)} core papers selected.")

    def _run_phase4_synthesis(self, inp: PipelineInput):
        print("\n" + "=" * 50)
        print("  PHASE 4: Synthesis Writer")
        print("=" * 50)

        if inp.mode == "brainstorm":
            # 从头生成段落
            method_name = inp.method_name or "unknown"
            self.sections = SynthesisWriter.write_related_work(
                self.core_papers,
                user_method_name=method_name,
                user_method_summary=inp.method_summary,
                length_preference=inp.length_preference,
            )
            self.final_draft = SynthesisWriter.assemble_full_draft(self.sections)
            print(f"  Generated {len(self.sections)} section(s).")
        else:
            # draft 模式：用 LLM Rewriter 改写 gap 区域，保留非 gap 原文
            report = self._verification_report
            rewritten = inp.raw_text
            self._change_records = []

            if report:
                gap_issues = [i for i in report.issues if i.severity == "gap"]
                if gap_issues and report.recommendations:
                    from agents.llm_rewriter import LLMRewriter
                    print("  REWRITE: Using LLM to rewrite gap areas...")
                    rewritten, self._change_records = LLMRewriter.rewrite_all_gaps(
                        rewritten, report.recommendations, gap_issues,
                    )
                    for rec in self._change_records:
                        print(f"    - Rewrote gap → ~\\cite{{{rec['cite_key']}}}: "
                              f"{rec['paper_title']}")

            # 插入验证摘要注释
            if report:
                halluc_part = (f", {report.hallucinations} hallucination(s)"
                                if report.hallucinations else " (0 hallucinations)")
                gap_part = (f", {len(self._change_records)} gap(s) rewritten by LLM"
                            if self._change_records else ", no gaps")
                summary_lines = [
                    "% ============================================================",
                    f"% Citation Verification & Rewrite — "
                    f"{inp.extraction.topic_hint.replace('_', ' ').title()}",
                    f"% Citations: {report.found_citations}/{report.total_citations} verified"
                    f"{halluc_part}{gap_part}",
                    f"% Verdict: {report.final_verdict}",
                    "% ============================================================",
                    "",
                ]
                old_header_re = re.compile(
                    r'^% =+\n% Citation Verification[^\n]*\n(?:%[^\n]*\n)*% =+\n',
                    re.MULTILINE,
                )
                rewritten = old_header_re.sub('', rewritten).lstrip('\n')
                rewritten = "\n".join(summary_lines + rewritten.split("\n"))

            self.final_draft = rewritten
            print(f"  Draft rewritten with {len(self._change_records)} gap fix(es).")

    def _run_phase5_fact_checker(self, inp: PipelineInput, task_dir: str):
        print("\n" + "=" * 50)
        print("  PHASE 5: Fact Checker")
        print("=" * 50)

        if inp.mode == "brainstorm":
            self.audit_report, corrected = FactChecker.audit(
                self.sections, self.core_papers,
            )
            print(f"  Verdict: {self.audit_report.final_verdict}")
        else:
            # draft 模式：完整验证（已在 _build_verification_report 中完成）
            report = self._verification_report
            if report:
                print(f"  Verified: {report.verified_claims} | "
                      f"Hallucinations: {report.hallucinations} | "
                      f"Existence errors: {report.existence_failures} | "
                      f"Stale: {report.stale_count}")
                print(f"  Gaps: {report.gap_count} | "
                      f"Verdict: {report.final_verdict}")

                # 如果有 gap 推荐，显示给用户
                if report.recommendations and not inp.auto_approve:
                    print("\n" + "-" * 50)
                    self._print_proposals(report.recommendations)
                    print("\n  [INFO] Use --auto-approve to skip confirmation.")

    def _run_phase6_cleanup(self, task_dir: str):
        print("\n" + "=" * 50)
        print("  PHASE 6: Cleanup Agent")
        print("=" * 50)

        try:
            from agents.cleanup_agent import CleanupAgent
            cleanup = CleanupAgent(workspace_dir=self.workspace_dir)

            # 销毁无效文献
            dois_to_destroy = []
            if self._verification_report:
                dois_to_destroy = self._verification_report.dois_to_destroy or []
            if dois_to_destroy:
                destroy_report = cleanup.destroy_records(
                    dois_to_destroy, self.archive, dry_run=False,
                )
                print(f"  Destroyed {len(dois_to_destroy)} invalid record(s).")

            # 版本锁定
            fact_table_path = cleanup.lock_version(
                archive_entries=self.core_papers,
                audit_report=self.audit_report,
                core_papers_status="approved",
            )
            print(f"  Fact table locked: {fact_table_path}")

        except Exception as e:
            print(f"  Cleanup skipped: {e}")

    # ------------------------------------------------------------------
    # Verification: draft 模式专用的完整验证流程
    # ------------------------------------------------------------------

    def _build_verification_report(
        self, extraction: ExtractionResult, found_map: dict,
    ) -> VerificationReport:
        """draft 模式：构建完整 VerificationReport。"""
        report = VerificationReport()
        report.total_citations = len(extraction.cite_keys)
        report.found_citations = len(found_map)
        report.missing_citations = len(extraction.cite_keys) - len(found_map)

        found_papers = list(found_map.values())

        # Existence Check
        existence_issues, dois_to_destroy = FactChecker._check_existence(
            self.archive, [],
        )
        report.existence_failures = len(existence_issues)
        report.dois_to_destroy = dois_to_destroy
        report.issues.extend(existence_issues)

        # Semantic Reconciliation
        citekey_to_entry: dict[str, ArchiveEntry] = {}
        for paper in found_papers:
            for entry in self.archive:
                if entry.doi == paper.doi:
                    if paper.cite_key:
                        citekey_to_entry[paper.cite_key] = entry
                    citekey_to_entry[entry.doi] = entry
                    break

        for cite_ref in extraction.citations:
            entry = citekey_to_entry.get(cite_ref.cite_key)
            if not entry:
                continue
            if cite_ref.full_context and entry.abstract_anchor:
                is_supported = FactChecker._is_claim_supported_by_anchor(
                    cite_ref.full_context, entry.abstract_anchor, entry.title,
                )
                if is_supported is False:
                    report.issues.append(AuditIssue(
                        severity="hallucination",
                        location=f"Line {cite_ref.location_line}",
                        description=(
                            f"HALLUCINATION for \"{entry.title}\": "
                            f"citation context \"{cite_ref.full_context[:100]}\" "
                            f"is NOT supported by the paper's abstract."
                        ),
                        suggestion="Verify this citation is correctly placed.",
                        related_doi=entry.doi,
                        anchor_evidence=entry.abstract_anchor,
                    ))
                    report.hallucinations += 1
                else:
                    report.verified_claims += 1
            elif entry.abstract_anchor:
                report.verified_claims += 1

        # Stale Citation Detection
        stale_issues = FactChecker.detect_stale_citations(self.archive)
        report.stale_count = len(stale_issues)
        report.issues.extend(stale_issues)

        # Gap Detection & Recommendation
        gap_issues, recommendations = FactChecker.detect_gaps(
            extraction.unsupported_claims, extraction.citations, self.archive,
        )
        report.gap_count = len(gap_issues)
        report.recommendations = recommendations
        report.issues.extend(gap_issues)

        # Final Verdict
        missing_keys = [k for k in extraction.cite_keys if k not in found_map]
        hallucinations = [i for i in report.issues if i.severity == "hallucination"]
        errors = [i for i in report.issues if i.severity == "error"]
        gaps = [i for i in report.issues if i.severity == "gap"]

        if missing_keys:
            report.final_verdict = (
                f"NEEDS ATTENTION — {len(missing_keys)} citation(s) not found. "
                f"{len(hallucinations)} hallucination(s). {len(gaps)} gap(s).")
        elif hallucinations:
            report.final_verdict = (
                f"HALLUCINATION DETECTED — {len(hallucinations)} claim(s).")
        elif gaps:
            report.final_verdict = f"GAPS FOUND — {len(gaps)} unsupported claim(s)."
        elif errors:
            report.final_verdict = f"NEEDS REVISION — {len(errors)} error(s)."
        elif report.stale_count > 0:
            report.final_verdict = f"PASS WITH NOTES — {report.stale_count} stale."
        else:
            report.final_verdict = "PASS — All citations verified, no gaps."

        return report

    # ------------------------------------------------------------------
    # Unified Output
    # ------------------------------------------------------------------

    def _write_unified_output(self, inp: PipelineInput, task_dir: str) -> dict:
        """统一输出：guide/ + output/ 结构化目录。"""
        guide_dir = os.path.join(task_dir, "guide")
        output_dir = os.path.join(task_dir, "output")
        os.makedirs(guide_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        result = {"task_dir": task_dir}

        if inp.mode == "draft":
            # draft 模式：输出改写后的 tex + 审计报告
            input_filename = self._detect_input_filename(inp.raw_text, inp.topic_name)
            tex_path = os.path.join(output_dir, input_filename)
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(self.final_draft)
            result["main_tex"] = tex_path

            report = self._verification_report
            if report:
                # Citation Audit Report
                extraction = inp.extraction
                found_map = SearchAgent.search_by_keys(extraction.cite_keys)
                report_text = FactChecker.format_verification_report(
                    report, extraction.topic_hint,
                    archive=self.archive, found_map=found_map,
                )
                report_name = f"Citation_Audit_Report_{inp.topic_name.replace(' ', '_')}.md"
                report_path = os.path.join(output_dir, report_name)
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(report_text)
                result["audit_report"] = report_path

                # Change Log
                change_log = self._generate_change_log(
                    inp.raw_text, self.final_draft, extraction,
                    found_map, report,
                    change_records=getattr(self, '_change_records', []),
                )
                change_log_path = os.path.join(output_dir, "change_log.md")
                with open(change_log_path, "w", encoding="utf-8") as f:
                    f.write(change_log)
                result["change_log"] = change_log_path

            print(f"\n  Task directory: {task_dir}/")
            print(f"    output/{input_filename}           → {tex_path}  [MAIN]")
            if "audit_report" in result:
                print(f"    output/{report_name}   → {report_path}")
            if "change_log" in result:
                print(f"    output/change_log.md           → {change_log_path}")
        else:
            # brainstorm 模式：输出生成的 tex + 审计报告
            final_path = os.path.join(output_dir, "related_work.tex")
            with open(final_path, "w", encoding="utf-8") as f:
                f.write(self.final_draft)
            result["main_tex"] = final_path

            if self.audit_report:
                report_text = FactChecker.format_report(self.audit_report)
                report_path = os.path.join(output_dir, "audit_report.md")
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(report_text)
                result["audit_report"] = report_path

            print(f"\n  Task directory: {task_dir}/")
            print(f"    output/related_work.tex  → {final_path}  [MAIN]")
            if "audit_report" in result:
                print(f"    output/audit_report.md → {report_path}")

        # 共用输出：references.bib + cross_reference.md + guide/README.md
        bibtex = Archivist.export_bibtex(self.core_papers)

        # draft 模式：合并原文引用的 BibTeX（found_map）
        if inp.mode == "draft" and inp.extraction:
            draft_found = SearchAgent.search_by_keys(inp.extraction.cite_keys)
            draft_bib = Archivist.export_bibtex_from_literature(
                list(draft_found.values()),
                exclude_dois={e.doi for e in self.core_papers},
            )
            if draft_bib:
                bibtex = draft_bib.rstrip() + "\n\n" + bibtex

        if self._verification_report and self._verification_report.recommendations:
            rec_bib = FactChecker.format_bibtex_for_recommendations(
                self._verification_report.recommendations,
            )
            if rec_bib:
                bibtex = bibtex.rstrip() + "\n\n% === Recommended Papers ===\n" + rec_bib
        bib_path = os.path.join(output_dir, "references.bib")
        with open(bib_path, "w", encoding="utf-8") as f:
            f.write(bibtex)
        result["references"] = bib_path
        print(f"    output/references.bib   → {bib_path}")

        if self.archive:
            if inp.extraction:
                sections = extraction_to_sections(inp.extraction)
            else:
                sections = self.sections
            cross_ref = FactChecker.build_cross_reference_table(
                sections, self.archive,
            )
            cross_path = os.path.join(output_dir, "cross_reference.md")
            with open(cross_path, "w", encoding="utf-8") as f:
                f.write(cross_ref)
            result["cross_reference"] = cross_path

        return result

    # ------------------------------------------------------------------
    # Backward-compatible wrappers
    # ------------------------------------------------------------------

    def verify_citations(
        self,
        input_text: str,
        topic_name: str = "",
        auto_approve: bool = False,
    ) -> VerificationReport:
        """向后兼容：委托到 run_pipeline()。"""
        inp = self.build_pipeline_input(
            input_text=input_text,
            topic_name=topic_name,
            auto_approve=auto_approve,
        )
        # draft 模式需要在 Phase 3 之后构建 verification_report
        topic_name = inp.topic_name
        task_dir = os.path.join(self.workspace_dir, sanitize_dirname(topic_name))
        os.makedirs(task_dir, exist_ok=True)

        # Phase 1
        self._run_phase1_topic_specialist(inp, task_dir)
        # Phase 2
        self._run_phase2_search(inp)
        # Phase 3
        self._run_phase3_archivist(inp)
        # Build verification report (draft 专用)
        found_map = SearchAgent.search_by_keys(inp.extraction.cite_keys)
        self._verification_report = self._build_verification_report(
            inp.extraction, found_map,
        )
        # Phase 4-6
        self._run_phase4_synthesis(inp)
        self._run_phase5_fact_checker(inp, task_dir)
        self._run_phase6_cleanup(task_dir)
        self._write_unified_output(inp, task_dir)

        report = self._verification_report
        print("\n" + "=" * 50)
        print(f"  VERDICT: {report.final_verdict}")
        print("=" * 50)
        return report

    def run_phase0(self, raw_input: str = "", draft_content: str = None):
        """向后兼容。"""
        self.requirement = BrainstormAgent.run(raw_input, draft_content)
        return self.requirement

    def run_phases_1_to_6(self, requirement: UserRequirement = None) -> dict:
        """向后兼容：委托到 run_pipeline()。"""
        if requirement:
            self.requirement = requirement
        if not self.requirement:
            raise RuntimeError("No requirement.")
        req = self.requirement
        inp = self.build_pipeline_input(
            raw_text=req.raw_input,
            method_name=req.method_name,
            method_summary=req.method_summary,
            target_venue=req.target_venue,
            year_range=tuple(req.year_range) if req.year_range else None,
            max_papers=req.max_papers,
            length_preference=req.length_preference,
        )
        return self.run_pipeline(inp)

    def run_all(self, raw_input: str = "", draft_content: str = None) -> dict:
        """向后兼容：委托到 run_pipeline()。"""
        self.run_phase0(raw_input, draft_content)
        return self.run_phases_1_to_6(self.requirement)

    def _print_proposals(self, recommendations: list[GapRecommendation]):
        for idx, rec in enumerate(recommendations, 1):
            print(f"\n  --- Proposal #{idx} ---")
            print(f"  Claim: \"{rec.claim.claim_text[:100]}\"")
            print(f"  Reason: {rec.reason}")
            print(f"  Candidates:")
            for p_idx, paper in enumerate(rec.recommended_papers, 1):
                print(f"    [{p_idx}] {paper.title} ({paper.year}, {paper.venue})")
                print(f"        {paper.core_contribution}")

    def _write_outputs(
        self,
        report: VerificationReport,
        extraction: ExtractionResult,
        found_map: dict,
        task_dir: str,
        topic_name: str,
        input_text: str,
    ):
        """结构化输出：guide/ + output/ 两个子文件夹"""
        guide_dir = os.path.join(task_dir, "guide")
        output_dir = os.path.join(task_dir, "output")
        os.makedirs(guide_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # ===== 0. output/rewritten.tex (主线：LLM 改写后的文本) =====
        rewritten_text, change_records = self._rewrite_text_with_records(
            input_text, extraction, found_map, report
        )
        # 检测原始文件名
        input_filename = self._detect_input_filename(input_text, topic_name)
        rewritten_path = os.path.join(output_dir, input_filename)
        with open(rewritten_path, "w", encoding="utf-8") as f:
            f.write(rewritten_text)

        # ===== 1. guide/README.md =====
        readme = self._generate_readme(report, extraction, topic_name, input_filename)
        readme_path = os.path.join(guide_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme)

        # ===== 2. output/Citation_Audit_Report (with URLs) =====
        report_text = FactChecker.format_verification_report(
            report, extraction.topic_hint,
            archive=self.archive, found_map=found_map,
        )
        report_name = f"Citation_Audit_Report_{topic_name.replace(' ', '_')}.md"
        report_path = os.path.join(output_dir, report_name)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        # ===== 3. output/references.bib (with URL) =====
        bibtex = Archivist.export_bibtex(self.archive)
        if report.recommendations:
            rec_bib = FactChecker.format_bibtex_for_recommendations(
                report.recommendations
            )
            if rec_bib:
                bibtex = bibtex.rstrip() + "\n\n% === Recommended Papers ===\n" + rec_bib
        bib_path = os.path.join(output_dir, "references.bib")
        with open(bib_path, "w", encoding="utf-8") as f:
            f.write(bibtex)

        # ===== 4. output/cross_reference.md =====
        if self.archive:
            cross_ref = FactChecker.build_cross_reference_table(
                extraction_to_sections(extraction), self.archive
            )
            cross_path = os.path.join(output_dir, "cross_reference.md")
            with open(cross_path, "w", encoding="utf-8") as f:
                f.write(cross_ref)

        # ===== 5. output/change_log.md =====
        change_log = self._generate_change_log(
            input_text, rewritten_text, extraction, found_map, report, change_records
        )
        change_log_path = os.path.join(output_dir, "change_log.md")
        with open(change_log_path, "w", encoding="utf-8") as f:
            f.write(change_log)

        print(f"\n  Task directory: {task_dir}/")
        print(f"    output/{input_filename}                   → {rewritten_path}  [MAIN]")
        print(f"    output/references.bib                    → {bib_path}")
        print(f"    output/{report_name}   → {report_path}")
        print(f"    output/change_log.md                      → {change_log_path}")
        print(f"    guide/README.md                           → {readme_path}")

    # ------------------------------------------------------------------
    # Rewrite: 改写输入文本
    # ------------------------------------------------------------------

    def _rewrite_text_with_records(
        self,
        original_text: str,
        extraction: ExtractionResult,
        found_map: dict,
        report: VerificationReport,
    ) -> tuple[str, list[dict]]:
        """改写文本并返回 (改写后文本, 变更记录)"""
        rewritten = original_text
        change_records = []

        # LLM 改写 gap 区域
        gap_issues = [i for i in report.issues if i.severity == "gap"]
        if gap_issues and report.recommendations:
            from agents.llm_rewriter import LLMRewriter
            print("\n  REWRITE: Using LLM to rewrite gap areas...")

            rewritten, change_records = LLMRewriter.rewrite_all_gaps(
                rewritten, report.recommendations, gap_issues,
            )
            for rec in change_records:
                print(f"    - Rewrote gap → ~\\cite{{{rec['cite_key']}}}: "
                      f"{rec['paper_title']}")

        # 插入验证摘要注释
        halluc_part = (f", {report.hallucinations} hallucination(s)"
                        if report.hallucinations else " (0 hallucinations)")
        gap_part = (f", {len(change_records)} gap(s) rewritten by LLM"
                     if change_records else ", no gaps")
        summary_lines = [
            "% ============================================================",
            f"% Citation Verification & Rewrite — "
            f"{extraction.topic_hint.replace('_', ' ').title()}",
            f"% Citations: {report.found_citations}/{report.total_citations} verified"
            f"{halluc_part}{gap_part}",
            f"% Verdict: {report.final_verdict}",
            "% ============================================================",
            "",
        ]

        # 剥离已有的旧注释头（避免重复）
        old_header_re = re.compile(
            r'^% =+\n% Citation Verification[^\n]*\n(?:%[^\n]*\n)*% =+\n',
            re.MULTILINE,
        )
        rewritten = old_header_re.sub('', rewritten).lstrip('\n')

        rewritten = "\n".join(summary_lines + rewritten.split("\n"))
        return rewritten, change_records

    def _detect_input_filename(self, input_text: str, topic_name: str) -> str:
        """从输入文本推断输出文件名"""
        # 检查是否有 \section 或 \subsection 标题
        section_match = re.search(r'\\section\{([^}]+)\}', input_text)
        if section_match:
            title = section_match.group(1)
            title = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', title)
            title = re.sub(r'[{}\\]', '', title).strip().lower()
            slug = re.sub(r'[^a-z0-9]+', '_', title).strip("_")
            if slug:
                return f"{slug}.tex"
        return f"{topic_name.replace(' ', '_')}.tex"

    def _generate_change_log(
        self,
        original_text: str,
        rewritten_text: str,
        extraction: ExtractionResult,
        found_map: dict,
        report: VerificationReport,
        change_records: list[dict] = None,
    ) -> str:
        """生成改写变更日志"""
        change_records = change_records or []
        lines = [
            "# Change Log",
            "",
            f"## Verification Summary",
            f"- Citations verified: {report.found_citations}/{report.total_citations}",
            f"- Hallucinations: {report.hallucinations}",
            f"- Gaps detected: {report.gap_count}",
            f"- Gaps rewritten by LLM: {len(change_records)}",
            f"- Stale citations: {report.stale_count}",
            "",
            "## LLM Rewrites (Gap → Full Paragraph)",
            "",
        ]

        if change_records:
            lines.append(
                "| # | Original Gap | Rewritten To | Cite Key | Paper |"
            )
            lines.append(
                "|---|---|---|---|---|"
            )
            for idx, rec in enumerate(change_records, 1):
                lines.append(
                    f"| {idx} | \"{rec['original_claim'][:50]}\" | "
                    f"\"{rec['rewritten_to'][:60]}\" | "
                    f"`{rec['cite_key']}` | "
                    f"[{rec['paper_title']}]({rec['paper_url']}) |"
                )
        else:
            lines.append("No gaps detected. No LLM rewrites needed.")

        lines.extend([
            "",
            "## Citation Verification Details",
            "",
            "| Cite Key | Title | URL | Status |",
            "|---|---|---|---|",
        ])
        for key, paper in found_map.items():
            url = paper.url or "N/A"
            doi_valid, _ = FactChecker.validate_doi_static(paper.doi)
            status = "Verified" if doi_valid else "Unverified"
            title_short = paper.title[:40] + ("..." if len(paper.title) > 40 else "")
            lines.append(f"| `{key}` | {title_short} | [{url}]({url}) | {status} |")

        if report.stale_count > 0:
            stale_issues = [i for i in report.issues if i.severity == "stale"]
            lines.extend(["", "## Stale Citation Warnings", ""])
            for issue in stale_issues:
                lines.append(f"- {issue.description}")
                lines.append(f"  Suggestion: {issue.suggestion}")

        lines.extend([
            "",
            "## No-Change Items",
            "",
            f"- All {report.found_citations} existing citations preserved",
            "- Text structure preserved outside gap areas",
            "",
        ])
        return "\n".join(lines)

    def _generate_readme(
        self,
        report: VerificationReport,
        extraction: ExtractionResult,
        topic_name: str,
        input_filename: str = "",
    ) -> str:
        """生成输出说明文档"""
        title = topic_name.replace("_", " ").title()
        lines = [
            f"# {title} — Citation Verification & Rewrite Output",
            "",
            "## Output Structure",
            "",
            "```",
            f"{topic_name}/",
            f"├── guide/",
            f"│   └── README.md              ← This file (how to use the output)",
            f"└── output/",
            f"    ├── *.tex                  ← MAIN: rewritten text with verified citations",
            f"    ├── references.bib         ← BibTeX entries with URLs",
            f"    ├── Citation_Audit_Report_{topic_name}.md",
            f"    ├── change_log.md          ← What was changed and why",
            f"    └── cross_reference.md",
            "```",
            "",
            "## Verification Summary",
            "",
            f"- **Input**: LaTeX text with {report.total_citations} citations",
            f"- **Found in KB**: {report.found_citations}/{report.total_citations}",
            f"- **Hallucinations**: {report.hallucinations}",
            f"- **Citation gaps**: {report.gap_count}",
            f"- **Stale citations**: {report.stale_count}",
            f"- **Verdict**: {report.final_verdict}",
            "",
            "## File Descriptions",
            "",
            "| File | Description |",
            "|---|---|",
            "| `output/*.tex` | **MAIN OUTPUT** — Rewritten text with verified citations and gap-filled references |",
            "| `output/references.bib` | BibTeX entries with URLs for all cited papers |",
            "| `output/Citation_Audit_Report_*.md` | Full audit: [1] Citation Audit, [2] Gap Alert, [3] Supplement Suggestions |",
            "| `output/change_log.md` | Detailed log of all changes made to the text |",
            "| `output/cross_reference.md` | Claim-to-anchor cross-reference table |",
            "",
            "## How to Use",
            "",
            "1. **Start with `output/*.tex`** — this is the rewritten text, ready for your manuscript",
            "2. Check `change_log.md` to understand what was changed and why",
            "3. Review `Citation_Audit_Report_*.md` for the full verification details",
            "4. Copy BibTeX entries from `references.bib` into your manuscript's .bib file",
            "5. Lines marked `% [AUTO-ADDED]` indicate automatically inserted citations — review these",
            "",
            "> **Note**: All recommended papers are from the verified knowledge base.",
            "> No hallucinated papers are ever suggested.",
        ]
        return "\n".join(lines)

    # ==================================================================
    # Legacy: Related Work Pipeline
    # ==================================================================

    def run_phase0(self, raw_input: str = "", draft_content: str = None) -> UserRequirement:
        self.requirement = BrainstormAgent.run(raw_input, draft_content)
        return self.requirement

    def run_phases_1_to_6(self, requirement: UserRequirement = None) -> dict:
        if requirement:
            self.requirement = requirement
        if not self.requirement:
            raise RuntimeError("No requirement.")

        req = self.requirement
        method_name = req.method_name or "unknown"
        task_dir = os.path.join(self.workspace_dir, sanitize_dirname(method_name))
        os.makedirs(task_dir, exist_ok=True)
        req.save(os.path.join(task_dir, "requirements.json"))

        print("\n" + "=" * 50)
        print("  PHASE 1: Topic Specialist")
        print("=" * 50)
        self.profile = TopicSpecialist.build_profile_from_requirement(req)
        self.search_task = TopicSpecialist.generate_search_task(self.profile)
        if req.target_venue:
            self.search_task.target_venues = [v.strip() for v in req.target_venue.split(",")]
        if req.year_range:
            self.search_task.year_range = tuple(req.year_range)

        print("\n" + "=" * 50)
        print("  PHASE 2: Search Agent")
        print("=" * 50)
        self.papers = SearchAgent.search(self.search_task)
        print(f"  Found {len(self.papers)} candidate papers.")

        print("\n" + "=" * 50)
        print("  PHASE 3: Archivist")
        print("=" * 50)
        self.archive, rejections = Archivist.build_archive(
            self.papers, req.method_summary, strict_mode=req.strict_mode
        )
        if rejections:
            for r in rejections:
                print(f"    REJECTED: {r}")
        self.core_papers = Archivist.select_core_papers(self.archive, max_count=req.max_papers)
        print(f"  {len(self.core_papers)} core papers selected.")

        print("\n" + "=" * 50)
        print("  PHASE 4: Synthesis Writer")
        print("=" * 50)
        self.sections = SynthesisWriter.write_related_work(
            self.core_papers,
            user_method_name=method_name,
            user_method_summary=req.method_summary,
            length_preference=req.length_preference,
        )
        self.final_draft = SynthesisWriter.assemble_full_draft(self.sections)

        print("\n" + "=" * 50)
        print("  PHASE 5: Fact-Checker")
        print("=" * 50)
        self.audit_report, corrected = FactChecker.audit(self.sections, self.core_papers)
        print(f"  Verdict: {self.audit_report.final_verdict}")

        final_path = os.path.join(task_dir, "related_work.tex")
        with open(final_path, "w", encoding="utf-8") as f:
            f.write(self.final_draft)
        report_text = FactChecker.format_report(self.audit_report)
        report_path = os.path.join(task_dir, "audit_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        bibtex = Archivist.export_bibtex(self.core_papers)
        bib_path = os.path.join(task_dir, "references.bib")
        with open(bib_path, "w", encoding="utf-8") as f:
            f.write(bibtex)

        print(f"\n  related_work.tex → {final_path}")
        print(f"  audit_report.md  → {report_path}")
        print(f"  references.bib   → {bib_path}")
        return {"task_dir": task_dir, "final_draft": final_path,
                "audit_report": report_path, "references": bib_path}

    def run_all(self, raw_input: str = "", draft_content: str = None) -> dict:
        self.run_phase0(raw_input, draft_content)
        return self.run_phases_1_to_6(self.requirement)
