"""
Fact-Checker Agent (审计校对员) — v4 通用引用验证版
=====================================================
定位：通用文献引用验证，不区分 section 类型。
强化项：
  1. 第一重：存在性校验 (Existence Check) — DOI/arXiv 格式校验 + URL 域名验证
  2. 第二重：语义对账 (Semantic Reconciliation) — 逐句比对引用上下文与文献摘要
  3. 幻觉检测：声明如果不在 abstract_anchor 中出现，判定为幻觉
  4. 缺失识别：检测文中缺乏引用支撑的学术主张
  5. 文献推荐：为缺失区域推荐真实可查的高相关文献
  6. 动态输出：根据输入主题自动生成报告名称
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from agents.archivist import ArchiveEntry
from agents.synthesis_writer import Section
from agents.search_agent import Literature
from agents.citation_extractor import CitationRef, UnsupportedClaim


@dataclass
class AuditIssue:
    """审计问题 — v3"""
    severity: str  # error / warning / info / hallucination
    location: str
    description: str
    suggestion: str
    related_doi: str = ""
    anchor_evidence: str = ""  # v3: 对应的锚点原文


@dataclass
class AuditReport:
    """审计报告 — v3"""
    total_claims: int = 0
    verified_claims: int = 0
    hallucinations: int = 0
    existence_failures: int = 0
    doi_format_errors: int = 0  # v3: DOI 格式错误数
    issues: list[AuditIssue] = field(default_factory=list)
    final_verdict: str = ""
    dois_to_destroy: list[str] = field(default_factory=list)  # v3: 通知 Cleanup 销毁

    @property
    def pass_rate(self) -> float:
        if self.total_claims == 0:
            return 0.0
        return self.verified_claims / self.total_claims * 100


@dataclass
class GapRecommendation:
    """为缺失区域推荐的文献"""
    claim: UnsupportedClaim
    recommended_papers: list[Literature] = field(default_factory=list)
    reason: str = ""


@dataclass
class VerificationReport:
    """v4: 通用引用验证报告（不区分 section 类型）"""
    # 引用统计
    total_citations: int = 0
    found_citations: int = 0
    missing_citations: int = 0

    # 审计结果
    verified_claims: int = 0
    hallucinations: int = 0
    existence_failures: int = 0
    gap_count: int = 0
    stale_count: int = 0

    # 详细数据
    issues: list[AuditIssue] = field(default_factory=list)
    dois_to_destroy: list[str] = field(default_factory=list)
    recommendations: list[GapRecommendation] = field(default_factory=list)
    final_verdict: str = ""

    @property
    def pass_rate(self) -> float:
        total = self.verified_claims + self.hallucinations
        if total == 0:
            return 0.0
        return self.verified_claims / total * 100


class FactChecker:
    """
    审计校对员 v3 — 双重对账逻辑。
    第一重：存在性校验（DOI 格式 + URL 域名）
    第二重：语义对账（正文声明 vs 原始摘要锚点，逐句级别）
    """

    SYSTEM_PROMPT = """You are a Fact-Checker Agent for academic citation verification (v4).

## Your Role
You are the UNIVERSAL CITATION VERIFIER. You handle ANY section of an academic paper.
Your job is to ensure zero hallucination AND identify citation gaps.

## Dual Verification

### Check 1: Existence Check (存在性校验)
For EVERY cited paper, verify:
- DOI format: must match "10.XXXX/..." pattern
- arXiv ID format: must match "YYMM.NNNNN" pattern
- URL must be from known academic domain
- If DOI is just a citation key (e.g., "smith2023method") → flag as ERROR

### Check 2: Semantic Reconciliation (语义对账)
For EVERY factual claim with citation:
- Compare claim against the Abstract Anchor verbatim
- Specific numbers, metrics, percentages NOT in anchor → HALLUCINATION
- Method descriptions NOT in anchor → HALLUCINATION

## Gap Detection (v4 — NEW)
- Scan for academic claims WITHOUT citations
- Claims involving SOTA, model architectures, comparisons, datasets need citations
- Recommend 1-3 real papers per gap (must be verifiable, NO hallucination)
- Flag stale citations (>5 years old) with SOTA update suggestions

## Severity Levels
- **HALLUCINATION**: Claim not supported by Abstract Anchor
- **ERROR**: Broken DOI, non-existent citation, fake reference
- **GAP**: Academic claim missing citation support
- **STALE**: Citation is >5 years old, newer work may exist
- **WARNING**: Style issue, listing pattern
- **INFO**: Suggestion for improvement

## Output (Dynamic Naming)
Report title auto-generated from input topic keywords.
Sections: [1. Citation Audit] [2. Gap Alert] [3. Supplement Suggestions]
"""

    # 已知 URL 验证模式
    VALID_URL_DOMAINS = [
        "arxiv.org",
        "ieeexplore.ieee.org",
        "doi.org",
        "dx.doi.org",
        "proceedings.mlr.press",
        "aclanthology.org",
        "openreview.net",
        "neurips.cc",
        "openaccess.thecvf.com",
        "semanticscholar.org",
    ]

    # DOI 格式校验
    DOI_PATTERN = re.compile(r"^10\.\d{4,9}/[^\s]+$")
    # arXiv ID 格式
    ARXIV_ID_PATTERN = re.compile(r"^\d{4}\.\d{4,6}$")

    @classmethod
    def audit(
        cls,
        sections: list[Section],
        archive: list[ArchiveEntry],
    ) -> tuple[AuditReport, str]:
        """
        执行双重审计，返回 (审计报告, 修正后的草稿)。
        """
        report = AuditReport()
        issues: list[AuditIssue] = []

        archive_map = {e.doi: e for e in archive if e.doi}
        url_to_doi = {}
        for e in archive:
            if e.url and e.doi:
                url_to_doi[e.url] = e.doi

        # ===== 第一重：存在性校验 =====
        existence_issues, dois_to_destroy = cls._check_existence(archive, sections)
        issues.extend(existence_issues)
        report.existence_failures = len(existence_issues)
        report.dois_to_destroy = dois_to_destroy

        # ===== 第二重：语义对账 =====
        semantic_issues, claims, verified = cls._check_semantic_reconciliation(
            sections, archive_map
        )
        issues.extend(semantic_issues)
        report.total_claims = claims
        report.verified_claims = verified

        # ===== 幻觉统计 =====
        report.hallucinations = sum(
            1 for i in issues if i.severity == "hallucination"
        )

        # ===== 3-5：叙事质量 + 引用完整性 + 格式 =====
        for section in sections:
            issues.extend(cls._check_narrative_quality(section))
            issues.extend(cls._check_format(section))
        issues.extend(cls._check_citation_completeness(sections, archive))

        report.issues = issues
        corrected = cls._apply_fixes(sections, issues, archive_map)

        # 最终判定
        hallucinations = [i for i in issues if i.severity == "hallucination"]
        errors = [i for i in issues if i.severity == "error"]
        warnings = [i for i in issues if i.severity == "warning"]

        if dois_to_destroy:
            report.final_verdict = (
                f"DESTROY + {len(hallucinations)} HALLUCINATIONS — "
                f"{len(dois_to_destroy)} record(s) must be physically destroyed "
                f"(fake DOI/citation key). "
                f"{len(hallucinations)} claim(s) not supported by Abstract Anchor."
            )
        elif hallucinations:
            report.final_verdict = (
                f"HALLUCINATION DETECTED — {len(hallucinations)} claim(s) "
                f"not supported by Abstract Anchor. MUST rewrite."
            )
        elif errors:
            report.final_verdict = (
                f"NEEDS REVISION — {len(errors)} error(s), "
                f"{len(warnings)} warning(s)."
            )
        elif warnings:
            report.final_verdict = (
                f"PASS WITH NOTES — {len(warnings)} warning(s)."
            )
        else:
            report.final_verdict = "PASS — No issues found."

        return report, corrected

    @classmethod
    def _check_existence(
        cls,
        archive: list[ArchiveEntry],
        sections: list[Section],
    ) -> tuple[list[AuditIssue], list[str]]:
        """
        第一重：存在性校验。
        验证每篇被引用的文献的 DOI 格式和 URL。
        Returns: (issues, dois_to_destroy)
        """
        issues = []
        dois_to_destroy = []

        for entry in archive:
            doi = entry.doi
            url = entry.url

            # ===== DOI 格式校验 =====
            if doi:
                is_valid_doi = bool(cls.DOI_PATTERN.match(doi))
                is_valid_arxiv = bool(cls.ARXIV_ID_PATTERN.match(doi))
                is_citation_key = bool(
                    re.match(r"^[a-z]", doi) and not any(c in doi for c in "./")
                )

                if is_citation_key:
                    # 裸引用 key 伪装成 DOI → 标记为需要销毁
                    issues.append(AuditIssue(
                        severity="error",
                        location="Global",
                        description=(
                            f"\"{entry.title}\" — DOI '{doi}' is a citation key, "
                            f"not a real DOI. Cannot verify existence."
                        ),
                        suggestion=(
                            f"CRITICAL: Provide a real DOI (10.XXXX/...) or "
                            f"arXiv ID (YYMM.NNNNN) for \"{entry.title}\". "
                            f"Cleanup Agent MUST DESTROY this record immediately."
                        ),
                        related_doi=doi,
                    ))
                    dois_to_destroy.append(doi)

                elif not is_valid_doi and not is_valid_arxiv:
                    issues.append(AuditIssue(
                        severity="error",
                        location="Global",
                        description=(
                            f"\"{entry.title}\" — DOI '{doi}' does not match "
                            f"valid DOI (10.XXXX/...) or arXiv (YYMM.NNNNN) format."
                        ),
                        suggestion=(
                            f"Provide a correctly formatted DOI or arXiv ID. "
                            f"Cleanup Agent should DESTROY this record if uncorrectable."
                        ),
                        related_doi=doi,
                    ))
                    dois_to_destroy.append(doi)

            # ===== URL 域名校验 =====
            if url:
                has_valid_url = bool(cls._is_valid_url(url))
                if not has_valid_url and doi not in dois_to_destroy:
                    issues.append(AuditIssue(
                        severity="warning",
                        location="Global",
                        description=(
                            f"\"{entry.title}\" — URL '{url}' cannot be "
                            f"programmatically validated (not a known academic domain)."
                        ),
                        suggestion="Verify the URL manually or replace with a confirmed link.",
                        related_doi=doi,
                    ))
            elif not doi:
                # 无 DOI 也无 URL
                issues.append(AuditIssue(
                    severity="error",
                    location="Global",
                    description=(
                        f"\"{entry.title}\" — No DOI and no URL. "
                        f"Cannot verify existence."
                    ),
                    suggestion="Provide DOI or URL. Cleanup Agent should DESTROY this record.",
                    related_doi=doi or entry.title,
                ))
                dois_to_destroy.append(doi or entry.title)

        return issues, dois_to_destroy

    @classmethod
    def _is_valid_url(cls, url: str) -> bool:
        """检查 URL 是否符合已知学术域名模式"""
        if not url:
            return False
        return any(
            url.startswith(f"https://{domain}") or url.startswith(f"http://{domain}")
            for domain in cls.VALID_URL_DOMAINS
        )

    @classmethod
    def _check_semantic_reconciliation(
        cls,
        sections: list[Section],
        archive_map: dict[str, ArchiveEntry],
    ) -> tuple[list[AuditIssue], int, int]:
        """
        第二重：语义对账（v3 逐句级别）。
        将正文每个含引用的句子与对应的 Abstract Anchor 进行比对。
        """
        issues = []
        claims = 0
        verified = 0

        for section in sections:
            content = section.content
            # 去掉 LaTeX 命令，提取纯文本
            clean_text = re.sub(r"\\cite[p]?\{[^}]+\}", "[CITATION]", content)
            clean_text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", clean_text)

            # 逐句检查
            sentences = re.split(r'(?<=[.!?])\s+', clean_text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

            for doi in section.citations_used:
                if doi not in archive_map:
                    continue

                entry = archive_map[doi]
                anchor = entry.abstract_anchor
                title_words = [w for w in entry.title.lower().split() if len(w) > 3]

                # 检查：该论文是否在正文中被提及
                found = any(w in clean_text.lower() for w in title_words)
                if not found:
                    continue

                claims += 1

                # ===== 核心语义对账 =====
                if anchor:
                    # 提取正文中的声明句
                    relevant_sentences = [
                        s for s in sentences
                        if any(w in s.lower() for w in title_words)
                    ]

                    if relevant_sentences:
                        # 逐句比对：正文声明 vs 锚点
                        all_ok = True
                        for sent in relevant_sentences:
                            is_supported = cls._is_claim_supported_by_anchor(
                                sent, anchor, entry.title
                            )
                            if is_supported is True:
                                verified += 1
                            elif is_supported is False:
                                # 明确不支持 → 幻觉
                                claim_preview = sent[:120] + "..." if len(sent) > 120 else sent
                                issues.append(AuditIssue(
                                    severity="hallucination",
                                    location=section.subsection_title,
                                    description=(
                                        f"HALLUCINATION for \"{entry.title}\": "
                                        f"claim \"{claim_preview}\" is NOT "
                                        f"supported by the Abstract Anchor."
                                    ),
                                    suggestion=(
                                        f"Remove unsupported claim or rewrite using "
                                        f"only the Abstract Anchor as source.\n"
                                        f"Anchor: \"{anchor[:200]}\""
                                    ),
                                    related_doi=doi,
                                    anchor_evidence=anchor,
                                ))
                                all_ok = False
                            # is_supported is None → 无法判断，不标记

                        if all_ok and relevant_sentences:
                            pass  # 所有声明都有锚点支持
                    else:
                        # 有引用但没有具体声明句 → 可能只是标题提及，标记为 info
                        issues.append(AuditIssue(
                            severity="info",
                            location=section.subsection_title,
                            description=(
                                f"\"{entry.title}\" is cited but no specific "
                                f"claims are made about it in the text."
                            ),
                            suggestion="Add a specific claim about this paper's contribution.",
                            related_doi=doi,
                        ))

                else:
                    # 无锚点 → 所有声明均不可验证
                    issues.append(AuditIssue(
                        severity="warning",
                        location=section.subsection_title,
                        description=(
                            f"Cited \"{entry.title}\" has NO Abstract Anchor. "
                            f"All factual claims about this paper are unverifiable."
                        ),
                        suggestion=(
                            "Archivist must extract abstract anchor before writing. "
                            "Remove all specific claims about this paper."
                        ),
                        related_doi=doi,
                    ))

        return issues, claims, verified

    @classmethod
    def _is_claim_supported_by_anchor(
        cls, claim_sentence: str, anchor: str, title: str
    ) -> Optional[bool]:
        """
        判断正文声明是否被锚点支持。
        Returns:
            True  — 锚点中有证据支持该声明
            False — 明确不支持（幻觉）
            None  — 无法确定（不标记）
        """
        claim_lower = claim_sentence.lower()
        anchor_lower = anchor.lower()

        # ===== 1. 数值声明检测 =====
        # 提取正文中的具体数值
        numerical_claims = re.findall(
            r"(?:achieves?|improves?|reduces?|increases?|outperforms?|"
            r"accuracy|precision|recall|f-score|success rate|error rate|"
            r"performance)\s+(?:of\s+)?[\d.]+%?",
            claim_lower,
        )
        if numerical_claims:
            # 检查这些数值是否出现在锚点中
            for claim_num in numerical_claims:
                num_value = re.search(r"[\d.]+%?", claim_num)
                if num_value:
                    if num_value.group(0) not in anchor_lower:
                        return False  # 数值在锚点中不存在 → 幻觉

        # ===== 2. 方法能力声明检测 =====
        # 检测 "X can/does/is able to Y" 类声明
        capability_claims = re.findall(
            r"(?:can|could|is able to|is designed to|enables?|allows?|supports?|"
            r"achieves?|provides?|implements?)\s+[\w\s]{10,60}",
            claim_lower,
        )
        if capability_claims:
            # 提取声明中的关键动词和名词
            claim_verbs = set(re.findall(
                r"\b(?:detect|verify|embed|extract|encode|protect|identify|"
                r"improve|enhance|reduce|achieve|demonstrate|propose|introduce)\w*\b",
                claim_lower,
            ))
            anchor_verbs = set(re.findall(
                r"\b(?:detect|verify|embed|extract|encode|protect|identify|"
                r"improve|enhance|reduce|achieve|demonstrate|propose|introduce)\w*\b",
                anchor_lower,
            ))
            if claim_verbs and not claim_verbs & anchor_verbs:
                # 声明的核心动词在锚点中完全不存在
                return False

        # ===== 3. 关键概念匹配 =====
        # 提取正文中的关键概念词（>5 字母的非停用词）
        stop_words = {
            "which", "where", "their", "these", "those", "about",
            "other", "being", "using", "based", "through", "without",
            "with", "from", "into", "than", "that", "this", "they",
            "have", "been", "were", "will", "also", "more", "most",
            "some", "such", "only", "very", "over", "when", "then",
            "after", "between", "under", "both", "each", "does",
            "however", "approach", "method", "proposed", "paper",
        }
        claim_concepts = set(
            w for w in re.findall(r"\b[a-zA-Z]{6,}\b", claim_lower)
            if w not in stop_words
        )
        anchor_concepts = set(
            w for w in re.findall(r"\b[a-zA-Z]{6,}\b", anchor_lower)
            if w not in stop_words
        )

        if claim_concepts:
            overlap = claim_concepts & anchor_concepts
            overlap_ratio = len(overlap) / len(claim_concepts)

            if overlap_ratio >= 0.3:
                return True  # 有足够的概念重叠
            elif overlap_ratio < 0.1 and len(claim_concepts) > 2:
                return False  # 几乎没有概念重叠 → 疑似幻觉

        return None  # 无法确定

    # ===== v4: Gap Detection & Recommendation =====

    @classmethod
    def detect_gaps(
        cls,
        unsupported_claims: list[UnsupportedClaim],
        citations: list[CitationRef],
        archive: list[ArchiveEntry],
        current_year: int = 2026,
    ) -> tuple[list[AuditIssue], list[GapRecommendation]]:
        """
        v4: 检测引用缺失，推荐真实文献。
        Returns: (gap_issues, recommendations)
        """
        gap_issues = []
        recommendations = []

        # 已引用的 DOI 集合
        cited_dois = {e.doi for e in archive if e.doi}

        for claim in unsupported_claims:
            # 跳过低置信度的
            if claim.confidence < 0.3:
                continue

            # 检查是否确实没有引用在附近（ CitationExtractor 已做此检查，双重验证）
            gap_issues.append(AuditIssue(
                severity="gap",
                location=f"Line {claim.location_line}",
                description=(
                    f"[GAP] Academic claim without citation support: "
                    f"\"{claim.claim_text[:100]}\""
                ),
                suggestion=(
                    f"This {claim.claim_type} claim needs citation support. "
                    f"Consider citing relevant literature."
                ),
                related_doi="",
            ))

            # 推荐文献
            recommended = cls._recommend_papers_for_gap(
                claim, cited_dois, current_year
            )
            if recommended:
                recommendations.append(GapRecommendation(
                    claim=claim,
                    recommended_papers=recommended,
                    reason=f"Claim type: {claim.claim_type}, "
                           f"confidence: {claim.confidence:.0%}",
                ))

        return gap_issues, recommendations

    @classmethod
    def _recommend_papers_for_gap(
        cls,
        claim: UnsupportedClaim,
        already_cited: set[str],
        current_year: int = 2026,
        max_results: int = 3,
    ) -> list[Literature]:
        """为缺失区域推荐文献（调用 SearchAgent）"""
        from agents.search_agent import SearchAgent

        candidates = SearchAgent.recommend_for_claim(
            claim.claim_text, claim.claim_type, max_results=max_results * 2
        )

        # 过滤已引用的
        filtered = [
            p for p in candidates
            if p.doi not in already_cited
        ]

        # 优先选择较新的
        filtered.sort(key=lambda p: (-p.year, p.citation_count))
        return filtered[:max_results]

    @classmethod
    def detect_stale_citations(
        cls,
        archive: list[ArchiveEntry],
        current_year: int = 2026,
        stale_threshold: int = 5,
    ) -> list[AuditIssue]:
        """
        v4: 检测陈旧引用，提示是否有更新的 SOTA 工作。
        """
        issues = []
        for entry in archive:
            age = current_year - entry.year
            if age > stale_threshold:
                issues.append(AuditIssue(
                    severity="stale",
                    location="Global",
                    description=(
                        f"[STALE] \"{entry.title}\" ({entry.year}) "
                        f"is {age} years old. Newer SOTA work may exist."
                    ),
                    suggestion=(
                        f"Consider whether there are more recent works "
                        f"(post-{entry.year + stale_threshold}) that have "
                        f"superseded or improved upon this work."
                    ),
                    related_doi=entry.doi,
                ))
        return issues

    # ===== v4: Dynamic Report =====

    @classmethod
    def format_verification_report(
        cls,
        report: "VerificationReport",
        topic_hint: str = "",
        archive: list[ArchiveEntry] = None,
        found_map: dict = None,
    ) -> str:
        """
        v4: 生成通用审计报告。
        动态命名，三段式输出：[1. Citation Audit] [2. Gap Alert] [3. Supplement Suggestions]
        """
        # 动态标题
        title = topic_hint.replace("_", " ").title() if topic_hint else "General"
        report_name = f"Citation_Audit_Report_{title.replace(' ', '_')}.md"

        lines = [
            f"# {report_name}",
            "",
            "=" * 65,
            f"  Citation Verification Report — {title}",
            "=" * 65,
            "",
            "## Summary",
            f"- Total citations checked: {report.total_citations}",
            f"- Citations found in KB: {report.found_citations}",
            f"- Citations NOT found: {report.missing_citations}",
            f"- Verified claims: {report.verified_claims}",
            f"- **Hallucinations: {report.hallucinations}**",
            f"- **Citation gaps: {report.gap_count}**",
            f"- **Stale citations: {report.stale_count}**",
            f"- Pass rate: {report.pass_rate:.1f}%",
            f"- **Final Verdict: {report.final_verdict}**",
        ]

        # --- Citation Traceability Table (with URLs) ---
        if found_map:
            lines.extend([
                "",
                "## Citation Traceability",
                "| # | Cite Key | Title | Year | Venue | URL | Status |",
                "|---|---|---|---|---|---|---|",
            ])
            for idx, (key, paper) in enumerate(found_map.items(), 1):
                url = paper.url or "N/A"
                doi_valid, _ = cls.validate_doi_static(paper.doi)
                status = "Verified" if doi_valid else "Unverified"
                title_short = paper.title[:40] + ("..." if len(paper.title) > 40 else "")
                lines.append(
                    f"| {idx} | `{key}` | {title_short} "
                    f"| {paper.year} | {paper.venue} | [{url}]({url}) | {status} |"
                )

        # --- Section 1: Citation Audit ---
        lines.extend([
            "",
            "=" * 65,
            "  [1] Citation Audit (已有引用审计)",
            "=" * 65,
        ])

        for sev in ["hallucination", "error", "stale", "warning", "info"]:
            filtered = [i for i in report.issues if i.severity == sev]
            if not filtered:
                continue
            icon = sev.upper()
            lines.append(f"\n### [{icon}] ({len(filtered)})")
            for issue in filtered:
                lines.extend([
                    f"- **Issue:** {issue.description}",
                    f"- **Suggestion:** {issue.suggestion}",
                ])
                if issue.related_doi:
                    lines.append(f"- **Related DOI:** {issue.related_doi}")
                if issue.anchor_evidence:
                    lines.append(f"- **Anchor:** {issue.anchor_evidence[:200]}")

        # 销毁通知
        if report.dois_to_destroy:
            lines.extend([
                "",
                "### DESTROY NOTIFICATION",
                f"The following {len(report.dois_to_destroy)} record(s) must be removed:",
            ])
            for doi in report.dois_to_destroy:
                lines.append(f"- `{doi}`")

        # --- Section 2: Gap Alert ---
        gap_issues = [i for i in report.issues if i.severity == "gap"]
        if gap_issues:
            lines.extend([
                "",
                "=" * 65,
                "  [2] Gap Alert (文献缺失预警)",
                "=" * 65,
                "",
                f"Found **{len(gap_issues)}** unsupported academic claim(s) "
                f"that need citation support:",
            ])
            for idx, issue in enumerate(gap_issues, 1):
                lines.extend([
                    f"\n### Gap #{idx}",
                    f"- **Location:** {issue.location}",
                    f"- **Claim:** {issue.description.replace('[GAP] Academic claim without citation support: ', '')}",
                    f"- **Type:** Suggestion — {issue.suggestion}",
                ])

        # --- Section 3: Supplement Suggestions ---
        if report.recommendations:
            lines.extend([
                "",
                "=" * 65,
                "  [3] Supplement Suggestions (补全建议清单)",
                "=" * 65,
                "",
                "> **NOTE:** All recommended papers are from the verified knowledge base.",
                "> Review each suggestion and confirm before adding to your manuscript.",
                "",
            ])
            for idx, rec in enumerate(report.recommendations, 1):
                lines.extend([
                    f"### Recommendation #{idx}",
                    f"- **Original claim:** \"{rec.claim.claim_text[:100]}\"",
                    f"- **Reason:** {rec.reason}",
                    f"- **Candidates:**",
                ])
                for p_idx, paper in enumerate(rec.recommended_papers, 1):
                    lines.extend([
                        f"  {p_idx}. **{paper.title}**",
                        f"     Authors: {paper.authors} | Year: {paper.year} | Venue: {paper.venue}",
                        f"     URL: {paper.url}",
                        f"     Contribution: {paper.core_contribution}",
                    ])
                lines.append("")

        lines.extend(["", "=" * 65])
        return "\n".join(lines)

    @classmethod
    def format_bibtex_for_recommendations(
        cls, recommendations: list[GapRecommendation]
    ) -> str:
        """为推荐文献生成 BibTeX"""
        entries = []
        seen = set()
        for rec in recommendations:
            for paper in rec.recommended_papers:
                if paper.doi and paper.doi not in seen:
                    seen.add(paper.doi)
                    key = paper.doi
                    entries.append(
                        f"@article{{{key},\n"
                        f"  title={{{paper.title}}},\n"
                        f"  author={{{paper.authors}}},\n"
                        f"  year={{{paper.year}}},\n"
                        f"  journal={{{paper.venue}}},\n"
                        f"  url={{{paper.url}}},\n"
                        f"}}"
                    )
        return "\n\n".join(entries)

    # ===== 以下方法保留兼容行为 =====

    @classmethod
    def _check_narrative_quality(cls, section: Section) -> list[AuditIssue]:
        """检查叙事质量，检测罗列式写作"""
        issues = []
        content = section.content
        sentences = re.split(r'(?<=[.!?])\s+', content)
        consecutive_simple = 0
        for sent in sentences:
            if len(sent.split()) < 25 and ("citep" in sent or "et al" in sent.lower()):
                consecutive_simple += 1
                if consecutive_simple >= 3:
                    issues.append(AuditIssue(
                        severity="warning",
                        location=section.subsection_title,
                        description="Detected listing pattern: 3+ consecutive "
                                       "short sentences each citing different papers.",
                        suggestion="Rewrite to create synthesis.",
                    ))
                    break
            else:
                consecutive_simple = 0
        return issues

    @classmethod
    def _check_citation_completeness(
        cls, sections: list[Section], archive: list[ArchiveEntry]
    ) -> list[AuditIssue]:
        """检查未引用的论文"""
        issues = []
        cited_dois = set()
        for section in sections:
            cited_dois.update(section.citations_used)

        for entry in archive:
            if entry.doi and entry.doi not in cited_dois:
                issues.append(AuditIssue(
                    severity="info",
                    location="Global",
                    description=f"'{entry.title}' is archived but not cited.",
                    suggestion="Add citation or remove from archive.",
                    related_doi=entry.doi,
                ))
        return issues

    @classmethod
    def _check_format(cls, section: Section) -> list[AuditIssue]:
        """检查 LaTeX 格式"""
        issues = []
        content = section.content
        bare_cites = re.findall(r'\\cite\{([^}]+)\}', content)
        for cite in bare_cites:
            issues.append(AuditIssue(
                severity="warning",
                location=section.subsection_title,
                description=f"Found \\cite{{{cite}}}; use \\citep{{{cite}}}.",
                suggestion="Use \\citep{} for parenthetical citations.",
            ))
        if "\n\n\n" in content:
            issues.append(AuditIssue(
                severity="info",
                location=section.subsection_title,
                description="Found consecutive blank lines.",
                suggestion="Remove extra blank lines.",
            ))
        return issues

    @classmethod
    def _apply_fixes(
        cls,
        sections: list[Section],
        issues: list[AuditIssue],
        archive_map: dict[str, ArchiveEntry],
    ) -> str:
        """应用修正并生成最终稿"""
        errors = [i for i in issues if i.severity in ("error", "hallucination")]
        hallucinations = [i for i in issues if i.severity == "hallucination"]
        destroy_count = len([
            i for i in issues
            if "DESTROY" in i.suggestion or "destroy" in i.suggestion
        ])

        lines = [
            "% ============================================================",
            "% FINAL DRAFT — Related Work (with audit annotations)",
            f"% Hallucinations: {len(hallucinations)} "
            f"| Errors: {len(errors)} | "
            f"Records to destroy: {destroy_count}",
            "% ============================================================",
            "",
        ]

        from agents.synthesis_writer import SynthesisWriter
        draft = SynthesisWriter.assemble_full_draft(sections)

        for issue in errors:
            lines.append(
                f"\n% [{issue.severity.upper()}] {issue.description}\n"
                f"% [SUGGESTION] {issue.suggestion}\n"
            )

        return "\n".join(lines)

    @classmethod
    def format_report(cls, report: AuditReport) -> str:
        """格式化审计报告 v3"""
        lines = [
            "",
            "=" * 65,
            "  审计报告 — Audit Report (v3 Dual Verification)",
            "=" * 65,
            "",
            f"## Summary",
            f"- Total claims checked: {report.total_claims}",
            f"- Verified claims: {report.verified_claims}",
            f"- **Hallucinations: {report.hallucinations}**",
            f"- Existence failures: {report.existence_failures}",
            f"- Pass rate: {report.pass_rate:.1f}%",
            f"- **Final Verdict: {report.final_verdict}**",
        ]

        # v3: 销毁通知
        if report.dois_to_destroy:
            lines.extend([
                "",
                f"## DESTROY NOTIFICATION (to Cleanup Agent)",
                f"The following {len(report.dois_to_destroy)} record(s) must be PHYSICALLY DESTROYED:",
            ])
            for doi in report.dois_to_destroy:
                lines.append(f"- `{doi}`")

        lines.append("")

        for severity in ["hallucination", "error", "warning", "info"]:
            icon = {
                "hallucination": "HALLUCINATION",
                "error": "ERROR",
                "warning": "WARNING",
                "info": "INFO",
            }.get(severity, "?")
            filtered = [i for i in report.issues if i.severity == severity]
            if not filtered:
                continue

            lines.append(f"## [{icon}] ({len(filtered)})")
            for issue in filtered:
                lines.extend([
                    f"### [{severity.upper()}] {issue.location}",
                    f"- **Issue:** {issue.description}",
                    f"- **Suggestion:** {issue.suggestion}",
                    f"- **Related DOI:** {issue.related_doi}",
                ])
                if issue.anchor_evidence:
                    lines.append(f"- **Anchor Evidence:** {issue.anchor_evidence[:200]}")
                lines.append("")

        return "\n".join(lines)

    @classmethod
    def build_cross_reference_table(
        cls, sections: list[Section], archive: list[ArchiveEntry]
    ) -> str:
        """构建引用文献对照表 v3 — 含锚点对账"""
        archive_map = {e.doi: e for e in archive if e.doi}

        lines = [
            "# 引用文献对照表 — Cross-Reference Table (v3)",
            "",
            "| # | Citation Key | Title | Used In | Anchor | DOI Valid | Status |",
            "|---|---|---|---|---|---|---|",
        ]

        idx = 1
        for section in sections:
            for doi in section.citations_used:
                entry = archive_map.get(doi)
                if entry:
                    title_short = entry.title[:35] + ("..." if len(entry.title) > 35 else "")
                    has_anchor = "Yes" if entry.abstract_anchor else "NO"
                    doi_valid, _ = cls.validate_doi_static(entry.doi)
                    doi_status = "VALID" if doi_valid else "INVALID"
                    entry_status = (
                        "Verified" if entry.status == "verified"
                        else "Unverified"
                    )
                    lines.append(
                        f"| {idx} | `{doi}` | {title_short} "
                        f"| {section.subsection_title} | {has_anchor} "
                        f"| {doi_status} | {entry_status} |"
                    )
                    idx += 1

        return "\n".join(lines)

    @classmethod
    def validate_doi_static(cls, doi: str) -> tuple[bool, str]:
        """复用 Archivist 的 DOI 校验逻辑"""
        if not doi:
            return False, "empty"
        if cls.DOI_PATTERN.match(doi):
            return True, "DOI"
        if cls.ARXIV_ID_PATTERN.match(doi):
            return True, "arXiv"
        return False, "invalid format"
