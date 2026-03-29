"""
Archivist Agent (文献档案员) — v3 真理守门人版
=====================================================
强化项：
  1. 硬性元数据：严格 DOI/arXiv 格式校验，无合法标识则拒绝入库
  2. 原始摘要锚点：智能提取 1-2 句核心原话（基于信息密度排序），作为写作唯一合法素材
  3. 状态标记：[已验证真伪] / [待核实] / [仅有标题-禁止引用]
  4. 去重检测：基于 DOI + 标题相似度双重去重
  5. 入库报告：输出完整的准入/拒绝日志
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from agents.search_agent import Literature


@dataclass
class ArchiveEntry:
    """归档条目 — v3"""
    title: str
    key_concept: str
    motivation: str
    citation_role: str
    url: str
    doi: str
    # 原始摘要锚点（从原文提取的 1-2 句直接引用，写作唯一合法素材）
    abstract_anchor: str = ""
    authors: str = ""
    year: int = 0
    venue: str = ""
    method_type: str = ""
    access_type: str = ""
    # 状态标记
    status: str = "unverified"  # verified / unverified / title_only
    # v3 新增：拒绝原因（仅 rejected 状态使用）
    rejection_reason: str = ""


@dataclass
class AdmissionReport:
    """入库准入报告 — v3"""
    total_input: int = 0
    admitted: int = 0
    rejected: int = 0
    duplicates_removed: int = 0
    rejection_details: list[str] = field(default_factory=list)
    statistics: dict = field(default_factory=dict)


class Archivist:
    """
    文献档案员 v3 — 真理守门人。
    准入主权：无 DOI/arXiv 不入库，无摘要锚点不升级为 verified。
    """

    SYSTEM_PROMPT = """You are an Archivist Agent for academic related-work writing.

## Your Role
You are the TRUTH GATEKEEPER of the pipeline.
No paper enters the archive without passing your checks.

## Hard Rules (v3 — Strict Enforcement)
1. **DOI/arXiv URL is MANDATORY.** No DOI, no URL → REJECT.
   - Valid DOI: starts with "10." followed by registrant/code (e.g., "10.1145/1234567")
   - Valid arXiv: "YYMM.NNNNN" or "YYMM.NNNNNN" format (e.g., "2301.10226")
   - Valid URL: must match known academic domains (arxiv.org, ieeexplore.ieee.org, etc.)
2. **Abstract Anchor is MANDATORY.** Extract 1-2 sentences verbatim
   from the original abstract as the ONLY legal source material.
   Anchor must be the most information-rich sentences (method + contribution),
   not just the first two sentences.
3. **Status Tagging:**
   - `verified` — DOI format valid + abstract anchor extracted from original text
   - `unverified` — Has DOI-like ID but format not strictly validated
   - `title_only` — Has title only, MUST NOT be cited in final draft
4. **No Duplicates.** Same DOI or highly similar title → reject later entry.

## Admission Criteria
A paper is REJECTED if:
- It has no DOI and no arXiv/IEEE URL
- Its DOI doesn't match "10.XXXX/..." or "YYMM.NNNNN" format
- It's a duplicate of an existing entry (same DOI or >80% title overlap)
"""

    # ===== DOI 格式校验 =====
    DOI_PATTERN = re.compile(r"^10\.\d{4,9}/[^\s]+$")
    # arXiv 编号格式: YYMM.NNNNN 或 YYMM.NNNNNN
    ARXIV_ID_PATTERN = re.compile(r"^\d{4}\.\d{4,6}$")

    # 已知真实 URL 模式（用于快速验证）
    KNOWN_URL_PATTERNS = [
        r"^https://arxiv\.org/abs/\d{4}\.\d{4,5}$",
        r"^https://arxiv\.org/html/\d{4}\.\d{5}",
        r"^https://arxiv\.org/pdf/\d{4}\.\d{4,5}",
        r"^https://(doi\.org|dx\.doi\.org)/10\.",
        r"^https://ieeexplore\.ieee\.org/document/\d+",
        r"^https://(proceedings\.mlr\.press|aclanthology\.org|openreview\.net)/",
    ]

    @classmethod
    def validate_doi(cls, doi: str) -> tuple[bool, str]:
        """
        校验 DOI 格式。
        Returns: (is_valid, reason)
        """
        if not doi:
            return False, "DOI is empty"

        # 检查是否是合法 DOI 格式 (10.XXXX/...)
        if cls.DOI_PATTERN.match(doi):
            return True, "Valid DOI format"

        # 检查是否是合法 arXiv ID (YYMM.NNNNN)
        if cls.ARXIV_ID_PATTERN.match(doi):
            return True, "Valid arXiv ID format"

        # 检查是否是裸引用 key (如 "zeng2025huref...")
        if re.match(r"^[a-z]", doi) and not any(c in doi for c in "./"):
            return False, f"'{doi}' is a citation key, not a real DOI/arXiv ID"

        return False, f"'{doi}' does not match DOI (10.XXXX/...) or arXiv (YYMM.NNNNN) format"

    @classmethod
    def validate_url(cls, url: str) -> tuple[bool, str]:
        """
        校验 URL 是否为已知学术域名。
        Returns: (is_valid, reason)
        """
        if not url:
            return False, "URL is empty"

        for pattern in cls.KNOWN_URL_PATTERNS:
            if re.match(pattern, url):
                return True, f"Matches known academic URL pattern"

        return False, f"'{url}' does not match known academic URL patterns"

    @classmethod
    def _extract_abstract_anchor(cls, paper: Literature) -> str:
        """
        智能提取 1-2 句核心原话作为摘要锚点。
        策略：按信息密度排序，优先选择包含方法描述和贡献的句子。
        这是 Writer 写作时唯一合法的素材来源。
        """
        if not paper.abstract:
            return ""

        # 分句
        sentences = [
            s.strip()
            for s in re.split(r'(?<=[.!?])\s+', paper.abstract)
            if s.strip() and len(s.strip()) > 15
        ]

        if not sentences:
            return ""

        if len(sentences) <= 2:
            return ". ".join(sentences)

        # ===== 智能选择：按信息密度评分 =====
        # 优先包含以下关键词的句子（描述方法和贡献）
        method_keywords = {
            "propose", "introduce", "present", "develop", "design",
            "achieve", "demonstrate", "show", "improve", "outperform",
            "method", "approach", "technique", "framework", "model",
            "fingerprint", "watermark", "verify", "detect", "embed",
            "novel", "efficient", "robust", "effective",
        }

        def sentence_score(s: str) -> float:
            words = set(s.lower().split())
            keyword_hits = len(words & method_keywords)
            # 长度适中（太短信息少，太长可能是背景铺垫）
            length_score = 1.0
            word_count = len(s.split())
            if 15 <= word_count <= 50:
                length_score = 1.5
            elif word_count > 50:
                length_score = 0.7
            # 包含数值的句子通常有具体贡献
            has_numbers = bool(re.search(r'\d+\.?\d*', s))
            number_bonus = 0.3 if has_numbers else 0.0
            return keyword_hits + length_score + number_bonus

        scored = sorted(sentences, key=sentence_score, reverse=True)

        # 取前 2 句（保持原句顺序）
        top_indices = sorted(
            range(len(sentences)),
            key=lambda i: sentence_score(sentences[i]),
            reverse=True,
        )[:2]
        top_indices.sort()  # 恢复原始顺序

        selected = [sentences[i] for i in top_indices]
        return ". ".join(selected) + "."

    @classmethod
    def _detect_duplicate(
        cls, paper: Literature, existing: list[ArchiveEntry]
    ) -> Optional[ArchiveEntry]:
        """
        检测是否与已有条目重复。
        优先按 DOI 精确匹配，其次按标题相似度（>80% token 重叠）。
        """
        # DOI 精确匹配
        if paper.doi:
            for e in existing:
                if e.doi and e.doi.lower() == paper.doi.lower():
                    return e

        # 标题相似度匹配
        title_words = set(w.lower() for w in paper.title.split() if len(w) > 2)
        if len(title_words) < 3:
            return None

        for e in existing:
            existing_words = set(w.lower() for w in e.title.split() if len(w) > 2)
            if not existing_words:
                continue
            overlap = len(title_words & existing_words) / max(len(title_words), len(existing_words))
            if overlap > 0.8:
                return e

        return None

    @classmethod
    def build_archive(
        cls,
        papers: list[Literature],
        user_method_summary: str = "",
        strict_mode: bool = True,
    ) -> tuple[list[ArchiveEntry], list[str]]:
        """
        为所有文献建立归档条目。

        Returns:
            (archive_entries, rejection_reasons)
            rejection_reasons 记录被拒绝的论文及原因
        """
        archive = []
        rejections = []

        for paper in papers:
            # ========== 硬性检查 1: DOI/arXiv URL 格式校验 ==========
            url = paper.url or ""
            doi = paper.doi or ""

            doi_valid, doi_reason = cls.validate_doi(doi)
            url_valid, url_reason = cls.validate_url(url)

            if strict_mode and not doi_valid and not url_valid:
                rejections.append(
                    f"REJECTED: \"{paper.title}\" — "
                    f"DOI 校验失败: {doi_reason} | URL 校验失败: {url_reason}"
                )
                continue

            # ========== 硬性检查 2: 去重检测 ==========
            duplicate = cls._detect_duplicate(paper, archive)
            if duplicate:
                rejections.append(
                    f"DUPLICATE: \"{paper.title}\" — "
                    f"与已入库的 \"{duplicate.title}\" 重复 "
                    f"(DOI: {duplicate.doi})"
                )
                continue

            # ========== 硬性检查 3: 原始摘要锚点 ==========
            anchor = cls._extract_abstract_anchor(paper)

            # ========== 状态标记 ==========
            if doi_valid and anchor:
                status = "verified"
            elif url_valid or doi_valid:
                status = "unverified"
            else:
                status = "title_only"

            # ========== 确定引用角色和动机 ==========
            role, motivation = cls._classify_paper(paper, user_method_summary)
            key_concept = cls._generate_key_concept(paper, anchor)

            entry = ArchiveEntry(
                title=paper.title,
                authors=paper.authors,
                year=paper.year,
                venue=paper.venue,
                method_type=paper.method_type,
                access_type=paper.access_type,
                key_concept=key_concept,
                abstract_anchor=anchor,
                motivation=motivation,
                citation_role=role,
                url=url,
                doi=doi,
                status=status,
            )
            archive.append(entry)

        # 排序
        role_order = {
            "foundational": 0, "predecessor": 1,
            "tool_provider": 2, "baseline": 3,
            "competitor": 4, "attack_target": 5,
        }
        archive.sort(key=lambda e: role_order.get(e.citation_role, 99))

        return archive, rejections

    @classmethod
    def _generate_key_concept(
        cls, paper: Literature, anchor: str
    ) -> str:
        """
        生成核心概念摘要，以锚点为主体。
        锚点 = 合法素材；其余 = 补充说明（标记为 [推断]）
        """
        parts = []

        # 句1: 原始锚点（合法素材）
        if anchor:
            parts.append(anchor)

        # 句2: 方法贡献
        if paper.core_contribution:
            parts.append(f"[推断] {paper.core_contribution}")

        # 句3: 局限性
        if paper.method_type == "passive":
            parts.append(
                "[推断] Lacks inherent unforgeability; "
                "behavioral signatures can potentially be "
                "replicated by any party with API access."
            )
        elif paper.method_type == "proactive":
            parts.append(
                "[推断] Vulnerable to removal attacks when model "
                "weights are accessible; fingerprint embedded "
                "internally can be erased via fine-tuning."
            )
        elif paper.method_type == "watermarking":
            parts.append(
                "[推断] Focuses on text generation watermarking "
                "rather than model ownership verification."
            )
        else:
            parts.append(
                "[推断] Addresses model security concerns "
                "that motivate ownership protection research."
            )

        return " ".join(parts)

    @classmethod
    def _classify_paper(cls, paper: Literature, user_method_summary: str) -> tuple[str, str]:
        """根据论文特征确定引用角色和动机"""
        title_lower = paper.title.lower()

        if paper.year <= 2022 and ("watermark" in title_lower or "fingerprint" in title_lower):
            return "foundational", (
                "Provides the foundational framework that established "
                "the research direction. "
                "Cited to establish historical context."
            )
        if paper.method_type == "passive" and paper.access_type in ("black-box", "both"):
            return "competitor", (
                "Represents the passive fingerprinting paradigm "
                "that we argue lacks inherent unforgeability. "
                "Cited as a competing approach whose limitations "
                "motivate our external secret design."
            )
        if paper.method_type == "passive" and paper.access_type == "white-box":
            return "baseline", (
                "Demonstrates white-box passive fingerprinting "
                "requiring model weight access. "
                "Cited to show this requirement is impractical."
            )
        if paper.method_type == "proactive":
            return "competitor", (
                "Represents the proactive fingerprinting paradigm "
                "that embeds secrets internally. "
                "Cited to demonstrate that internal embeddings "
                "are vulnerable to fine-tuning removal attacks, "
                "which our external secret design overcomes."
            )
        if paper.method_type == "watermarking":
            return "tool_provider", (
                "Provides watermarking techniques relevant to the "
                "broader ownership protection landscape. "
                "Cited to distinguish our fingerprinting approach "
                "from text watermarking methods."
            )
        if "extraction" in title_lower or "membership" in title_lower:
            return "defense_context", (
                "Demonstrates the threat landscape that motivates "
                "ownership verification. "
                "Cited to establish why model protection is critical."
            )

        return "predecessor", (
            "Related prior work in the model protection domain. "
            "Cited for comprehensive coverage."
        )

    # ========== 以下方法保留兼容行为 ==========

    @classmethod
    def select_core_papers(
        cls, archive: list[ArchiveEntry], max_count: int = 8
    ) -> list[ArchiveEntry]:
        """从档案中筛选核心文献，仅允许 verified/unverified 状态。"""
        by_type: dict[str, list[ArchiveEntry]] = {}
        for e in archive:
            # title_only 的文献不允许选入核心清单
            if e.status == "title_only":
                continue
            by_type.setdefault(e.method_type, []).append(e)

        for mtype, pool in by_type.items():
            if mtype == "foundational":
                pool.sort(key=lambda e: e.year)
            else:
                pool.sort(key=lambda e: (-e.year, e.title))

        main_types = ["passive", "proactive"]
        other_types = [t for t in by_type if t not in main_types]
        per_main = max_count // len(main_types)
        remaining = max_count - per_main * len(main_types)

        selected: list[ArchiveEntry] = []
        for mtype in main_types:
            pool = by_type.get(mtype, [])
            take = min(per_main, len(pool))
            selected.extend(pool[:take])

        for mtype in other_types:
            if remaining <= 0:
                break
            pool = by_type.get(mtype, [])
            take = min(remaining, len(pool))
            selected.extend(pool[:take])
            remaining -= take

        if remaining > 0:
            for mtype in main_types:
                if remaining <= 0:
                    break
                pool = by_type.get(mtype, [])
                already = sum(1 for s in selected if s.method_type == mtype)
                extra = pool[already : already + remaining]
                selected.extend(extra)
                remaining -= len(extra)

        type_order = {
            "passive": 0, "foundational": 0,
            "proactive": 1, "watermarking": 2, "other": 3,
        }
        selected.sort(key=lambda e: (type_order.get(e.method_type, 9), e.year))
        return selected

    @classmethod
    def format_core_papers(cls, papers: list[ArchiveEntry]) -> str:
        """格式化核心文献清单"""
        lines = [
            "",
            "=" * 70,
            "  ARCHIVIST v3 — 核心文献确认清单 (Truth Gatekeeper Mode)",
            "=" * 70,
            "",
            f"  共选出 {len(papers)} 篇核心文献。",
            f"  仅含 verified/unverified 状态论文（title_only 已排除）。",
            "",
        ]

        for i, p in enumerate(papers, 1):
            role_label = {
                "competitor": "竞争者 (直接对比对象)",
                "foundational": "奠基者 (开创性工作)",
                "baseline": "基准方法 (对比基准)",
                "tool_provider": "工具提供者",
                "predecessor": "前驱工作",
            }.get(p.citation_role, p.citation_role)

            # 状态标签
            status_label = {
                "verified": "已验证真伪",
                "unverified": "待核实",
                "title_only": "仅有标题-禁止引用",
            }.get(p.status, p.status)

            # DOI 校验信息
            doi_valid, _ = cls.validate_doi(p.doi)
            url_valid, _ = cls.validate_url(p.url)
            validation_tag = "VALID" if doi_valid else ("URL-OK" if url_valid else "WARN")

            lines.extend([
                f"  {'─' * 66}",
                f"  {i}. [{p.method_type.upper()}] [{status_label}] [{validation_tag}] {p.title}",
                f"     Authors: {p.authors}  |  Year: {p.year}  |  Venue: {p.venue}",
                f"     DOI: {p.doi}" if p.doi else "     DOI: N/A",
                f"     URL: {p.url}" if p.url else "     URL: N/A",
                f"     Citation Role: {role_label}",
                f"     Motivation: {p.motivation}",
                f"     Key Concept:",
            ])

            # 先输出原始锚点（合法素材），再输出推断部分
            if p.abstract_anchor:
                lines.append(f"     Abstract Anchor (合法素材):")
                for sent in p.abstract_anchor.split(". "):
                    if sent.strip():
                        lines.append(f"       >> {sent.strip()}.")
            else:
                lines.append(f"     Abstract Anchor: [无 — 所有声明均不可验证]")

            infer_parts = [
                s.strip() for s in p.key_concept.split("[推断]")
                if s.strip()
            ]
            if infer_parts:
                lines.append(f"     Key Concept (推断补充):")
                for part in infer_parts:
                    lines.append(f"       - {part}")

            lines.append("")

        lines.extend([
            "  " + "=" * 70,
            "  说明：[推断] 标记的内容为 Agent 推断，非原文引用。",
            "        Writer 必须仅以 Abstract Anchor 作为事实依据。",
            "        [VALID] = DOI 格式校验通过 | [URL-OK] = URL 域名已知 | [WARN] = 需人工核实",
            "  " + "=" * 70,
            "",
        ])
        return "\n".join(lines)

    @classmethod
    def format_archive_table(cls, archive: list[ArchiveEntry]) -> str:
        """生成 Markdown 格式的文献索引表"""
        lines = [
            "# 文献档案索引表 (Literature Archive) — v3",
            "",
            f"> Total entries: {len(archive)}",
            "> Generated by Archivist Agent v3 | Truth Gatekeeper Mode",
            "> [已验证] = DOI/URL 已确认  [待核实] = 有链接未人工确认  [禁止引用] = 仅有标题",
            "",
        ]

        role_labels = {
            "foundational": "Foundational Works — 奠基性工作",
            "predecessor": "Predecessors — 前驱工作",
            "tool_provider": "Tool Providers — 工具提供者",
            "baseline": "Baselines — 基准方法",
            "competitor": "Competitors — 竞争者",
            "attack_target": "Attack Targets — 攻击目标",
            "defense_context": "Defense Context — 防御上下文",
        }
        status_icons = {
            "verified": "已验证",
            "unverified": "待核实",
            "title_only": "禁止引用",
        }

        current_role = None
        for i, entry in enumerate(archive, 1):
            if entry.citation_role != current_role:
                current_role = entry.citation_role
                label = role_labels.get(current_role, current_role)
                lines.extend(["", f"## {label}", ""])

            status = status_icons.get(entry.status, entry.status)

            lines.extend([
                f"### {i}. {entry.title}  [{status}]",
                "",
                f"- **Authors:** {entry.authors}",
                f"- **Year/Venue:** {entry.year} | {entry.venue}",
                f"- **Type:** {entry.method_type} | **Access:** {entry.access_type}",
                f"- **DOI:** {entry.doi}" if entry.doi else "- **DOI:** N/A",
                f"- **URL:** {entry.url}" if entry.url else "- **URL:** N/A",
                f"- **Abstract Anchor (合法素材):**",
                f"  > {entry.abstract_anchor}" if entry.abstract_anchor else "  > N/A — 不可验证",
                f"- **Key Concept:** {entry.key_concept}",
                f"- **Motivation:** {entry.motivation}",
                f"- **Citation Role:** `{entry.citation_role}`",
                "",
            ])

        return "\n".join(lines)

    @classmethod
    def export_bibtex(cls, archive: list[ArchiveEntry]) -> str:
        """导出 BibTeX 引用"""
        entries = []
        for e in archive:
            cite_key = e.doi.split(",")[0] if e.doi else e.title.lower().replace(" ", "_")[:20]
            lines = [
                f"@article{{{cite_key},",
                f"  title={{{e.title}}},",
                f"  author={{{e.authors}}},",
                f"  year={{{e.year}}},",
                f"  journal={{{e.venue}}},",
            ]
            if e.url:
                lines.append(f"  url={{{e.url}}},")
            if e.doi and e.doi.startswith("10."):
                lines.append(f"  doi={{{e.doi}}},")
            lines.append("}")
            entries.append("\n".join(lines))
        return "\n\n".join(entries)

    @classmethod
    def export_bibtex_from_literature(
        cls,
        papers: list,
        exclude_dois: set[str] = None,
    ) -> str:
        """从 Literature 对象列表导出 BibTeX，可选排除已有 DOI。"""
        exclude = exclude_dois or set()
        entries = []
        for p in papers:
            if p.doi and p.doi in exclude:
                continue
            cite_key = p.cite_key or (p.doi.split(",")[0] if p.doi else p.title.lower().replace(" ", "_")[:20])
            lines = [
                f"@article{{{cite_key},",
                f"  title={{{p.title}}},",
                f"  author={{{p.authors}}},",
                f"  year={{{p.year}}},",
                f"  journal={{{p.venue}}},",
            ]
            if p.url:
                lines.append(f"  url={{{p.url}}},")
            if p.doi and p.doi.startswith("10."):
                lines.append(f"  doi={{{p.doi}}},")
            lines.append("}")
            entries.append("\n".join(lines))
        return "\n\n".join(entries)
