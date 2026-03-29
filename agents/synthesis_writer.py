"""
Synthesis Writer Agent (合成撰写员)
====================================
职责：调用档案员的数据进行分类综述，按逻辑流撰写相关工作章节。
要求：必须按逻辑流（演进式、分类式、对比式）撰写，严禁罗列。
"""

from dataclasses import dataclass
from typing import Optional
from agents.archivist import ArchiveEntry


@dataclass
class Section:
    """章节段落"""
    subsection_title: str
    narrative_strategy: str  # evolutionary / taxonomic / comparative
    content: str
    citations_used: list[str]  # 使用的 DOI 列表


class SynthesisWriter:
    """
    合成撰写员：基于档案数据，按逻辑流撰写相关工作章节。
    """

    SYSTEM_PROMPT = """You are a Synthesis Writer Agent for academic related-work writing.

## Your Role
You are the FOURTH agent in the pipeline. You receive the literature archive from the
Archivist Agent and must write a high-quality, logically structured Related Work section.

## CRITICAL RULES
1. **NO LISTING.** Do NOT write "Method A does X. Method B does Y. Method C does Z."
   Instead, write narrative prose that connects papers logically.
2. **THREE narrative strategies:**
   - **Evolutionary**: Show how the field progressed over time (A → B → C)
   - **Taxonomic**: Organize by categories, showing within-category evolution
   - **Comparative**: Explicitly compare approaches, highlighting trade-offs
3. **Every claim must be traceable** to an ArchiveEntry's Key Concept.
4. **Lead with limitations** of prior work to naturally motivate our contribution.
5. **Smooth transitions** between subsections and between papers.
6. **LaTeX format** with proper \\citep{} commands.

## Writing Style
- Academic but accessible
- Each paragraph should have a clear topic sentence
- Use transition phrases: "Building upon...", "In contrast to...", "While X addresses...",
  "A fundamental limitation of...", "Recent work has shifted towards..."
- End each subsection by identifying the gap that our method fills

## Structure
1. Opening paragraph: frame the overall research landscape
2. Subsection 1: Category 1 (with internal narrative flow)
3. Subsection 2: Category 2 (with internal narrative flow)
4. [Optional] Subsection 3+: Additional categories
5. Closing paragraph: identify the collective gap and preview our contribution
"""

    @classmethod
    def write_related_work(
        cls,
        archive: list[ArchiveEntry],
        user_method_name: str = "VeriPatent",
        user_method_summary: str = "",
        subsections: Optional[list[str]] = None,
        length_preference: str = "concise",
    ) -> list[Section]:
        """
        基于档案撰写相关工作章节。

        Args:
            archive: 文献档案列表
            user_method_name: 用户方法名称
            user_method_summary: 用户方法简介
            subsections: 自定义子章节标题，None 则自动生成
        """
        if subsections is None:
            subsections = cls._suggest_subsections(archive)

        sections = []

        for title in subsections:
            # 筛选该子章节相关的档案
            relevant = cls._filter_for_subsection(archive, title)

            # 确定叙述策略
            strategy = cls._determine_strategy(title)

            # 生成段落内容
            content = cls._compose_section(
                title=title,
                entries=relevant,
                strategy=strategy,
                user_method_name=user_method_name,
                user_method_summary=user_method_summary,
            )

            citations = [e.doi for e in relevant if e.doi]

            sections.append(Section(
                subsection_title=title,
                narrative_strategy=strategy,
                content=content,
                citations_used=citations,
            ))

        return sections

    @classmethod
    def _suggest_subsections(cls, archive: list[ArchiveEntry]) -> list[str]:
        """基于档案内容建议子章节结构"""
        types = set(e.method_type for e in archive)
        subsections = []

        if "passive" in types:
            subsections.append("Passive Fingerprinting")
        if "watermarking" in types:
            subsections.append("Text Watermarking")
        if "proactive" in types:
            subsections.append("Proactive Fingerprinting")
        if "other" in types:
            subsections.append("Model Security and Ownership Verification")

        return subsections

    @classmethod
    def _filter_for_subsection(
        cls, archive: list[ArchiveEntry], title: str
    ) -> list[ArchiveEntry]:
        """根据子章节标题筛选相关档案"""
        title_lower = title.lower()
        mapping = {
            "passive fingerprinting": "passive",
            "text watermarking": "watermarking",
            "proactive fingerprinting": "proactive",
            "model security": "other",
        }

        target_type = None
        for key, mtype in mapping.items():
            if key in title_lower:
                target_type = mtype
                break

        if target_type:
            return [e for e in archive if e.method_type == target_type]
        return archive

    @classmethod
    def _determine_strategy(cls, title: str) -> str:
        """确定叙述策略"""
        title_lower = title.lower()
        if "passive" in title_lower:
            return "taxonomic"  # 按白盒/黑盒分类
        if "proactive" in title_lower:
            return "evolutionary"  # 按演进逻辑
        if "watermarking" in title_lower:
            return "comparative"  # 对比式
        return "evolutionary"

    @classmethod
    def _compose_section(
        cls,
        title: str,
        entries: list[ArchiveEntry],
        strategy: str,
        user_method_name: str,
        user_method_summary: str,
    ) -> str:
        """
        撰写单个子章节内容。
        返回 LaTeX 格式的段落。
        """
        if not entries:
            return f"% No entries found for {title}"

        lines = [f"\\subsection{{{title}}}"]
        lines.append("")

        if strategy == "taxonomic":
            lines.extend(cls._write_taxonomic(entries, user_method_name))
        elif strategy == "evolutionary":
            lines.extend(cls._write_evolutionary(entries, user_method_name))
        elif strategy == "comparative":
            lines.extend(cls._write_comparative(entries, user_method_name))
        else:
            lines.extend(cls._write_evolutionary(entries, user_method_name))

        return "\n".join(lines)

    @classmethod
    def _write_taxonomic(
        cls, entries: list[ArchiveEntry], user_method_name: str
    ) -> list[str]:
        """分类式叙述：按白盒/黑盒分类，展示各自方法"""
        white_box = [e for e in entries if e.access_type == "white-box"]
        black_box = [e for e in entries if e.access_type in ("black-box", "both")]

        lines = []

        # 开篇
        lines.append(
            "Passive methods extract inherent and unique characteristics from a model's "
            "behavior or parameters without modifying the model."
        )

        # 白盒部分
        if white_box:
            wb_names = " \\citep{" + ", ".join(e.doi for e in white_box if e.doi) + "}"
            empty_cite = " \\citep{}"
            suffix = wb_names if wb_names != empty_cite else ""
            lines.append(f"For white-box settings{suffix} ")
            for e in white_box:
                cite = f"\\citep{{{e.doi}}}" if e.doi else ""
                lines.append(
                    f"{e.title} {cite} {e.key_concept.split('.')[1] if '.' in e.key_concept else ''}."
                )
            lines.append(
                "However, these approaches require white-box access to model weights, "
                "a requirement often impractical in ownership disputes due to the risk of model theft."
            )

        # 黑盒部分
        if black_box:
            lines.append(
                "For black-box settings, methods optimize specific input patterns to elicit "
                "predefined, unique responses from the model. "
            )
            for e in black_box:
                cite = f"\\citep{{{e.doi}}}" if e.doi else ""
                lines.append(f"{e.title} {cite} {e.key_concept.split('.')[1] if '.' in e.key_concept else ''}.")

        # 收尾：点出局限性
        lines.append(
            "A fundamental limitation of passive techniques is their lack of inherent unforgeability; "
            "any party with API access could potentially discover or replicate such behavioral signatures, "
            "enabling false ownership claims. "
            "Building upon biosensor detection principles, "
            f"our approach, {user_method_name}, implements multi-layered verification "
            "that is resistant to simple replication attempts."
        )

        return lines

    @classmethod
    def _write_evolutionary(
        cls, entries: list[ArchiveEntry], user_method_name: str
    ) -> list[str]:
        """演进式叙述：展示方法随时间的演进"""
        lines = []

        # 按年份排序
        sorted_entries = sorted(entries, key=lambda e: (e.year, e.title))

        # 开篇
        lines.append(
            "Proactive methods actively embed fingerprints during the model's training "
            "or modification phase, aiming to bind the fingerprint exclusively to the legitimate trainer."
        )

        # 早期工作
        early = [e for e in sorted_entries if e.year <= 2023]
        if early:
            lines.append("Early work, such as ")
            for e in early:
                cite = f"\\citep{{{e.doi}}}" if e.doi else e.title
                lines.append(f"{cite} {e.key_concept.split('.')[1] if '.' in e.key_concept else ''}.")
            lines.append("")

        # 中期改进
        mid = [e for e in sorted_entries if 2024 <= e.year <= 2024]
        if mid:
            lines.append("Subsequent work improved upon these foundations. ")
            for e in mid:
                cite = f"\\citep{{{e.doi}}}" if e.doi else e.title
                lines.append(f"{cite} {e.key_concept.split('.')[1] if '.' in e.key_concept else ''}.")

        # 近期方法
        recent = [e for e in sorted_entries if e.year >= 2025]
        if recent:
            lines.append("More recent approaches include ")
            for e in recent:
                cite = f"\\citep{{{e.doi}}}" if e.doi else e.title
                lines.append(f"{cite} {e.key_concept.split('.')[1] if '.' in e.key_concept else ''}.")

        # 收尾：共同缺陷 + 我们的贡献
        lines.append(
            "A common weakness across most proactive methods is their reliance on embedding "
            "the fingerprint solely within the model. Under a realistic threat model where "
            "adversaries may access model weights, such internal embeddings offer limited "
            "security guarantees and are vulnerable to removal via fine-tuning or other attacks."
        )
        lines.append("")
        lines.append(
            f"In contrast to these methods, our approach, {user_method_name}, employs an "
            "external encoder with a secret key to establish an external secret, thereby "
            "decoupling the core fingerprint secret from the model parameters. This design "
            "enhances resilience against white-box attacks and enables robust black-box "
            "verification even under sophisticated output manipulation."
        )

        return lines

    @classmethod
    def _write_comparative(
        cls, entries: list[ArchiveEntry], user_method_name: str
    ) -> list[str]:
        """对比式叙述：显式比较不同方法"""
        lines = []

        lines.append(
            "Text watermarking represents a related but distinct approach to model ownership "
            "protection, focusing on embedding detectable signals in generated text rather than "
            "in the model itself."
        )

        for e in entries:
            cite = f"\\citep{{{e.doi}}}" if e.doi else e.title
            lines.append(f"{e.title} {cite} {e.key_concept.split('.')[1] if '.' in e.key_concept else ''}.")

        lines.append(
            "While these watermarking techniques are effective for proving text provenance, "
            "they address a fundamentally different threat model than fingerprinting: "
            "watermarking verifies that specific text was generated by a watermarked model, "
            "whereas fingerprinting verifies model ownership regardless of output."
        )

        return lines

    @classmethod
    def assemble_full_draft(cls, sections: list[Section]) -> str:
        """组装完整草稿"""
        lines = ["\\section{Related Work}\\label{sec:related}", ""]

        for section in sections:
            lines.append(section.content)
            lines.append("")

        return "\n".join(lines)
