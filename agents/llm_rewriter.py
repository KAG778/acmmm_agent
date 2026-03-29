"""
LLM Rewrite Agent (文本改写员)
====================================
职责：使用 LLM 将 gap 区域改写为完整学术段落，自然融入引用。
约束：仅使用论文摘要作为素材，禁止幻觉；保留原始排版风格（短行换行）。
"""

import os
import re
import textwrap
from anthropic import Anthropic

# conda env: agent_llm
_CONDA_ENV = "/home/wangmeiyi/miniconda3/envs/agent_llm/bin/python3"
_MODEL = "claude-sonnet-4-20250514"  # 实际由 API 路由


class LLMRewriter:
    """使用 LLM 将 unsupported claim 改写为带引用的学术段落。"""

    # 学术文本中安全的断行位置（优先级从高到低）
    _BREAK_PATTERNS = re.compile(
        r',\s+|;\s+|(?<=[.!?])\s+|(?<=[–—])\s+'
    )

    # LaTeX 不可拆分单元（\cite{key}, \ref{key}, (text) 等）
    _LATEX_UNIT_RE = re.compile(
        r'\\cite\{[^}]*\}|\\ref\{[^}]*\}|\\[a-zA-Z]+\{[^}]*\}'
        r'|\([^)]*\)|~\\cite\{[^}]*\}'
    )

    @classmethod
    def _detect_line_width(cls, context: str) -> int:
        """从上下文文本推断原始行宽（中位数），最小 60 字符。"""
        if not context:
            return 72
        widths = [len(line) for line in context.split("\n") if line.strip()]
        if not widths:
            return 72
        widths.sort()
        median = widths[len(widths) // 2]
        return max(median, 60)

    @classmethod
    def _tokenize_for_wrap(cls, text: str) -> list[str]:
        """将文本拆分为 token 列表，保护 LaTeX 结构不被拆分。"""
        tokens = []
        pos = 0
        for m in cls._LATEX_UNIT_RE.finditer(text):
            # LaTeX 单元前的普通文本按空格拆分
            before = text[pos:m.start()]
            if before.strip():
                tokens.extend(before.split())
            # LaTeX 单元整体保留为一个 token
            tokens.append(m.group())
            pos = m.end()
        # 剩余普通文本
        remaining = text[pos:]
        if remaining.strip():
            tokens.extend(remaining.split())
        return tokens

    @classmethod
    def _wrap_to_context_style(cls, text: str, target_width: int) -> str:
        """将 LLM 输出按上下文行宽风格换行。

        策略：
        1. LaTeX 结构（\\cite{}, \\ref{}, 括号）作为不可拆分单元
        2. 优先在标点后（逗号/句号/分号/破折号）换行
        3. 其次在词边界处换行
        4. 保证每行不超过 target_width
        """
        tokens = cls._tokenize_for_wrap(text)
        if not tokens:
            return text

        lines, current_line = [], ""
        for token in tokens:
            # 在标点处检查是否应该断行
            if current_line:
                test = current_line + " " + token
            else:
                test = token

            if len(test) <= target_width:
                current_line = test
            else:
                # 当前行放不下，先输出
                if current_line:
                    lines.append(current_line)
                # 如果单个 token 超长，直接输出（不截断 LaTeX 单元）
                current_line = token

        if current_line:
            lines.append(current_line)

        return "\n".join(lines)

    REWRITE_PROMPT = """You are an academic paper editor. Your task is to MINIMALLY edit an unsupported claim so it naturally incorporates a citation.

## Strict Rules
1. PRESERVE the original text as much as possible — only change what is needed to insert the citation.
2. Do NOT expand, paraphrase, or add new information beyond the original claim.
3. Insert ~\\cite{{{cite_key}}} at the most natural position (usually after the entity being cited).
4. Match the original writing style, tone, and level of detail exactly.
5. Do NOT change numerical values, abbreviations, or technical terms.
6. Do NOT include LaTeX section commands (\\section, \\subsection, etc.) in your output.
7. Output ONLY the rewritten claim text — no section headers, no explanations.

## Context
- **Surrounding text (style reference):**
{context_before}

- **Original claim (PRESERVE this structure):**
{claim_text}

- **Paper to cite:**
  Cite key: {cite_key}

Rewritten text:"""

    @classmethod
    def rewrite_gap(
        cls,
        claim_text: str,
        context_before: str,
        context_after: str,
        paper_title: str,
        paper_authors: str,
        paper_year: int,
        paper_venue: str,
        paper_abstract: str,
        paper_contribution: str,
        cite_key: str,
    ) -> str:
        """
        调用 LLM 改写单个 gap。
        Returns: 改写后的文本（纯文本，不含 LaTeX 结构标签）。
        """
        prompt = cls.REWRITE_PROMPT.format(
            context_before=context_before or "(start of section)",
            claim_text=claim_text,
            paper_title=paper_title,
            paper_authors=paper_authors,
            paper_year=paper_year,
            paper_venue=paper_venue,
            paper_abstract=paper_abstract,
            paper_contribution=paper_contribution,
            cite_key=cite_key,
        )

        try:
            client = Anthropic()
            response = client.messages.create(
                model=_MODEL,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            rewritten = response.content[0].text.strip()
            # 清理可能的引号包裹
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1]
            if rewritten.startswith("'") and rewritten.endswith("'"):
                rewritten = rewritten[1:-1]
            # 后处理：按上下文行宽风格换行
            width = cls._detect_line_width(context_before)
            rewritten = cls._wrap_to_context_style(rewritten, width)
            # 清理 LLM 常见格式问题
            rewritten = cls._clean_llm_output(rewritten)
            return rewritten
        except Exception as e:
            return f"~\\cite{{{cite_key}}} % [LLM rewrite failed: {e}]"

    @classmethod
    def _clean_llm_output(cls, text: str) -> str:
        """清理 LLM 输出中的常见格式问题。"""
        # 修复逗号/句号前多余的空格: "word , next" → "word, next"
        text = re.sub(r'\s+([,;:.!?])', r'\1', text)
        # 修复左括号前的多余空格: "word ( text" → "word (text" — 仅在空格两边都是非空时
        text = re.sub(r'\(\s+', '(', text)
        return text

    @classmethod
    def rewrite_all_gaps(
        cls,
        original_text: str,
        recommendations: list,  # list of GapRecommendation
        gap_issues: list,  # list of AuditIssue with severity="gap"
    ) -> tuple[str, list[dict]]:
        """
        批量改写所有 gap 区域，替换原文中的 unsupported claim。
        Returns: (改写后的完整文本, 改写记录列表)
        """
        import re

        if not recommendations:
            return original_text, []

        rewritten_text = original_text
        change_records = []

        # 构建映射：gap_issue → recommendation
        # 通过 claim 文本匹配
        gap_to_rec = {}
        for rec in recommendations:
            if rec.recommended_papers:
                key = rec.claim.claim_text[:80]
                gap_to_rec[key] = rec

        # 逐个处理 gap
        for issue in gap_issues:
            # 提取 claim 文本
            claim = issue.description
            if "[GAP]" in claim:
                claim = claim.replace("[GAP] Academic claim without citation support: ", "").strip()
            # 去除外层引号（description 格式: "...claim..."）
            if len(claim) >= 2 and claim[0] == '"' and claim[-1] == '"':
                claim = claim[1:-1]

            # 找到匹配的 recommendation
            matched_rec = None
            for key, rec in gap_to_rec.items():
                if key in claim or claim[:60] in key:
                    matched_rec = rec
                    break

            if not matched_rec or not matched_rec.recommended_papers:
                continue

            best_paper = matched_rec.recommended_papers[0]
            cite_key = best_paper.cite_key or best_paper.doi

            # 获取上下文（原文中 claim 前后的文本）
            claim_clean = re.sub(r'[{}\\]', '', claim).strip()
            lines = rewritten_text.split("\n")

            # 全文匹配：处理跨行 claim
            full_text_stripped = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', rewritten_text)
            pos = full_text_stripped.find(claim_clean[:40])
            if pos == -1:
                continue

            # 映射字符偏移 → 行号范围
            prefix = full_text_stripped[:pos]
            start_line_idx = prefix.count('\n')
            # claim_clean 可能被 description 截断到100字符，
            # 因此从 start_line_idx 向后扩展到句末（行尾含 '.' 或无标点短行）
            claim_end_in_stripped = pos + len(claim_clean)
            remaining = full_text_stripped[pos:claim_end_in_stripped]
            end_line_idx = start_line_idx + remaining.count('\n')
            # 扩展到段落结束（空行或 LaTeX 结构命令）
            if end_line_idx < len(lines) - 1:
                # claim_clean 可能被截断，end_line_idx 未必覆盖完整段落
                # 学术段落内句号不是段尾标记，需要看到空行/结构命令才停
                for j in range(end_line_idx + 1, len(lines)):
                    candidate = lines[j].strip()
                    # 空行 = 段落结束
                    if not candidate:
                        break
                    # LaTeX 结构命令 = 新段落/节开始
                    if candidate.startswith('\\') and re.match(
                        r'\\(section|subsection|subsubsection|paragraph|begin|end)\b',
                        candidate,
                    ):
                        break
                    end_line_idx = j

            ctx_before = ""
            if start_line_idx > 0:
                # 扩展到 6 行上下文以获得更准确的行宽估计
                ctx_before = "\n".join(lines[max(0, start_line_idx - 6):start_line_idx])

            ctx_after = ""
            if end_line_idx < len(lines) - 1:
                ctx_after = "\n".join(lines[end_line_idx + 1:end_line_idx + 3])

            # 构建完整 claim 文本（包括截断描述之外的后续行）
            full_claim_lines = lines[start_line_idx:end_line_idx + 1]
            full_claim = " ".join(l.strip() for l in full_claim_lines)

            # 调用 LLM 改写
            new_paragraph = cls.rewrite_gap(
                claim_text=full_claim,
                context_before=ctx_before,
                context_after=ctx_after,
                paper_title=best_paper.title,
                paper_authors=best_paper.authors,
                paper_year=best_paper.year,
                paper_venue=best_paper.venue,
                paper_abstract=best_paper.abstract,
                paper_contribution=best_paper.core_contribution,
                cite_key=cite_key,
            )

            # 替换原文中的 gap 区域（支持跨行）
            lines[start_line_idx:end_line_idx + 1] = [new_paragraph]
            rewritten_text = "\n".join(lines)
            lines = rewritten_text.split("\n")

            change_records.append({
                "original_claim": claim[:100],
                "rewritten_to": new_paragraph[:100],
                "cite_key": cite_key,
                "paper_title": best_paper.title,
                "paper_url": best_paper.url,
            })

        return rewritten_text, change_records
