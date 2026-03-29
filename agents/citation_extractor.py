"""
Citation Extractor (通用引用提取器)
====================================
职责：从任意 LaTeX 文本中提取所有 \cite{} 标签、引用上下文、以及未支撑的学术主张。
输入：任何 LaTeX 文本（相关工作、实验设置、方法描述……）
输出：结构化的引用清单 + 上下文 + 潜在缺失。
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CitationRef:
    """单条 \cite{} 提取结果"""
    cite_key: str = ""          # e.g. "zhang2022opt"
    context_before: str = ""    # 引用前 1-2 句
    context_after: str = ""     # 引用后 1-2 句
    full_context: str = ""      # 包含引用的完整句子
    location_line: int = 0
    raw_latex: str = ""    # 原始 \cite{...} 文本


@dataclass
class UnsupportedClaim:
    """未支撑的学术主张（引用真空）"""
    claim_text: str
    location_line: int = 0
    claim_type: str = ""   # model_architecture / sota / comparison / dataset / method
    confidence: float = 0.0


@dataclass
class ExtractionResult:
    """完整提取结果"""
    cite_keys: list[str] = field(default_factory=list)
    citations: list[CitationRef] = field(default_factory=list)
    unsupported_claims: list[UnsupportedClaim] = field(default_factory=list)
    topic_hint: str = ""   # 从文本内容推断的主题


class CitationExtractor:
    """
    通用引用提取器。
    不区分 section 类型——任何 LaTeX 文本均可处理。
    """

    # 学术主张信号词（模型架构、SOTA、对比实验等）
    CLAIM_SIGNALS = {
        "model_architecture": [
            r"\b(?:based on|uses?|employs?|adopts?|leverages?)\s+[\w\s]{5,30}(?:model|architecture|encoder|decoder|transformer|LLM|GPT|BERT|T5|LLaMA|OPT|Mistral|Pythia)",
            r"\b(?:we use|we adopt|we employ|we utilize)\b.{10,80}",
        ],
        "sota": [
            r"\b(state.of.the.art|SOTA|best.performing|outperforms?\s+(?:all|existing|previous|prior))\b",
            r"\b(achieves?\s+(?:the\s+)?(?:best|highest|superior|competitive))\b",
        ],
        "comparison": [
            r"\b(compare[sd]?\s+with|compared?\s+to|versus|vs\.?)\s+[\w\s]{5,40}",
            r"\b(baseline|competitor|counterpart)\b.{5,40}",
        ],
        "dataset": [
            r"\b(?:dataset|corpus|benchmark)\s+(?:of|from|containing|with)\s+[\w\s]{5,50}",
            r"\b(?:we (?:construct|collect|use|adopt|build))\s+(?:a\s+)?(?:\w+\s+)?(?:dataset|corpus)\b",
        ],
        "method": [
            r"\b(?:proposes?|introduces?|presents?|develops?)\s+(?:a\s+)?(?:\w+\s+){0,3}(?:method|approach|technique|framework|mechanism)\b",
            r"\b(?:our (?:method|approach|framework|model|contribution))\b",
        ],
    }

    # v4: 语义断点扫描触发词（无 \cite 时的 Potential Gap 信号）
    SEMANTIC_BREAKPOINT_TRIGGERS = {
        "sota_gap": [
            r"\b(?:recent (?:progress|advances?|developments?|work|studies?))\b",
            r"\b(?:standard practice|commonly used|widely adopted)\b",
            r"\b(?:state.of.the.art|SOTA)\b",
            r"\b(?:inspired by|motivated by|following)\b.{10,50}",
            r"\b(?:previous (?:studies?|works?|research|efforts?))\s+(?:have|has)\b",
            r"\b(?:existing (?:methods?|approaches?|techniques?|works?))\b",
        ],
        "technical_definition": [
            r"\b(?:is (?:a |an |the )){1,2}(?:well.known|popular|standard|classical|traditional)\b.{10,60}",
            r"\b(?:widely (?:used|adopted|recognized|accepted|studied))\b",
            r"\b(?:common (?:practice|approach|technique|strategy))\b",
        ],
        "comparison_gap": [
            r"\b(?:outperforms?|surpasses?|exceeds?)\s+(?:the\s+)?(?:prior|existing|previous|current|conventional)\b",
            r"\b(?:superior (?:performance|results?|accuracy|quality))\b",
            r"\b(?:competitive (?:with|against|to))\b",
        ],
    }

    @classmethod
    def extract(cls, text: str) -> ExtractionResult:
        """
        从任意 LaTeX 文本中提取引用信息。
        """
        result = ExtractionResult()

        # Step 1: 提取所有 \cite{} 和 \citep{} / \citet{} 标签
        result.citations = cls._extract_citations(text)
        result.cite_keys = list(dict.fromkeys(
            c.cite_key for c in result.citations
        ))

        # Step 2: 检测未支撑的学术主张
        result.unsupported_claims = cls._detect_unsupported_claims(text)

        # Step 3: 推断主题
        result.topic_hint = cls._infer_topic(text)

        return result

    @classmethod
    def _extract_citations(cls, text: str) -> list[CitationRef]:
        """提取所有 \cite{} 引用及上下文"""
        citations = []
        lines = text.split("\n")

        cite_pattern = re.compile(r'\\cite[a-z]*\{([^}]+)\}')

        for line_idx, line in enumerate(lines, 1):
            # 找到本行所有 cite 的位置
            cite_matches = list(cite_pattern.finditer(line))

            for cite_idx, match in enumerate(cite_matches):
                raw_keys = match.group(1)
                for key in raw_keys.split(","):
                    key = key.strip()
                    if not key:
                        continue

                    ref = CitationRef(
                        cite_key=key,
                        raw_latex=match.group(0),
                        location_line=line_idx,
                    )

                    # 提取引用局部上下文：前一个 cite 到本 cite 之间的文本
                    # 这样 "A~\cite{x}, B~\cite{y}" 对 x 的 context 是 "A", 对 y 是 "B"
                    prev_end = cite_matches[cite_idx - 1].end() if cite_idx > 0 else 0
                    next_start = cite_matches[cite_idx + 1].start() if cite_idx + 1 < len(cite_matches) else len(line)

                    before_text = line[prev_end:match.start()]
                    after_text = line[match.end():next_start]

                    # 清理 LaTeX 残留
                    before_text = re.sub(r'~|\\[a-zA-Z]+\{[^}]*\}|[{}]', ' ', before_text).strip()
                    after_text = re.sub(r'~|\\[a-zA-Z]+\{[^}]*\}|[{}]', ' ', after_text).strip()

                    ref.context_before = before_text
                    ref.context_after = after_text
                    ref.full_context = f"{before_text} {after_text}".strip()

                    citations.append(ref)

        return citations

    @classmethod
    def _extract_full_sentence(cls, text: str, pos: int) -> str:
        """从 pos 位置向前后扩展，提取完整句子"""
        # 向前找句首
        start = pos
        while start > 0 and text[start - 1] not in ".!?\n":
            start -= 1
        if start > 0:
            start += 1  # 跳过句末标点

        # 向后找句尾
        end = pos
        while end < len(text) and text[end] not in ".!?":
            end += 1
        if end < len(text):
            end += 1  # 包含句末标点

        return text[start:end].strip()

    @classmethod
    def _detect_unsupported_claims(cls, text: str) -> list[UnsupportedClaim]:
        """
        检测文本中缺乏引用支撑的学术主张。
        使用双层扫描：
          1. CLAIM_SIGNALS — 学术主张信号（模型、SOTA、对比、数据集、方法）
          2. SEMANTIC_BREAKPOINT_TRIGGERS — 语义断点（recent progress, standard practice 等）
        """
        claims = []
        lines = text.split("\n")

        # 找出所有有 \cite{} 的行号集合（含相邻行）
        cited_lines = set()
        cite_pattern = re.compile(r'\\cite[a-z]*\{[^}]+\}')
        for i, line in enumerate(lines):
            if cite_pattern.search(line):
                cited_lines.add(i)
                cited_lines.add(i - 1)
                cited_lines.add(i + 1)

        # 分句（先清理 LaTeX 命令，避免 \\subsection{} 被当作句子开头）
        clean_text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
        clean_text = re.sub(r'[{}\\]', '', clean_text)
        sentences = re.split(r'(?<=[.!?])\s+', clean_text)
        for sent in sentences:
            sent_stripped = sent.strip()
            if len(sent_stripped) < 20:
                continue

            # 跳过有引用的句子（引用已被清理，检查原文）
            # 找回原文中的对应句子
            sent_preview = re.sub(r'\s+', ' ', sent_stripped)[:50]
            original_line = cls._find_line_number(text, sent_preview)

            # 跳过纯 LaTeX 结构行（无实质内容）
            if not re.search(r'[a-zA-Z]{3,}', sent_stripped):
                continue

            line_num = original_line
            # 如果该行附近没有引用
            if line_num is not None and line_num in cited_lines:
                continue
            # 跨行句子可能无法定位到单行，检查句子关键词是否出现在有引用的行附近
            if line_num is None:
                sent_keywords = set(re.findall(r'[a-zA-Z]{5,}', sent_stripped.lower()))
                for cl in cited_lines:
                    if 0 <= cl < len(lines):
                        line_words = set(re.findall(r'[a-zA-Z]{5,}', lines[cl].lower()))
                        if sent_keywords & line_words:
                            line_num = cl + 1
                            break
                if line_num is not None and line_num in cited_lines:
                    continue

            # --- Layer 1: 学术主张信号 ---
            matched_type = None
            for claim_type, patterns in cls.CLAIM_SIGNALS.items():
                for pat in patterns:
                    if re.search(pat, sent_stripped, re.IGNORECASE):
                        matched_type = claim_type
                        break
                if matched_type:
                    break

            # --- Layer 2: 语义断点扫描 ---
            if not matched_type:
                for gap_type, patterns in cls.SEMANTIC_BREAKPOINT_TRIGGERS.items():
                    for pat in patterns:
                        if re.search(pat, sent_stripped, re.IGNORECASE):
                            matched_type = gap_type
                            break
                    if matched_type:
                        break

            if matched_type:
                confidence = cls._estimate_confidence(sent_stripped, matched_type)
                if confidence >= 0.3:
                    claims.append(UnsupportedClaim(
                        claim_text=sent_stripped,
                        location_line=line_num or 0,
                        claim_type=matched_type,
                        confidence=round(confidence, 2),
                    ))

        # 去重
        seen = set()
        unique = []
        for c in claims:
            if c.claim_text[:60] not in seen:
                seen.add(c.claim_text[:60])
                unique.append(c)

        return unique

    @classmethod
    def _find_line_number(cls, text: str, sentence: str) -> Optional[int]:
        """找到句子在原始文本中的行号（大小写不敏感匹配）"""
        lines = text.split("\n")
        sent_start = sentence[:40].lower()
        for i, line in enumerate(lines):
            if sent_start in line.lower():
                return i + 1
        return None

    @classmethod
    def _estimate_confidence(cls, sentence: str, claim_type: str) -> float:
        """
        估算学术主张需要引用的置信度。
        包含具体模型名/数据集名/数值 → 高置信度
        """
        score = 0.4  # 基础分：匹配了信号词

        # 包含具体模型名 → 高置信度
        model_names = re.findall(
            r'\b(?:OPT|GPT-J|GPT-2|GPT-3|GPT-4|BERT|T5|mT5|LLaMA|Mistral|Pythia|'
            r'Gemini|Claude|ChatGLM|Vicuna|Alpaca|Falcon|Qwen)\b[\w.-]*',
            sentence, re.IGNORECASE,
        )
        score += len(model_names) * 0.2

        # 包含具体数值
        numbers = re.findall(r'\d+(?:\.\d+)?(?:%|B|M|K)?', sentence)
        score += min(len(numbers) * 0.05, 0.15)

        # 包含具体数据集名
        dataset_names = re.findall(
            r'\b(?:AG News|DailyDialog|SQuAD|GLUE|SuperGLUE|MMLU|HumanEval|'
            r'Wikitext|CNN/DailyMail|arXiv|USPTO)\b',
            sentence, re.IGNORECASE,
        )
        score += len(dataset_names) * 0.15

        # 包含具体方法名（大写开头+技术词汇）
        method_signals = re.findall(
            r'\b(?:fine-tun|pre-train|transfer learn|attention mechanism|'
            r'cross-entropy|beam search|prompt engineer|RLHF|DPO|LoRA)\b',
            sentence, re.IGNORECASE,
        )
        score += len(method_signals) * 0.1

        return min(score, 1.0)

    @classmethod
    def _infer_topic(cls, text: str) -> str:
        """从文本内容推断主题（用于动态输出命名）"""
        # 检查 section 标题
        section_match = re.search(r'\\(?:section|subsection)\{([^}]+)\}', text)
        if section_match:
            title = section_match.group(1)
            # 清理 LaTeX 命令
            title = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', title)
            title = re.sub(r'[{}\\]', '', title).strip()
            if title:
                # 提取关键词
                words = [w for w in title.split() if len(w) > 3]
                if words:
                    return "_".join(words[:3]).replace(" ", "_")

        # 从内容关键词推断
        keyword_counts: dict[str, int] = {}
        domain_keywords = {
            "fingerprint": "fingerprinting",
            "watermark": "watermarking",
            "ownership": "ownership_verification",
            "detection": "detection",
            "generation": "generation",
            "classification": "classification",
            "summarization": "summarization",
            "translation": "translation",
            "dialogue": "dialogue",
            "patent": "patent",
            "LLM": "llm",
            "transformer": "transformer",
            "experiment": "experiment",
            "training": "training",
            "evaluation": "evaluation",
        }
        text_lower = text.lower()
        for kw, topic in domain_keywords.items():
            count = text_lower.count(kw.lower())
            if count > 0:
                keyword_counts[topic] = keyword_counts.get(topic, 0) + count

        if keyword_counts:
            return max(keyword_counts, key=keyword_counts.get)

        return "general"
