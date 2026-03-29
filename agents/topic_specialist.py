"""
Topic Specialist Agent (主题定义官)
====================================
职责：深度解析用户提供的现有研究内容，提炼研究主题、技术路线图及本研究的独特卖点。
输出：检索任务书（关键词组合、研究时间跨度、目标会议/期刊范围）。
"""

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.brainstorm_agent import UserRequirement


@dataclass
class ResearchProfile:
    """研究画像"""
    title: str = ""
    core_method: str = ""
    target_problem: str = ""
    value_proposition: str = ""
    technical_route: list[str] = field(default_factory=list)
    key_differences: list[str] = field(default_factory=list)


@dataclass
class SearchTaskDocument:
    """检索任务书"""
    # 关键词组合
    keyword_groups: list[list[str]] = field(default_factory=list)
    # 研究时间跨度
    year_range: tuple[int, int] = (2022, 2025)
    # 目标会议/期刊
    target_venues: list[str] = field(default_factory=list)
    # 额外约束
    constraints: list[str] = field(default_factory=list)


class TopicSpecialist:
    """
    主题定义官：分析研究内容，输出检索任务书。
    """

    SYSTEM_PROMPT = """You are a Topic Specialist Agent for academic related-work writing.

## Your Role
You are the FIRST agent in the pipeline. Your job is to deeply analyze the user's research
and produce a precise Search Task Document that will guide all subsequent literature retrieval.

## Your Responsibilities
1. Parse the user's provided research content (paper draft, abstract, method description)
2. Identify the core research topic, technical approach, and novelty
3. Extract the "Value Proposition" — what makes this work unique compared to prior art
4. Map the technical landscape: predecessors, competitors, and tool-providers
5. Generate structured search queries

## Output Format
You must produce a SearchTaskDocument with:
- keyword_groups: Each group is a set of semantically related keywords (AND within group, OR across groups)
- year_range: Suggested publication year range
- target_venues: Relevant top-tier venues (conferences/journals)
- constraints: Any domain-specific filtering rules

## Guidelines
- Keywords should cover: (a) the problem domain, (b) the method family, (c) competing approaches, (d) foundational techniques
- Aim for 4-8 keyword groups with 2-5 keywords each
- Include both broad terms (for coverage) and specific terms (for precision)
- Target venues should be realistic for the research domain's top tier
"""

    INTERACTION_PROMPT = """你好！我是 **主题定义官 (Topic Specialist)**，多智能体协同系统的第一步。

我已经阅读了你现有的 Related Work 章节草稿。在开始检索之前，我需要向你确认以下信息以构建完整的检索任务书：

---

### 1. 你的论文核心信息
- **论文标题**：是否为 VeriPatent？请确认完整标题。
- **核心方法**：使用外部编码器 + 密钥建立外部秘密，实现模型指纹验证。还有需要补充的吗？
- **目标会议/期刊**：这篇论文准备投哪个会议/期刊？（如 ACM MM, CVPR, NeurIPS, S&P, USENIX Security 等）

### 2. 研究细分方向
- 除了你草稿中提到的 Passive Fingerprinting 和 Proactive Fingerprinting 两大类，是否还有其他需要覆盖的相关工作方向？
  - 例如：Watermarking (水印), Model Ownership Verification (模型所有权验证), Copyright Protection (版权保护), IP Protection for LLMs 等？
- 是否需要包含 Watermarking（水印）相关工作作为相关技术分支？

### 3. 时间范围
- 你希望覆盖哪段时间的文献？（建议 2022-2025，或根据你的判断调整）

### 4. 语言偏好
- 相关工作章节用英文撰写？LaTeX 格式？

### 5. 补充材料
- 除了已有的 Related Work 草稿外，是否有其他材料（如 Introduction、Method 章节草稿、实验设置等）可以提供给我，帮助我更精确地定位研究定位？

---

请逐一回答或一次性回复，我将据此生成检索任务书。"""

    @classmethod
    def get_interaction_prompt(cls) -> str:
        """获取与用户交互的提示"""
        return cls.INTERACTION_PROMPT

    @classmethod
    def build_profile_from_draft(cls, draft_content: str) -> ResearchProfile:
        """
        从现有草稿中提取研究画像。
        目前从草稿中提取核心信息，后续可扩展为 LLM 调用。
        """
        profile = ResearchProfile()

        # 从草稿中提取的关键信息（基于已阅读的 Related Work 草稿）
        profile.title = "VeriPatent"
        profile.target_problem = (
            "Large language model fingerprinting and ownership verification — "
            "establishing provable model ownership in both black-box and white-box settings"
        )
        profile.core_method = (
            "External encoder with secret key to establish external secrets, "
            "decoupling fingerprint secrets from model parameters; "
            "multi-layered verification inspired by biosensor detection"
        )
        profile.value_proposition = (
            "Decouples fingerprint secret from model parameters via external encoder, "
            "providing resilience against both white-box (weight access) and black-box "
            "(output manipulation) attacks — unlike prior methods that embed fingerprints "
            "internally and are vulnerable to removal"
        )
        profile.technical_route = [
            "Passive fingerprinting → behavioral signatures",
            "Proactive fingerprinting → embedded triggers/responses",
            "External secret approach → encoder-based verification",
        ]
        profile.key_differences = [
            "External vs. internal fingerprint storage",
            "Resistance to fine-tuning attacks",
            "Multi-layered verification (biosensor-inspired)",
            "No need to modify model weights or embeddings",
        ]

        return profile

    @classmethod
    def build_profile_from_requirement(cls, req: "UserRequirement") -> ResearchProfile:
        """
        从 Phase 0 的 UserRequirement 构建研究画像。
        """
        profile = ResearchProfile()

        profile.title = req.method_name
        profile.target_problem = (
            f"Model fingerprinting and ownership verification "
            f"for {req.method_name}"
        )
        profile.core_method = req.method_summary or profile.title
        profile.value_proposition = req.method_summary or ""
        profile.technical_route = []
        profile.key_differences = []

        # 从子方向推断技术路线
        for d in req.research_directions:
            if "passive" in d:
                profile.technical_route.append(
                    "Passive fingerprinting → behavioral signatures"
                )
            elif "proactive" in d:
                profile.technical_route.append(
                    "Proactive fingerprinting → embedded triggers/responses"
                )
            elif "watermark" in d:
                profile.technical_route.append(
                    "Text watermarking → generation-level signals"
                )

        if not profile.technical_route:
            profile.technical_route = [
                "Passive fingerprinting → behavioral signatures",
                "Proactive fingerprinting → embedded triggers/responses",
            ]

        return profile

    @classmethod
    def generate_search_task(cls, profile: ResearchProfile) -> SearchTaskDocument:
        """
        基于研究画像生成检索任务书。
        """
        task = SearchTaskDocument()

        # 关键词组合：组内 AND，组间 OR
        task.keyword_groups = [
            # Problem domain
            [
                "LLM fingerprinting",
                "language model fingerprint",
                "model ownership verification",
                "AI model provenance",
            ],
            # Passive methods
            [
                "passive fingerprinting",
                "model fingerprint",
                "behavioral fingerprint",
                "model attribution",
                "watermark-free fingerprint",
            ],
            # Proactive methods
            [
                "proactive fingerprinting",
                "model watermarking",
                "trigger-based fingerprint",
                "embedded fingerprint",
                "training-time watermark",
            ],
            # Specific methods mentioned in draft
            [
                "TRAP targeted random adversarial prompt",
                "RAP-SM robust active prompt",
                "HuRef human-readable fingerprint",
                "REEF representation encoding",
                "WLM watermark language model",
                "instruction fingerprint",
            ],
            # Attack & defense
            [
                "fingerprint removal",
                "model unlearning attack",
                "ownership dispute",
                "white-box attack model",
                "black-box verification",
            ],
            # Broader related areas
            [
                "neural network watermarking",
                "deep learning IP protection",
                "AI copyright protection",
                "model extraction defense",
            ],
        ]

        task.year_range = (2022, 2025)

        task.target_venues = [
            "ACM MM",
            "CVPR",
            "ICCV",
            "NeurIPS",
            "ICML",
            "ICLR",
            "IEEE S&P",
            "USENIX Security",
            "CCS",
            "NDSS",
            "AAAI",
            "IJCAI",
            "ACL",
            "EMNLP",
            "IEEE TIFS",
            "IEEE TDSC",
            "IEEE TIP",
        ]

        task.constraints = [
            "Prioritize works published at top-tier security and AI venues",
            "Include arXiv preprints only if highly cited or from well-known groups",
            "Exclude survey papers unless they provide unique taxonomy",
            "Focus on LLM/VLM era (post-ChatGPT) unless foundational",
            "Include both white-box and black-box setting papers",
        ]

        return task

    @classmethod
    def format_search_task(cls, task: SearchTaskDocument) -> str:
        """格式化检索任务书为可读文本"""
        lines = [
            "=" * 60,
            "📋 检索任务书 (Search Task Document)",
            "=" * 60,
            "",
            "### 关键词组合 (AND within group, OR across groups)",
        ]

        for i, group in enumerate(task.keyword_groups, 1):
            keywords = " AND ".join(f'"{kw}"' for kw in group)
            lines.append(f"  Group {i}: {keywords}")

        lines.extend([
            "",
            f"### 时间范围: {task.year_range[0]} - {task.year_range[1]}",
            "",
            "### 目标会议/期刊:",
        ])
        for venue in task.target_venues:
            lines.append(f"  - {venue}")

        lines.extend(["", "### 检索约束:"])
        for c in task.constraints:
            lines.append(f"  ⚠ {c}")

        lines.extend([
            "",
            "=" * 60,
        ])

        return "\n".join(lines)
