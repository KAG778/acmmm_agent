"""
Brainstorm Agent (需求定义官) — Phase 0
========================================
流水线最前面的需求收集阶段。

职责：
  1. 接收用户原始输入（一句话 / 草稿 / 提问）
  2. 提取已知的 key points 并展示
  3. 询问用户是否逐项确认
  4. 逐项问答：基础信息 → 研究子方向 → 约束条件 → 严格程度
  5. 输出结构化 UserRequirement，供所有下游 Agent 使用
"""

import re
import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class UserRequirement:
    """Phase 0 输出 — 用户需求"""
    raw_input: str = ""
    method_name: str = ""
    method_summary: str = ""
    target_venue: str = ""
    research_directions: list[str] = field(default_factory=list)
    year_range: tuple[int, int] = (2022, 2025)
    max_papers: int = 8
    draft_content: str = ""
    strict_mode: bool = True
    length_preference: str = "concise"  # concise / standard / comprehensive
    extra_requirements: list[str] = field(default_factory=list)
    confirmed: bool = False

    def to_dict(self) -> dict:
        return {
            "raw_input": self.raw_input,
            "method_name": self.method_name,
            "method_summary": self.method_summary,
            "target_venue": self.target_venue,
            "research_directions": self.research_directions,
            "year_range": list(self.year_range),
            "max_papers": self.max_papers,
            "strict_mode": self.strict_mode,
            "length_preference": self.length_preference,
            "extra_requirements": self.extra_requirements,
            "confirmed": self.confirmed,
        }

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "UserRequirement":
        return cls(
            raw_input=d.get("raw_input", ""),
            method_name=d.get("method_name", ""),
            method_summary=d.get("method_summary", ""),
            target_venue=d.get("target_venue", ""),
            research_directions=d.get("research_directions", []),
            year_range=tuple(d.get("year_range", [2022, 2025])),
            max_papers=d.get("max_papers", 8),
            strict_mode=d.get("strict_mode", True),
            length_preference=d.get("length_preference", "concise"),
            extra_requirements=d.get("extra_requirements", []),
            confirmed=d.get("confirmed", False),
        )

    @classmethod
    def load(cls, path: str) -> Optional["UserRequirement"]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return cls.from_dict(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            return None


class BrainstormAgent:
    """
    需求定义官 — Phase 0。
    从用户原始输入中提取需求，逐项确认后输出 UserRequirement。
    """

    # 已知会议/期刊列表（用于自动识别）
    KNOWN_VENUES = {
        "acm mm", "cvpr", "iccv", "neurips", "icml", "iclr",
        "ieee sp", "usenix security", "ccs", "ndss",
        "aaai", "ijcai", "acl", "emnlp",
    }

    # 已知研究子方向
    KNOWN_DIRECTIONS = {
        "passive", "passive fingerprinting", "fingerprinting",
        "proactive", "proactive fingerprinting",
        "watermarking", "text watermarking",
        "model ownership", "ownership verification",
        "ip protection", "copyright protection",
    }

    # 已知篇幅关键词
    LENGTH_KEYWORDS = {
        "concise": {"短", "精简", "concise", "短小", "不要太长", "简短", "简洁"},
        "comprehensive": {"详", "全面", "comprehensive", "详细", "完整", "长"},
    }

    @classmethod
    def run(
        cls,
        raw_input: str = "",
        draft_content: str = None,
    ) -> UserRequirement:
        """
        主入口：解析输入 → 展示理解 → 确认 → Q&A → 返回 UserRequirement。
        """
        req = UserRequirement(raw_input=raw_input or "")
        if draft_content:
            req.draft_content = draft_content

        # Step 1-2: 解析 & 展示
        parsed = cls._parse_raw_input(raw_input, draft_content)
        cls._display_understanding(parsed)

        # Step 3: 确认是否需要补充
        print()
        need_details = cls._ask_yes_no(
            "  是否需要补充其他信息？ [y/n]: "
        )

        if need_details:
            # Step 4: 逐项问答
            req = cls._run_qa(req, parsed)
        else:
            # 用解析结果直接填充
            req = cls._fill_from_parsed(req, parsed)

        req.confirmed = True
        print()
        print("  ✓ 需求确认完成，进入 Phase 1...")
        print()
        return req

    @classmethod
    def _parse_raw_input(
        cls, raw_input: str, draft_content: str = None
    ) -> dict:
        """
        从用户原始输入中提取已知信息。
        Returns dict with possible keys: method_name, target_venue,
        method_summary, directions, year_range, max_papers,
        length_preference, extra.
        """
        parsed = {}
        text = raw_input.lower()

        # 提取方法名：常见模式 "帮我写 X 的相关工作" / "X method" 等
        # 寻找首字母大写或引号包围的名称
        patterns = [
            r"帮我写\s+(.+?)\s*(?:的|相关|related)",
            r"write\s+(?:the\s+)?(?:related\s+work\s+(?:for|on)\s+)(\S+)",
            r"for\s+(\S+?)(?:\s|$|,|，)",
        ]
        for pat in patterns:
            m = re.search(pat, raw_input, re.IGNORECASE)
            if m:
                parsed["method_name"] = m.group(1).strip()
                break

        # 提取会议/期刊
        input_lower = raw_input.lower()
        for venue in cls.KNOWN_VENUES:
            if venue in input_lower:
                parsed["target_venue"] = venue.upper()
                break

        # 提取篇幅偏好（先匹配长关键词，避免短关键词子串误匹配）
        for pref, keywords in cls.LENGTH_KEYWORDS.items():
            matched = False
            for kw in keywords:
                if kw in input_lower:
                    parsed["length_preference"] = pref
                    matched = True
                    break
            if matched:
                break

        # 提取子方向
        directions = []
        for d in cls.KNOWN_DIRECTIONS:
            if d in input_lower:
                directions.append(d)
        if directions:
            parsed["directions"] = directions

        return parsed

    @classmethod
    def _display_understanding(cls, parsed: dict) -> None:
        """打印 "我的理解" 列表"""
        print()
        print("  " + "=" * 50)
        print("  PHASE 0: 需求定义")
        print("  " + "=" * 50)

        if not parsed:
            print("  未能从输入中自动提取信息，请手动填写。")
            return

        print("  我提取到以下信息：")
        mapping = {
            "method_name": "方法名",
            "target_venue": "目标会议",
            "length_preference": "篇幅偏好",
            "directions": "研究子方向",
        }
        for key, label in mapping.items():
            if key in parsed:
                val = parsed[key]
                if isinstance(val, list):
                    val = ", ".join(val)
                print(f"    {label}: {val}")

    @classmethod
    def _ask_yes_no(cls, prompt: str) -> bool:
        """问一个 y/n 问题"""
        while True:
            ans = input(prompt).strip().lower()
            if ans in ("y", "yes", "是"):
                return True
            if ans in ("n", "no", "否", ""):
                return False
            print("    请输入 y/n: ")

    @classmethod
    def _ask_question(cls, prompt: str, default: str = "") -> str:
        """问一个问题，支持默认值"""
        if default:
            ans = input(f"  {prompt} [{default}]: ").strip()
        else:
            ans = input(f"  {prompt}: ").strip()
        return ans if ans else default

    @classmethod
    def _fill_from_parsed(
        cls, req: UserRequirement, parsed: dict
    ) -> UserRequirement:
        """用解析结果填充 UserRequirement（用户选择不逐项确认时）"""
        if "method_name" in parsed:
            req.method_name = parsed["method_name"]
        if "target_venue" in parsed:
            req.target_venue = parsed["target_venue"]
        if "length_preference" in parsed:
            req.length_preference = parsed["length_preference"]
        if "directions" in parsed:
            req.research_directions = parsed["directions"]
        return req

    @classmethod
    def _run_qa(
        cls, req: UserRequirement, parsed: dict
    ) -> UserRequirement:
        """逐项问答，填充 UserRequirement"""
        print()

        # Q1: 方法名
        default_name = parsed.get("method_name", "")
        req.method_name = cls._ask_question(
            "Q1/6: 论文标题 / 方法名", default_name
        )

        # Q2: 目标会议
        default_venue = parsed.get("target_venue", "")
        req.target_venue = cls._ask_question(
            "Q2/6: 目标会议 / 期刊", default_venue
        )

        # Q3: 核心方法描述
        default_summary = parsed.get("method_summary", "")
        req.method_summary = cls._ask_question(
            "Q3/6: 核心方法描述（1-2句）", default_summary
        )

        # Q4: 子方向
        default_dirs = "passive, proactive"
        dirs_input = cls._ask_question(
            "Q4/6: 需要覆盖的子方向（逗号分隔）", default_dirs
        )
        req.research_directions = [
            d.strip().lower()
            for d in dirs_input.split(",")
            if d.strip()
        ]

        # Q5: 约束条件（合为一组）
        print("  Q5/6: 约束条件")
        year_default = f"{req.year_range[0]}-{req.year_range[1]}"
        year_input = cls._ask_question("    年份范围", year_default)
        m = re.match(r"(\d{4})\s*[-–]\s*(\d{4})", year_input)
        if m:
            req.year_range = (int(m.group(1)), int(m.group(2)))

        req.max_papers = int(cls._ask_question(
            "    核心文献数", str(req.max_papers)
        ))

        length_default = parsed.get("length_preference", "concise")
        req.length_preference = cls._ask_question(
            "    篇幅 (concise/standard/comprehensive)", length_default
        )

        # Q6: 严格程度
        strict = cls._ask_yes_no(
            "  Q6/6: 严格模式（拒绝无 DOI 的文献） [y/n]: "
        )
        req.strict_mode = strict

        return req
