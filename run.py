#!/usr/bin/env python3
"""
Unified Academic Writing System (v5)
====================================
Usage:
    # Draft 模式（最常用）：传入 LaTeX 文件，审核增强
    python3 run.py experiment_setup.tex
    python3 run.py experiment_setup.tex --topic 实验设置 --auto-approve

    # Brainstorm 模式：从头生成
    python3 run.py --brainstorm "帮我写 VeriPatent 相关工作"
    python3 run.py --brainstorm "帮我写 VeriPatent 相关工作" --venue ACM MM

    # Legacy 兼容
    python3 run.py --method VeriPatent --draft related_work.tex
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.orchestrator import PipelineOrchestrator
from agents.brainstorm_agent import UserRequirement


def main():
    parser = argparse.ArgumentParser(
        description="Unified Academic Writing System (v5)"
    )

    # === Draft 模式：位置参数 ===
    parser.add_argument(
        "input_file",
        nargs="?",
        type=str,
        default=None,
        help="LaTeX file to verify and enhance (draft mode).",
    )

    # === Brainstorm 模式 ===
    parser.add_argument(
        "--brainstorm", "-b",
        nargs="?",
        const="",
        default=None,
        help="Brainstorm mode: generate from scratch.",
    )

    # === Legacy 兼容 ===
    parser.add_argument(
        "--method", "-m",
        type=str, default=None,
        help="Method name (legacy: use with --draft).",
    )
    parser.add_argument(
        "--draft", "-d",
        type=str, default=None,
        help="Additional draft content (legacy: use with --method).",
    )

    # === 通用修饰符 ===
    parser.add_argument(
        "--topic", "-t",
        type=str, default=None,
        help="Topic name for output file naming (auto-detected if omitted).",
    )
    parser.add_argument(
        "--auto-approve", "-a",
        action="store_true",
        help="Skip interactive confirmation.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str, default=None,
        help="Output root directory (default: workspace/).",
    )
    parser.add_argument(
        "--method-summary", "-s",
        type=str, default=None,
        help="Method summary for brainstorm mode.",
    )
    parser.add_argument(
        "--venue", "-v",
        type=str, default=None,
        help="Target venue.",
    )

    args = parser.parse_args()
    output_dir = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "workspace"
    )

    orchestrator = PipelineOrchestrator(workspace_dir=output_dir)

    # ============================================================
    # Draft 模式：传入 LaTeX 文件
    # ============================================================
    if args.input_file:
        if not os.path.exists(args.input_file):
            print(f"Error: File not found: {args.input_file}")
            sys.exit(1)

        print(f"  Loaded: {args.input_file}")
        inp = PipelineOrchestrator.build_pipeline_input(
            file_path=args.input_file,
            topic_name=args.topic or "",
            auto_approve=args.auto_approve,
        )

        # draft 模式走 verify_citations（内部调用完整 pipeline）
        orchestrator.verify_citations(
            inp.raw_text,
            topic_name=inp.topic_name,
            auto_approve=inp.auto_approve,
        )
        return

    # ============================================================
    # Brainstorm 模式
    # ============================================================
    if args.brainstorm is not None:
        raw_input = args.brainstorm

        if args.draft and os.path.exists(args.draft):
            with open(args.draft, "r", encoding="utf-8") as f:
                draft_content = f.read()
        else:
            draft_content = None

        if not raw_input:
            print()
            raw_input = input("  请描述你的需求: ").strip()
            if not raw_input:
                print("  Error: 需求不能为空。")
                sys.exit(1)

        orchestrator.run_all(raw_input=raw_input, draft_content=draft_content)
        return

    # ============================================================
    # Legacy: --method + --draft
    # ============================================================
    if args.method:
        if not args.draft or not os.path.exists(args.draft):
            print("Error: --method requires --draft <path>")
            sys.exit(1)

        with open(args.draft, "r", encoding="utf-8") as f:
            draft_content = f.read()

        req = UserRequirement(
            raw_input=f"--method {args.method} with draft {args.draft}",
            method_name=args.method,
            draft_content=draft_content,
        )

        orchestrator.run_phases_1_to_6(req)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
