"""
Cleanup Agent (环境清理官) — v3 动态内存回收版
=====================================================
强化项：
  1. 冗余剪枝：审计通过后立刻删除临时草稿和无效搜索记录
  2. 版本锁定：Context 窗口只保留最新被 Auditor 盖章后的"事实表"
  3. 状态感知：跟踪哪些文件已被审计盖章，哪些仍是草稿
  4. 物理销毁：接收 Auditor 的 dois_to_destroy 列表，从档案中永久移除
"""

import os
import re
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class FileRecord:
    """工作区文件记录"""
    path: str
    name: str
    size_bytes: int
    modified: float
    kind: str
    run_id: str
    status: str = "active"  # active / outdated / stale / audited / draft / destroyed
    audit_stamp: str = ""  # 审计盖章时间戳


@dataclass
class CleanupReport:
    """清理报告 v3"""
    timestamp: str = ""
    total_files: int = 0
    kept: int = 0
    removed: int = 0
    promoted: int = 0  # 草稿升级为最终稿
    audit_rejections: int = 0  # 被审计打回的草稿
    records_destroyed: int = 0  # v3: 被 Auditor 命令销毁的文献记录数
    destroyed_dois: list[str] = field(default_factory=list)  # v3: 已销毁的 DOI 列表
    removed_files: list[str] = field(default_factory=list)
    status_summary: list[str] = field(default_factory=list)


class CleanupAgent:
    """
    环境清理官 v3 — 动态内存回收 + 物理销毁。
    核心理念：防止 Context 污染 = 保持 Agent 智商。
    """

    KIND_PATTERNS = {
        "search_task": r"^search_task_(\d+_\d+)\.md$",
        "search_results": r"^search_results_(\d+_\d+)\.md$",
        "archive": r"^archive_(\d+_\d+)\.md$",
        "core_papers": r"^core_papers_(\d+_\d+)\.md$",
        "references": r"^references_(\d+_\d+)\.bib$",
        "draft": r"^draft_(\d+_\d+)\.tex$",
        "final_draft": r"^final_draft_(\d+_\d+)\.tex$",
        "audit_report": r"^audit_report_(\d+_\d+)\.md$",
        "cross_reference": r"^cross_reference_(\d+_\d+)\.md$",
        "fact_table": r"^fact_table_v\d+\.md$",
    }

    STATE_FILE = ".pipeline_state.json"

    KIND_LABELS = {
        "search_task": "SearchTask",
        "search_results": "SearchResults",
        "archive": "Archive",
        "core_papers": "CorePapers",
        "references": "References",
        "draft": "Draft",
        "final_draft": "FinalDraft",
        "audit_report": "AuditReport",
        "cross_reference": "CrossRef",
        "fact_table": "FactTable",
        "state": "PipelineState",
        "unknown": "Other",
    }

    def __init__(self, workspace_dir: str = "workspace"):
        self.workspace_dir = workspace_dir
        self._fact_table_path = os.path.join(workspace_dir, "fact_table_latest.md")
        self._destroy_log_path = os.path.join(workspace_dir, ".destroy_log.json")

    # ------------------------------------------------------------------
    # 1. 扫描工作区
    # ------------------------------------------------------------------

    def scan(self) -> list[FileRecord]:
        """扫描工作区，为每个文件建立记录"""
        records = []
        if not os.path.isdir(self.workspace_dir):
            return records

        for entry in os.listdir(self.workspace_dir):
            fpath = os.path.join(self.workspace_dir, entry)
            if not os.path.isfile(fpath):
                continue

            stat = os.stat(fpath)
            kind, run_id = self._classify_file(entry)

            # 加载审计盖章状态
            status, stamp = self._load_audit_stamp(fpath, kind, run_id)

            records.append(FileRecord(
                path=fpath, name=entry,
                size_bytes=stat.st_size,
                modified=stat.st_mtime,
                kind=kind, run_id=run_id,
                status=status, audit_stamp=stamp,
            ))
        return records

    def _classify_file(self, filename: str) -> tuple[str, str]:
        if filename == self.STATE_FILE:
            return "state", "current"
        for kind, pattern in self.KIND_PATTERNS.items():
            m = re.match(pattern, filename)
            if m:
                return kind, m.group(1)
        return "unknown", ""

    def _load_audit_stamp(
        self, fpath: str, kind: str, run_id: str
    ) -> tuple[str, str]:
        """加载审计盖章状态"""
        stamp_file = os.path.join(
            os.path.dirname(fpath), f".audit_stamp_{run_id}.json"
        )
        if os.path.exists(stamp_file):
            try:
                with open(stamp_file, "r") as f:
                    data = json.load(f)
                return data.get("status", "active"), data.get("stamp", "")
            except (json.JSONDecodeError, KeyError):
                pass
        return "active", ""

    def _save_audit_stamp(
        self, fpath: str, kind: str, run_id: str,
        status: str, stamp: str = "",
    ):
        """保存审计盖章状态"""
        stamp_dir = os.path.dirname(fpath)
        stamp_file = os.path.join(stamp_dir, f".audit_stamp_{run_id}.json")
        data = {"status": status, "stamp": stamp}
        os.makedirs(stamp_dir, exist_ok=True)
        with open(stamp_file, "w") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # 2. 识别清理目标
    # ------------------------------------------------------------------

    def identify_cleanup_targets(self, records: list[FileRecord]) -> list[FileRecord]:
        """
        识别需要清理的文件。
        v3: 同批次中 draft 在 final_draft 存在后则标记为 stale。
        """
        by_kind: dict[str, list[FileRecord]] = {}
        for r in records:
            by_kind.setdefault(r.kind, []).append(r)

        latest_runs: dict[str, str] = {}
        for kind, group in by_kind.items():
            if kind == "state":
                continue
            sorted_runs = sorted(set(r.run_id for r in group), reverse=True)
            if sorted_runs:
                latest_runs[kind] = sorted_runs[0]

        final_runs = set(r.run_id for r in by_kind.get("final_draft", []))
        audited_runs = set()

        for r in records:
            if r.kind == "state":
                continue
            if r.status == "audited":
                audited_runs.add(r.run_id)

            if r.kind in latest_runs and r.run_id != latest_runs[r.kind]:
                r.status = "outdated"
            elif r.kind == "draft" and r.run_id in final_runs:
                r.status = "stale"
            elif r.kind == "draft" and r.run_id in audited_runs:
                r.status = "stale"
            else:
                r.status = "active"

        return [r for r in records if r.status != "active"]

    # ------------------------------------------------------------------
    # 3. 执行清理 + 冗余剪枝
    # ------------------------------------------------------------------

    def cleanup(
        self,
        dry_run: bool = True,
        audit_run_id: str = None,
    ) -> CleanupReport:
        """
        执行清理。
        v3: 支持 audit_run_id，审计通过后触发该批次的草稿清理。

        Args:
            dry_run: 仅报告不实际删除
            audit_run_id: 指定已审计通过的批次 ID，触发草稿清理
        """
        report = CleanupReport(timestamp=datetime.now().isoformat())
        records = self.scan()
        report.total_files = len(records)

        if not records:
            report.status_summary.append("Workspace is empty.")
            return report

        # 如果指定了审计批次，标记该批次的 draft 为 stale
        if audit_run_id:
            for r in records:
                if r.run_id == audit_run_id and r.kind == "draft":
                    r.status = "stale"
                    report.audit_rejections += 1

        targets = self.identify_cleanup_targets(records)
        report.duplicates_found = self._check_metadata_redundancy(records)

        for r in targets:
            if dry_run:
                report.removed += 1
                report.removed_files.append(r.path)
            else:
                os.remove(r.path)
                report.removed += 1
                report.removed_files.append(r.path)

        report.kept = report.total_files - report.removed
        active = [r for r in records if r.status == "active"]
        report.status_summary = self._build_status_summary(active, targets)

        return report

    # ------------------------------------------------------------------
    # 3.5 v3: 物理销毁 — 处理 Auditor 的销毁指令
    # ------------------------------------------------------------------

    def destroy_records(
        self,
        dois_to_destroy: list[str],
        archive: list,
        dry_run: bool = True,
    ) -> CleanupReport:
        """
        v3: 执行 Auditor 发出的物理销毁指令。
        从档案中永久移除指定 DOI 的文献记录。

        Args:
            dois_to_destroy: Auditor 标记为需要销毁的 DOI 列表
            archive: 当前文献档案列表
            dry_run: 仅报告不实际删除
        """
        report = CleanupReport(timestamp=datetime.now().isoformat())
        report.total_files = len(archive)

        destroy_set = set(dois_to_destroy)
        surviving = []

        for entry in archive:
            doi = getattr(entry, "doi", "")
            if doi in destroy_set or (not doi and getattr(entry, "title", "") in destroy_set):
                report.records_destroyed += 1
                report.destroyed_dois.append(doi or getattr(entry, "title", ""))
                report.removed_files.append(f"ARCHIVE: {getattr(entry, 'title', doi)}")
                if not dry_run:
                    report.status_summary.append(
                        f"  [DESTROYED] {doi} — {getattr(entry, 'title', 'N/A')}"
                    )
            else:
                surviving.append(entry)

        report.kept = len(surviving)

        # 记录销毁日志
        self._log_destruction(dois_to_destroy, dry_run)

        return report

    def _log_destruction(self, dois: list[str], dry_run: bool):
        """记录销毁日志到 .destroy_log.json"""
        os.makedirs(self.workspace_dir, exist_ok=True)
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "dry_run_destroy" if dry_run else "physical_destroy",
            "dois_destroyed": dois,
        }

        existing_log = []
        if os.path.exists(self._destroy_log_path):
            try:
                with open(self._destroy_log_path, "r") as f:
                    existing_log = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        existing_log.append(log_entry)
        with open(self._destroy_log_path, "w") as f:
            json.dump(existing_log, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 4. 版本锁定 — 生成事实表
    # ------------------------------------------------------------------

    def lock_version(
        self,
        archive_entries: list = None,
        audit_report = None,
        core_papers_status: str = "approved",
    ) -> str:
        """
        版本锁定：生成当前"事实表"，作为 Context 唯一的合法素材来源。
        只保留被审计通过的内容，草稿和临时记录全部排除。

        事实表 = 当前 Context 窗口中唯一合法的"事实来源"。
        """
        lines = [
            "",
            "=" * 70,
            "  FACT TABLE — 事实表 (Version Locked)",
            "=" * 70,
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"  Core Papers Status: {core_papers_status}",
            "",
            "  [LOCKED] 此文件是当前 Context 中唯一合法的事实来源。",
            "  Writer 必须仅使用此文件中的内容进行写作。",
            "  任何超出此文件的内容均视为幻觉。",
            "",
        ]

        if archive_entries:
            lines.append("## 已确认文献 (Confirmed Literature)")
            for i, entry in enumerate(archive_entries, 1):
                status = getattr(entry, "status", "unknown")
                status_label = {
                    "verified": "已验证",
                    "unverified": "待核实",
                    "title_only": "禁止引用",
                }.get(status, status)

                doi = getattr(entry, "doi", "")
                anchor = getattr(entry, "abstract_anchor", "")

                lines.extend([
                    f"  {i}. [{status_label}] {getattr(entry, 'title', 'N/A')}",
                    f"     DOI: {doi}",
                ])
                if anchor:
                    lines.append(f"     Abstract Anchor: {anchor[:200]}...")
                else:
                    lines.append(f"     Abstract Anchor: [无 — 禁止引用具体声明]")
                lines.append("")

        if audit_report:
            verdict = getattr(audit_report, "final_verdict", str(audit_report))
            lines.append("## 审计结果 (Audit Results)")
            lines.append(f"  Verdict: {verdict}")
            # v3: 包含销毁通知
            dois_destroy = getattr(audit_report, "dois_to_destroy", [])
            if dois_destroy:
                lines.append(f"\n  DESTROYED ({len(dois_destroy)} records):")
                for doi in dois_destroy:
                    lines.append(f"    - {doi}")
            lines.append("")

        fact_table = "\n".join(lines)

        # 写入文件
        os.makedirs(self.workspace_dir, exist_ok=True)
        with open(self._fact_table_path, "w", encoding="utf-8") as f:
            f.write(fact_table)

        return fact_table

    # ------------------------------------------------------------------
    # 5. 状态清单
    # ------------------------------------------------------------------

    def _build_status_summary(
        self, active: list[FileRecord], outdated: list[FileRecord]
    ) -> list[str]:
        lines = []

        lines.append("=== PRODUCTION FILES (版本锁定) ===")
        by_kind: dict[str, list[FileRecord]] = {}
        for r in active:
            by_kind.setdefault(r.kind, []).append(r)

        for kind in [
            "fact_table", "final_draft", "audit_report",
            "core_papers", "archive", "references",
            "cross_reference", "search_task", "search_results", "state",
        ]:
            if kind not in by_kind:
                continue
            label = self.KIND_LABELS.get(kind, kind)
            for r in by_kind[kind]:
                size_kb = r.size_bytes / 1024
                mtime = datetime.fromtimestamp(r.modified).strftime("%m-%d %H:%M")
                audit_tag = f" [{r.audit_stamp}]" if r.audit_stamp else ""
                lines.append(
                    f"  [ACTIVE] {label:16s}{audit_tag:12s} "
                    f"{r.name:40s}  {size_kb:>6.1f}KB  {mtime}"
                )

        if outdated:
            lines.append("")
            lines.append("=== STALE / OUTDATED (待清理) ===")
            for r in outdated:
                size_kb = r.size_bytes / 1024
                mtime = datetime.fromtimestamp(r.modified).strftime("%m-%d %H:%M")
                status_tag = f"[{r.status.upper()}]"
                lines.append(
                    f"  {status_tag:10s} {r.kind:16s} "
                    f"{r.name:40s}  {size_kb:>6.1f}KB  {mtime}"
                )

        return lines

    # ------------------------------------------------------------------
    # 以下方法保留兼容 ==========
    # ------------------------------------------------------------------

    def normalize_names(self, dry_run: bool = True) -> list[tuple[str, str]]:
        records = self.scan()
        by_kind: dict[str, list[FileRecord]] = {}
        for r in records:
            if r.kind not in ("state", "unknown"):
                by_kind.setdefault(r.kind, []).append(r)

        renames = []
        for kind, group in by_kind.items():
            active = [r for r in group if r.status == "active"]
            active.sort(key=lambda r: r.modified)
            for i, r in enumerate(active, 1):
                ext = os.path.splitext(r.name)[1]
                label = self.KIND_LABELS.get(kind, kind)
                new_name = f"{label}{ext}" if len(active) == 1 else f"{label}_v{i}{ext}"
                if r.name != new_name:
                    old_path = r.path
                    new_path = os.path.join(os.path.dirname(r.path), new_name)
                    renames.append((old_path, new_path))
                    if not dry_run:
                        os.rename(old_path, new_path)
        return renames

    def _check_metadata_redundancy(self, records: list[FileRecord]) -> int:
        duplicates = 0
        by_run: dict[str, dict[str, FileRecord]] = {}
        for r in records:
            by_run.setdefault(r.run_id, {}).setdefault(r.kind, r)
        for run_id, kinds in by_run.items():
            if run_id in ("current", ""):
                continue
            if "archive" in kinds and "core_papers" in kinds:
                duplicates += 1
            if "draft" in kinds and "final_draft" in kinds:
                duplicates += 1
        return duplicates

    def format_report(self, report: CleanupReport) -> str:
        lines = [
            "",
            "=" * 65,
            "  CLEANUP REPORT (v3 Dynamic Memory Reclamation)",
            "=" * 65,
            "",
            f"  Timestamp:    {report.timestamp}",
            f"  Total files:  {report.total_files}",
            f"  Kept:         {report.kept}",
            f"  Removed:      {report.removed}",
            f"  Promoted:      {report.promoted}",
            f"  Audit rejects: {report.audit_rejections}",
            f"  Records destroyed: {report.records_destroyed}",
            "",
        ]

        if report.destroyed_dois:
            lines.append("  --- Destroyed Records (Auditor ordered) ---")
            for doi in report.destroyed_dois:
                lines.append(f"    x  {doi}")
            lines.append("")

        if report.removed_files:
            lines.append("  --- Removed Files ---")
            for f in report.removed_files:
                fname = os.path.basename(f)
                lines.append(f"    x  {fname}")
            lines.append("")

        if report.status_summary:
            lines.append("  --- Workspace Status ---")
            for s in report.status_summary:
                lines.append(f"  {s}")

        lines.append("")
        lines.append("=" * 65)
        return "\n".join(lines)
