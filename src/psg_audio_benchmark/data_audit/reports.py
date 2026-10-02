"""Markdown report renderers for the Stage-2 audit (prompt section 5 & 6).

Every report distinguishes four evidence levels (prompt section 5):

* **[文献假设]** literature-reported hypothesis (e.g. 50 patients, 48/16 kHz,
  ~36 airflow) -- never copied as a measured conclusion.
* **[实测]** a fact measured from real files during this audit.
* **[推断]** an inference drawn from measured facts.
* **[待决]** an open question deferred to a later stage or to a human.

The audit only populates the measured/inference sections when the license gate
has passed; when the gate is blocked, the structure/data-audit/data-dictionary
reports are emitted in a "not_generated: license gate blocked" stub form and the
``license_gate_blocked.md`` + completion report carry the explanation.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .manifests import EVENT_DIST_PRELIMINARY_NOTE

_LIT = "[文献假设]"
_OBS = "[实测]"
_INF = "[推断]"
_OPEN = "[待决]"


def _lines(*items: str) -> str:
    return "\n".join(items) + "\n"


# ---------------------------------------------------------------------------
# License gate reports
# ---------------------------------------------------------------------------

def render_license_gate_blocked(summary: Any) -> str:
    g = summary.license_gate
    missing = ", ".join(g.missing_items) if g.missing_items else "(未给出具体缺项)"
    notes = "\n".join(f"- {n}" for n in g.notes) if g.notes else "- （无附加说明）"
    return _lines(
        "# 阶段 2 许可确认门：BLOCKED",
        "",
        f"- run_id：`{summary.run_id}`",
        f"- 访问日期：{summary.access_date}",
        f"- 状态：**BLOCKED**（许可门未通过，未扫描/读取 raw 内容）",
        f"- 证据文件相对路径：`{g.evidence_relpath}`",
        f"- 缺项：{missing}",
        "",
        "## 附加说明",
        notes,
        "",
        "## 许可门行为（提示词 §1）",
        "",
        "- 在读取任何真实患者数据内容前，必须存在人工依据 Science Data Bank 正式页面/条款填写的",
        f"  `{g.evidence_relpath}`，且 `license_status` 不再是 `TBD_AFTER_MANUAL_LICENSE_REVIEW`。",
        "- 当前证据缺失/不完整或许可仍 TBD → 只生成本文件并停止；**不得扫描/读取 raw 内容**。",
        "- 不得通过修改测试、硬编码 `true` 或伪造确认文件绕过此门。",
        "",
        "## 人工操作项（解除门所需）",
        "",
        "1. 登录 Science Data Bank（scidb.cn），定位数据 DOI `10.57760/sciencedb.19070`。",
        "2. 依据正式页面/条款填写 `docs/license_evidence/primary_dataset_license_confirmation.md`",
        "   的全部字段：DOI、版本、访问日期、使用条款、非商业科研依据、派生特征/再分发策略、",
        "   证据截图相对位置、填写人、确认日期、不确定项。",
        "3. 重跑 `python scripts/02_data_audit.py --config config/config.yaml`；门通过后即产出完整审计。",
        "",
        "## 本次未做的事",
        "",
        "- 未生成 patient/file/availability/annotation/event 清单；未读取音频/CSV/annotation 内容。",
        "- raw 目录前后轻量快照均未采集（因未进入扫描阶段），raw 绝对未被本流程触碰。",
    )


def render_license_gate_passed(summary: Any) -> str:
    g = summary.license_gate
    return _lines(
        "# 阶段 2 许可确认门：PASSED",
        "",
        f"- run_id：`{summary.run_id}`",
        f"- 证据文件相对路径：`{g.evidence_relpath}`",
        f"- 数据 DOI：`{g.data_doi}`",
        f"- 数据版本：`{g.dataset_version}`",
        f"- 访问日期：{g.access_date}",
        f"- 派生 license_status：`{g.license_status}`",
        "",
        "> 许可门通过仅授权本次只读审计；不改变 raw 只读约束，不等于允许公开再分发原始音频。",
        "> 再分发/派生发布策略以证据文件填写为准。",
    )


# ---------------------------------------------------------------------------
# Structure exploration report (prompt 6 Q1-Q3, Q8)
# ---------------------------------------------------------------------------

def render_data_structure_exploration(summary: Any) -> str:
    if not getattr(summary, "license_gate_passed", False):
        return _lines(
            "# 数据结构探索报告（Stage 2）",
            "",
            f"- run_id：`{summary.run_id}`",
            "- 状态：**NOT_GENERATED — license gate blocked**。",
            "- 许可门未通过，未扫描 raw；本报告在门通过并重跑后生成。",
            "",
            f"详见 `{'license_gate_blocked.md'}`。",
        )

    p = summary.audio_summary or {}
    return _lines(
        "# 数据结构探索报告（Stage 2）",
        "",
        f"- run_id：`{summary.run_id}`",
        f"- 访问日期：{summary.access_date}",
        "",
        "## 1. 患者目录识别（提示词 §6 Q1）",
        "",
        f"- {_OBS} 候选患者目录数：{summary.n_patients}",
        f"- {_OBS} 患者 ID 解析规则版本：见 `patient_manifest.csv.mapping_rule_version`。",
        f"- {_OBS} 患者 ID 解析问题数：{len(summary.patient_issues)}（详见 `patient_id_resolution_issues.csv`）。",
        f"- {_LIT} 文献假设约 50 名患者 —— 以本审计实测为准。",
        "",
        "## 2. 文件语义与模态（提示词 §6 Q2）",
        "",
        f"- {_OBS} 文件总数：{summary.n_files}（分类证据见 `file_manifest_enriched.csv`）。",
        f"- {_OBS} 模态可用性矩阵：见 `signal_availability_matrix.csv`。",
        "- 录音笔文件数分布（提示词禁止假设恒为 2 段）：见 `patient_manifest.csv.recorder_file_count`。",
        "",
        "## 3. 音频实测（提示词 §6 Q3）",
        "",
        f"- {_OBS} 容器：{p.get('containers', {})}.",
        f"- {_OBS} 采样率分布：{p.get('sample_rates', {})}.",
        f"- {_OBS} 声道分布：{p.get('channels', {})}.",
        f"- {_OBS} 采样位深分布：{p.get('sample_widths', {})}.",
        f"- {_OBS} header 时长范围（秒）：min={p.get('dur_min')} / max={p.get('dur_max')}.",
        f"- {_OBS} 不可读/异常音频：{p.get('n_unreadable', 0)}.",
        f"- {_LIT} 文献假设手机 48 kHz stereo、录音笔 16 kHz —— 以实测为准，禁止照搬。",
        f"- {_OPEN} header 时长不是真实同步关系；时钟对齐留给后续阶段。",
        "",
        "## 4. 目录/文件命名异常（提示词 §6 Q8）",
        "",
        _format_anomalies(summary.anomalies),
    )


# ---------------------------------------------------------------------------
# Data audit report (prompt 6 Q4-Q7)
# ---------------------------------------------------------------------------

def render_data_audit_report(summary: Any) -> str:
    if not getattr(summary, "license_gate_passed", False):
        return _lines(
            "# 数据审计报告（Stage 2）",
            "",
            f"- run_id：`{summary.run_id}`",
            "- 状态：**NOT_GENERATED — license gate blocked**。",
            "- 许可门通过并重跑后生成 CSV/JSON/音频结构审计结论。",
        )

    c = summary.csv_summary or {}
    a = summary.annotation_summary or {}
    return _lines(
        "# 数据审计报告（Stage 2）",
        "",
        f"- run_id：`{summary.run_id}`",
        f"- 访问日期：{summary.access_date}",
        "",
        "## 1. CSV schema / 时间列 / 采样率 / 缺失（提示词 §6 Q4）",
        "",
        f"- {_OBS} CSV 文件数：{c.get('n_csv', 0)}.",
        f"- {_OBS} 推断角色分布：{c.get('roles', {})}.",
        f"- {_OBS} 采样率估计分布：{c.get('sample_rates', {})}.",
        f"- {_OBS} 基础单元格缺失比例范围：{c.get('missing_min')} ~ {c.get('missing_max')}.",
        f"- {_OBS} 无时间列/时间不可解析文件数：{c.get('n_no_time', 0)}（报告结构问题，未臆测采样率）。",
        f"- {_INF} 缺失统计仅基础单元格；未做插补。",
        "",
        "## 2. JSON/TXT annotation 结构（提示词 §6 Q5）",
        "",
        f"- {_OBS} annotation 文件数：{a.get('n_annotations', 0)}（格式分布：{a.get('formats', {})}）。",
        f"- {_OBS} 是否含 `record_start`：{a.get('has_record_start')}；`awake_intervals`：{a.get('has_awake')}；`events`：{a.get('has_events')}。",
        f"- {_OBS} 事件对象字段集与变体（如 `evnet_start` 拼写）：见 `annotation_structure_inventory.csv`。",
        f"- {_OBS} raw event type 频数（结构审计）：见 `event_distribution_preliminary.csv`（{EVENT_DIST_PRELIMINARY_NOTE}）。",
        f"- {_INF} 结构可支持下一阶段受控解析，但本阶段不写最终标准化事件表。",
        "",
        "## 3. 时间范围表面可比较性（提示词 §6 Q6）",
        "",
        f"- {_INF} 音频/CSV/annotation 时间范围是否表面可比较：{a.get('time_surface_comparable', '未判定')}.",
        f"- {_OPEN} 真正对齐须在同步阶段核验；本审计不假装已对齐。",
        "",
        "## 4. airflow 子集（提示词 §6 Q7）",
        "",
        f"- {_OBS} 含 airflow(Flow_DR) 患者数：{summary.airflow_patients}.",
        f"- {_LIT} 文献假设约 36/50 —— 以实测为准。",
        f"- {_INF} airflow 仅在可用子集使用；缺失患者不删除。",
        "",
        "## 5. 异常与待决",
        "",
        _format_anomalies(summary.anomalies),
    )


# ---------------------------------------------------------------------------
# Data dictionary (observed)
# ---------------------------------------------------------------------------

def render_data_dictionary(summary: Any) -> str:
    if not getattr(summary, "license_gate_passed", False):
        return _lines(
            "# 观测数据字典（Stage 2）",
            "",
            f"- run_id：`{summary.run_id}`",
            "- 状态：**NOT_GENERATED — license gate blocked**。",
        )
    c = summary.csv_summary or {}
    a = summary.annotation_summary or {}
    return _lines(
        "# 观测数据字典（Stage 2，基于实测文件结构）",
        "",
        f"- run_id：`{summary.run_id}`",
        "",
        "> 本字典来自本次审计的实测结构，非文献照抄。字段含义以实测为准。",
        "",
        "## CSV 生理/睡眠结构文件",
        "",
        f"- 观测到的列名集合（去重）：见 `file_manifest_enriched.csv.csv_*` 与各 CSV 的 `csv_columns`。",
        f"- 推断角色：{c.get('roles', {})}.",
        f"- 缺失值标记：`-`（基础单元格缺失，未插补）。",
        "",
        "## Annotation JSON 事件对象",
        "",
        f"- 观测字段集：{a.get('event_field_set', [])}.",
        f"- 变体：{a.get('field_variants', [])}（注意 `evnet_start` 为原文拼写）。",
        f"- raw event type 取值（结构审计）：{a.get('raw_event_types', [])}.",
        "",
        "## 音频",
        "",
        "- 见 `data_structure_exploration_report.md` 第 3 节（容器/采样率/声道/位深/时长）。",
    )


# ---------------------------------------------------------------------------
# Completion report
# ---------------------------------------------------------------------------

def render_completion_report(summary: Any, config_hash: str = "") -> str:
    gate_passed = getattr(summary, "license_gate_passed", False)
    status_line = summary.overall_status
    gate = summary.license_gate
    products = getattr(summary, "product_paths", {}) or {}
    prod_lines = "\n".join(f"- `{k}`：{v}" for k, v in sorted(products.items())) or "- （许可门 blocked，未生成清单）"
    return _lines(
        "# 阶段 2 完成报告（phase_02_completion_report）",
        "",
        f"- run_id：`{summary.run_id}`",
        f"- config_hash：`{config_hash or '(n/a)'}`",
        f"- 访问日期：{summary.access_date}",
        f"- 阶段 2 状态：**{status_line}**",
        f"- 许可门状态：`{gate.status}`（license_status=`{gate.license_status}`）",
        f"- 阶段 1 基线 production run：`{summary.stage1_run_id}`",
        "",
        "## 许可门",
        "",
        f"- 证据相对路径：`{gate.evidence_relpath}`",
        f"- 数据 DOI：`{gate.data_doi or '(blocked,未核验)'}`；版本：`{gate.dataset_version or '(blocked)'}`",
        f"- 缺项：{', '.join(gate.missing_items) if gate.missing_items else '无'}",
        "",
        "## 审计产物路径",
        "",
        prod_lines,
        "",
        "## 准入下一阶段（标注受控解析与时间轴同步）",
        "",
        (
            f"- {_OPEN} 工程准入满足（许可门已通过，且本次已产出真实患者/模态/结构审计）；"
            "是否进入阶段 3（受控 annotation 解析与时间轴同步）由人工复核本次审计结论后决定。"
            if gate_passed
            else f"- {_OPEN} 当前 **未满足**：许可门未通过（缺项：{', '.join(gate.missing_items) or '未给出'}）。"
            "需先人工依据 Science Data Bank 正式条款补齐许可证据并重跑，产出真实患者/模态/结构审计后，方可评估进入阶段 3。"
        ),
        "- 本次不自行进入下一阶段。",
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _format_anomalies(anomalies: Optional[Iterable[str]]) -> str:
    if not anomalies:
        return "- （本审计未记录异常）"
    return "\n".join(f"- {a}" for a in list(anomalies)[:200])


__all__ = [
    "render_license_gate_blocked",
    "render_license_gate_passed",
    "render_data_structure_exploration",
    "render_data_audit_report",
    "render_data_dictionary",
    "render_completion_report",
]
