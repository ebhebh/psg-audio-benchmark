"""Markdown report renderers for Stage 3 (prompt sections 3.2, 5, 7).

Three reports + the input-QC erratum text. Every claim is tagged:

* **[实测]** measured from real files during this run.
* **[配置规则]** derived from a versioned config rule (e.g. the 30 s threshold,
  the event-type mapping).
* **[待决]** deferred to a later stage or a human.

No conclusion is written as a clinical diagnosis; no audio time anchor is
assumed.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .event_mapping import EventTypeMapping
from .schema import Stage3Summary

_OBS = "[实测]"
_CFG = "[配置规则]"
_OPEN = "[待决]"


def _lines(*items: str) -> str:
    return "\n".join(items) + "\n"


def _kv(d: Dict[str, int]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(d.items())) if d else "(无)"


# ---------------------------------------------------------------------------
# annotation_parse_report.md
# ---------------------------------------------------------------------------

def render_annotation_parse_report(summary: Stage3Summary) -> str:
    s = summary
    reject_lines = (
        "\n".join(f"- `{k}`：{v}" for k, v in sorted(s.rejection_reason_counts.items()))
        if s.rejection_reason_counts
        else "- （无被拒绝事件）"
    )
    return _lines(
        "# 阶段 3 受控 annotation 解析报告",
        "",
        f"- run_id：`{s.run_id}`",
        f"- 访问日期：{s.access_date}",
        f"- 阶段 3 状态：**{s.overall_status}**",
        f"- 许可门：{'PASSED' if s.license_gate_passed else 'BLOCKED'}（license_status=`{s.license_status}`）",
        f"- 阶段 2 输入 run_id：`{s.stage2_production_run_id}`",
        "",
        "## 1. 患者级解析（提示词 §5）",
        "",
        f"- {_OBS} 患者数：{s.n_patients}。",
        f"- {_OBS} 解析成功（含 record_start/awake/events）的患者数：{s.n_patients_parsed_ok}。",
        f"- {_OBS} 出现文件级解析失败/异常的患者数：{s.n_patients_with_parse_failures}。",
        "",
        "## 2. 事件计数（raw 与 standardized）",
        "",
        f"- {_OBS} raw 事件总数（文件内 events 数之和）：{s.n_events_raw}。",
        f"- {_OBS} 进入 parsed_events 的标准化事件数：{s.n_events_standardized}。",
        f"- {_OBS} 被拒绝/仅警告的事件数：{s.n_events_rejected}。",
        f"- {_OBS} raw 事件类型频数：{_kv(s.raw_event_type_counts)}。",
        f"- {_OBS} 保守标准化后类型频数：{_kv(s.standardized_type_counts)}。",
        "",
        "## 3. 拒绝原因",
        "",
        reject_lines,
        "",
        "## 4. 原始类型与保守映射",
        "",
        f"- {_OBS} 实测原始 `event_type` 仅含：{', '.join(sorted(s.raw_event_type_counts)) or '(无)'}。",
        f"- {_CFG} 保守映射：`hypo`→`hypopnea`；`osa`→`apnea_unspecified`（**不是** obstructive_apnea）。",
        f"- {_CFG} 映射版本：`{s.mapping_version}`；详见 `event_type_mapping_report.md`。",
        f"- {_OPEN} 在获得原始标注说明书或作者确认前，不得据 `osa` 声称区分阻塞性/中枢性/混合性。",
        "",
        "## 5. awake intervals",
        "",
        f"- {_OBS} awake raw 区间总数：{s.awake_raw_total}；canonical 区间总数：{s.awake_canonical_total}。",
        f"- {_OBS} 完全重复去重数：{s.awake_exact_duplicates_removed}；重叠/相邻合并吸收数：{s.awake_overlaps_or_adjacent_merged}。",
        f"- {_OBS} 跨午夜 canonical 区间数：{s.awake_cross_midnight_intervals}；跨午夜患者数：{s.patients_cross_midnight}。",
        f"- {_OBS} 事件与醒期重叠：共 {s.events_overlapping_awake} 个事件至少部分落在某 awake 区间（仅统计，本阶段不删醒期、不建标签窗）。",
        "",
        "## 6. 不变性与隔离",
        "",
        f"- {_OBS} raw 前后轻量快照是否一致（raw_modified）：{s.raw_modified}。",
        f"- {_OBS} 告警总数（synchronization_warnings.csv 行数）：{s.warnings_total}。",
    )


# ---------------------------------------------------------------------------
# time_synchronization_report.md
# ---------------------------------------------------------------------------

def render_time_synchronization_report(summary: Stage3Summary) -> str:
    s = summary
    sig_lines = (
        "\n".join(
            f"- {_OBS} `{m}`：可对齐（verified）CSV 文件数 {s.signal_verified_counts.get(m, 0)} / "
            f"共 {s.signal_modality_counts.get(m, 0)} 个。"
            for m in sorted(s.signal_modality_counts)
        )
        or "- （无 CSV 信号）"
    )
    audio_lines = "\n".join(
        f"- {_OBS} `{st}`：{n} 个音频文件。"
        for st, n in sorted(s.audio_status_counts.items())
    ) or "- （无音频文件）"
    short_lines = (
        "\n".join(f"- `{p}`" for p in s.short_audio_files)
        if s.short_audio_files
        else "- （无）"
    )
    return _lines(
        "# 阶段 3 时间轴同步报告",
        "",
        f"- run_id：`{s.run_id}`",
        f"- 访问日期：{s.access_date}",
        "",
        "> 时间契约见 `docs/time_axis_contract.md`。本阶段证据优先，绝不假定音频采集起点。",
        "",
        "## 1. 统一时间表示（提示词 §4.1）",
        "",
        f"- {_OBS} 每位患者的 record_start 已解析为 seconds_of_day / day_offset / cumulative；详见 `record_time_anchors.parquet`。",
        f"- {_CFG} 跨午夜采用唯一、可审计的 +86400 rollover 规则，记录 day_offset；不做字符串模糊比较。",
        f"- {_CFG} `record_start` 仅作为 PSG 标注时钟锚点；配置项 `record_start_is_audio_anchor` 恒为 false，禁止以之代替音频采集起点。",
        "",
        "## 2. CSV 各模态可验证数（提示词 §4.1, §7.6）",
        "",
        sig_lines,
        f"- {_OBS} 详见 `signal_time_ranges.parquet`（首末时间、样本数、相对/绝对时长、跨午夜、verification/quality 状态）。",
        "",
        "## 3. 音频时间锚点严格判定（提示词 §4.2）",
        "",
        audio_lines,
        f"- {_OBS} 因短时长被排除的音频文件数：{s.n_audio_excluded_too_short}。",
        f"- {_CFG} 排除阈值 `minimum_source_audio_duration_seconds` = 30.0 s（输入 QC，非删除）。",
        f"- {_OBS} 被排除文件（相对路径）：",
        short_lines,
        "",
        "## 4. 音频“未解决”的阻断含义（提示词 §4.2, §7.9）",
        "",
        f"- {_OBS} 实测：所有 WAV 容器非音频元数据仅含 `LIST/INFO` 的 `ISFT`（编码器软件），无 `bext`、无 `ICRD`、无可校准时钟。",
        f"- {_OBS} 结论：全部音频标记为 `unresolved_no_trustworthy_audio_time_anchor`（短时长者另标记 `excluded_audio_too_short_for_analysis`）。",
        f"- {_CFG} 此状态**禁止**把 PSG 事件直接映射到音频窗口，且**不允许**后续阶段静默以 record_start 代替音频起点。",
        f"- {_OPEN} 音频窗口构建在获得可信时间锚点（如作者确认的采集时间字段）之前持续 BLOCKED；详见完成汇报第 9 节。",
        "",
        "## 5. 同步告警",
        "",
        f"- {_OBS} 全部告警见 `synchronization_warnings.csv`（共 {s.warnings_total} 行）。",
        "- 状态分类始终区分 verified / unresolved / excluded / not_applicable，不以布尔掩盖。",
    )


# ---------------------------------------------------------------------------
# event_type_mapping_report.md
# ---------------------------------------------------------------------------

def render_event_type_mapping_report(summary: Stage3Summary, mapping: EventTypeMapping) -> str:
    s = summary
    observed_lines = "\n".join(
        f"- `{raw}`（实测 {s.raw_event_type_counts.get(raw, 0)} 次）→ `{mapping.mapped[raw]}`（{mapping.status.get(raw)}）"
        for raw in sorted(mapping.mapped)
    )
    reserved_lines = "\n".join(
        f"- `{r}`：reserved_placeholder，本阶段**不生成**（无原始证据）。"
        for r in sorted(mapping.reserved)
    ) or "- （无）"
    return _lines(
        "# 阶段 3 事件类型映射报告",
        "",
        f"- run_id：`{s.run_id}`",
        f"- 映射版本：`{mapping.mapping_version}`",
        f"- 映射来源：{mapping.mapping_source.strip()}",
        "",
        "## 1. 实测原始值（提示词 §3.2）",
        "",
        f"- {_OBS} 本数据集原始 `event_type` 仅观察到：{', '.join(f'`{t}`' for t in sorted(mapping.observed_raw_types))}。",
        "- 未直接观察到 central / mixed / obstructive 等显式亚型。",
        "",
        "## 2. 本项目保守标准化",
        "",
        observed_lines,
        "",
        f"- {_CFG} `osa` → `apnea_unspecified`，**不是** `obstructive_apnea`。在没有原始标注说明书或作者确认前，不得据 `osa` 区分阻塞性/中枢性/混合性。",
        f"- {_CFG} 二分类候选字段 `{mapping.binary_field}`：仅对本数据集观察到的 hypo/osa 事件置 1；不作为临床诊断标签。",
        "",
        "## 3. 尚不可作出的亚型结论",
        "",
        reserved_lines,
        f"- {_OPEN} 任何 central/mixed/obstructive 的显式映射须待原始证据后另起版本，本次不得凭空生成。",
    )


# ---------------------------------------------------------------------------
# phase_02_input_qc_erratum.md (text body; the runner writes the file)
# ---------------------------------------------------------------------------

def render_phase02_input_qc_erratum(
    *,
    threshold_seconds: float,
    affected_files_relpaths: List[str],
    stage2_production_run_id: str,
    access_date: str,
) -> str:
    files_lines = "\n".join(f"- `{p}`" for p in affected_files_relpaths) or "- （无）"
    return _lines(
        "# 阶段 2 输入 QC 勘误（ERRATUM）：短时长音频未列入异常",
        "",
        f"- 勘误日期（UTC）：{access_date}",
        f"- 触发提示词：`Project_Materials/06_阶段3_受控标注解析与时间轴同步_Claude_Code提示词.md` §2",
        f"- 关联阶段 2 production run：`{stage2_production_run_id}`",
        "",
        "---",
        "",
        "## 1. 发现方式",
        "",
        "独立审计确认 `data/raw/V5/Data/19/19_recorder_1.wav` 为 2,126 字节、0.128 秒。阶段 2 的",
        "`file_manifest_enriched.csv` 记录了该时长（`audio_header_duration_s=0.128`，`audio_readability_status=readable`），",
        "却未将其列为异常；`data_audit_report.md` “异常与待决”一节写为“（本审计未记录异常）”，即“异常音频 0”。",
        "该表述不完整：一个无法构成任何 30 秒主窗口的音频文件应在输入 QC 中被显式标记。",
        "",
        "## 2. 阈值（配置规则，非删除阈值）",
        "",
        f"- {_CFG} `config.yaml.annotation_sync.minimum_source_audio_duration_seconds` = {threshold_seconds}。",
        "- 含义：低于该时长的音频**不能**构成任何 30 秒主窗口，是输入 QC 阈值，**不是**删除/移动/修改阈值。",
        "",
        "## 3. 受影响文件",
        "",
        f"- {_OBS} 满足 `duration_seconds < {threshold_seconds}` 的音频文件数：{len(affected_files_relpaths)}。",
        files_lines,
        "",
        "## 4. 对下游的处置（保留一切原始产物）",
        "",
        "- 该文件保留于 `data/raw` 原始区与阶段 2 的 `file_manifest_enriched.csv`，**不删除、不移动、不修改**。",
        "- 阶段 2 的固定生产产物与历史 run-dir **不被重跑、覆盖或改写**；本勘误为补充记录，不覆盖原报告。",
        "- 在阶段 3 的音频可用性表中，该文件标记为 `excluded_audio_too_short_for_analysis`，排除其音频对齐/窗口候选。",
        "- 阶段 2 通用异常判定代码本次**未修改**（因此不涉及为阶段 2 新增回归测试的范畴）；阶段 3 在自身产物中应用该阈值，并由阶段 3 测试覆盖（合成短/正常 WAV）。",
    )


__all__ = [
    "render_annotation_parse_report",
    "render_time_synchronization_report",
    "render_event_type_mapping_report",
    "render_phase02_input_qc_erratum",
]
