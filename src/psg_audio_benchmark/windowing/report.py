"""Markdown report renderers for Stage 4 (prompt sections 5, 7).

Two reports + the Stage-3 cross-midnight count erratum. Every claim is tagged:

* **[实测]** measured from real files/tables during this run.
* **[配置规则]** derived from the resolved windowing config.
* **[待决]** deferred to a later stage or a human.

No conclusion is a clinical diagnosis; no audio time anchor is assumed; excluded
windows are never reported as negative.
"""

from __future__ import annotations

from typing import Dict

from .schema import WindowingSummary

_OBS = "[实测]"
_CFG = "[配置规则]"
_OPEN = "[待决]"


def _lines(*items: str) -> str:
    return "\n".join(items) + "\n"


def _kv(d: Dict[str, int]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(d.items())) if d else "(无)"


# ---------------------------------------------------------------------------
# window_label_report.md
# ---------------------------------------------------------------------------

def render_window_label_report(summary: WindowingSummary) -> str:
    s = summary
    return _lines(
        "# 阶段 4 窗口标签报告（CSV/标注侧）",
        "",
        f"- run_id：`{s.run_id}`",
        f"- 输入 Stage-3 run_id：`{s.input_run_id}`",
        f"- 访问日期：{s.access_date}",
        f"- 阶段 4 状态：**{s.overall_status}**",
        "",
        "> 窗口契约见 `docs/window_label_contract.md` 与 `windowing_config_resolved.yaml`。"
        "本阶段仅构建 CSV/标注侧窗口索引与研究标签，不含音频、特征、建模、划分与评价。",
        "",
        "## 1. 窗口标签语义（研究标签，非诊断）",
        "",
        f"- {_CFG} 主协议：固定长度 30.0 s、步长 30.0 s、左闭右开 `[start, end)`，坐标为相对 record_start 秒。",
        f"- {_CFG} 只有“核心覆盖完整且 awake 合格”的窗口才可得研究标签；正窗（≥1 保留事件起点）=1，负窗（0 保留事件起点）=0。",
        f"- {_CFG} 排除窗（覆盖不足或 awake 重叠）的 `binary_event_label` 为空，**绝不记作阴性**。",
        f"- {_OPEN} 所有标签均为“数据集评分呼吸事件研究标签”，**不是**临床诊断、不是 PSG 替代、不区分阻塞性/中枢性/混合性。",
        "",
        "## 2. 总体窗口计数",
        "",
        f"- {_OBS} 患者数：{s.n_patients}；拥有核心信号域的患者数：{s.n_patients_with_core_domain}。",
        f"- {_OBS} 候选窗口：{s.total_candidate_windows}；核心覆盖完整窗口：{s.total_core_complete_windows}。",
        f"- {_OBS} awake 排除窗口：{s.total_awake_excluded_windows}；覆盖不足排除窗口：{s.total_coverage_excluded_windows}；排除窗口合计：{s.total_excluded_windows}。",
        f"- {_OBS} 正窗：{s.total_positive_windows}；负窗：{s.total_negative_windows}；可用窗口（正+负）：{s.total_usable_windows}。",
        "",
        "## 3. 患者级窗口分布",
        "",
        f"- {_OBS} 各患者可用窗口数 min/median/max：{s.per_patient_window_min}/{s.per_patient_window_median}/{s.per_patient_window_max}。",
        f"- {_OBS} 零可用窗口患者数：{s.n_patients_zero_usable}；"
        + ("患者：" + ", ".join(f"`{p}`" for p in sorted(s.zero_usable_patients)) if s.zero_usable_patients else "（无）")
        + "。",
        "",
        "## 4. 事件归宿（不得隐去未关联事件）",
        "",
        f"- {_OBS} 事件总数：{s.total_events}；进入标签窗口的关联事件（linked）：{s.total_linked_events}。",
        f"- {_OBS} 事件归宿分类：{_kv(s.event_destiny_counts)}。",
        "- awake/域外/无有效窗口事件均单独统计，**不作为负例，也不丢失**。",
    )


# ---------------------------------------------------------------------------
# windowing_qc_report.md
# ---------------------------------------------------------------------------

def render_windowing_qc_report(summary: WindowingSummary) -> str:
    s = summary
    sig_lines = "\n".join(
        f"- {_OBS} `{m}`：{s.sleep_structure_status_counts.get(m, 0)} 个（观察状态，非本阶段 verified 依据）。"
        for m in sorted(s.sleep_structure_status_counts)
    ) or "- （无）"
    return _lines(
        "# 阶段 4 科学与工程 QC 报告",
        "",
        f"- run_id：`{s.run_id}`",
        f"- 输入 Stage-3 run_id：`{s.input_run_id}`",
        "",
        "## 1. 按患者汇总的窗口计数",
        "",
        f"- {_OBS} 候选窗口：{s.total_candidate_windows}；核心覆盖完整：{s.total_core_complete_windows}。",
        f"- {_OBS} awake 排除：{s.total_awake_excluded_windows}；覆盖不足排除：{s.total_coverage_excluded_windows}；排除合计：{s.total_excluded_windows}。",
        f"- {_OBS} 最终正窗：{s.total_positive_windows}；负窗：{s.total_negative_windows}；可用：{s.total_usable_windows}。",
        f"- {_OBS} 各患者窗口数 min/median/max：{s.per_patient_window_min}/{s.per_patient_window_median}/{s.per_patient_window_max}。",
        f"- {_OBS} 零可用窗口患者数：{s.n_patients_zero_usable}（详见 `patient_window_summary.csv` 的 `zero_reason`）。",
        "",
        "## 2. 事件归宿",
        "",
        f"- {_OBS} 已关联事件：{s.total_linked_events}。",
        f"- {_OBS} 各拒绝/旁置原因：{_kv(s.event_destiny_counts)}。",
        f"- {_OBS} 事件逐条归宿见 `window_event_links.parquet`（含未关联事件，window_index=-1）。",
        "",
        "## 3. CSV 模态窗口覆盖",
        "",
        f"- {_OBS} heart_rate 窗口覆盖：{s.hr_window_coverage}；spo2 窗口覆盖：{s.spo2_window_coverage}；airflow 窗口覆盖（可选）：{s.airflow_window_coverage}。",
        sig_lines,
        f"- {_CFG} `sleep_structure` 仅作观察性上下文/coverage 列，本阶段状态为 `not_verified`，不得据此声称睡眠分期对齐完成。",
        "",
        "## 4. 音频窗口 eligible=0 的证据与阻断",
        "",
        f"- {_OBS} 音频文件总数：{s.audio_total_files}；音频状态分布：{_kv(s.audio_status_counts)}。",
        f"- {_OBS} `audio_window_eligible=true` 的窗口数：{s.audio_window_eligible_count}（恒为 0）。",
        f"- {_CFG} 阻断原因：`no_trustworthy_audio_time_anchor_unresolved`；禁止以 record_start/mtime/文件名/顺序/时长代替音频锚点。",
        f"- {_OPEN} 音频窗口构建在获得可信时间锚点前持续 BLOCKED，须单独单列，不得混入本 CSV 窗口索引。",
        "",
        "## 5. 不变性与隔离",
        "",
        f"- {_OBS} raw 前后轻量快照是否一致（raw_modified）：{s.raw_modified}；raw 是否被扫描（仅快照）：{s.raw_scanned}。",
        f"- {_OBS} 本阶段不读取原始 CSV/音频内容，仅消费 Stage-3 派生 parquet；阶段 1/2/3 历史 run 与固定产物不被改写。",
        f"- {_OBS} 告警/异常：{len(s.anomalies)}（{'; '.join(s.anomalies[:5]) if s.anomalies else '无'}）。",
        "",
        "## 6. 研究标签限定（重申）",
        "",
        f"- {_OPEN} 全部标签为研究标签、非诊断；本阶段不执行随机窗口划分，也不输出 train/val/test split；后续实验仍只能患者级划分。",
    )


# ---------------------------------------------------------------------------
# phase_03_cross_midnight_count_erratum.md (text body; runner writes the file)
# ---------------------------------------------------------------------------

def render_phase03_cross_midnight_erratum(
    *,
    correct_intervals: int,
    correct_patients: int,
    reported_patients: int,
    correct_patient_ids,
    input_run_id: str,
    erratum_date: str,
) -> str:
    pid_list = ", ".join(f"`{p}`" for p in sorted(correct_patient_ids)) if correct_patient_ids else "（无）"
    return _lines(
        "# 阶段 3 跨午夜患者计数勘误（ERRATUM）",
        "",
        f"- 勘误日期（UTC）：{erratum_date}",
        "- 触发提示词：`Project_Materials/07_阶段4_受限CSV标注侧窗口索引与标签构建_Claude_Code提示词.md` §2",
        f"- 关联阶段 3 production run：`{input_run_id}`",
        "",
        "---",
        "",
        "## 1. 发现方式",
        "",
        "独立审计以 `awake_intervals_canonical.parquet` 为准复核跨午夜（cross-midnight）统计：",
        f"`cross_midnight=true` 的 canonical 区间为 **{correct_intervals} 条，涉及 {correct_patients} 位患者**。",
        "",
        f"阶段 3 的 `annotation_parse_report.md` 第 5 节曾写为“跨午夜 canonical 区间数：{correct_intervals}；跨午夜患者数：{reported_patients}”，",
        f"其中“跨午夜患者数 {reported_patients}”**与表不一致**。根因：阶段 3 `_aggregate` 的 `patients_cross_midnight`",
        "把 awake canonical 跨午夜患者与 **CSV 信号跨午夜**（`signal_time_ranges.cross_midnight`）患者合并计数，",
        f"导致该叙述性计数被信号侧跨午夜 inflate 到 {reported_patients}。awake canonical 侧的正确值为 {correct_patients}。",
        "",
        "## 2. 正确数值（依据 `awake_intervals_canonical.parquet`）",
        "",
        f"- 跨午夜 canonical 区间数：**{correct_intervals}**（与原报告一致，未变）。",
        f"- 跨午夜患者数：**{correct_patients}**（原报告误记为 {reported_patients}）。",
        f"- 涉及患者：{pid_list}。",
        "",
        "## 3. 影响评估",
        "",
        "- 本勘误**仅更正叙述性聚合计数**；不改变 `awake_intervals_canonical.parquet` 等任何已生成的表。",
        "- 阶段 3 的历史 run-dir 与固定产物**不被重跑、覆盖或改写**。",
        "- 该误记不影响事件计数、CSV 时间轴验证或音频判定等其他结论。",
        "- 阶段 4 的窗口构建不依赖该聚合叙述计数（直接读取 canonical 表的逐条区间），不受影响。",
        "",
        "## 4. 回归测试",
        "",
        "- 阶段 4 新增回归测试 `test_stage4_cross_midnight_erratum.py`：直接读取 Stage-3 input run 的",
        f"`awake_intervals_canonical.parquet`，断言 `cross_midnight=true` 区间数为 {correct_intervals}、涉及患者数为 {correct_patients}。",
        "- 该测试防止该聚合统计再次被信号侧跨午夜污染。",
    )


__all__ = [
    "render_window_label_report",
    "render_windowing_qc_report",
    "render_phase03_cross_midnight_erratum",
]
