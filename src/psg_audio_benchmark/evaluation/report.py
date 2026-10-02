"""Markdown report renderers for Stage 6.

Three reports (all carry the prominent scope disclaimer):

* ``cohort_definition_report.md`` — cohort membership rules, counts, per-patient
  burden, exclusion accounting, the airflow discrepancy note, audio block.
* ``split_balance_report.md`` — outer/inner fold balance, airflow inheritance,
  patient-isolation verification, balance warnings.
* ``phase_06_completion_report.md`` — overall status, inputs, the deferred-
  transform list, next-stage admission.
"""

from __future__ import annotations

from typing import List

from .schema import (
    AUDIO_BLOCK_REASON,
    AUDIO_COHORT_COUNT,
    PATIENT_SPLIT_VERSION,
    SplitSummary,
)

_DISCLAIMER = (
    "> **范围声明（显示于每个阶段 6 报告）：** 这是**单一公开数据集**内的"
    "患者级**内部**验证方案，**不是**外部验证，**不是**临床部署/诊断系统，"
    "**不是** PSG 替代；结果仅用于 CSV 生理、PSG 锚定的回顾性基准，"
    "**不适用于音频**。音频路线仍 BLOCKED。"
)


def _pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def render_cohort_definition_report(s: SplitSummary) -> str:
    excl = "\n".join(f"  - `{k}`：{v}" for k, v in sorted(s.exclusions_by_scope.items())) or "  - （无）"
    return (
        "# 阶段 6 队列定义报告（cohort_definition_report）\n\n"
        f"{_DISCLAIMER}\n\n"
        f"- run_id：`{s.run_id}`\n"
        f"- patient_split_version：`{PATIENT_SPLIT_VERSION}`\n"
        f"- 输入 Stage-4 run_id：`{s.input_stage4_run_id}`（verified={s.input_stage4_verified}）\n"
        f"- 输入 Stage-5 run_id：`{s.input_stage5_run_id}`（verified={s.input_stage5_verified}）\n\n"
        "## 1. 队列成员规则（stable-key join，不含模型矩阵）\n\n"
        "- **核心队列（main analysis）**：Stage-4 正/负候选窗口且 Stage-5 "
        "`core_hr_spo2_available=True`（HR 与 SpO2 均可用）；窗口三模态/标签按 "
        "`run_id/window_id/patient_id/window_index` 稳定键连接，不复制任何特征数值。\n"
        "- **airflow 子队列（secondary analysis）**：核心队列 ∩ `airflow_available=True`"
        "（严格：窗口须 HR+SpO2+airflow 同时可用）；其外层折**继承**核心队列，不另行随机划分。\n"
        "- Stage-4 `excluded` 窗口、HR/SpO2 低覆盖、airflow 可选缺失窗口逐条留痕于 "
        "`cohort_exclusions.csv`，**绝不**伪装为负例。\n\n"
        "## 2. 实测计数\n\n"
        f"- Stage-4 窗口总数：{s.total_stage4_windows}；候选（正+负）：{s.total_candidate_windows}；"
        f"Stage-4 excluded：{s.total_excluded_stage4_windows}。\n"
        f"- **核心队列**：患者 {s.core_patients}；窗口 {s.core_windows}；"
        f"正 {s.core_positive} / 负 {s.core_negative}（positive rate {_pct(s.core_positive / s.core_windows) if s.core_windows else 'n/a'}）。\n"
        f"- **airflow 子队列（核心∩airflow，严格）**：患者 {s.airflow_patients}；窗口 {s.airflow_windows}；"
        f"正 {s.airflow_positive} / 负 {s.airflow_negative}。\n\n"
        "## 3. airflow 口径差异（显式记录，便于审计）\n\n"
        f"- 提示词给定基线「28,409 窗口 / 34 患者」= 候选窗口中 `airflow_available=True`（**不要求** HR/SpO2 同时可用）"
        f"：实测 {s.airflow_available_alone_windows} 窗口 / {s.airflow_available_alone_patients} 患者。\n"
        f"- 本阶段采用**严格**口径「核心 ∩ airflow」（定义文字「在核心队列基础上要求 airflow available」）："
        f"实测 {s.airflow_windows} 窗口 / {s.airflow_patients} 患者。\n"
        f"- 差异 {s.airflow_available_alone_windows - s.airflow_windows} 个窗口为「airflow 可用但 HR/SpO2 不可用」，"
        f"无法进入 HR+SpO2+airflow 模型，故不计入 airflow 子队列。患者数两种口径均为 {s.airflow_patients}，"
        f"且为 50 位核心患者的严格子集，外层折继承不受影响。\n\n"
        "## 4. 每位患者标签负担（核心队列）\n\n"
        f"- 每患者窗口数 min/median/max：{s.core_windows_per_patient_min} / "
        f"{s.core_windows_per_patient_median:.1f} / {s.core_windows_per_patient_max}。\n"
        f"- 每患者 positive rate min/median/max：{s.core_pos_rate_per_patient_min:.4f} / "
        f"{s.core_pos_rate_per_patient_median:.4f} / {s.core_pos_rate_per_patient_max:.4f}。\n"
        "- 标签负担高度偏态；划分只能近似平衡，患者隔离优先（见 split_balance_report）。\n\n"
        "## 5. 排除窗口计数（按 scope）\n\n"
        f"{excl}\n\n"
        "## 6. 音频阻断\n\n"
        f"- audio cohort count = **{s.audio_cohort_count}**（恒为 0）；阻断原因：`{s.audio_block_reason}`；"
        "本阶段不读取任何 `*.wav`，不生成音频 cohort/split。\n"
    )


def _outer_table(s: SplitSummary) -> str:
    if not s.outer_fold_stats:
        return "（无外层折统计）"
    lines = ["| outer fold | train 患者 | test 患者 | train 窗口 | test 窗口 | train pos rate | test pos rate |",
             "|---|---|---|---|---|---|---|"]
    for st in s.outer_fold_stats:
        lines.append(
            f"| {st['outer_fold']} | {st['n_train_patients']} | {st['n_test_patients']} | "
            f"{st['n_train_windows']} | {st['n_test_windows']} | "
            f"{_pct(st['train_positive_rate'])} | {_pct(st['test_positive_rate'])} |"
        )
    return "\n".join(lines)


def _inner_table(s: SplitSummary) -> str:
    if not s.inner_fold_stats:
        return "（无内层折统计）"
    lines = ["| outer | inner | inner-train 患者 | inner-val 患者 | inner-val 窗口 | inner-val pos rate |",
             "|---|---|---|---|---|---|"]
    for st in s.inner_fold_stats:
        lines.append(
            f"| {st['outer_fold']} | {st['inner_fold']} | {st['n_inner_train_patients']} | "
            f"{st['n_inner_val_patients']} | {st['n_inner_val_windows']} | "
            f"{_pct(st['inner_val_positive_rate'])} |"
        )
    return "\n".join(lines)


def _airflow_table(s: SplitSummary) -> str:
    if not s.airflow_fold_stats:
        return "（无 airflow 折统计）"
    lines = ["| outer fold | airflow 患者 | airflow 窗口 | airflow pos rate |", "|---|---|---|---|"]
    for st in s.airflow_fold_stats:
        lines.append(
            f"| {st['outer_fold']} | {st['n_airflow_patients']} | {st['n_airflow_windows']} | "
            f"{_pct(st['airflow_positive_rate'])} |"
        )
    return "\n".join(lines)


def render_split_balance_report(s: SplitSummary) -> str:
    warnings = "\n".join(f"- ⚠️ {w}" for w in s.balance_warnings) or "- （无平衡告警）"
    anomalies = "\n".join(f"- {a}" for a in s.anomalies) or "- （无异常）"
    return (
        "# 阶段 6 划分平衡报告（split_balance_report）\n\n"
        f"{_DISCLAIMER}\n\n"
        f"- run_id：`{s.run_id}`\n"
        f"- 划分单位：**患者**；外层 {len(s.outer_fold_stats)} 折，固定种子；内层每外层折内 "
        f"{len(s.inner_fold_stats)//max(len(s.outer_fold_stats),1)} 折（患者级）。\n"
        "- 平衡仅使用模型训练前的患者级汇总（窗口数、正/负、positive rate、可用模态）；"
        "不使用任何模型输出或逐窗口随机化。\n\n"
        "## 1. 外层折（核心队列）\n\n"
        f"{_outer_table(s)}\n\n"
        "## 2. 内层折（每个外层折内，仅 outer-train 患者；outer-test 绝不进入 inner）\n\n"
        f"{_inner_table(s)}\n\n"
        "## 3. airflow 子队列继承（每外层折可用性）\n\n"
        f"{_airflow_table(s)}\n\n"
        "## 4. 患者隔离与泄漏验证\n\n"
        "- [断言] 同一患者的全部窗口严格在同一外层 test 折；任一外层 train/test 患者集合无交集。\n"
        "- [断言] 每位核心患者恰有一个外层折；outer-train 的每位患者恰有一个该外层折对应的 inner 验证折。\n"
        "- [断言] outer-test 患者绝不进入该外层折的任意 inner 折（零泄漏）。\n"
        "- [断言] airflow 患者的外层折与核心完全相同，绝不生成独立随机 split。\n"
        "- [断言] 固定 seed + 排序约定下结果可复现；改 seed 不破坏患者隔离。\n\n"
        "## 5. 不平衡告警\n\n"
        f"{warnings}\n\n"
        "## 6. 异常/告警项\n\n"
        f"{anomalies}\n"
    )


def render_phase_06_completion_report(s: SplitSummary) -> str:
    status = s.overall_status
    prods = "\n".join(f"- `{k}`：{v}" for k, v in sorted(s.product_paths.items())) or "- （无）"
    return (
        "# 阶段 6 完成报告（phase_06_completion_report）\n\n"
        f"{_DISCLAIMER}\n\n"
        f"- run_id：`{s.run_id}`\n"
        f"- config_hash：`{s.config_hash or '(n/a)'}`\n"
        f"- patient_split_version：`{PATIENT_SPLIT_VERSION}`\n"
        f"- 访问日期：{s.access_date}\n"
        f"- 阶段 6 状态：**{status}**\n"
        f"- 许可门：{'PASSED' if s.license_gate_passed else 'BLOCKED'}（license_status=`{s.license_status}`）\n"
        f"- 输入 Stage-4 run_id：`{s.input_stage4_run_id}`（verified={s.input_stage4_verified}）\n"
        f"- 输入 Stage-5 run_id：`{s.input_stage5_run_id}`（verified={s.input_stage5_verified}）\n\n"
        "## 关键计数\n\n"
        f"- 核心队列：患者 {s.core_patients}；窗口 {s.core_windows}；正 {s.core_positive} / 负 {s.core_negative}。\n"
        f"- airflow 子队列（核心∩airflow）：患者 {s.airflow_patients}；窗口 {s.airflow_windows}；"
        f"正 {s.airflow_positive} / 负 {s.airflow_negative}。\n"
        f"- audio cohort count = **{AUDIO_COHORT_COUNT}**（恒为 0）；阻断原因：`{AUDIO_BLOCK_REASON}`。\n\n"
        "## 本阶段未做的拟合（全部留待模型阶段折内拟合）\n\n"
        f"- 无插补（no_imputation={s.no_imputation}）。\n"
        f"- 无标准化/归一化（no_normalization={s.no_normalization}）。\n"
        f"- 无特征选择（no_feature_selection={s.no_feature_selection}）。\n"
        f"- 无类别重采样（no_class_resampling={s.no_class_resampling}）。\n"
        f"- 无阈值优化（no_threshold_optimization={s.no_threshold_optimization}）。\n"
        f"- 无模型/无性能评价（no_model_matrix={s.no_model_matrix}, no_performance_metrics={s.no_performance_metrics}）。\n\n"
        "## 产物路径\n\n"
        f"{prods}\n\n"
        "## 准入下一阶段\n\n"
        "- [待决] **下一阶段（受限 CSV 路线）**：仅可讨论核心 CSV 队列上的传统模型基线；"
        "模型内任何变换（imputer/scaler/selector/resampler/阈值）必须折内拟合，不得全队列预计算。\n"
        "- [待决] **音频路线：BLOCKED** —— audio cohort=0，所有音频仍 unresolved/excluded，须单独单列。\n"
        "- 本次不自行进入下一阶段；等待审计。\n"
    )


__all__: List[str] = [
    "render_cohort_definition_report",
    "render_split_balance_report",
    "render_phase_06_completion_report",
]
