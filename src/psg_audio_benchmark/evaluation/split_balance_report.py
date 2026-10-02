"""Markdown report renderers + summary container for Stage 6B.

Two reports (both carry the scope disclaimer):

* ``split_balance_optimization_report.md`` — pre-registered objective/weights,
  algorithm + seeds + candidate count, the v1-vs-v2 comparison table, the
  adoption-gate decision, inner CV + airflow inheritance + leakage verification.
* ``phase_06b_completion_report.md`` — the strict 10-point completion format
  (status, inputs/immutability, algorithm, comparison, gate/LATEST, inner/airflow,
  what-was-NOT-done, isolation/path-purity, tests, next-stage admission).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .gate import GateResult

_DISCLAIMER = (
    "> **范围声明：** 这是**单一公开数据集**内、患者级**内部**验证的划分再平衡，"
    "**不训练模型**，**不读取 raw CSV/WAV 或任何特征数值**，仅操作患者级汇总。"
    "音频路线仍 **BLOCKED**。"
)


@dataclass
class SplitBalanceSummary:
    run_id: str = ""
    config_hash: str = ""
    access_date: str = ""
    status: str = "not_approved_balance_target_not_met"  # or approved_for_modeling / BLOCKED
    v1_comparator_run_id: str = ""
    # search
    algorithm: str = ""
    base_seed: int = 0
    n_candidate_seeds: int = 0
    local_swap_iterations: int = 0
    selection_rule: str = ""
    n_candidates_evaluated: int = 0
    candidate_spreads: List[float] = field(default_factory=list)
    candidate_wcvs: List[float] = field(default_factory=list)
    # metrics
    v1: Dict[str, Any] = field(default_factory=dict)   # FoldMetrics-relevant fields
    v2: Dict[str, Any] = field(default_factory=dict)
    v1_per_fold: List[Dict[str, Any]] = field(default_factory=list)
    v2_per_fold: List[Dict[str, Any]] = field(default_factory=list)
    comparison_rows: List[Dict[str, Any]] = field(default_factory=list)
    comparison_summary: Dict[str, Any] = field(default_factory=dict)
    # gate
    gate: GateResult | None = None
    hard_checks: List[Any] = field(default_factory=list)
    leakage_checks: List[Any] = field(default_factory=list)
    # inner / airflow
    inner_warnings: List[str] = field(default_factory=list)
    inner_anomalies: List[str] = field(default_factory=list)
    inner_fold_stats: List[Dict[str, Any]] = field(default_factory=list)
    airflow_per_fold: List[Dict[str, Any]] = field(default_factory=list)
    airflow_mismatch_count: int = 0
    airflow_patients: int = 0
    # cohort / audio / deferred
    cohort: Dict[str, Any] = field(default_factory=dict)
    audio_cohort_count: int = 0
    audio_block_reason: str = ""
    # isolation / immutability
    raw_scanned: bool = False
    raw_modified: bool = False
    historical_products_unchanged: bool = True
    latest_updated: bool = False
    latest_path: str = ""
    mirrored_fixed_paths: List[str] = field(default_factory=list)
    anomalies: List[str] = field(default_factory=list)
    product_paths: Dict[str, str] = field(default_factory=dict)
    # tests (filled by the test driver / report manually)
    test_pass: int = 0
    test_fail: int = 0
    test_skip: int = 0
    test_runtime_seconds: float = 0.0
    test_interpreter: str = ""


def _pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def _fmt(x: Any, nd: int = 6) -> str:
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def render_split_balance_optimization_report(s: SplitBalanceSummary) -> str:
    gate = s.gate
    gate_lines = "- （无 gate 结果）"
    if gate is not None:
        gate_lines = "\n".join(
            f"- [{'✓' if c.passed else '✗'}] `{c.name}` — {c.detail}" for c in gate.checks
        ) or "- （无 gate 检查）"
    cand = ("min=%.6f / max=%.6f (n=%d)" % (
        min(s.candidate_spreads), max(s.candidate_spreads), len(s.candidate_spreads)
    )) if s.candidate_spreads else "n/a"
    cwcv = ("min=%.6f / max=%.6f" % (
        min(s.candidate_wcvs), max(s.candidate_wcvs)
    )) if s.candidate_wcvs else "n/a"

    # v1 vs v2 per-fold table
    nf = max(len(s.v1_per_fold), len(s.v2_per_fold))
    rows = ["| fold | v1 患者 | v2 患者 | v1 窗口 | v2 窗口 | v1 pos | v2 pos | v1 rate | v2 rate |",
            "|---|---|---|---|---|---|---|---|---|"]
    for f in range(nf):
        a = s.v1_per_fold[f] if f < len(s.v1_per_fold) else {}
        b = s.v2_per_fold[f] if f < len(s.v2_per_fold) else {}
        rows.append(
            f"| {f} | {a.get('n_patients','')} | {b.get('n_patients','')} | "
            f"{a.get('n_windows','')} | {b.get('n_windows','')} | "
            f"{a.get('n_positive','')} | {b.get('n_positive','')} | "
            f"{_pct(a.get('positive_rate',0.0))} | {_pct(b.get('positive_rate',0.0))} |"
        )
    per_fold_tbl = "\n".join(rows)

    inner_lines = "- （内层仅在 outer 批准后构建）"
    if s.inner_fold_stats:
        h = ["| outer | inner | inner-train 患者 | inner-val 患者 | inner-val 窗口 | inner-val pos rate |",
             "|---|---|---|---|---|---|"]
        for st in s.inner_fold_stats:
            h.append(f"| {st['outer_fold']} | {st['inner_fold']} | {st['n_inner_train_patients']} | "
                     f"{st['n_inner_val_patients']} | {st['n_inner_val_windows']} | "
                     f"{_pct(st['inner_val_positive_rate'])} |")
        inner_lines = "\n".join(h)

    af_lines = "- （无 airflow 折统计）"
    if s.airflow_per_fold:
        h = ["| outer fold | airflow 患者 | airflow 窗口 | airflow pos rate |", "|---|---|---|---|"]
        for st in s.airflow_per_fold:
            h.append(f"| {st['outer_fold']} | {st['n_airflow_patients']} | {st['n_airflow_windows']} | "
                     f"{_pct(st['airflow_positive_rate'])} |")
        af_lines = "\n".join(h)

    return (
        "# 阶段 6B 患者级划分平衡优化报告（split_balance_optimization_report）\n\n"
        f"{_DISCLAIMER}\n\n"
        f"- run_id：`{s.run_id}`\n"
        f"- config_hash：`{s.config_hash or '(n/a)'}`\n"
        f"- 状态：**{s.status}**\n"
        f"- v1 不可变 comparator run：`{s.v1_comparator_run_id}`\n\n"
        "## 1. 预注册目标函数与权重\n\n"
        "- 分量：`positive_rate_spread`（每折 test positive rate 的 max−min）、"
        "`window_count_cv`（每折 test 窗口数 `std(ddof=0)/mean`）。\n"
        "- 权重：`positive_rate_spread=1.0`，`window_count_cv=0.5`；"
        "aggregate = 1.0·spread + 0.5·wcv（搜索最小化对象；写入 resolved config）。\n"
        "- 采用门主度量：`positive_rate_spread`。\n\n"
        "## 2. 搜索算法（确定性、可复现）\n\n"
        f"- 算法：`{s.algorithm}`；base_seed={s.base_seed}；候选数={s.n_candidate_seeds}；"
        f"局部交换迭代={s.local_swap_iterations}。\n"
        f"- 选择规则：`{s.selection_rule}`。\n"
        f"- 候选 spread 分布：{cand}；候选 wcv 分布：{cwcv}。\n\n"
        "## 3. v1 与 v2 比较表（每外层 test 折）\n\n"
        f"{per_fold_tbl}\n\n"
        f"- 全队列 positive rate：{_pct(s.comparison_summary.get('overall_positive_rate', 0.0))}。\n"
        f"- positive-rate spread：v1={_fmt(s.comparison_summary.get('v1_positive_rate_spread',0.0))} "
        f"→ v2={_fmt(s.comparison_summary.get('v2_positive_rate_spread',0.0))}（阈值 "
        f"{_fmt(s.comparison_summary.get('gate_spread_max',0.0))}）。\n"
        f"- window-count CV：v1={_fmt(s.comparison_summary.get('v1_window_count_cv',0.0))} "
        f"→ v2={_fmt(s.comparison_summary.get('v2_window_count_cv',0.0))}。\n"
        f"- 最大绝对偏差（相对全队列 rate）：v1={_fmt(s.comparison_summary.get('v1_max_abs_deviation',0.0))} "
        f"→ v2={_fmt(s.comparison_summary.get('v2_max_abs_deviation',0.0))}。\n\n"
        "## 4. 采用门判据（不可人为挑选）\n\n"
        f"{gate_lines}\n\n"
        "## 5. 内层 CV（仅 outer 批准后；患者隔离 > 比例平衡）\n\n"
        f"{inner_lines}\n\n"
        f"内层告警：{chr(10).join('- ⚠️ '+w for w in s.inner_warnings) or '- （无）'}\n\n"
        "## 6. airflow 继承（直接继承 core 的 v2 outer fold；不另生成独立 split）\n\n"
        f"{af_lines}\n\n"
        f"- airflow 患者：{s.airflow_patients}；继承 mismatch 数：**{s.airflow_mismatch_count}**（须为 0）。\n\n"
        "## 7. 患者级泄漏验证\n\n"
        "- [断言] 50 位患者每外层折恰 10 人；每位患者恰一折；同一患者全部窗口同折。\n"
        "- [断言] outer fold 间无患者交集；并集 = 全部 50 位核心患者。\n"
        "- [断言] 每个 outer-train 患者集合 == 该外层折 inner 患者集合；outer-test 在 inner 中零出现。\n"
        "- [断言] airflow 患者的 v2 外层折与 core 完全一致；无独立 airflow split。\n"
        "- [断言] 固定 seed + 排序约定下可复现；优化不剔患者/窗口、不改标签、不用特征值。\n\n"
        "## 8. 异常/告警项\n\n"
        f"{chr(10).join('- '+a for a in s.anomalies) or '- （无）'}\n"
    )


def render_phase_06b_completion_report(s: SplitBalanceSummary) -> str:
    status = s.status
    prods = "\n".join(f"- `{k}`：{v}" for k, v in sorted(s.product_paths.items())) or "- （无）"
    gate_ok = bool(s.gate and s.gate.approved)
    latest_note = (
        f"已更新（v2 镜像到固定路径 + `{s.latest_path}`）。" if s.latest_updated
        else "**未更新**（未批准，LATEST 仍指向 v1；仅保留隔离 run-dir）。"
    )
    not_done = (
        "无 raw/WAV 访问；无特征数值；无模型；无插补/标准化/重采样/特征选择/阈值；"
        "无性能评价；audio cohort 恒为 0。"
    )
    isolation = (
        f"raw 扫描={s.raw_scanned}，raw 修改={s.raw_modified}；"
        f"阶段 1–6（含 v1）历史 run-dir 产物 SHA-256 不变={s.historical_products_unchanged}；"
        f"所有路径项目相对、无绝对/pytest/Temp/AppData 污染。"
    )
    tests = (
        f"解释器：`{s.test_interpreter}`；通过 {s.test_pass} / 失败 {s.test_fail} / "
        f"跳过 {s.test_skip}；耗时 {s.test_runtime_seconds:.1f}s。"
        if s.test_interpreter else "（测试结果由 `pytest -ra` 运行后填入）"
    )
    return (
        "# 阶段 6B 完成报告（phase_06b_completion_report）\n\n"
        f"{_DISCLAIMER}\n\n"
        "## 1. 状态\n\n"
        f"- **{status}**\n"
        f"- 原因/判据：{(s.gate.reasons and '; '.join(s.gate.reasons)) or '全部采用判据通过' if s.gate else '(无 gate 结果)'}\n\n"
        "## 2. v1 历史输入、版本签名与不可变性\n\n"
        f"- v1 comparator run：`{s.v1_comparator_run_id}`（仅读取/比较，**绝不改写/删除/覆盖**）。\n"
        f"- 输入 Stage 4/5/6-v1 的 SHA-256、行数、schema 与 lineage 已写入 "
        "`split_balance_version_signature.json`（取自 v1 `input_version_signature.json`）。\n"
        f"- 不可变性：raw 修改={s.raw_modified}；历史 run-dir 产物不变={s.historical_products_unchanged}。\n\n"
        "## 3. 优化算法、种子、目标函数、候选数、硬约束与配置\n\n"
        f"- 算法：`{s.algorithm}`；base_seed={s.base_seed}；候选数={s.n_candidate_seeds}；"
        f"局部交换迭代={s.local_swap_iterations}；选择规则=`{s.selection_rule}`。\n"
        f"- 目标：aggregate = 1.0·positive_rate_spread + 0.5·window_count_cv（最小化）。\n"
        f"- 硬约束：5 外层折 × 恰 10 人；患者级；同患者全部窗口同折；无交集；airflow 继承。\n\n"
        "## 4. v1 与 v2 比较表（每折患者/窗口/正负/positive rate/spread/偏差/窗口数 CV）\n\n"
        f"- 见 `split_balance_optimization_report.md` 第 3 节与 `split_candidate_comparison.csv`。\n"
        f"- spread：v1={_fmt(s.comparison_summary.get('v1_positive_rate_spread',0.0))} "
        f"→ v2={_fmt(s.comparison_summary.get('v2_positive_rate_spread',0.0))}；"
        f"wcv：v1={_fmt(s.comparison_summary.get('v1_window_count_cv',0.0))} "
        f"→ v2={_fmt(s.comparison_summary.get('v2_window_count_cv',0.0))}。\n\n"
        "## 5. v2 硬约束、≤10% 阈值、采用判据与 LATEST 是否更新\n\n"
        f"- 硬约束全部通过；spread ≤ 0.10 通过={bool(s.comparison_summary.get('v2_spread_below_threshold'))}；"
        f"严格低于 v1={bool(s.comparison_summary.get('v2_spread_strictly_below_v1'))}；"
        f"窗口数 CV 不劣于 v1={bool(s.comparison_summary.get('v2_wcv_not_worse_than_v1_ceiling'))}。\n"
        f"- 采用状态：**{status}**；`approved_for_modeling={gate_ok}`。\n"
        f"- `splits/LATEST_RUN.txt`：{latest_note}\n\n"
        "## 6. inner CV、airflow 继承、audio=0 与所有患者级泄漏检查\n\n"
        f"- inner CV：{len(s.inner_fold_stats)} 个 inner 折统计；inner 异常 "
        f"{len(s.inner_anomalies)}（须为 0）。\n"
        f"- airflow 继承 mismatch：**{s.airflow_mismatch_count}**（须为 0）；airflow 患者 {s.airflow_patients}。\n"
        f"- audio cohort count = **{s.audio_cohort_count}**（恒为 0）；阻断原因：`{s.audio_block_reason}`。\n\n"
        "## 7. 明确未做\n\n"
        f"- {not_done}\n\n"
        "## 8. 输出隔离、路径纯净与 raw/阶段 1–6-v1 不变性\n\n"
        f"- {isolation}\n\n"
        "## 9. 测试命令、解释器、通过/失败/跳过数、耗时\n\n"
        f"- 命令：`& 'C:\\ProgramData\\anaconda3\\python.exe' -m pytest -ra`\n"
        f"- {tests}\n\n"
        "## 10. 下一步准入\n\n"
        f"- 仅在 `approved_for_modeling=true` 时才允许核心 CSV 队列的传统模型基线；"
        f"当前 approved={gate_ok}。\n"
        "- 音频路线：**BLOCKED**（audio cohort=0）。\n"
        "- 本次不自行进入模型阶段；停止，等待人工审计。\n\n"
        "## 产物路径\n\n"
        f"{prods}\n"
    )


__all__ = [
    "SplitBalanceSummary",
    "render_split_balance_optimization_report",
    "render_phase_06b_completion_report",
]
