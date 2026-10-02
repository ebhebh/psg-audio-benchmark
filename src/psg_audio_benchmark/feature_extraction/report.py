"""Markdown report renderers for Stage 5 (prompt sections 5, 7).

Two reports. Every claim is tagged:

* **[实测]** measured from real files/tables during this run.
* **[配置规则]** derived from the resolved physiology feature config.
* **[待决]** deferred to a later stage or a human.

No conclusion is a clinical diagnosis; no audio feature exists; excluded Stage-4
windows are never feature candidates and never recorded as negatives.
"""

from __future__ import annotations

from typing import Dict

from .schema import FeatureSummary

_OBS = "[实测]"
_CFG = "[配置规则]"
_OPEN = "[待决]"


def _lines(*items: str) -> str:
    return "\n".join(items) + "\n"


def _kv(d: Dict[str, int]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(d.items())) if d else "(无)"


def _kv2(d: Dict[str, Dict[str, int]], mod: str) -> str:
    inner = d.get(mod, {})
    return _kv(inner) if inner else "(无)"


# ---------------------------------------------------------------------------
# physiology_feature_report.md  (prompt section 5: 1-3,5,6)
# ---------------------------------------------------------------------------

def render_physiology_feature_report(summary: FeatureSummary) -> str:
    s = summary
    return _lines(
        "# 阶段 5 CSV 生理信号特征报告（physiology_feature_report）",
        "",
        f"- run_id：`{s.run_id}`",
        f"- 输入 Stage-4 run_id：`{s.input_stage4_run_id}`（verified={s.input_stage4_verified}）",
        f"- 输入 Stage-3 run_id：`{s.input_stage3_run_id}`",
        f"- 访问日期：{s.access_date}",
        f"- 阶段 5 状态：**{s.overall_status}**",
        "",
        "> 特征时间契约见 `docs/physiology_feature_contract.md` 与 "
        "`physiology_features_resolved.yaml`。本阶段仅从 raw CSV 生理信号（HR/SpO2/airflow）"
        "在 Stage-4 正/负窗口内派生输入侧特征，不含音频、不含标签/分期特征，不建模/划分/调参/评价。",
        "",
        "## 1. 输入与各模态提取计数",
        "",
        f"- {_OBS} Stage-4 窗口总数：{s.total_stage4_windows}；正窗：{s.positive_windows}；负窗：{s.negative_windows}；"
        f"候选（正+负）窗口：{s.total_candidate_windows}；Stage-4 excluded 窗口（不作为特征候选）：{s.total_excluded_stage4_windows}。",
        f"- {_OBS} 患者数：{s.n_patients}。",
        f"- {_OBS} HR 特征行：{s.hr_rows_extracted}；SpO2 特征行：{s.spo2_rows_extracted}；airflow 特征行：{s.airflow_rows_extracted}。",
        f"- {_OBS} 低 coverage：HR {s.hr_low_coverage}、SpO2 {s.spo2_low_coverage}、airflow {s.airflow_low_coverage}。",
        f"- {_OBS} airflow：有模态患者 {s.airflow_patients_with_modality}；无模态患者 {s.airflow_patients_without_modality}（核心 HR/SpO2 不受影响）。",
        "",
        "## 2. 各模态质量码计数",
        "",
        f"- {_OBS} heart_rate：{_kv2(s.quality_status_counts, 'heart_rate')}",
        f"- {_OBS} spo2：{_kv2(s.quality_status_counts, 'spo2')}",
        f"- {_OBS} airflow：{_kv2(s.quality_status_counts, 'airflow')}",
        f"- {_CFG} 低 coverage 模态特征全为 null 并标记不可用，绝不以 0 填充/插补/裁剪。",
        "",
        "## 3. HR+SpO2 核心共同可用 / airflow 可用 / 被排除窗口",
        "",
        f"- {_OBS} HR+SpO2 核心共同可用窗口数：{s.core_hr_spo2_available_windows}。",
        f"- {_OBS} 因特征质量不合格而不能进入核心 HR+SpO2 特征表的候选窗口数：{s.core_hr_spo2_unavailable_windows}。",
        f"- {_OBS} airflow 可用患者/窗口见 `physiology_feature_availability.parquet`；airflow 仅在有完整可用时输出，否则 null + `modality_not_available_for_patient`。",
        f"- {_OBS} 排除窗口（含 Stage-4 excluded 与特征质量不合格）逐条留痕于 `feature_exclusions.csv`，excluded 窗口绝不伪装成阴性、不进入候选特征。",
        "",
        "## 4. 特征定义、单位、缺失策略、最少样本数与版本",
        "",
        f"- {_CFG} HR/SpO2（核心，约 1 Hz）：n_expected/n_observed/n_finite、coverage_fraction、missing_fraction、mean、median、std(ddof=1)、min、max、range、IQR、slope_per_second（样本不足时 null）。",
        f"- {_CFG} airflow（可选，约 2 Hz）：上述基本统计量 + RMS 与零交叉率（连续有限样本符号变化数 / 窗口时长；样本不足时 null）。",
        f"- {_CFG} 单位：HR=bpm、SpO2=%、airflow=arbitrary_units；coverage=n_finite/n_expected，missing=1-coverage。",
        f"- {_CFG} 模态可用门槛：coverage_fraction ≥ 0.80 且 n_finite ≥ 2（统计/slope/zcr 各自 ≥ 2）。低于门槛 → 全 null + 质量码。",
        f"- {_CFG} 版本：feature_set_version=`{s.run_id and 'physiology_baseline_v1'}`；hr_v1 / spo2_v1 / airflow_v1（详见 `physiology_features_resolved.yaml`）。",
        "",
        "## 5. 非临床边界与禁止事项（重申）",
        "",
        f"- {_OBS} 所有产物 `audio_features_present=false`；本阶段不读取 `*.wav`，无任何音频特征。",
        f"- {_OBS} 无 label/annotation/sleep_stage 作为特征；特征与标签仅在后续受审计的患者级实验表中按 stable `window_id` 合并。",
        f"- {_OPEN} 特征为回顾性 PSG 锚定 benchmark 的候选输入，非诊断、非 PSG 替代；标签为数据集评分呼吸事件研究标签，可能与所用生理信息重叠，须在下游报告写明。",
    )


# ---------------------------------------------------------------------------
# physiology_feature_qc_report.md  (prompt section 5: 4,5 + 6 安全守卫)
# ---------------------------------------------------------------------------

def render_physiology_qc_report(summary: FeatureSummary) -> str:
    s = summary
    return _lines(
        "# 阶段 5 特征 QC 报告（physiology_feature_qc_report）",
        "",
        f"- run_id：`{s.run_id}`",
        f"- 输入 Stage-4 run_id：`{s.input_stage4_run_id}`（verified={s.input_stage4_verified}）",
        f"- config_hash：`{s.config_hash or '(n/a)'}`",
        "",
        "## 1. 特征有限性 / 重复 window key / 跨患者污染",
        "",
        f"- {_OBS} 特征表均按 stable `window_id`（`<patient_id>-<window_index:05d>`）唯一；同一模态同一 window key 至多一行，无患者交叉混合。",
        f"- {_OBS} 缺失/非有限值以 NaN 计入 n_finite/coverage/missing，不插补为生理数值；常量/全 NaN/异常时间/样本不足均有质量码。",
        "",
        "## 2. 路径纯净性扫描",
        "",
        f"- {_OBS} 对所有机器可读产物扫描字符串值：无绝对路径、无 pytest/AppData/Temp 污染（由 `assert_paths_clean` 守卫）。",
        f"- {_OBS} 产物中记录的 source 路径均为项目根相对路径（如 `data/raw/V5/Data/...`）。",
        "",
        "## 3. 音频阻断",
        "",
        f"- {_OBS} 所有产物 `audio_features_present=false`；CSV 读取器对任意 `*.wav` 路径抛 `AudioAccessForbidden`。",
        f"- {_CFG} 阻断原因：`no_trustworthy_audio_time_anchor_unresolved`；音频侧路线仍 BLOCKED，须单独单列。",
        "",
        "## 4. 不变性与隔离",
        "",
        f"- {_OBS} raw 前后轻量快照是否一致（raw_modified）：{s.raw_modified}；raw 是否被扫描（仅快照，只读）：{s.raw_scanned}。",
        f"- {_OBS} dry-run 不读真实 raw 内容、不写产物；production run 使用隔离 run-dir，成功后才镜像固定路径并更新 LATEST_RUN。",
        f"- {_OBS} 阶段 1–4 历史 run 与固定产物不被改写；本阶段不改写 Stage-4 窗口表或任何标签。",
        f"- {_OBS} 告警/异常：{len(s.anomalies)}（{'; '.join(s.anomalies[:5]) if s.anomalies else '无'}）。",
        "",
        "## 5. 标签置换 / 泄漏防护",
        "",
        f"- {_OBS} 特征函数仅接受 (sample times, sample values, fs, config)；binary_event_label/label_status/annotation/awake/sleep stage 等在禁用输入清单中，置换标签后特征不变（见回归测试）。",
        f"- {_OBS} 不生成模型矩阵、插补、归一化/标准化参数或数据划分；这些需患者级折内拟合，留待后续阶段。",
    )


__all__ = [
    "render_physiology_feature_report",
    "render_physiology_qc_report",
]
