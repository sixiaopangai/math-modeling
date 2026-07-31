#!/usr/bin/env python3
"""Generate the frozen publication figures from full-pipeline outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from utils.plot_style import (
    PALETTE,
    add_panel_labels,
    apply_publication_style,
    export_figure,
    publication_subplots,
)
from water_quality_model import PROJECT_ROOT


FIGURES = PROJECT_ROOT / "figures"
RESULTS = PROJECT_ROOT / "results"
GRADE_ORDER = ("SAFE", "LOW", "MEDIUM", "HIGH")
GRADE_COLORS = {
    "SAFE": PALETTE["positive"],
    "LOW": PALETTE["sky"],
    "MEDIUM": PALETTE["secondary"],
    "HIGH": PALETTE["contrast"],
}


def save(fig, stem: str) -> None:
    export_figure(fig, FIGURES / stem)
    plt.close(fig)


def plot_raw_q1(timestamp, ntu) -> None:
    mask = timestamp >= pd.Timestamp("2025-12-01")
    fig, ax = publication_subplots(width="report", aspect=0.48)
    ax.plot(timestamp[mask], ntu[mask], color=PALETTE["primary"], label="实测 NTU")
    ax.axhline(1.0, color=PALETTE["contrast"], linestyle="--", linewidth=0.9, label="1 NTU 限值")
    ax.axvspan(pd.Timestamp("2026-02-01"), pd.Timestamp("2026-03-01"), color=PALETTE["neutral"], alpha=0.12)
    ax.set(title="出厂水浊度与缺失区间", xlabel="自然时间", ylabel="出厂水浊度 (NTU)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper left")
    save(fig, "raw_q1_target_missing")


def plot_raw_q2(timestamp, rw_ntu, filt_ntu) -> None:
    mask = (timestamp >= pd.Timestamp("2026-01-01")) & (timestamp < pd.Timestamp("2026-04-01"))
    fig, ax = publication_subplots(width="report", aspect=0.48)
    ax.plot(timestamp[mask], rw_ntu[mask], color=PALETTE["neutral"], linestyle="--", label="原水")
    ax.plot(timestamp[mask], filt_ntu[mask], color=PALETTE["primary"], linestyle="-", label="滤后水")
    ax.set_yscale("log")
    ax.set(title="原水与滤后水浊度", xlabel="自然时间", ylabel="浊度 (NTU, 对数刻度)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper right")
    save(fig, "raw_q2_input_output_series")


def plot_raw_q3(timestamp, filt_ntu, ntu) -> None:
    mask = (timestamp >= pd.Timestamp("2026-01-01")) & (timestamp < pd.Timestamp("2026-02-01"))
    fig, ax = publication_subplots(width="report", aspect=0.48)
    ax.plot(timestamp[mask], filt_ntu[mask], color=PALETTE["primary"], linestyle="-", label="滤后水")
    ax.plot(timestamp[mask], ntu[mask], color=PALETTE["contrast"], linestyle="--", label="出厂水")
    ax.axhline(1.0, color=PALETTE["neutral"], linestyle="--", linewidth=0.8)
    ax.set(title="清水池前后浊度传播", xlabel="自然时间", ylabel="浊度 (NTU)")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.legend(loc="upper right")
    save(fig, "raw_q3_filtered_treated_series")


def plot_raw_q4(daily: pd.DataFrame) -> None:
    dates = pd.to_datetime(daily["operational_day"])
    fig, ax = publication_subplots(width="report", aspect=0.48)
    for source, label, color, marker in (
        ("observed", "实测", PALETTE["primary"], "o"),
        ("predicted", "预测", PALETTE["secondary"], "s"),
    ):
        mask = daily["source"] == source
        ax.scatter(dates[mask], daily.loc[mask, "max_ntu"], s=12, color=color, marker=marker, label=label)
    ax.axhline(1.0, color=PALETTE["contrast"], linestyle="--", linewidth=0.9, label="1 NTU 限值")
    ax.set(title="营运日最大浊度", xlabel="营运日", ylabel="日最大出厂水浊度 (NTU)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper right")
    save(fig, "raw_q4_daily_maximum")


def plot_process_q1(factors: pd.DataFrame) -> None:
    table = factors.head(8).sort_values("permutation_delta_rmse_ntu")
    fig, ax = publication_subplots(width="single", aspect=0.95)
    positions = np.arange(len(table))
    selected = table["selected_main_factor"].to_numpy(dtype=bool)
    ax.scatter(
        table.loc[~selected, "permutation_delta_rmse_ntu"],
        positions[~selected],
        s=24,
        color=PALETTE["neutral"],
        marker="x",
        linewidths=0.9,
        label="未通过稳健门槛",
    )
    ax.scatter(
        table.loc[selected, "permutation_delta_rmse_ntu"],
        positions[selected],
        s=27,
        color=GRADE_COLORS["SAFE"],
        marker="o",
        label="主要因素",
    )
    ax.axvline(0.0, color=PALETTE["dark"], linewidth=0.7)
    ax.set_yticks(positions, table["factor"])
    ax.set(title="因素置换贡献", xlabel="验证 RMSE 增量 (NTU)", ylabel="候选因素")
    ax.legend(loc="lower right")
    save(fig, "process_q1_factor_screening")


def plot_process_q2(search: pd.DataFrame) -> None:
    grouped = search.groupby(["rw_ntu_lag_steps", "alum_lag_steps"])["mean_validation_rmse_ntu"].min()
    matrix = grouped.unstack().reindex(index=range(4), columns=range(4)).to_numpy()
    fig, ax = publication_subplots(width="single", aspect=0.9)
    normalizer = Normalize(vmin=float(np.nanmin(matrix)), vmax=float(np.nanmax(matrix)))
    colormap = plt.get_cmap("viridis")
    for row in range(4):
        for column in range(4):
            value = float(matrix[row, column])
            scaled = float(normalizer(value))
            ax.add_patch(
                Rectangle(
                    (column - 0.5, row - 0.5), 1.0, 1.0,
                    facecolor=colormap(scaled), edgecolor="white", linewidth=0.6,
                )
            )
            ax.text(
                column, row, f"{value:.3f}",
                ha="center", va="center",
                color="white" if scaled < 0.55 else PALETTE["dark"],
                fontsize=6.4,
            )
    ax.set_xticks(range(4), ["0", "2", "4", "6"])
    ax.set_yticks(range(4), ["0", "2", "4", "6"])
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(3.5, -0.5)
    ax.set_aspect("equal")
    ax.set(title="时滞组合验证误差", xlabel="ALUM 时滞 (h)", ylabel="R/W NTU 时滞 (h)")
    save(fig, "process_q2_lag_search")


def plot_process_q3(tau: pd.DataFrame, kernel: np.ndarray, tau_hours: int) -> None:
    fig, axes = publication_subplots(1, 2, width="report", aspect=0.48, width_ratios=[1.25, 1.0])
    axes[0].plot(np.arange(len(kernel)) * 2, kernel, color=PALETTE["primary"], marker="o")
    axes[0].set(title="离散 RTD 核", xlabel="滞后 (h)", ylabel="质量权重")
    axes[1].plot(tau["tau_hours"], tau["mean_validation_rmse_ntu"], color=PALETTE["contrast"], marker="o")
    selected = tau[tau["tau_hours"] == tau_hours].iloc[0]
    axes[1].scatter([tau_hours], [selected["mean_validation_rmse_ntu"]], color=PALETTE["dark"], s=28, zorder=3)
    axes[1].set(title="时间尺度选择", xlabel="有效混合时间尺度 (h)", ylabel="验证 RMSE (NTU)")
    add_panel_labels(axes)
    save(fig, "process_q3_rtd_selection")


def plot_process_q4(daily: pd.DataFrame) -> None:
    fig, ax = publication_subplots(width="single", aspect=0.92)
    for grade in GRADE_ORDER:
        mask = daily["central_grade"] == grade
        ax.scatter(
            daily.loc[mask, "longest_exceedance_hours"],
            daily.loc[mask, "max_excess_ntu"],
            s=16,
            color=GRADE_COLORS[grade],
            label=grade,
            alpha=0.8,
        )
    ax.axvline(4.0, color=PALETTE["neutral"], linestyle=":", linewidth=0.8)
    ax.axvline(8.0, color=PALETTE["dark"], linestyle="--", linewidth=0.8)
    ax.axhline(0.5, color=PALETTE["neutral"], linestyle=":", linewidth=0.8)
    ax.axhline(1.0, color=PALETTE["dark"], linestyle="--", linewidth=0.8)
    ax.set(title="风险分级矩阵", xlabel="最长连续超标时长 (h)", ylabel="最大超标幅度 (NTU)")
    ax.legend(loc="lower right", ncol=2)
    save(fig, "process_q4_risk_matrix")


def plot_result_q1(prediction: pd.DataFrame) -> None:
    fig, ax = publication_subplots(width="report", aspect=0.52)
    slots = np.arange(12)
    for index, (day, table) in enumerate(prediction.groupby("营运日")):
        table = table.reset_index(drop=True)
        color = (PALETTE["primary"], PALETTE["secondary"], PALETTE["contrast"])[index]
        linestyle = ("-", "--", ":")[index]
        ax.plot(slots, table["条件预测_NTU"], color=color, linestyle=linestyle, label=day)
        ax.fill_between(slots, table["95%下限_NTU"], table["95%上限_NTU"], color=color, alpha=0.10)
    ax.axhline(1.0, color=PALETTE["neutral"], linestyle="--", linewidth=0.8)
    ax.set_xticks(slots, ["07", "09", "11", "13", "15", "17", "19", "21", "23", "01", "03", "05"])
    ax.set(title="指定营运日条件预测", xlabel="营运时次", ylabel="出厂水浊度 (NTU)")
    ax.legend(loc="upper right")
    save(fig, "result_q1_february_predictions")


def plot_result_q2(external: pd.DataFrame) -> None:
    external["timestamp"] = pd.to_datetime(external["timestamp"])
    fig, ax = publication_subplots(width="report", aspect=0.48)
    ax.plot(
        external["timestamp"], external["actual_filt_ntu"],
        color=PALETTE["dark"], linestyle="-", label="实测"
    )
    ax.plot(
        external["timestamp"], external["predicted_filt_ntu"],
        color=PALETTE["primary"], linestyle="--", label="ARX"
    )
    ax.set(title="滤后水外部验证", xlabel="自然时间", ylabel="滤后水浊度 (NTU)")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.legend(loc="upper right")
    save(fig, "result_q2_external_validation")


def plot_result_q3(accuracy: pd.DataFrame) -> None:
    fig, ax = publication_subplots(width="single", aspect=0.9)
    styles = {
        "hybrid": (PALETTE["primary"], "o", "-"),
        "persistence": (PALETTE["neutral"], "s", "--"),
        "physical": (PALETTE["contrast"], "^", ":"),
    }
    for model, table in accuracy.groupby("model"):
        color, marker, linestyle = styles[model]
        ax.plot(table["horizon_hours"], table["rmse"], color=color, marker=marker, linestyle=linestyle, label=model)
    ax.set(title="分时距外部回测", xlabel="预测时距 (h)", ylabel="RMSE (NTU)")
    ax.set_xticks([2, 4, 6, 8, 10, 12])
    ax.legend(loc="center right")
    save(fig, "result_q3_horizon_accuracy")


def plot_result_q4(daily: pd.DataFrame) -> None:
    dates = pd.to_datetime(daily["operational_day"])
    modal = daily["modal_grade"].map({grade: index for index, grade in enumerate(GRADE_ORDER)}).to_numpy()
    low = daily["grade_95_low"].map({grade: index for index, grade in enumerate(GRADE_ORDER)}).to_numpy()
    high = daily["grade_95_high"].map({grade: index for index, grade in enumerate(GRADE_ORDER)}).to_numpy()
    fig, ax = publication_subplots(width="report", aspect=0.46)
    ax.vlines(dates, low, high, color=PALETTE["neutral"], alpha=0.35, linewidth=0.8)
    for grade_index, grade in enumerate(GRADE_ORDER):
        mask = modal == grade_index
        ax.scatter(dates[mask], modal[mask], s=13, color=GRADE_COLORS[grade], label=grade)
    ax.set_yticks(range(4), GRADE_ORDER)
    ax.set_ylim(-0.35, 3.35)
    ax.set(title="九十个营运日风险等级", xlabel="营运日", ylabel="风险等级")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper right", ncol=4)
    save(fig, "result_q4_daily_grades")


def figure_contracts(summary: dict, factors: pd.DataFrame) -> list[dict]:
    selected = factors.loc[factors["selected_main_factor"], "factor"].astype(str).tolist()
    selected_text = ", ".join(selected) if selected else "no candidate"
    common = {
        "final_width": "report 6.3 in or single 3.5 in",
        "formats": "editable SVG + 300 DPI PNG + grayscale QA",
        "risk_check": "no silent row deletion; no causal interpretation; inspect color and grayscale",
    }
    rows = [
        ("raw_q1_target_missing", "Q1", "raw", "February is a complete target gap between observed January and March", "single-panel time series", "5460 timestamps; plotted from 2025-12 onward"),
        ("raw_q2_input_output_series", "Q2", "raw", "filtration reduces turbidity by orders of magnitude", "single-panel log time series", "2026-01 through 2026-03 upstream and filtered observations"),
        ("raw_q3_filtered_treated_series", "Q3", "raw", "filtered and treated turbidity have distinct short-run dynamics", "single-panel paired time series", "2026-01 observed values only"),
        ("raw_q4_daily_maximum", "Q4", "raw", "daily maxima distinguish observed and predicted periods against the 1 NTU limit", "single-panel point time series", "90 operational days; daily maxima"),
        ("process_q1_factor_screening", "Q1", "process", f"{selected_text} pass the positive-importance and 75% fold-direction screen", "single-panel ranked dot plot", "2026-01 posterior validation; 24 h block permutation; four rolling direction folds"),
        ("process_q2_lag_search", "Q2", "process", "validation error changes across independently enumerated input delays", "single-panel annotated vector matrix", "256 delay combinations x 5 ridge values; minima aggregated over two delays"),
        ("process_q3_rtd_selection", "Q3", "process", f"the mass-conserving RTD selects tau={summary['q3']['tau_hours']} h", "main kernel panel plus validation panel", "95%-mass exponential kernel and four rolling validation folds"),
        ("process_q4_risk_matrix", "Q4", "process", "amplitude and duration jointly determine operational risk", "single-panel decision scatter", "90 central daily estimates; designed thresholds shown"),
        ("result_q1_february_predictions", "Q1", "result", "February conditional estimates carry wide cross-period uncertainty", "single-panel three-day forecast with intervals", "36 requested points; rolling residual 95% intervals"),
        ("result_q2_external_validation", "Q2", "result", "ARX follows normal levels but misses isolated filtered-water spikes", "single-panel external validation time series", "2026-03 external observations"),
        ("result_q3_horizon_accuracy", "Q3", "result", "the hybrid model only surpasses persistence at the longest horizon", "single-panel multi-model error curve", "2026-03 rolling origins; even horizons only"),
        ("result_q4_daily_grades", "Q4", "result", "most days are safe while February grade ranges remain uncertain", "single-panel ordinal strip with intervals", "90 days; 500-path February uncertainty"),
    ]
    return [
        {
            "stem": stem,
            "question": question,
            "category": category,
            "core_conclusion": conclusion,
            "narrative_prototype": prototype,
            "evidence_mapping": evidence,
            "layout_rationale": "one dominant panel unless two independent Q3 process diagnostics require unequal panels",
            "chart_reason": "encoding matches time, matrix, ranking, decision, or forecast semantics",
            "statistics": evidence,
            "label_strategy": "short panel title, axes with units, at most one non-overlapping legend",
            "source_trace": "results files generated by water_quality_full.py --full",
            **common,
        }
        for stem, question, category, conclusion, prototype, evidence in rows
    ]


def run() -> None:
    apply_publication_style(language="zh", width="report")
    data = np.load(RESULTS / "plot_data.npz", allow_pickle=True)
    timestamp = pd.to_datetime(data["timestamp"], unit="us")
    ntu = data["ntu"].astype(float)
    filt_ntu = data["filt_ntu"].astype(float)
    rw_ntu = data["rw_ntu"].astype(float)
    daily = pd.read_csv(RESULTS / "q4_daily.csv")
    factors = pd.read_csv(RESULTS / "q1_factors.csv")
    q2_search = pd.read_csv(RESULTS / "plot_q2_search.csv")
    q2_external = pd.read_csv(RESULTS / "plot_q2_external.csv")
    tau = pd.read_csv(RESULTS / "plot_q3_tau.csv")
    accuracy = pd.read_csv(RESULTS / "q3_horizon_accuracy.csv")
    q1_prediction = pd.read_csv(RESULTS / "q1_predictions.csv")
    summary = json.loads((RESULTS / "full_summary.json").read_text(encoding="utf-8"))
    kernel_steps = int(np.ceil(np.log(0.05) / np.log(np.exp(-2 / summary["q3"]["tau_hours"]))))
    kernel = np.exp(-2 / summary["q3"]["tau_hours"]) ** np.arange(kernel_steps)
    kernel = kernel / kernel.sum()

    plot_raw_q1(timestamp, ntu)
    plot_raw_q2(timestamp, rw_ntu, filt_ntu)
    plot_raw_q3(timestamp, filt_ntu, ntu)
    plot_raw_q4(daily)
    plot_process_q1(factors)
    plot_process_q2(q2_search)
    plot_process_q3(tau, kernel, summary["q3"]["tau_hours"])
    plot_process_q4(daily)
    plot_result_q1(q1_prediction)
    plot_result_q2(q2_external)
    plot_result_q3(accuracy)
    plot_result_q4(daily)
    contracts = figure_contracts(summary, factors)
    (RESULTS / "图表契约.json").write_text(
        json.dumps(contracts, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not args.all:
        parser.error("Use --all")
    run()
    print(json.dumps({"logical_figures": 12, "directory": str(FIGURES)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
