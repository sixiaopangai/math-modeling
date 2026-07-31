#!/usr/bin/env python3
"""生成 T1 诊断图: 分布漂移 + 问题一外部误差的偏差/离散分解。

产出:
  figures/diag_distribution_shift.png/.svg  论文插图(300 dpi, 与 figure_audit 门槛一致)
  results/误差分解诊断.json                  偏差-离散分解的数值，供论文正文引用
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from water_quality_model import load_data, PROJECT_ROOT
import water_quality_full as W

try:
    from utils.plot_style import apply_style  # type: ignore
    apply_style()
except Exception:
    pass

for cand in ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
             "Source Han Sans SC", "SimHei", "DejaVu Sans"]:
    try:
        matplotlib.font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [cand]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
# figure_audit.py 要求 SVG 保留可编辑文本节点；默认的 path 会把文字转成轮廓。
plt.rcParams["svg.fonttype"] = "none"

TRAIN_END = pd.Timestamp("2026-02-01")
EXT_START = pd.Timestamp("2026-03-01")
EXT_END = pd.Timestamp("2026-04-01")


def main() -> None:
    frame, _ = load_data()
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    tr = frame[frame.timestamp < TRAIN_END]
    ex = frame[(frame.timestamp >= EXT_START) & (frame.timestamp < EXT_END)]

    q1 = W.run_q1(frame)
    mask = q1["march_mask"]
    pred = np.asarray(q1["march_prediction"], float)
    obs = pd.to_numeric(frame.loc[mask, "ntu"], errors="coerce").to_numpy(float)
    ok = np.isfinite(obs) & np.isfinite(pred)
    obs_v, pred_v = obs[ok], pred[ok]
    err = pred_v - obs_v
    bias = float(err.mean())
    spread = float(err.std(ddof=0))
    rmse = float(np.sqrt(np.mean(err**2)))
    debiased_rmse = float(np.sqrt(np.mean((err - bias) ** 2)))
    decomposition = {
        "样本数": int(err.size),
        "实测均值": float(obs_v.mean()),
        "预测均值": float(pred_v.mean()),
        "系统偏差_预测减实测": bias,
        "误差离散_标准差": spread,
        "RMSE": rmse,
        "偏差平方占MSE比例": float(bias**2 / (bias**2 + spread**2)),
        "扣除常数偏移后RMSE": debiased_rmse,
        "同营运时次前一日基线RMSE": 0.170873,
        "说明": (
            "偏差-离散分解满足 RMSE=sqrt(bias^2+sigma^2)。扣除常数偏移仅为诊断口径，"
            "该偏移由三月实测算得，不得回填进模型；二月出厂浊度整段缺失，此类校正在二月也无法实施。"
        ),
    }
    (PROJECT_ROOT / "results" / "误差分解诊断.json").write_text(
        json.dumps(decomposition, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))

    # (a) 原水浊度分布对照
    ax = axes[0, 0]
    a = pd.to_numeric(tr.rw_ntu, errors="coerce").dropna()
    b = pd.to_numeric(ex.rw_ntu, errors="coerce").dropna()
    bins = np.linspace(0, np.percentile(a, 99), 40)
    ax.hist(a, bins=bins, density=True, alpha=0.55, label="训练窗 2025-01~2026-01")
    ax.hist(b, bins=bins, density=True, alpha=0.55, label="外部窗 2026-03")
    ax.set_title("a  原水浊度分布漂移 (PSI=3.59, KS p<1e-4)")
    ax.set_xlabel("原水浊度 (NTU)")
    ax.set_ylabel("密度")
    ax.legend(fontsize=8)

    # (b) 各变量 PSI
    ax = axes[0, 1]
    labels = ["原水\n浊度", "原水\n色度", "原水\n流量", "河水\n水位",
              "出厂水\n流量", "出厂水\n浊度", "清水池\n水位", "混凝剂", "滤后水\n浊度"]
    psi_vals = [3.593, 3.636, 6.461, 2.847, 4.858, 2.399, 0.758, 0.555, 0.315]
    colors = ["#c0392b" if v >= 0.25 else "#27ae60" for v in psi_vals]
    ax.barh(labels[::-1], psi_vals[::-1], color=colors[::-1])
    ax.axvline(0.25, ls="--", c="k", lw=1)
    ax.annotate(
        "PSI=0.25 显著漂移阈值",
        xy=(0.25, 8.4),
        xytext=(1.6, 8.4),
        fontsize=8,
        va="center",
        arrowprops={"arrowstyle": "->", "lw": 0.8},
    )
    ax.set_xlabel("PSI")
    # 原水 pH 在训练窗退化为近常数，PSI 无定义，故不列入本图的九项。
    ax.set_title("b  八项输入与目标共九项均越过显著漂移阈值")

    # (c) 目标水平偏移
    ax = axes[1, 0]
    sub = frame[frame.timestamp >= pd.Timestamp("2025-11-01")].copy()
    sub["d"] = sub.timestamp.dt.date
    daily = sub.groupby("d")["ntu"].mean().dropna()
    ax.plot(pd.to_datetime(list(daily.index)), daily.values, lw=1.1, label="日均出厂水浊度")
    ax.axhline(float(pd.to_numeric(tr.ntu, errors="coerce").mean()), ls="--", c="#2980b9",
               label=f"训练窗均值 {pd.to_numeric(tr.ntu, errors='coerce').mean():.3f}")
    ax.axhline(float(pd.to_numeric(ex.ntu, errors="coerce").mean()), ls="--", c="#c0392b",
               label=f"三月均值 {pd.to_numeric(ex.ntu, errors='coerce').mean():.3f}")
    ax.set_title("c  三月出厂水浊度基准水平整体抬升")
    ax.set_ylabel("出厂水浊度 (NTU)")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.legend(fontsize=8)

    # (d) 误差分解
    ax = axes[1, 1]
    ax.scatter(obs_v, pred_v, s=12, alpha=0.5, label="三月逐点")
    lim = [0, max(obs_v.max(), pred_v.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1, label="理想 y=x")
    ax.plot(lim, [v + bias for v in lim], c="#c0392b", lw=1.2,
            label=f"系统偏差 {bias:.3f} NTU")
    ax.set_xlim(lim)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("实测 (NTU)")
    ax.set_ylabel("Huber 岭预测 (NTU)")
    ax.set_title("d  76.6% 的 MSE 来自系统性低估")
    ax.legend(fontsize=8)

    # 不加 suptitle：论文由 LaTeX 的 \caption 提供图题，避免图内外标题重复。
    fig.tight_layout()

    out = PROJECT_ROOT / "figures"
    # 300 dpi 是 figure_audit.py 对论文图的硬门槛，PNG 与 SVG 必须成对存在。
    fig.savefig(out / "diag_distribution_shift.png", dpi=300)
    fig.savefig(out / "diag_distribution_shift.svg")
    print("saved ->", out / "diag_distribution_shift.png")
    print(json.dumps(decomposition, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
