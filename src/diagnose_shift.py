#!/usr/bin/env python3
"""T1.1 训练窗与外部窗（2026-03）的分布漂移诊断。

产出:
  results/分布漂移诊断.csv     逐变量分布对照 + KS + PSI
  results/指标口径诊断.json    R2 分母退化判据与 skill score
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from water_quality_model import load_data, PROJECT_ROOT

TRAIN_END = pd.Timestamp("2026-02-01")   # 训练/开发窗上界(不含)
EXT_START = pd.Timestamp("2026-03-01")   # 外部窗
EXT_END = pd.Timestamp("2026-04-01")

VARS = [
    ("rw_ntu", "原水浊度"),
    ("rw_clr", "原水色度"),
    ("rw_ph", "原水pH"),
    ("rw_flow", "原水流量"),
    ("river_level", "河水水位"),
    ("filt_ntu", "滤后水浊度"),
    ("alum", "混凝剂投加量"),
    ("cw_level", "清水池水位"),
    ("tw_flow", "出厂水流量"),
    ("ntu", "出厂水浊度"),
]


def psi(train: np.ndarray, ext: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index，按训练窗分位数分箱。"""
    train = train[np.isfinite(train)]
    ext = ext[np.isfinite(ext)]
    if train.size < 20 or ext.size < 20:
        return float("nan")
    edges = np.unique(np.quantile(train, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(train, bins=edges)[0] / train.size
    q = np.histogram(ext, bins=edges)[0] / ext.size
    eps = 1e-6
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def describe(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: float("nan") for k in
                ["n", "mean", "std", "p5", "p25", "p50", "p75", "p95", "min", "max"]}
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": int(x.size), "mean": float(x.mean()), "std": float(x.std(ddof=1)),
        "p5": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
        "p75": float(q[3]), "p95": float(q[4]),
        "min": float(x.min()), "max": float(x.max()),
    }


def main() -> None:
    frame, _ = load_data()
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    train_mask = frame["timestamp"] < TRAIN_END
    ext_mask = (frame["timestamp"] >= EXT_START) & (frame["timestamp"] < EXT_END)

    rows = []
    for col, label in VARS:
        if col not in frame.columns:
            continue
        tr = pd.to_numeric(frame.loc[train_mask, col], errors="coerce").to_numpy(float)
        ex = pd.to_numeric(frame.loc[ext_mask, col], errors="coerce").to_numpy(float)
        d_tr, d_ex = describe(tr), describe(ex)
        tr_f = tr[np.isfinite(tr)]
        ex_f = ex[np.isfinite(ex)]
        if tr_f.size > 20 and ex_f.size > 20:
            ks_stat, ks_p = ks_2samp(tr_f, ex_f)
        else:
            ks_stat, ks_p = float("nan"), float("nan")
        row = {"变量": label, "字段": col}
        row.update({f"训练_{k}": v for k, v in d_tr.items()})
        row.update({f"三月_{k}": v for k, v in d_ex.items()})
        row["KS统计量"] = float(ks_stat)
        row["KS_p值"] = float(ks_p)
        row["PSI"] = psi(tr, ex)
        row["标准差比_三月除训练"] = (
            d_ex["std"] / d_tr["std"] if d_tr["std"] and np.isfinite(d_tr["std"]) else float("nan")
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / "results" / "分布漂移诊断.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    # ---- 目标变量的 R2 分母诊断 ----
    y_tr = pd.to_numeric(frame.loc[train_mask, "ntu"], errors="coerce").to_numpy(float)
    y_ex = pd.to_numeric(frame.loc[ext_mask, "ntu"], errors="coerce").to_numpy(float)
    y_tr = y_tr[np.isfinite(y_tr)]
    y_ex = y_ex[np.isfinite(y_ex)]

    # 论文中已报告的外部 RMSE（问题一）
    reported = {
        "Huber岭": 0.392112,
        "RFF岭": 0.371614,
        "同营运时次前一日基线": 0.170873,
    }
    sd_ex = float(y_ex.std(ddof=1))
    baseline_rmse = reported["同营运时次前一日基线"]
    diag = {
        "训练窗": {"n": int(y_tr.size), "std": float(y_tr.std(ddof=1)),
                   "mean": float(y_tr.mean()), "min": float(y_tr.min()), "max": float(y_tr.max())},
        "三月外部窗": {"n": int(y_ex.size), "std": sd_ex,
                       "mean": float(y_ex.mean()), "min": float(y_ex.min()), "max": float(y_ex.max())},
        "目标标准差比_三月除训练": sd_ex / float(y_tr.std(ddof=1)),
        "RMSE相对三月目标标准差之比": {k: v / sd_ex for k, v in reported.items()},
        "技能分_相对前一营运日基线": {
            k: 1.0 - (v ** 2) / (baseline_rmse ** 2) for k, v in reported.items()
        },
        "说明": (
            "R2 = 1 - MSE/(n/(n-1)*sd^2)。当三月目标标准差远小于训练窗时，"
            "同等绝对误差会被放大为更负的 R2，此时负 R2 主要反映分母退化而非模型完全失效；"
            "技能分以前一营运日基线为分母，是更贴合本题的对照口径。"
        ),
    }
    (PROJECT_ROOT / "results" / "指标口径诊断.json").write_text(
        json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    pd.set_option("display.width", 200)
    print(out[["变量", "训练_std", "三月_std", "标准差比_三月除训练", "KS统计量", "KS_p值", "PSI"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(json.dumps(diag, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
