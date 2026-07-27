#!/usr/bin/env python3
"""Full computation pipeline for the 2026 APMCM water-quality problem."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import chi2, spearmanr

from water_quality_model import (
    MODEL_LAMBDAS,
    NUMERIC_NAMES,
    PROJECT_ROOT,
    SEED,
    STEP_HOURS,
    RidgeModel,
    classify_day,
    fit_ridge,
    load_data,
    metric_dict,
    q1_feature_frame,
    q2_design,
    rtd_kernel,
)


GRADE_ORDER = ("SAFE", "LOW", "MEDIUM", "HIGH")
ROLLING_FOLDS = (
    (pd.Timestamp("2025-09-01"), pd.Timestamp("2025-10-01")),
    (pd.Timestamp("2025-10-01"), pd.Timestamp("2025-11-01")),
    (pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-01")),
    (pd.Timestamp("2025-12-01"), pd.Timestamp("2026-01-01")),
)
Q1_VARIABLES = (
    "river_level",
    "rw_flow",
    "rw_ntu",
    "rw_clr",
    "rw_ph",
    "filt_ntu",
    "cw_level",
    "treated_ph",
    "cl2",
    "alum",
    "tw_flow",
)


@dataclass
class Preprocessor:
    median: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=float).copy()
        data = np.where(np.isnan(data), self.median, data)
        data = np.clip(data, self.lower, self.upper)
        return (data - self.mean) / self.scale


@dataclass
class RFFModel:
    preprocessor: Preprocessor
    weight: np.ndarray
    phase: np.ndarray
    ridge_model: RidgeModel
    gamma: float
    ridge: float

    def features(self, values: np.ndarray) -> np.ndarray:
        standardized = self.preprocessor.transform(values)
        return math.sqrt(2.0 / self.weight.shape[1]) * np.cos(standardized @ self.weight + self.phase)

    def predict(self, values: np.ndarray) -> np.ndarray:
        return self.ridge_model.predict(self.features(values))


def fit_preprocessor(values: np.ndarray) -> Preprocessor:
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values, axis=0)
    median = np.where(np.isnan(median), 0.0, median)
    lower = np.nanpercentile(values, 0.5, axis=0)
    upper = np.nanpercentile(values, 99.5, axis=0)
    lower = np.where(np.isnan(lower), median, lower)
    upper = np.where(np.isnan(upper), median, upper)
    clean = np.where(np.isnan(values), median, values)
    clean = np.clip(clean, lower, upper)
    mean = np.mean(clean, axis=0)
    scale = np.std(clean, axis=0)
    scale = np.where(scale < 1e-10, 1.0, scale)
    return Preprocessor(median, lower, upper, mean, scale)


def gamma_base(standardized: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    count = min(1000, len(standardized))
    sample = standardized[rng.choice(len(standardized), count, replace=False)]
    distances = pdist(sample, metric="sqeuclidean")
    positive = distances[distances > 1e-12]
    median_distance = float(np.median(positive)) if len(positive) else 1.0
    return 1.0 / (2.0 * median_distance)


def fit_rff_model(
    values: np.ndarray,
    target: np.ndarray,
    gamma_factor: float,
    ridge: float,
    seed: int,
) -> RFFModel:
    preprocessor = fit_preprocessor(values)
    standardized = preprocessor.transform(values)
    gamma = gamma_base(standardized, seed) * gamma_factor
    rng = np.random.default_rng(seed)
    weight = rng.normal(0.0, math.sqrt(2.0 * gamma), size=(standardized.shape[1], 256))
    phase = rng.uniform(0.0, 2.0 * math.pi, size=256)
    features = math.sqrt(2.0 / 256.0) * np.cos(standardized @ weight + phase)
    ridge_model = fit_ridge(features, target, ridge, robust=False)
    return RFFModel(preprocessor, weight, phase, ridge_model, gamma, ridge)


def rolling_masks(frame: pd.DataFrame, available: pd.Series | np.ndarray):
    available_array = np.asarray(available, dtype=bool)
    for start, end in ROLLING_FOLDS:
        train = available_array & (frame["timestamp"].to_numpy() < start.to_datetime64())
        validation = available_array & (frame["timestamp"].to_numpy() >= start.to_datetime64()) & (
            frame["timestamp"].to_numpy() < end.to_datetime64()
        )
        yield train, validation


def block_spearman_ci(x: np.ndarray, y: np.ndarray, seed: int, repeats: int = 200) -> tuple[float, float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    estimate = float(spearmanr(x, y).statistic)
    block_count = len(x) // 12
    if block_count < 2:
        return estimate, float("nan"), float("nan")
    x, y = x[: block_count * 12], y[: block_count * 12]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        selected = rng.integers(0, block_count, size=block_count)
        indices = np.concatenate([np.arange(item * 12, item * 12 + 12) for item in selected])
        values.append(float(spearmanr(x[indices], y[indices]).statistic))
    return estimate, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def run_q1(frame: pd.DataFrame) -> dict[str, Any]:
    features, feature_names = q1_feature_frame(frame)
    target_log = np.log1p(frame["ntu"])
    available = target_log.notna().to_numpy()

    ridge_scores = []
    for ridge in MODEL_LAMBDAS:
        fold_scores = []
        for train, validation in rolling_masks(frame, available):
            model = fit_ridge(features.loc[train].to_numpy(), target_log.loc[train].to_numpy(), ridge, robust=True)
            prediction = np.maximum(0.0, np.expm1(model.predict(features.loc[validation].to_numpy())))
            fold_scores.append(metric_dict(frame.loc[validation, "ntu"].to_numpy(), prediction)["rmse"])
        ridge_scores.append({"ridge": ridge, "mean_rmse": float(np.mean(fold_scores))})
    best_ridge = min(ridge_scores, key=lambda item: (item["mean_rmse"], item["ridge"]))["ridge"]

    rff_scores = []
    for gamma_factor in (0.5, 1.0, 2.0):
        for ridge in MODEL_LAMBDAS:
            fold_scores = []
            for fold_index, (train, validation) in enumerate(rolling_masks(frame, available)):
                model = fit_rff_model(
                    features.loc[train].to_numpy(),
                    target_log.loc[train].to_numpy(),
                    gamma_factor,
                    ridge,
                    SEED + fold_index,
                )
                prediction = np.maximum(0.0, np.expm1(model.predict(features.loc[validation].to_numpy())))
                fold_scores.append(metric_dict(frame.loc[validation, "ntu"].to_numpy(), prediction)["rmse"])
            rff_scores.append(
                {"gamma_factor": gamma_factor, "ridge": ridge, "mean_rmse": float(np.mean(fold_scores))}
            )
    best_rff = min(rff_scores, key=lambda item: (item["mean_rmse"], item["ridge"], item["gamma_factor"]))

    final_train = available & (frame["timestamp"] < pd.Timestamp("2026-02-01")).to_numpy()
    march = available & (frame["timestamp"] >= pd.Timestamp("2026-03-01")).to_numpy()
    main_model = fit_ridge(
        features.loc[final_train].to_numpy(), target_log.loc[final_train].to_numpy(), best_ridge, robust=True
    )
    rff_model = fit_rff_model(
        features.loc[final_train].to_numpy(),
        target_log.loc[final_train].to_numpy(),
        best_rff["gamma_factor"],
        best_rff["ridge"],
        SEED,
    )
    main_march = np.maximum(0.0, np.expm1(main_model.predict(features.loc[march].to_numpy())))
    rff_march = np.maximum(0.0, np.expm1(rff_model.predict(features.loc[march].to_numpy())))
    actual_march = frame.loc[march, "ntu"].to_numpy()
    previous_day = frame["ntu"].shift(12).loc[march].to_numpy()
    baseline_valid = np.isfinite(previous_day)

    jan_validation = available & (frame["timestamp"] >= pd.Timestamp("2026-01-01")).to_numpy() & (
        frame["timestamp"] < pd.Timestamp("2026-02-01")
    ).to_numpy()
    pre_jan = available & (frame["timestamp"] < pd.Timestamp("2026-01-01")).to_numpy()
    interval_model = fit_ridge(
        features.loc[pre_jan].to_numpy(), target_log.loc[pre_jan].to_numpy(), best_ridge, robust=True
    )
    jan_log_prediction = interval_model.predict(features.loc[jan_validation].to_numpy())
    rolling_residuals = []
    rolling_q1_models = []
    for train, validation in rolling_masks(frame, available):
        rolling_model = fit_ridge(
            features.loc[train].to_numpy(), target_log.loc[train].to_numpy(), best_ridge, robust=True
        )
        rolling_q1_models.append((train, rolling_model))
        rolling_residuals.extend(
            target_log.loc[validation].to_numpy()
            - rolling_model.predict(features.loc[validation].to_numpy())
        )
    interval_residual = np.asarray(rolling_residuals)
    residual_quantiles = np.quantile(interval_residual, [0.025, 0.975])

    selected_days = {date(2026, 2, 1), date(2026, 2, 10), date(2026, 2, 20)}
    requested = frame["op_day"].isin(selected_days).to_numpy()
    requested_log = main_model.predict(features.loc[requested].to_numpy())
    prediction_table = pd.DataFrame(
        {
            "营运日": frame.loc[requested, "op_day"].astype(str).to_numpy(),
            "自然时间": frame.loc[requested, "timestamp"].dt.strftime("%Y-%m-%d %H:%M").to_numpy(),
            "时次": frame.loc[requested, "time"].to_numpy(),
            "条件预测_NTU": np.maximum(0.0, np.expm1(requested_log)),
            "95%下限_NTU": np.maximum(0.0, np.expm1(requested_log + residual_quantiles[0])),
            "95%上限_NTU": np.maximum(0.0, np.expm1(requested_log + residual_quantiles[1])),
            "适用性说明": "条件估计；2026-03外测泛化门禁失败",
        }
    )

    validation_indices = np.flatnonzero(jan_validation)
    jan_actual = frame.loc[jan_validation, "ntu"].to_numpy()
    jan_prediction = np.maximum(0.0, np.expm1(jan_log_prediction))
    base_rmse = metric_dict(jan_actual, jan_prediction)["rmse"]
    rng = np.random.default_rng(SEED)
    block_count = len(validation_indices) // 12
    block_order = rng.permutation(block_count)
    permutation_index = np.concatenate(
        [np.arange(item * 12, item * 12 + 12) for item in block_order]
    )
    x_validation = features.loc[jan_validation].to_numpy()
    factors = []
    for variable_index, variable in enumerate(Q1_VARIABLES):
        columns = [
            index
            for index, name in enumerate(feature_names)
            if name == variable or name.startswith(f"{variable}_lag")
        ]
        permuted = x_validation.copy()
        permuted[:, columns] = permuted[permutation_index][:, columns]
        perm_prediction = np.maximum(0.0, np.expm1(interval_model.predict(permuted)))
        importance = metric_dict(jan_actual, perm_prediction)["rmse"] - base_rmse

        train_values = frame.loc[pre_jan, variable].to_numpy(dtype=float)
        q25, q75 = np.nanquantile(train_values, [0.25, 0.75])
        low = np.nanmedian(features.loc[pre_jan].to_numpy(), axis=0)[None, :]
        high = low.copy()
        for column in columns:
            name = feature_names[column]
            low[0, column] = np.log1p(q25) if variable in {"rw_ntu", "rw_clr", "filt_ntu"} else q25
            high[0, column] = np.log1p(q75) if variable in {"rw_ntu", "rw_clr", "filt_ntu"} else q75
        effect = float(np.expm1(interval_model.predict(high))[0] - np.expm1(interval_model.predict(low))[0])
        rho, rho_low, rho_high = block_spearman_ci(
            frame.loc[pre_jan, variable].to_numpy(dtype=float),
            frame.loc[pre_jan, "ntu"].to_numpy(dtype=float),
            SEED + variable_index,
        )
        fold_effect_signs = []
        fold_spearman_signs = []
        fold_agreements = []
        for fold_train, fold_model in rolling_q1_models:
            fold_variable = frame.loc[fold_train, variable].to_numpy(dtype=float)
            fold_q25, fold_q75 = np.nanquantile(fold_variable, [0.25, 0.75])
            fold_low = np.nanmedian(features.loc[fold_train].to_numpy(), axis=0)[None, :]
            fold_high = fold_low.copy()
            for column in columns:
                fold_low[0, column] = (
                    np.log1p(fold_q25) if variable in {"rw_ntu", "rw_clr", "filt_ntu"} else fold_q25
                )
                fold_high[0, column] = (
                    np.log1p(fold_q75) if variable in {"rw_ntu", "rw_clr", "filt_ntu"} else fold_q75
                )
            fold_effect = float(
                np.expm1(fold_model.predict(fold_high))[0]
                - np.expm1(fold_model.predict(fold_low))[0]
            )
            fold_rho = float(
                spearmanr(
                    fold_variable,
                    frame.loc[fold_train, "ntu"].to_numpy(dtype=float),
                    nan_policy="omit",
                ).statistic
            )
            effect_sign = int(np.sign(fold_effect))
            rho_sign = int(np.sign(fold_rho)) if np.isfinite(fold_rho) else 0
            fold_effect_signs.append(effect_sign)
            fold_spearman_signs.append(rho_sign)
            fold_agreements.append(effect_sign != 0 and effect_sign == rho_sign)
        factors.append(
            {
                "factor": variable,
                "permutation_delta_rmse_ntu": float(importance),
                "q25_to_q75_effect_ntu": effect,
                "direction": "positive" if effect > 0 else "negative" if effect < 0 else "flat",
                "spearman_rho": rho,
                "spearman_ci_low": rho_low,
                "spearman_ci_high": rho_high,
                "fold_effect_signs": ",".join(str(value) for value in fold_effect_signs),
                "fold_spearman_signs": ",".join(str(value) for value in fold_spearman_signs),
                "fold_direction_agreement_count": int(sum(fold_agreements)),
                "fold_direction_agreement_rate": float(np.mean(fold_agreements)),
                "importance_split": "2026-01 posterior validation",
            }
        )
    factor_table = pd.DataFrame(factors).sort_values("permutation_delta_rmse_ntu", ascending=False).reset_index(drop=True)
    factor_table["importance_rank"] = np.arange(1, len(factor_table) + 1)
    factor_table["direction_consistent"] = factor_table["fold_direction_agreement_rate"] >= 0.75
    factor_table["selected_main_factor"] = (
        (factor_table["importance_rank"] <= 6)
        & (factor_table["permutation_delta_rmse_ntu"] > 0)
        & factor_table["direction_consistent"]
    )

    function_rows = [
        {
            "feature": "intercept",
            "source_variable": "constant",
            "lag_hours": 0,
            "input_transform": "constant",
            "standardized_coefficient": float(main_model.coef[0]),
            "coefficient_per_transformed_unit": float(main_model.coef[0]),
            "imputation_median": np.nan,
            "winsor_lower": np.nan,
            "winsor_upper": np.nan,
            "standardization_mean": np.nan,
            "standardization_scale": np.nan,
            "function_definition": "NTU_hat=max(0, exp(beta0 + sum(beta_j*z_j))-1)",
        }
    ]
    for feature_index, feature_name in enumerate(feature_names):
        if "_lag" in feature_name:
            source_variable, lag_text = feature_name.rsplit("_lag", 1)
            lag_hours = int(lag_text) * STEP_HOURS
        else:
            source_variable = feature_name
            lag_hours = 0
        transform = "log1p" if source_variable in {"rw_ntu", "rw_clr", "filt_ntu"} else "identity"
        if source_variable in {"hour_sin", "hour_cos", "year_sin", "year_cos"}:
            transform = "cyclic"
        function_rows.append(
            {
                "feature": feature_name,
                "source_variable": source_variable,
                "lag_hours": lag_hours,
                "input_transform": transform,
                "standardized_coefficient": float(main_model.coef[feature_index + 1]),
                "coefficient_per_transformed_unit": float(
                    main_model.coef[feature_index + 1] / main_model.scale[feature_index]
                ),
                "imputation_median": float(main_model.median[feature_index]),
                "winsor_lower": float(main_model.lower[feature_index]),
                "winsor_upper": float(main_model.upper[feature_index]),
                "standardization_mean": float(main_model.mean[feature_index]),
                "standardization_scale": float(main_model.scale[feature_index]),
                "function_definition": "z_j=(clip(g_j(x_j),lower,upper)-mean)/scale",
            }
        )
    function_parameters = pd.DataFrame(function_rows)

    curve_rows = []
    baseline_features = np.nanmedian(features.loc[final_train].to_numpy(), axis=0)[None, :]
    selected_lookup = factor_table.set_index("factor")["selected_main_factor"].to_dict()
    for variable in Q1_VARIABLES:
        columns = [
            index
            for index, name in enumerate(feature_names)
            if name == variable or name.startswith(f"{variable}_lag")
        ]
        grid = np.linspace(
            float(np.nanquantile(frame.loc[final_train, variable], 0.05)),
            float(np.nanquantile(frame.loc[final_train, variable], 0.95)),
            25,
        )
        for value in grid:
            curve_features = baseline_features.copy()
            transformed_value = (
                np.log1p(value) if variable in {"rw_ntu", "rw_clr", "filt_ntu"} else value
            )
            curve_features[0, columns] = transformed_value
            curve_rows.append(
                {
                    "factor": variable,
                    "factor_value": float(value),
                    "conditional_ntu": float(
                        max(0.0, np.expm1(main_model.predict(curve_features))[0])
                    ),
                    "selected_main_factor": bool(selected_lookup[variable]),
                    "other_features": "held at training medians; current and lagged terms moved together",
                }
            )
    partial_effects = pd.DataFrame(curve_rows)

    evaluation = pd.DataFrame(
        [
            {"model": "Huber-ridge", **metric_dict(actual_march, main_march)},
            {"model": "RFF-ridge", **metric_dict(actual_march, rff_march)},
            {
                "model": "same-hour previous operational day",
                **metric_dict(actual_march[baseline_valid], previous_day[baseline_valid]),
            },
        ]
    )
    evaluation["split"] = "2026-03 external"
    baseline_rmse = float(
        evaluation.loc[evaluation["model"] == "same-hour previous operational day", "rmse"].iloc[0]
    )
    evaluation["meets_generalization_gate"] = (
        (evaluation["r2"] >= 0) & (evaluation["rmse"] < baseline_rmse)
    )
    evaluation["interpretation"] = np.where(
        evaluation["meets_generalization_gate"],
        "passes external gate",
        "fails external gate; do not claim generalization",
    )
    return {
        "prediction": prediction_table,
        "evaluation": evaluation,
        "factors": factor_table,
        "function_parameters": function_parameters,
        "partial_effects": partial_effects,
        "ridge_search": pd.DataFrame(ridge_scores),
        "rff_search": pd.DataFrame(rff_scores),
        "best_ridge": float(best_ridge),
        "best_rff": best_rff,
        "main_model": main_model,
        "features": features,
        "march_mask": march,
        "march_prediction": main_march,
    }


def q2_design_order(
    frame: pd.DataFrame, delays: tuple[int, int, int, int], order: int
) -> tuple[pd.DataFrame, pd.Series]:
    target = np.log1p(frame["filt_ntu"])
    data: dict[str, pd.Series] = {f"target_lag{lag}": target.shift(lag) for lag in range(1, order + 1)}
    variables = ("rw_ntu", "rw_ph", "alum", "rw_flow")
    for variable, delay in zip(variables, delays):
        series = frame[variable].shift(delay)
        if variable == "rw_ntu":
            series = np.log1p(series)
        data[variable] = series
        data[f"{variable}_sq"] = series**2
    data["ntu_alum"] = np.log1p(frame["rw_ntu"].shift(delays[0])) * frame["alum"].shift(delays[2])
    data["ph_alum"] = frame["rw_ph"].shift(delays[1]) * frame["alum"].shift(delays[2])
    return pd.DataFrame(data), target


def run_q2(frame: pd.DataFrame) -> dict[str, Any]:
    search_rows = []
    best: tuple[float, tuple[int, int, int, int], float] | None = None
    for delays in np.ndindex(4, 4, 4, 4):
        design, target = q2_design(frame, delays)
        complete = design.notna().all(axis=1) & target.notna()
        for ridge in MODEL_LAMBDAS:
            scores = []
            for train, validation in rolling_masks(frame, complete):
                model = fit_ridge(design.loc[train].to_numpy(), target.loc[train].to_numpy(), ridge)
                predicted = np.maximum(0.0, np.expm1(model.predict(design.loc[validation].to_numpy())))
                scores.append(metric_dict(frame.loc[validation, "filt_ntu"].to_numpy(), predicted)["rmse"])
            mean_rmse = float(np.mean(scores))
            search_rows.append(
                {
                    "rw_ntu_lag_steps": delays[0],
                    "rw_ph_lag_steps": delays[1],
                    "alum_lag_steps": delays[2],
                    "rw_flow_lag_steps": delays[3],
                    "ridge": ridge,
                    "mean_validation_rmse_ntu": mean_rmse,
                }
            )
            key = (mean_rmse, sum(delays), delays, ridge)
            if best is None or key < (best[0], sum(best[1]), best[1], best[2]):
                best = (mean_rmse, delays, ridge)
    if best is None:
        raise RuntimeError("Q2 search failed")

    delays = best[1]
    order_rows = []
    order_best = None
    for order in (3, 6, 9, 12):
        order_design, order_target = q2_design_order(frame, delays, order)
        order_complete = order_design.notna().all(axis=1) & order_target.notna()
        for order_ridge in MODEL_LAMBDAS:
            scores = []
            for fold_train, fold_validation in rolling_masks(frame, order_complete):
                order_model = fit_ridge(
                    order_design.loc[fold_train].to_numpy(),
                    order_target.loc[fold_train].to_numpy(),
                    order_ridge,
                )
                order_prediction = np.maximum(
                    0.0,
                    np.expm1(order_model.predict(order_design.loc[fold_validation].to_numpy())),
                )
                scores.append(
                    metric_dict(frame.loc[fold_validation, "filt_ntu"].to_numpy(), order_prediction)["rmse"]
                )
            mean_rmse = float(np.mean(scores))
            order_rows.append({"ar_order": order, "ridge": order_ridge, "mean_validation_rmse_ntu": mean_rmse})
            key = (mean_rmse, order, order_ridge)
            if order_best is None or key < order_best:
                order_best = key
    ridge = float(order_best[2])
    selected_order = int(order_best[1])
    design, target = q2_design_order(frame, delays, selected_order)
    complete = design.notna().all(axis=1) & target.notna()
    train = complete & (frame["timestamp"] < pd.Timestamp("2026-03-01"))
    march = complete & (frame["timestamp"] >= pd.Timestamp("2026-03-01"))
    model = fit_ridge(design.loc[train].to_numpy(), target.loc[train].to_numpy(), ridge)
    prediction = np.maximum(0.0, np.expm1(model.predict(design.loc[march].to_numpy())))
    actual = frame.loc[march, "filt_ntu"].to_numpy()
    metrics = metric_dict(actual, prediction)
    residual = actual - prediction

    acf = []
    centered = residual - np.mean(residual)
    denominator = np.sum(centered**2)
    for lag in range(1, 13):
        acf.append(float(np.sum(centered[:-lag] * centered[lag:]) / denominator))
    n = len(residual)
    q_stat = float(n * (n + 2) * sum(value**2 / (n - lag) for lag, value in enumerate(acf, 1)))
    q_pvalue = float(chi2.sf(q_stat, 12))

    cross_rows = []
    target_diff = np.log1p(frame["filt_ntu"]).diff()
    for variable in ("rw_ntu", "rw_ph", "alum", "rw_flow"):
        source = np.log1p(frame[variable]) if variable == "rw_ntu" else frame[variable]
        source_diff = source.diff()
        correlations = []
        for lag in range(7):
            pair = pd.concat((source_diff.shift(lag), target_diff), axis=1).dropna()
            correlations.append(float(pair.corr().iloc[0, 1]))
        peak = int(np.nanargmax(np.abs(correlations)))
        cross_rows.append(
            {
                "variable": variable,
                "arx_lag_steps": delays[("rw_ntu", "rw_ph", "alum", "rw_flow").index(variable)],
                "crosscorr_peak_steps": peak,
                "crosscorr_peak_hours": peak * STEP_HOURS,
                "crosscorr_at_peak": correlations[peak],
                "identifiable": abs(peak - delays[("rw_ntu", "rw_ph", "alum", "rw_flow").index(variable)]) <= 2,
            }
        )

    delay_table = pd.DataFrame(cross_rows)
    delay_table["arx_lag_hours"] = delay_table["arx_lag_steps"] * STEP_HOURS
    delay_table["selected_ridge"] = ridge
    delay_table["selected_ar_order"] = selected_order
    parameter_table = pd.DataFrame(
        {
            "term": ["intercept", *design.columns],
            "standardized_coefficient": model.coef,
            "direction": np.where(model.coef >= 0, "positive", "negative"),
        }
    )
    accuracy = pd.DataFrame(
        [
            {
                "split": "rolling validation mean",
                "rmse": order_best[0],
                "mae": np.nan,
                "r2": np.nan,
                "ljung_box_q12": np.nan,
                "ljung_box_pvalue": np.nan,
                "max_abs_residual_acf_lag1_6": np.nan,
            },
            {
                "split": "2026-03 external",
                **metrics,
                "ljung_box_q12": q_stat,
                "ljung_box_pvalue": q_pvalue,
                "max_abs_residual_acf_lag1_6": float(np.max(np.abs(acf[:6]))),
            },
        ]
    )
    accuracy["diagnostic_status"] = np.where(
        accuracy["max_abs_residual_acf_lag1_6"].isna(),
        "not applicable",
        np.where(
            accuracy["max_abs_residual_acf_lag1_6"] > 0.3,
            "strong residual autocorrelation remains after AR-order test",
            "residual autocorrelation gate passed",
        ),
    )
    external = pd.DataFrame(
        {
            "timestamp": frame.loc[march, "timestamp"].to_numpy(),
            "actual_filt_ntu": actual,
            "predicted_filt_ntu": prediction,
            "residual_ntu": residual,
        }
    )
    search = pd.DataFrame(search_rows)
    return {
        "delays": delay_table,
        "parameters": parameter_table,
        "accuracy": accuracy,
        "external": external,
        "acf": pd.DataFrame({"lag_steps": np.arange(1, 13), "acf": acf}),
        "search": search,
        "order_sensitivity": pd.DataFrame(order_rows),
        "best_delays": delays,
        "best_ridge": float(ridge),
        "selected_order": selected_order,
    }


def tau_at(frame: pd.DataFrame, index: int, tau0: float, median_level: float, median_flow: float) -> float:
    level = frame.at[index, "cw_level"]
    flow = frame.at[index, "tw_flow"]
    level = median_level if not np.isfinite(level) else level
    flow = median_flow if not np.isfinite(flow) or flow <= 0 else flow
    return float(np.clip(tau0 * (level / median_level) / (flow / median_flow), 2.0, 24.0))


def physical_at(values: np.ndarray, index: int, kernel: np.ndarray) -> float:
    total = 0.0
    fallback = values[max(0, index)]
    if not np.isfinite(fallback):
        fallback = float(np.nanmedian(values))
    for lag, weight in enumerate(kernel):
        source_index = max(0, index - lag)
        value = values[source_index]
        total += weight * (fallback if not np.isfinite(value) else value)
    return float(total)


def physical_baseline(
    frame: pd.DataFrame, tau0: float, median_level: float, median_flow: float
) -> np.ndarray:
    values = frame["filt_ntu"].to_numpy(dtype=float)
    output = np.empty(len(frame))
    for index in range(len(frame)):
        output[index] = physical_at(values, index, rtd_kernel(tau_at(frame, index, tau0, median_level, median_flow)))
    return output


def physical_forecast(
    frame: pd.DataFrame,
    origin: int,
    horizon_steps: int,
    tau0: float,
    median_level: float,
    median_flow: float,
) -> float:
    values = frame["filt_ntu"].to_numpy(dtype=float)
    last = values[origin]
    if not np.isfinite(last):
        last = values[: origin + 1][np.isfinite(values[: origin + 1])][-1]
    future_index = origin + horizon_steps
    tau = tau_at(frame, origin, tau0, median_level, median_flow)
    kernel = rtd_kernel(tau)
    total = 0.0
    for lag, weight in enumerate(kernel):
        source_index = future_index - lag
        value = last if source_index > origin else values[max(0, source_index)]
        if not np.isfinite(value):
            value = last
        total += weight * value
    return float(total)


STATE_NAMES = (
    *(f"residual_lag{lag}" for lag in range(6)),
    "log_filt_current",
    "log_filt_lag1",
    "delta_log_filt",
    "log_rw_ntu_current",
    "log_rw_ntu_lag1",
    "delta_log_rw_ntu",
    "alum",
    "rw_flow",
    "tw_flow",
    "cw_level",
    "alum_missing",
    "rw_flow_missing",
    "tw_flow_missing",
    "cw_level_missing",
    "hour_sin",
    "hour_cos",
    "year_sin",
    "year_cos",
)


def state_feature(
    frame: pd.DataFrame,
    origin: int,
    residual_state: np.ndarray,
    override: dict[str, float] | None = None,
) -> np.ndarray:
    override = override or {}
    residual_part = np.asarray([residual_state[origin - lag] for lag in range(6)], dtype=float)
    filt_now, filt_previous = frame.at[origin, "filt_ntu"], frame.at[origin - 1, "filt_ntu"]
    rw_now = override.get("rw_ntu", frame.at[origin, "rw_ntu"])
    rw_previous = frame.at[origin - 1, "rw_ntu"]
    alum = override.get("alum", frame.at[origin, "alum"])
    continuous = np.asarray(
        [
            np.log1p(filt_now),
            np.log1p(filt_previous),
            np.log1p(filt_now) - np.log1p(filt_previous),
            np.log1p(rw_now),
            np.log1p(rw_previous),
            np.log1p(rw_now) - np.log1p(rw_previous),
            alum,
            frame.at[origin, "rw_flow"],
            frame.at[origin, "tw_flow"],
            frame.at[origin, "cw_level"],
        ],
        dtype=float,
    )
    missing = np.isnan(continuous[[6, 7, 8, 9]]).astype(float)
    timestamp = frame.at[origin, "timestamp"]
    hour = timestamp.hour + timestamp.minute / 60.0
    day_of_year = timestamp.dayofyear
    cycles = np.asarray(
        [
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            math.sin(2 * math.pi * day_of_year / 365.25),
            math.cos(2 * math.pi * day_of_year / 365.25),
        ]
    )
    return np.concatenate((residual_part, continuous, missing, cycles))


def build_state_samples(
    frame: pd.DataFrame,
    residual_reference: np.ndarray,
    horizon: int,
    tau0: float,
    median_level: float,
    median_flow: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features, targets, origins, target_values = [], [], [], []
    y = frame["ntu"].to_numpy(dtype=float)
    for origin in range(6, len(frame) - horizon):
        if not np.all(np.isfinite(residual_reference[origin - 5 : origin + 1])):
            continue
        target_index = origin + horizon
        if not np.isfinite(y[target_index]):
            continue
        physical = physical_forecast(frame, origin, horizon, tau0, median_level, median_flow)
        features.append(state_feature(frame, origin, residual_reference))
        targets.append(math.log1p(y[target_index]) - math.log1p(physical))
        origins.append(origin)
        target_values.append(y[target_index])
    return np.asarray(features), np.asarray(targets), np.asarray(origins), np.asarray(target_values)


def predict_log_direct(
    frame: pd.DataFrame,
    model: RidgeModel,
    origin: int,
    horizon: int,
    residual_state: np.ndarray,
    tau0: float,
    median_level: float,
    median_flow: float,
    override: dict[str, float] | None = None,
) -> float:
    feature = state_feature(frame, origin, residual_state, override)[None, :]
    physical = physical_forecast(frame, origin, horizon, tau0, median_level, median_flow)
    return float(math.log1p(physical) + model.predict(feature)[0])


def moving_block_errors(error_vectors: np.ndarray, length: int, paths: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.empty((paths, length, error_vectors.shape[1]), dtype=np.float32)
    block = 12
    for path in range(paths):
        position = 0
        while position < length:
            start = int(rng.integers(0, len(error_vectors)))
            count = min(block, length - position)
            indices = (start + np.arange(count)) % len(error_vectors)
            output[path, position : position + count] = error_vectors[indices]
            position += count
    return output


def path_state_matrix(frame: pd.DataFrame, origin: int, path_residual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    base = state_feature(frame, origin, reference)
    matrix = np.tile(base, (path_residual.shape[0], 1))
    matrix[:, :6] = np.column_stack([path_residual[:, origin - lag] for lag in range(6)])
    return matrix


def run_q3(frame: pd.DataFrame) -> dict[str, Any]:
    early = frame["timestamp"] < pd.Timestamp("2025-09-01")
    median_level = float(frame.loc[early, "cw_level"].median())
    median_flow = float(frame.loc[early, "tw_flow"].median())
    tau_rows = []
    y_log = np.log1p(frame["ntu"])
    available = y_log.notna().to_numpy()
    for tau0 in range(2, 25, 2):
        physical = physical_baseline(frame, tau0, median_level, median_flow)
        predictor = np.log1p(physical)[:, None]
        scores = []
        for train, validation in rolling_masks(frame, available):
            model = fit_ridge(predictor[train], y_log.loc[train].to_numpy(), 0.1)
            predicted = np.maximum(0.0, np.expm1(model.predict(predictor[validation])))
            scores.append(metric_dict(frame.loc[validation, "ntu"].to_numpy(), predicted)["rmse"])
        tau_rows.append({"tau_hours": tau0, "mean_validation_rmse_ntu": float(np.mean(scores))})
    tau0 = int(min(tau_rows, key=lambda item: item["mean_validation_rmse_ntu"])["tau_hours"])

    final_train = frame["timestamp"] < pd.Timestamp("2026-02-01")
    median_level = float(frame.loc[final_train, "cw_level"].median())
    median_flow = float(frame.loc[final_train, "tw_flow"].median())
    physical = physical_baseline(frame, tau0, median_level, median_flow)
    actual_y = frame["ntu"].to_numpy(dtype=float)
    residual_reference = np.log1p(actual_y) - np.log1p(physical)

    selected_lambdas: dict[int, float] = {}
    sample_cache = {}
    for horizon in range(1, 7):
        x, target, origins, target_values = build_state_samples(
            frame, residual_reference, horizon, tau0, median_level, median_flow
        )
        target_timestamps = frame.loc[origins + horizon, "timestamp"].to_numpy()
        scores = []
        for ridge in MODEL_LAMBDAS:
            fold_rmse = []
            for start, end in ROLLING_FOLDS:
                train = target_timestamps < start.to_datetime64()
                validation = (target_timestamps >= start.to_datetime64()) & (target_timestamps < end.to_datetime64())
                model = fit_ridge(x[train], target[train], ridge, robust=True)
                residual_prediction = model.predict(x[validation])
                prediction = []
                for value, origin in zip(residual_prediction, origins[validation]):
                    baseline = physical_forecast(frame, int(origin), horizon, tau0, median_level, median_flow)
                    prediction.append(max(0.0, math.expm1(math.log1p(baseline) + value)))
                fold_rmse.append(metric_dict(target_values[validation], np.asarray(prediction))["rmse"])
            scores.append((float(np.mean(fold_rmse)), ridge))
        selected_lambdas[horizon] = min(scores)[1]
        sample_cache[horizon] = (x, target, origins, target_values, target_timestamps)

    final_models = {}
    interim_models = {}
    for horizon in range(1, 7):
        x, target, origins, _, target_timestamps = sample_cache[horizon]
        final_mask = target_timestamps < np.datetime64("2026-02-01")
        interim_mask = target_timestamps < np.datetime64("2026-01-01")
        final_models[horizon] = fit_ridge(
            x[final_mask], target[final_mask], selected_lambdas[horizon], robust=True
        )
        interim_models[horizon] = fit_ridge(
            x[interim_mask], target[interim_mask], selected_lambdas[horizon], robust=True
        )

    january_start = int(frame.index[frame["timestamp"] == pd.Timestamp("2026-01-01 07:00")][0])
    february_start = int(frame.index[frame["timestamp"] == pd.Timestamp("2026-02-01 07:00")][0])
    march_start = int(frame.index[frame["timestamp"] == pd.Timestamp("2026-03-01 07:00")][0])
    end_index = len(frame) - 1
    bootstrap_rows = []
    for origin in range(january_start, february_start - 6):
        if not np.all(np.isfinite(residual_reference[origin - 5 : origin + 1])):
            continue
        row = []
        valid = True
        for horizon in range(1, 7):
            target_index = origin + horizon
            if not np.isfinite(actual_y[target_index]):
                valid = False
                break
            predicted_log = predict_log_direct(
                frame,
                interim_models[horizon],
                origin,
                horizon,
                residual_reference,
                tau0,
                median_level,
                median_flow,
            )
            row.append(math.log1p(actual_y[target_index]) - predicted_log)
        if valid:
            bootstrap_rows.append(row)
    error_vectors = np.asarray(bootstrap_rows)
    if len(error_vectors) < 100:
        raise AssertionError("Insufficient joint validation residual vectors for Q3 bootstrap")

    star_y = actual_y.copy()
    star_r = residual_reference.copy()
    for index in range(6, february_start):
        if np.isfinite(star_y[index]):
            continue
        predicted_log = predict_log_direct(
            frame, final_models[1], index - 1, 1, star_r, tau0, median_level, median_flow
        )
        star_y[index] = max(0.0, math.expm1(predicted_log))
        star_r[index] = math.log1p(star_y[index]) - math.log1p(physical[index])
    for index in range(february_start, march_start):
        predicted_log = predict_log_direct(
            frame, final_models[1], index - 1, 1, star_r, tau0, median_level, median_flow
        )
        star_y[index] = max(0.0, math.expm1(predicted_log))
        star_r[index] = math.log1p(star_y[index]) - math.log1p(physical[index])
    for index in range(march_start, len(frame)):
        if np.isfinite(actual_y[index]):
            star_y[index] = actual_y[index]
            star_r[index] = residual_reference[index]

    minimum_origin = february_start - 6
    error_length = end_index - minimum_origin + 1
    sampled_errors = moving_block_errors(error_vectors, error_length, 500, SEED)
    path_y = np.tile(star_y, (500, 1)).astype(np.float32)
    path_r = np.tile(star_r, (500, 1)).astype(np.float32)
    for target_index in range(february_start, end_index + 1):
        origin = target_index - 1
        matrix = path_state_matrix(frame, origin, path_r, star_r)
        baseline = physical_forecast(frame, origin, 1, tau0, median_level, median_flow)
        predicted_log = math.log1p(baseline) + final_models[1].predict(matrix)
        predicted_log += sampled_errors[:, origin - minimum_origin, 0]
        values = np.maximum(0.0, np.expm1(predicted_log))
        path_y[:, target_index] = values.astype(np.float32)
        path_r[:, target_index] = (np.log1p(values) - math.log1p(physical[target_index])).astype(np.float32)

    def direct_paths(origin: int, horizon: int) -> np.ndarray:
        matrix = path_state_matrix(frame, origin, path_r, star_r)
        baseline = physical_forecast(frame, origin, horizon, tau0, median_level, median_flow)
        predicted_log = math.log1p(baseline) + final_models[horizon].predict(matrix)
        predicted_log += sampled_errors[:, origin - minimum_origin, horizon - 1]
        return np.maximum(0.0, np.expm1(predicted_log))

    selected_dates = (date(2026, 2, 1), date(2026, 2, 10), date(2026, 2, 20))
    designated_rows = []
    for selected_date in selected_dates:
        day_rows = frame.index[(frame["op_day"] == selected_date) & frame["time"].isin(
            ("0700", "0900", "1100", "1300", "1500", "1700", "1900")
        )]
        for target_index in day_rows:
            origin = int(target_index) - 6
            paths = direct_paths(origin, 6)
            quantiles = np.quantile(paths, [0.025, 0.5, 0.975])
            hydraulic = physical_forecast(frame, origin, 6, tau0, median_level, median_flow)
            designated_rows.append(
                {
                    "营运日": str(selected_date),
                    "预测起点": frame.at[origin, "timestamp"],
                    "目标时间": frame.at[target_index, "timestamp"],
                    "时距_h": 12,
                    "预测中位数_NTU": quantiles[1],
                    "95%下限_NTU": quantiles[0],
                    "95%上限_NTU": quantiles[2],
                    "水力基线_NTU": hydraulic,
                    "残差修正贡献_NTU": quantiles[1] - hydraulic,
                    "轨迹数": 500,
                }
            )
    designated = pd.DataFrame(designated_rows)

    horizon_rows = []
    for selected_date in selected_dates:
        origin = int(frame.index[(frame["op_day"] == selected_date) & (frame["time"] == "0700")][0])
        even = {0: path_y[:, origin].astype(float)}
        for horizon in range(1, 7):
            even[2 * horizon] = direct_paths(origin, horizon)
        for hour in range(1, 13):
            if hour % 2 == 0:
                values = even[hour]
                interpolated = False
            else:
                lower_hour, upper_hour = hour - 1, hour + 1
                values = np.expm1(
                    0.5 * (np.log1p(even[lower_hour]) + np.log1p(even[upper_hour]))
                )
                interpolated = True
            quantiles = np.quantile(values, [0.025, 0.5, 0.975])
            horizon_rows.append(
                {
                    "营运日": str(selected_date),
                    "预测起点": frame.at[origin, "timestamp"],
                    "时距_h": hour,
                    "目标时间": frame.at[origin, "timestamp"] + pd.Timedelta(hours=hour),
                    "interpolated_horizon": interpolated,
                    "预测中位数_NTU": quantiles[1],
                    "95%下限_NTU": quantiles[0],
                    "95%上限_NTU": quantiles[2],
                }
            )
    all_horizons = pd.DataFrame(horizon_rows)

    accuracy_rows = []
    external_predictions = []
    for horizon in range(1, 7):
        actual_values, hybrid_values, persistence_values, physical_values = [], [], [], []
        lower_values, upper_values = [], []
        error_low, error_high = np.quantile(error_vectors[:, horizon - 1], [0.025, 0.975])
        for origin in range(march_start, len(frame) - horizon):
            target_index = origin + horizon
            if frame.at[target_index, "timestamp"] < pd.Timestamp("2026-03-01") or not np.isfinite(actual_y[target_index]):
                continue
            predicted_log = predict_log_direct(
                frame,
                final_models[horizon],
                origin,
                horizon,
                star_r,
                tau0,
                median_level,
                median_flow,
            )
            hybrid = max(0.0, math.expm1(predicted_log))
            baseline = physical_forecast(frame, origin, horizon, tau0, median_level, median_flow)
            actual_values.append(actual_y[target_index])
            hybrid_values.append(hybrid)
            persistence_values.append(star_y[origin])
            physical_values.append(baseline)
            lower_values.append(max(0.0, math.expm1(predicted_log + error_low)))
            upper_values.append(max(0.0, math.expm1(predicted_log + error_high)))
            external_predictions.append(
                {
                    "origin": frame.at[origin, "timestamp"],
                    "target": frame.at[target_index, "timestamp"],
                    "horizon_hours": horizon * 2,
                    "actual_ntu": actual_y[target_index],
                    "hybrid_ntu": hybrid,
                    "lower_ntu": lower_values[-1],
                    "upper_ntu": upper_values[-1],
                }
            )
        actual_array = np.asarray(actual_values)
        for model_name, predictions in (
            ("hybrid", hybrid_values),
            ("persistence", persistence_values),
            ("physical", physical_values),
        ):
            metrics = metric_dict(actual_array, np.asarray(predictions))
            accuracy_rows.append(
                {
                    "horizon_hours": horizon * 2,
                    "model": model_name,
                    **metrics,
                    "coverage_95": float(
                        np.mean((actual_array >= np.asarray(lower_values)) & (actual_array <= np.asarray(upper_values)))
                    )
                    if model_name == "hybrid"
                    else np.nan,
                    "n": len(actual_array),
                }
            )
    accuracy = pd.DataFrame(accuracy_rows)
    persistence_rmse = accuracy.loc[accuracy["model"] == "persistence"].set_index("horizon_hours")["rmse"]
    accuracy["beats_persistence"] = accuracy.apply(
        lambda row: bool(row["rmse"] < persistence_rmse.loc[row["horizon_hours"]])
        if row["model"] == "hybrid"
        else np.nan,
        axis=1,
    )

    sensitivity_rows = []
    scenarios = (
        ("R/W NTU +25%", "rw_ntu", 0.25),
        ("R/W NTU +100%", "rw_ntu", 1.0),
        ("ALUM +10%", "alum", 0.10),
        ("ALUM -10%", "alum", -0.10),
    )
    for selected_date in selected_dates:
        origin = int(frame.index[(frame["op_day"] == selected_date) & (frame["time"] == "0700")][0])
        for horizon in range(1, 7):
            base_log = predict_log_direct(
                frame, final_models[horizon], origin, horizon, star_r, tau0, median_level, median_flow
            )
            base = max(0.0, math.expm1(base_log))
            for scenario, variable, fraction in scenarios:
                original = frame.at[origin, variable]
                override = {variable: original * (1.0 + fraction)}
                scenario_log = predict_log_direct(
                    frame,
                    final_models[horizon],
                    origin,
                    horizon,
                    star_r,
                    tau0,
                    median_level,
                    median_flow,
                    override,
                )
                value = max(0.0, math.expm1(scenario_log))
                sensitivity_rows.append(
                    {
                        "operational_day": str(selected_date),
                        "origin": frame.at[origin, "timestamp"],
                        "horizon_hours": horizon * 2,
                        "scenario": scenario,
                        "baseline_ntu": base,
                        "scenario_ntu": value,
                        "delta_ntu": value - base,
                        "elasticity": ((value - base) / base / fraction) if base > 0 and fraction != 0 else np.nan,
                    }
                )

    return {
        "tau_search": pd.DataFrame(tau_rows),
        "tau_hours": tau0,
        "median_level": median_level,
        "median_flow": median_flow,
        "physical": physical,
        "selected_lambdas": selected_lambdas,
        "models": final_models,
        "star_y": star_y,
        "star_r": star_r,
        "path_y": path_y,
        "path_r": path_r,
        "designated": designated,
        "all_horizons": all_horizons,
        "accuracy": accuracy,
        "external": pd.DataFrame(external_predictions),
        "sensitivity": pd.DataFrame(sensitivity_rows),
        "error_vectors": error_vectors,
        "kernel": rtd_kernel(tau0),
    }


def classify_custom(values: np.ndarray, low_a: float, med_a: float, low_d: float, med_d: float) -> str:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 9:
        return "INSUFFICIENT"
    excess = np.maximum(values - 1.0, 0.0)
    run = longest = 0
    for flag in excess > 0:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    amplitude = float(np.max(excess))
    duration = float(longest * STEP_HOURS)
    if amplitude == 0:
        return "SAFE"
    if amplitude <= low_a and duration <= low_d:
        return "LOW"
    if amplitude <= med_a and duration <= med_d:
        return "MEDIUM"
    return "HIGH"


def run_q4(frame: pd.DataFrame, q3: dict[str, Any]) -> dict[str, pd.DataFrame]:
    start, end = date(2026, 1, 1), date(2026, 3, 31)
    days = [item for item in sorted(frame["op_day"].unique()) if start <= item <= end]
    rows = []
    for operational_day in days:
        indices = frame.index[frame["op_day"] == operational_day].to_numpy()
        observed = frame.loc[indices, "ntu"].to_numpy(dtype=float)
        predicted_mask = ~np.isfinite(observed)
        central_values = np.where(predicted_mask, q3["star_y"][indices], observed)
        central = classify_day(central_values)
        if predicted_mask.any():
            path_grades = [classify_day(values)["grade"] for values in q3["path_y"][:, indices]]
            counts = {grade: path_grades.count(grade) for grade in GRADE_ORDER}
            probabilities = {grade: counts[grade] / 500.0 for grade in GRADE_ORDER}
            codes = np.asarray([GRADE_ORDER.index(grade) for grade in path_grades])
            modal_grade = max(GRADE_ORDER, key=lambda grade: (counts[grade], -GRADE_ORDER.index(grade)))
            lower_grade = GRADE_ORDER[int(np.quantile(codes, 0.025, method="nearest"))]
            upper_grade = GRADE_ORDER[int(np.quantile(codes, 0.975, method="nearest"))]
        else:
            probabilities = {grade: float(grade == central["grade"]) for grade in GRADE_ORDER}
            modal_grade = lower_grade = upper_grade = central["grade"]
        source = "observed" if not predicted_mask.any() else "predicted" if predicted_mask.all() else "mixed"
        rows.append(
            {
                "operational_day": str(operational_day),
                "source": source,
                "valid_points": int(np.isfinite(central_values).sum()),
                "predicted_ratio": float(np.mean(predicted_mask)),
                "max_ntu": central.get("max_ntu", np.nan),
                "max_excess_ntu": central.get("max_excess_ntu", np.nan),
                "longest_exceedance_hours": central.get("longest_exceedance_hours", np.nan),
                "exceedance_load_ntu_hours": central.get("exceedance_load_ntu_hours", np.nan),
                "central_grade": central["grade"],
                "modal_grade": modal_grade,
                "grade_95_low": lower_grade,
                "grade_95_high": upper_grade,
                **{f"p_{grade.lower()}": probabilities[grade] for grade in GRADE_ORDER},
            }
        )
    daily = pd.DataFrame(rows)
    counts = daily["modal_grade"].value_counts().reindex(GRADE_ORDER, fill_value=0)
    summary = pd.DataFrame(
        {
            "grade": GRADE_ORDER,
            "days": counts.to_numpy(),
            "proportion": counts.to_numpy() / len(daily),
        }
    )

    base_grades = daily["central_grade"].to_numpy()
    robust_rows = []
    central_series = np.where(
        np.isfinite(frame["ntu"].to_numpy(dtype=float)), frame["ntu"].to_numpy(dtype=float), q3["star_y"]
    )
    for low_a in (0.4, 0.5, 0.6):
        for med_a in (0.8, 1.0, 1.2):
            for low_d in (2.0, 4.0, 6.0):
                for med_d in (6.0, 8.0, 10.0):
                    grades = []
                    for operational_day in days:
                        indices = frame.index[frame["op_day"] == operational_day].to_numpy()
                        grades.append(classify_custom(central_series[indices], low_a, med_a, low_d, med_d))
                    robust_rows.append(
                        {
                            "low_amplitude": low_a,
                            "medium_amplitude": med_a,
                            "low_duration_hours": low_d,
                            "medium_duration_hours": med_d,
                            "unchanged_rate": float(np.mean(np.asarray(grades) == base_grades)),
                        }
                    )
    return {
        "daily": daily,
        "march": daily[daily["operational_day"].str.startswith("2026-03")].copy(),
        "summary": summary,
        "robustness": pd.DataFrame(robust_rows),
    }


def data_quality_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for raw_name, name in NUMERIC_NAMES.items():
        values = frame[name]
        rows.append(
            {
                "source_field": raw_name,
                "clean_field": name,
                "rows": len(frame),
                "missing_clean": int(values.isna().sum()),
                "missing_rate": float(values.isna().mean()),
                "invalid_count": int(frame[f"{name}_invalid"].sum()),
                "minimum": float(values.min()) if values.notna().any() else np.nan,
                "median": float(values.median()) if values.notna().any() else np.nan,
                "maximum": float(values.max()) if values.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_dataframe_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10g")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, date)):
        return value.isoformat()
    return value


def write_outputs(frame: pd.DataFrame, inputs: list[dict[str, Any]], q1, q2, q3, q4) -> None:
    results = PROJECT_ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    quality = data_quality_table(frame)
    input_table = pd.DataFrame(inputs)
    workbook = results / "建模结果.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        q1["prediction"].to_excel(writer, sheet_name="Q1预测", index=False)
        q1["evaluation"].to_excel(writer, sheet_name="Q1模型评价", index=False)
        q1["factors"].to_excel(writer, sheet_name="Q1因素筛选", index=False)
        q1["function_parameters"].to_excel(writer, sheet_name="Q1函数参数", index=False)
        q1["partial_effects"].to_excel(writer, sheet_name="Q1偏效应", index=False)
        q2["delays"].to_excel(writer, sheet_name="Q2时滞与参数", index=False)
        q2["parameters"].to_excel(writer, sheet_name="Q2时滞与参数", index=False, startrow=len(q2["delays"]) + 3)
        q2["accuracy"].to_excel(writer, sheet_name="Q2拟合精度", index=False)
        q2["order_sensitivity"].to_excel(writer, sheet_name="Q2_AR阶次敏感性", index=False)
        pd.concat(
            (
                q3["designated"].assign(table_type="specified_12h_target"),
                q3["all_horizons"].assign(table_type="daily_origin_1_12h"),
            ),
            ignore_index=True,
            sort=False,
        ).to_excel(writer, sheet_name="Q3预测", index=False)
        q3["accuracy"].to_excel(writer, sheet_name="Q3分时距精度", index=False)
        q3["sensitivity"].to_excel(writer, sheet_name="Q3敏感性", index=False)
        q4["summary"].to_excel(writer, sheet_name="Q4等级占比", index=False)
        q4["robustness"].to_excel(writer, sheet_name="Q4稳健性", index=False)
        q4["march"].to_excel(writer, sheet_name="Q4三月逐日", index=False)
        q4["daily"].to_excel(writer, sheet_name="Q4全期逐日", index=False)
        quality.to_excel(writer, sheet_name="数据质量", index=False)
        input_table.to_excel(writer, sheet_name="输入哈希", index=False)

    csv_tables = {
        "q1_predictions.csv": q1["prediction"],
        "q1_evaluation.csv": q1["evaluation"],
        "q1_factors.csv": q1["factors"],
        "q1_function_parameters.csv": q1["function_parameters"],
        "q1_partial_effects.csv": q1["partial_effects"],
        "q2_delays.csv": q2["delays"],
        "q2_accuracy.csv": q2["accuracy"],
        "q2_ar_order_sensitivity.csv": q2["order_sensitivity"],
        "q3_designated_predictions.csv": q3["designated"],
        "q3_horizon_predictions.csv": q3["all_horizons"],
        "q3_horizon_accuracy.csv": q3["accuracy"],
        "q3_sensitivity.csv": q3["sensitivity"],
        "q4_daily.csv": q4["daily"],
        "q4_summary.csv": q4["summary"],
        "data_quality.csv": quality,
    }
    for filename, table in csv_tables.items():
        write_dataframe_csv(table, results / filename)

    np.savez_compressed(
        results / "q3_uncertainty_500_paths.npz",
        path_y=q3["path_y"],
        path_r=q3["path_r"],
        validation_error_vectors=q3["error_vectors"],
        seed=np.asarray([SEED]),
    )
    summary = {
        "seed": SEED,
        "data": {
            "rows": len(frame),
            "unique_timestamps": int(frame["timestamp"].nunique()),
            "time_repairs": int(frame["time_repaired"].sum()),
            "input_hashes": inputs,
        },
        "q1": {
            "best_ridge": q1["best_ridge"],
            "best_rff": q1["best_rff"],
            "external_metrics": q1["evaluation"].to_dict(orient="records"),
            "top_factors": q1["factors"].head(6).to_dict(orient="records"),
            "selected_main_factors": q1["factors"].loc[
                q1["factors"]["selected_main_factor"], "factor"
            ].tolist(),
            "function_relation": {
                "formula": "NTU_hat=max(0, exp(beta0 + sum(beta_j*z_j))-1)",
                "parameter_table": "results/q1_function_parameters.csv",
                "partial_effect_table": "results/q1_partial_effects.csv",
                "parameter_count_including_intercept": int(len(q1["function_parameters"])),
            },
        },
        "q2": {
            "delays_steps": list(q2["best_delays"]),
            "ridge": q2["best_ridge"],
            "selected_ar_order": q2["selected_order"],
            "external_metrics": q2["accuracy"].to_dict(orient="records"),
        },
        "q3": {
            "tau_hours": q3["tau_hours"],
            "kernel_mass": float(np.sum(q3["kernel"])),
            "ridge_by_horizon": q3["selected_lambdas"],
            "paths": int(q3["path_y"].shape[0]),
            "external_metrics": q3["accuracy"].to_dict(orient="records"),
        },
        "q4": {
            "days": len(q4["daily"]),
            "grade_summary": q4["summary"].to_dict(orient="records"),
            "robustness_mean_unchanged": float(q4["robustness"]["unchanged_rate"].mean()),
            "robustness_min_unchanged": float(q4["robustness"]["unchanged_rate"].min()),
        },
    }
    (results / "full_summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )

    plot_data = {
        "timestamp": frame["timestamp"].astype("int64").to_numpy(),
        "op_day": frame["op_day"].astype(str).to_numpy(),
        "ntu": frame["ntu"].to_numpy(dtype=float),
        "filt_ntu": frame["filt_ntu"].to_numpy(dtype=float),
        "rw_ntu": frame["rw_ntu"].to_numpy(dtype=float),
        "physical": q3["physical"],
        "star_y": q3["star_y"],
        "q1_march_mask": q1["march_mask"],
        "q1_march_prediction": q1["march_prediction"],
    }
    np.savez_compressed(results / "plot_data.npz", **plot_data)
    q1["factors"].to_json(results / "plot_q1_factors.json", orient="records")
    q2["search"].to_csv(results / "plot_q2_search.csv", index=False)
    q2["external"].to_csv(results / "plot_q2_external.csv", index=False)
    q3["tau_search"].to_csv(results / "plot_q3_tau.csv", index=False)


def run_full() -> dict[str, Any]:
    frame, inputs = load_data(PROJECT_ROOT)
    q1 = run_q1(frame)
    q2 = run_q2(frame)
    q3 = run_q3(frame)
    q4 = run_q4(frame, q3)
    write_outputs(frame, inputs, q1, q2, q3, q4)
    return {
        "rows": len(frame),
        "q1_external_r2": float(q1["evaluation"].iloc[0]["r2"]),
        "q2_delays_hours": q2["delays"]["arx_lag_hours"].astype(int).tolist(),
        "q3_tau_hours": q3["tau_hours"],
        "q3_paths": int(q3["path_y"].shape[0]),
        "q4_days": len(q4["daily"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if not args.full:
        parser.error("Use --full")
    result = run_full()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
