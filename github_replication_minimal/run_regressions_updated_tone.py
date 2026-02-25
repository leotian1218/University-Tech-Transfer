from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm


warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "merged_autm_tone_data_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "regression_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def parse_numeric(series: pd.Series, percent: bool = False) -> pd.Series:
    s = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    out = pd.to_numeric(s, errors="coerce")
    if percent:
        out = out / 100.0
    return out


def prepare_panel() -> pd.DataFrame:
    raw = pd.read_csv(INPUT_FILE)

    panel = pd.DataFrame(
        {
            "institution": raw["Institution_std"].astype(str).str.strip(),
            "year": pd.to_numeric(raw["Year"], errors="coerce"),
            "patents": pd.to_numeric(raw["New Pat App Fld"], errors="coerce"),
            "tone_mean": pd.to_numeric(raw["Mean_Tone_Score"], errors="coerce"),
            "tone_median": pd.to_numeric(raw["Median_Tone_Score"], errors="coerce"),
            "royalty": pd.to_numeric(
                raw["royalty_adjusted_numerical"], errors="coerce"
            ).fillna(parse_numeric(raw["Royalty Share"], percent=True)),
            "res_exp_m": pd.to_numeric(raw["Tot Res Exp"], errors="coerce") / 1_000_000,
            "faculty": pd.to_numeric(raw["faculty2_corrected"], errors="coerce").fillna(
                pd.to_numeric(raw["faculty2"], errors="coerce")
            ),
            "tto_age": pd.to_numeric(raw["TLO Age"], errors="coerce"),
            "tto_staff": parse_numeric(raw["Lic FTEs"]),
            "r1": pd.to_numeric(raw["Carnegie R1"], errors="coerce"),
        }
    )

    panel = panel.dropna(subset=["institution", "year"]).copy()
    panel = panel.groupby(["institution", "year"], as_index=False).mean(numeric_only=True)
    panel = panel.sort_values(["institution", "year"]).reset_index(drop=True)

    # If policy text is unavailable for some years, carry policy tone between observed policy years.
    panel["tone_mean_ffill"] = panel.groupby("institution")["tone_mean"].transform(
        lambda s: s.ffill().bfill()
    )

    for base in ["patents", "tone_mean", "tone_median", "tone_mean_ffill"]:
        panel[f"{base}_l1"] = panel.groupby("institution")[base].shift(1)
        panel[f"{base}_l2"] = panel.groupby("institution")[base].shift(2)
        panel[f"{base}_f1"] = panel.groupby("institution")[base].shift(-1)

    panel["y_log_patents"] = np.log1p(panel["patents"])
    panel["log_res_exp"] = np.log(panel["res_exp_m"])
    panel["log_faculty"] = np.log(panel["faculty"])
    panel["log_tto_age"] = np.log1p(panel["tto_age"].clip(lower=0))
    panel["log_tto_staff"] = np.log1p(panel["tto_staff"].clip(lower=0))
    panel["high_staff"] = (
        panel["tto_staff"] >= panel["tto_staff"].median(skipna=True)
    ).astype(float)

    return panel


def fit_fe_ols(formula: str, data: pd.DataFrame, two_way_cluster: bool = False):
    if two_way_cluster:
        groups = pd.DataFrame(
            {
                "institution_id": data["institution"].cat.codes,
                "year": data["year"],
            }
        )
        model = smf.ols(formula, data=data).fit(
            cov_type="cluster", cov_kwds={"groups": groups}
        )
    else:
        model = smf.ols(formula, data=data).fit(
            cov_type="cluster", cov_kwds={"groups": data["institution"]}
        )
    return model


def fit_fe_poisson(formula: str, data: pd.DataFrame):
    return smf.glm(formula, data=data, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": data["institution"]}, maxiter=200
    )


def stars(pvalue: float) -> str:
    if pd.isna(pvalue):
        return ""
    if pvalue < 0.01:
        return "***"
    if pvalue < 0.05:
        return "**"
    if pvalue < 0.1:
        return "*"
    return ""


def extract_terms(model, model_name: str, terms: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for term in terms:
        if term in model.params.index:
            rows.append(
                {
                    "model": model_name,
                    "term": term,
                    "coef": float(model.params[term]),
                    "std_err": float(model.bse[term]),
                    "pvalue": float(model.pvalues[term]),
                    "signif": stars(float(model.pvalues[term])),
                    "nobs": int(model.nobs),
                    "r2": float(getattr(model, "rsquared", np.nan)),
                }
            )
    return pd.DataFrame(rows)


def linear_combo(model, weights: dict[str, float]) -> tuple[float, float, float]:
    params = model.params
    cov = model.cov_params()

    idx = []
    w = []
    for key, wt in weights.items():
        if key in params.index:
            idx.append(key)
            w.append(wt)

    if not idx:
        return np.nan, np.nan, np.nan

    vec = np.array(w)
    beta = params[idx].to_numpy()
    vcov = cov.loc[idx, idx].to_numpy()

    estimate = float(vec @ beta)
    variance = float(vec @ vcov @ vec)
    variance = max(variance, 0.0)
    se = float(np.sqrt(variance))

    if se == 0 or np.isnan(se):
        pval = np.nan
    else:
        z = estimate / se
        pval = float(2 * (1 - norm.cdf(abs(z))))

    return estimate, se, pval


def marginal_effect_table(
    model, tone_sd: float, royalty_points: dict[str, float]
) -> pd.DataFrame:
    rows: list[dict] = []
    tone_var = "tone_mean_l1"

    scenarios = [
        ("Non-R1, Low Staff", 0.0, 0.0),
        ("R1, Low Staff", 1.0, 0.0),
        ("Non-R1, High Staff", 0.0, 1.0),
    ]

    for royalty_label, royalty_val in royalty_points.items():
        for scenario_name, r1_val, hs_val in scenarios:
            weights = {
                tone_var: 1.0,
                f"{tone_var}:royalty": royalty_val,
                f"{tone_var}:r1": r1_val,
                f"{tone_var}:high_staff": hs_val,
            }
            me, se, pval = linear_combo(model, weights)
            log_impact = me * tone_sd
            pct_impact = (np.exp(log_impact) - 1) * 100
            rows.append(
                {
                    "scenario": scenario_name,
                    "royalty_level": royalty_label,
                    "royalty_value": royalty_val,
                    "marginal_dlog_patents_dtone": me,
                    "std_err": se,
                    "pvalue": pval,
                    "signif": stars(pval),
                    "one_sd_tone_effect_pct": pct_impact,
                }
            )

    return pd.DataFrame(rows)


def sample_for_main(panel: pd.DataFrame) -> pd.DataFrame:
    required = [
        "y_log_patents",
        "patents",
        "tone_mean_l1",
        "patents_l1",
        "log_res_exp",
        "log_faculty",
        "log_tto_age",
        "log_tto_staff",
        "royalty",
        "high_staff",
        "r1",
    ]
    out = panel.dropna(subset=required).copy()
    out["year"] = out["year"].astype(int)
    out["institution"] = out["institution"].astype("category")
    return out


def main():
    panel = prepare_panel()
    main_sample = sample_for_main(panel)

    controls = "patents_l1 + log_res_exp + log_faculty + log_tto_age + log_tto_staff"
    fe = "C(institution) + C(year)"

    formula_base = f"y_log_patents ~ tone_mean_l1 + {controls} + {fe}"
    formula_full = (
        "y_log_patents ~ tone_mean_l1 + "
        f"{controls} + royalty + tone_mean_l1:royalty + "
        f"tone_mean_l1:high_staff + tone_mean_l1:r1 + {fe}"
    )

    m_base = fit_fe_ols(formula_base, main_sample)
    m_full = fit_fe_ols(formula_full, main_sample)

    key_terms = [
        "tone_mean_l1",
        "patents_l1",
        "log_res_exp",
        "log_faculty",
        "log_tto_age",
        "log_tto_staff",
        "royalty",
        "tone_mean_l1:royalty",
        "tone_mean_l1:high_staff",
        "tone_mean_l1:r1",
    ]
    table2 = pd.concat(
        [
            extract_terms(m_base, "Table2_Base", key_terms),
            extract_terms(m_full, "Table2_Full", key_terms),
        ],
        ignore_index=True,
    )
    table2.to_csv(OUTPUT_DIR / "table2_main_models.csv", index=False)

    tone_sd = float(main_sample["tone_mean_l1"].std())
    royalty_points = {
        "P35": float(main_sample["royalty"].quantile(0.35)),
        "Mean": float(main_sample["royalty"].mean()),
        "P48": float(main_sample["royalty"].quantile(0.48)),
    }
    table3 = marginal_effect_table(m_full, tone_sd=tone_sd, royalty_points=royalty_points)
    table3.to_csv(OUTPUT_DIR / "table3_marginal_effects.csv", index=False)

    # Table 4 style robustness checks.
    robustness_rows: list[dict] = []

    placebo = main_sample.dropna(subset=["tone_mean_f1"]).copy()
    formula_placebo = (
        "y_log_patents ~ tone_mean_f1 + tone_mean_l1 + "
        f"{controls} + royalty + tone_mean_l1:royalty + "
        f"tone_mean_l1:high_staff + tone_mean_l1:r1 + {fe}"
    )
    m_placebo = fit_fe_ols(formula_placebo, placebo)
    robustness_rows.append(
        {
            "model": "Placebo_FutureTone",
            "main_term": "tone_mean_f1",
            "coef": float(m_placebo.params.get("tone_mean_f1", np.nan)),
            "std_err": float(m_placebo.bse.get("tone_mean_f1", np.nan)),
            "pvalue": float(m_placebo.pvalues.get("tone_mean_f1", np.nan)),
            "signif": stars(float(m_placebo.pvalues.get("tone_mean_f1", np.nan))),
            "nobs": int(m_placebo.nobs),
            "r2_or_pseudo_r2": float(getattr(m_placebo, "rsquared", np.nan)),
        }
    )

    q05, q95 = main_sample["tone_mean_l1"].quantile([0.05, 0.95])
    trimmed = main_sample[
        (main_sample["tone_mean_l1"] >= q05) & (main_sample["tone_mean_l1"] <= q95)
    ].copy()
    m_trim = fit_fe_ols(formula_full, trimmed)
    robustness_rows.append(
        {
            "model": "Trimmed_Tone_5_95",
            "main_term": "tone_mean_l1",
            "coef": float(m_trim.params.get("tone_mean_l1", np.nan)),
            "std_err": float(m_trim.bse.get("tone_mean_l1", np.nan)),
            "pvalue": float(m_trim.pvalues.get("tone_mean_l1", np.nan)),
            "signif": stars(float(m_trim.pvalues.get("tone_mean_l1", np.nan))),
            "nobs": int(m_trim.nobs),
            "r2_or_pseudo_r2": float(getattr(m_trim, "rsquared", np.nan)),
        }
    )

    alt = panel.dropna(
        subset=[
            "y_log_patents",
            "patents",
            "tone_mean_l2",
            "patents_l1",
            "log_res_exp",
            "log_faculty",
            "log_tto_age",
            "log_tto_staff",
            "royalty",
            "high_staff",
            "r1",
        ]
    ).copy()
    alt["year"] = alt["year"].astype(int)
    alt["institution"] = alt["institution"].astype("category")
    formula_alt = (
        "y_log_patents ~ tone_mean_l2 + "
        f"{controls} + royalty + tone_mean_l2:royalty + "
        f"tone_mean_l2:high_staff + tone_mean_l2:r1 + {fe}"
    )
    m_alt = fit_fe_ols(formula_alt, alt)
    robustness_rows.append(
        {
            "model": "AltLag_L2",
            "main_term": "tone_mean_l2",
            "coef": float(m_alt.params.get("tone_mean_l2", np.nan)),
            "std_err": float(m_alt.bse.get("tone_mean_l2", np.nan)),
            "pvalue": float(m_alt.pvalues.get("tone_mean_l2", np.nan)),
            "signif": stars(float(m_alt.pvalues.get("tone_mean_l2", np.nan))),
            "nobs": int(m_alt.nobs),
            "r2_or_pseudo_r2": float(getattr(m_alt, "rsquared", np.nan)),
        }
    )

    formula_poisson = (
        "patents ~ tone_mean_l1 + "
        f"{controls} + royalty + tone_mean_l1:royalty + "
        f"tone_mean_l1:high_staff + tone_mean_l1:r1 + {fe}"
    )
    m_pois = fit_fe_poisson(formula_poisson, main_sample)
    robustness_rows.append(
        {
            "model": "Poisson_FE",
            "main_term": "tone_mean_l1",
            "coef": float(m_pois.params.get("tone_mean_l1", np.nan)),
            "std_err": float(m_pois.bse.get("tone_mean_l1", np.nan)),
            "pvalue": float(m_pois.pvalues.get("tone_mean_l1", np.nan)),
            "signif": stars(float(m_pois.pvalues.get("tone_mean_l1", np.nan))),
            "nobs": int(m_pois.nobs),
            "r2_or_pseudo_r2": float(getattr(m_pois, "prsquared", np.nan)),
        }
    )

    pd.DataFrame(robustness_rows).to_csv(OUTPUT_DIR / "table4_robustness.csv", index=False)

    # Extra checks for reviewer-facing discussion.
    extra_rows: list[dict] = []

    # Baseline with median tone score.
    median_sample = panel.dropna(
        subset=[
            "y_log_patents",
            "tone_median_l1",
            "patents_l1",
            "log_res_exp",
            "log_faculty",
            "log_tto_age",
            "log_tto_staff",
        ]
    ).copy()
    median_sample["year"] = median_sample["year"].astype(int)
    median_sample["institution"] = median_sample["institution"].astype("category")
    formula_median = f"y_log_patents ~ tone_median_l1 + {controls} + {fe}"
    m_median = fit_fe_ols(formula_median, median_sample)
    extra_rows.append(
        {
            "check": "Baseline_MedianTone",
            "term": "tone_median_l1",
            "coef": float(m_median.params["tone_median_l1"]),
            "std_err": float(m_median.bse["tone_median_l1"]),
            "pvalue": float(m_median.pvalues["tone_median_l1"]),
            "signif": stars(float(m_median.pvalues["tone_median_l1"])),
            "nobs": int(m_median.nobs),
        }
    )

    # Two-way clustered SE (institution + year).
    m_twoway = fit_fe_ols(formula_base, main_sample, two_way_cluster=True)
    extra_rows.append(
        {
            "check": "Baseline_TwoWayCluster",
            "term": "tone_mean_l1",
            "coef": float(m_twoway.params["tone_mean_l1"]),
            "std_err": float(m_twoway.bse["tone_mean_l1"]),
            "pvalue": float(m_twoway.pvalues["tone_mean_l1"]),
            "signif": stars(float(m_twoway.pvalues["tone_mean_l1"])),
            "nobs": int(m_twoway.nobs),
        }
    )

    # Excluding COVID years.
    no_covid = main_sample[~main_sample["year"].between(2020, 2023)].copy()
    m_no_covid = fit_fe_ols(formula_base, no_covid)
    extra_rows.append(
        {
            "check": "Baseline_NoCOVID_2020_2023",
            "term": "tone_mean_l1",
            "coef": float(m_no_covid.params["tone_mean_l1"]),
            "std_err": float(m_no_covid.bse["tone_mean_l1"]),
            "pvalue": float(m_no_covid.pvalues["tone_mean_l1"]),
            "signif": stars(float(m_no_covid.pvalues["tone_mean_l1"])),
            "nobs": int(m_no_covid.nobs),
        }
    )

    # Baseline with carried-forward tone score.
    ffill_sample = panel.dropna(
        subset=[
            "y_log_patents",
            "tone_mean_ffill_l1",
            "patents_l1",
            "log_res_exp",
            "log_faculty",
            "log_tto_age",
            "log_tto_staff",
        ]
    ).copy()
    ffill_sample["year"] = ffill_sample["year"].astype(int)
    ffill_sample["institution"] = ffill_sample["institution"].astype("category")
    formula_ffill = f"y_log_patents ~ tone_mean_ffill_l1 + {controls} + {fe}"
    m_ffill = fit_fe_ols(formula_ffill, ffill_sample)
    extra_rows.append(
        {
            "check": "Baseline_FfilledTone",
            "term": "tone_mean_ffill_l1",
            "coef": float(m_ffill.params["tone_mean_ffill_l1"]),
            "std_err": float(m_ffill.bse["tone_mean_ffill_l1"]),
            "pvalue": float(m_ffill.pvalues["tone_mean_ffill_l1"]),
            "signif": stars(float(m_ffill.pvalues["tone_mean_ffill_l1"])),
            "nobs": int(m_ffill.nobs),
        }
    )

    # Baseline by R1 status.
    for label, grp in [("Baseline_R1_only", 1), ("Baseline_NonR1_only", 0)]:
        sub = main_sample[main_sample["r1"] == grp].copy()
        if sub["institution"].nunique() < 10:
            continue
        sub["institution"] = sub["institution"].astype("category")
        model = fit_fe_ols(formula_base, sub)
        extra_rows.append(
            {
                "check": label,
                "term": "tone_mean_l1",
                "coef": float(model.params["tone_mean_l1"]),
                "std_err": float(model.bse["tone_mean_l1"]),
                "pvalue": float(model.pvalues["tone_mean_l1"]),
                "signif": stars(float(model.pvalues["tone_mean_l1"])),
                "nobs": int(model.nobs),
            }
        )

    pd.DataFrame(extra_rows).to_csv(OUTPUT_DIR / "extra_checks.csv", index=False)

    sample_summary = pd.DataFrame(
        [
            {"sample": "Raw_panel_rows", "value": len(panel)},
            {"sample": "Raw_panel_universities", "value": panel["institution"].nunique()},
            {"sample": "Raw_panel_years", "value": panel["year"].nunique()},
            {"sample": "Main_sample_rows", "value": len(main_sample)},
            {
                "sample": "Main_sample_universities",
                "value": main_sample["institution"].nunique(),
            },
            {"sample": "Main_sample_years", "value": main_sample["year"].nunique()},
            {
                "sample": "Tone_sd_main_sample",
                "value": float(main_sample["tone_mean_l1"].std()),
            },
            {
                "sample": "Royalty_mean_main_sample",
                "value": float(main_sample["royalty"].mean()),
            },
            {
                "sample": "Royalty_p35_main_sample",
                "value": float(main_sample["royalty"].quantile(0.35)),
            },
            {
                "sample": "Royalty_p48_main_sample",
                "value": float(main_sample["royalty"].quantile(0.48)),
            },
        ]
    )
    sample_summary.to_csv(OUTPUT_DIR / "sample_summary.csv", index=False)

    print("Saved outputs to:", OUTPUT_DIR)
    print("Main sample:", len(main_sample), "obs,", main_sample["institution"].nunique(), "universities")
    print(
        "Base tone coef:",
        round(float(m_base.params["tone_mean_l1"]), 4),
        "p=",
        round(float(m_base.pvalues["tone_mean_l1"]), 4),
    )
    print(
        "Full tone coef:",
        round(float(m_full.params["tone_mean_l1"]), 4),
        "p=",
        round(float(m_full.pvalues["tone_mean_l1"]), 4),
    )
    print(
        "Full tone x high_staff:",
        round(float(m_full.params["tone_mean_l1:high_staff"]), 4),
        "p=",
        round(float(m_full.pvalues["tone_mean_l1:high_staff"]), 4),
    )


if __name__ == "__main__":
    main()
