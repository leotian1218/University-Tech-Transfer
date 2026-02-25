from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "merged_autm_tone_data_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "regression_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE)
    panel = pd.DataFrame(
        {
            "institution": df["Institution_std"].astype(str).str.strip(),
            "year": pd.to_numeric(df["Year"], errors="coerce"),
            "patents": pd.to_numeric(df["New Pat App Fld"], errors="coerce"),
            "tone": pd.to_numeric(df["Mean_Tone_Score"], errors="coerce"),
            "res_exp_m": pd.to_numeric(df["Tot Res Exp"], errors="coerce") / 1_000_000,
            "faculty": pd.to_numeric(df["faculty2_corrected"], errors="coerce").fillna(
                pd.to_numeric(df["faculty2"], errors="coerce")
            ),
            "tto_age": pd.to_numeric(df["TLO Age"], errors="coerce"),
            "tto_staff": pd.to_numeric(df["Lic FTEs"], errors="coerce"),
        }
    )
    panel = panel.dropna(subset=["institution", "year"]).copy()
    panel = panel.groupby(["institution", "year"], as_index=False).mean(numeric_only=True)
    panel = panel.sort_values(["institution", "year"]).reset_index(drop=True)

    panel["tone_l1"] = panel.groupby("institution")["tone"].shift(1)
    panel["pat_l1"] = panel.groupby("institution")["patents"].shift(1)
    panel["y_log_pat"] = np.log1p(panel["patents"])
    panel["year"] = panel["year"].astype(int)
    panel["year_c"] = panel["year"] - panel["year"].mean()

    panel["log_res_exp"] = np.log(panel["res_exp_m"])
    panel["log_faculty"] = np.log(panel["faculty"])
    panel["log_tto_age"] = np.log1p(panel["tto_age"].clip(lower=0))
    panel["log_tto_staff"] = np.log1p(panel["tto_staff"].clip(lower=0))

    panel["res_exp_b"] = panel["res_exp_m"] / 1000.0
    panel["faculty_1k"] = panel["faculty"] / 1000.0
    panel["tto_age_10y"] = panel["tto_age"] / 10.0
    panel["tto_staff_10"] = panel["tto_staff"] / 10.0

    return panel


def run_nolag_and_functional_forms(panel: pd.DataFrame):
    rows: list[dict] = []

    def run(name: str, formula: str, data: pd.DataFrame, target: str):
        model = smf.ols(formula, data=data).fit(
            cov_type="cluster", cov_kwds={"groups": data["institution"]}
        )
        rows.append(
            {
                "model": name,
                "nobs": int(model.nobs),
                "r2": float(model.rsquared),
                "term": target,
                "coef": float(model.params.get(target, np.nan)),
                "se": float(model.bse.get(target, np.nan)),
                "p": float(model.pvalues.get(target, np.nan)),
            }
        )

    # lag + log controls
    d_lag_log = panel.dropna(
        subset=[
            "y_log_pat",
            "tone_l1",
            "pat_l1",
            "log_res_exp",
            "log_faculty",
            "log_tto_age",
            "log_tto_staff",
        ]
    ).copy()
    d_lag_log["institution"] = d_lag_log["institution"].astype("category")

    run(
        "M1_lag_log_controls",
        "y_log_pat ~ tone_l1 + pat_l1 + log_res_exp + log_faculty + log_tto_age + log_tto_staff + C(institution) + C(year)",
        d_lag_log,
        "tone_l1",
    )

    # no-lag + log controls
    d_nolag_log = panel.dropna(
        subset=[
            "y_log_pat",
            "tone",
            "pat_l1",
            "log_res_exp",
            "log_faculty",
            "log_tto_age",
            "log_tto_staff",
        ]
    ).copy()
    d_nolag_log["institution"] = d_nolag_log["institution"].astype("category")

    run(
        "M2_noLag_log_controls",
        "y_log_pat ~ tone + pat_l1 + log_res_exp + log_faculty + log_tto_age + log_tto_staff + C(institution) + C(year)",
        d_nolag_log,
        "tone",
    )

    # lag + level controls
    d_lag_lvl = panel.dropna(
        subset=[
            "y_log_pat",
            "tone_l1",
            "pat_l1",
            "res_exp_b",
            "faculty_1k",
            "tto_age_10y",
            "tto_staff_10",
        ]
    ).copy()
    d_lag_lvl["institution"] = d_lag_lvl["institution"].astype("category")

    run(
        "M3_lag_level_controls",
        "y_log_pat ~ tone_l1 + pat_l1 + res_exp_b + faculty_1k + tto_age_10y + tto_staff_10 + C(institution) + C(year)",
        d_lag_lvl,
        "tone_l1",
    )

    # no-lag + level controls
    d_nolag_lvl = panel.dropna(
        subset=[
            "y_log_pat",
            "tone",
            "pat_l1",
            "res_exp_b",
            "faculty_1k",
            "tto_age_10y",
            "tto_staff_10",
        ]
    ).copy()
    d_nolag_lvl["institution"] = d_nolag_lvl["institution"].astype("category")

    run(
        "M4_noLag_level_controls",
        "y_log_pat ~ tone + pat_l1 + res_exp_b + faculty_1k + tto_age_10y + tto_staff_10 + C(institution) + C(year)",
        d_nolag_lvl,
        "tone",
    )

    # mixed controls
    d_lag_mix = panel.dropna(
        subset=[
            "y_log_pat",
            "tone_l1",
            "pat_l1",
            "log_res_exp",
            "faculty_1k",
            "tto_age_10y",
            "tto_staff_10",
        ]
    ).copy()
    d_lag_mix["institution"] = d_lag_mix["institution"].astype("category")

    run(
        "M5_lag_mixed_controls",
        "y_log_pat ~ tone_l1 + pat_l1 + log_res_exp + faculty_1k + tto_age_10y + tto_staff_10 + C(institution) + C(year)",
        d_lag_mix,
        "tone_l1",
    )

    d_nolag_mix = panel.dropna(
        subset=[
            "y_log_pat",
            "tone",
            "pat_l1",
            "log_res_exp",
            "faculty_1k",
            "tto_age_10y",
            "tto_staff_10",
        ]
    ).copy()
    d_nolag_mix["institution"] = d_nolag_mix["institution"].astype("category")

    run(
        "M6_noLag_mixed_controls",
        "y_log_pat ~ tone + pat_l1 + log_res_exp + faculty_1k + tto_age_10y + tto_staff_10 + C(institution) + C(year)",
        d_nolag_mix,
        "tone",
    )

    # no-lag with institution-specific trend
    run(
        "M7_noLag_log_controls_plus_instTrend",
        "y_log_pat ~ tone + pat_l1 + log_res_exp + log_faculty + log_tto_age + log_tto_staff + C(institution) + C(year) + C(institution):year_c",
        d_nolag_log,
        "tone",
    )

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "no_lag_and_log_vs_level_comparison.csv", index=False
    )


def run_control_shape_diagnostics(panel: pd.DataFrame):
    shape_rows: list[dict] = []
    vars_ = ["res_exp_m", "faculty", "tto_age", "tto_staff"]
    for var in vars_:
        s = pd.to_numeric(panel[var], errors="coerce").dropna()
        if s.empty:
            continue
        shape_rows.append(
            {
                "var": var,
                "n": int(s.shape[0]),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "p1": float(s.quantile(0.01)),
                "p10": float(s.quantile(0.10)),
                "p50": float(s.quantile(0.50)),
                "p90": float(s.quantile(0.90)),
                "p99": float(s.quantile(0.99)),
                "max": float(s.max()),
                "skew": float(s.skew()),
            }
        )
    pd.DataFrame(shape_rows).to_csv(OUTPUT_DIR / "control_variable_shape_stats.csv", index=False)

    panel2 = panel.sort_values(["institution", "year"]).copy()
    for var in vars_:
        panel2[f"d_{var}"] = panel2.groupby("institution")[var].diff()
        panel2[f"gpct_{var}"] = panel2.groupby("institution")[var].pct_change()

    change_rows: list[dict] = []
    for var in vars_:
        d = pd.to_numeric(panel2[f"d_{var}"], errors="coerce").dropna()
        g = (
            pd.to_numeric(panel2[f"gpct_{var}"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if not d.empty:
            change_rows.append(
                {
                    "var": var,
                    "metric": "delta_level",
                    "p50": float(d.quantile(0.5)),
                    "p90": float(d.quantile(0.9)),
                    "p99": float(d.quantile(0.99)),
                    "share_zero": float((d == 0).mean()),
                    "n": int(len(d)),
                }
            )
        if not g.empty:
            change_rows.append(
                {
                    "var": var,
                    "metric": "pct_change",
                    "p50": float(g.quantile(0.5)),
                    "p90": float(g.quantile(0.9)),
                    "p99": float(g.quantile(0.99)),
                    "share_zero": float((g == 0).mean()),
                    "n": int(len(g)),
                }
            )
    pd.DataFrame(change_rows).to_csv(
        OUTPUT_DIR / "control_variable_within_change_stats.csv", index=False
    )


def run_control_within_variation(panel: pd.DataFrame):
    rows: list[dict] = []
    for var in ["res_exp_m", "faculty", "tto_age", "tto_staff"]:
        g = panel.dropna(subset=[var]).groupby("institution")[var]
        n_inst = g.ngroups
        varying = int((g.nunique() > 1).sum())

        tmp = panel[["institution", "year", var]].dropna().sort_values(["institution", "year"])
        tmp["d"] = tmp.groupby("institution")[var].diff()
        d = tmp["d"].dropna()

        rows.append(
            {
                "var": var,
                "institutions_with_data": n_inst,
                "institutions_within_varying": varying,
                "within_varying_share": varying / n_inst if n_inst else np.nan,
                "year_pairs": int(len(d)),
                "share_zero_delta": float((d == 0).mean()) if len(d) else np.nan,
                "median_abs_delta": float(d.abs().median()) if len(d) else np.nan,
                "p90_abs_delta": float(d.abs().quantile(0.9)) if len(d) else np.nan,
            }
        )

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "control_within_variation_diagnostics.csv", index=False
    )


def run_drop_tests(panel: pd.DataFrame):
    d = panel.dropna(
        subset=[
            "y_log_pat",
            "tone_l1",
            "pat_l1",
            "log_res_exp",
            "log_faculty",
            "log_tto_age",
            "log_tto_staff",
        ]
    ).copy()
    d["institution"] = d["institution"].astype("category")

    specs = {
        "full_controls": (
            "y_log_pat ~ tone_l1 + pat_l1 + log_res + log_fac + log_age + log_staff + "
            "C(institution) + C(year)"
        ),
        "drop_age": (
            "y_log_pat ~ tone_l1 + pat_l1 + log_res + log_fac + log_staff + "
            "C(institution) + C(year)"
        ),
        "drop_faculty": (
            "y_log_pat ~ tone_l1 + pat_l1 + log_res + log_age + log_staff + "
            "C(institution) + C(year)"
        ),
        "drop_age_faculty": (
            "y_log_pat ~ tone_l1 + pat_l1 + log_res + log_staff + C(institution) + C(year)"
        ),
        "drop_age_faculty_staff": (
            "y_log_pat ~ tone_l1 + pat_l1 + log_res + C(institution) + C(year)"
        ),
    }

    # local aliases
    d = d.rename(
        columns={
            "log_res_exp": "log_res",
            "log_faculty": "log_fac",
            "log_tto_age": "log_age",
            "log_tto_staff": "log_staff",
        }
    )

    rows: list[dict] = []
    for name, formula in specs.items():
        model = smf.ols(formula, data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["institution"]}
        )
        rows.append(
            {
                "spec": name,
                "coef_tone_l1": float(model.params["tone_l1"]),
                "se": float(model.bse["tone_l1"]),
                "p": float(model.pvalues["tone_l1"]),
                "nobs": int(model.nobs),
                "r2": float(model.rsquared),
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "control_necessity_drop_tests.csv", index=False)


def main():
    panel = load_panel()
    run_nolag_and_functional_forms(panel)
    run_control_shape_diagnostics(panel)
    run_control_within_variation(panel)
    run_drop_tests(panel)
    print("Saved no-lag and functional-form diagnostic outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
