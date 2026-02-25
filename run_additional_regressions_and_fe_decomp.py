from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


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


def load_panel() -> pd.DataFrame:
    raw = pd.read_csv(INPUT_FILE)

    panel = pd.DataFrame(
        {
            "institution": raw["Institution_std"].astype(str).str.strip(),
            "year": pd.to_numeric(raw["Year"], errors="coerce"),
            "patents": pd.to_numeric(raw["New Pat App Fld"], errors="coerce"),
            "tone": pd.to_numeric(raw["Mean_Tone_Score"], errors="coerce"),
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
            "private": pd.to_numeric(raw["Private"], errors="coerce"),
            "medschool": pd.to_numeric(raw["MEDSCHOOL"], errors="coerce"),
            "app_per_fac": pd.to_numeric(raw["appperfac2"], errors="coerce"),
            "disc_per_fac": pd.to_numeric(raw["discperfaculty2"], errors="coerce"),
            "inv_disclosures": pd.to_numeric(raw["Inv Dis Rec"], errors="coerce"),
        }
    )

    panel = panel.dropna(subset=["institution", "year"]).copy()
    panel = panel.groupby(["institution", "year"], as_index=False).mean(numeric_only=True)
    panel = panel.sort_values(["institution", "year"]).reset_index(drop=True)

    panel["tone_l1"] = panel.groupby("institution")["tone"].shift(1)
    panel["patents_l1"] = panel.groupby("institution")["patents"].shift(1)
    panel["y_log_pat"] = np.log1p(panel["patents"])
    panel["log_res_exp"] = np.log(panel["res_exp_m"])
    panel["log_faculty"] = np.log(panel["faculty"])
    panel["log_tto_age"] = np.log1p(panel["tto_age"].clip(lower=0))
    panel["log_tto_staff"] = np.log1p(panel["tto_staff"].clip(lower=0))
    panel["high_staff"] = (
        panel["tto_staff"] >= panel["tto_staff"].median(skipna=True)
    ).astype(float)

    return panel


def run_additional_variable_models(panel: pd.DataFrame):
    required = [
        "y_log_pat",
        "tone_l1",
        "patents_l1",
        "log_res_exp",
        "log_faculty",
        "log_tto_age",
        "log_tto_staff",
        "royalty",
        "r1",
        "private",
        "medschool",
    ]
    main = panel.dropna(subset=required).copy()
    main["institution"] = main["institution"].astype("category")
    main["year"] = main["year"].astype(int)

    controls = "patents_l1 + log_res_exp + log_faculty + log_tto_age + log_tto_staff"
    fe = "C(institution) + C(year)"

    models = {
        "A1_baseline_FE": f"y_log_pat ~ tone_l1 + {controls} + {fe}",
        "A2_add_royalty": f"y_log_pat ~ tone_l1 + royalty + {controls} + {fe}",
        "A3_add_institution_types": (
            f"y_log_pat ~ tone_l1 + royalty + private + medschool + {controls} + {fe}"
        ),
        "A4_interact_r1": f"y_log_pat ~ tone_l1 + royalty + tone_l1:r1 + {controls} + {fe}",
        "A5_interact_royalty": (
            f"y_log_pat ~ tone_l1 + royalty + tone_l1:royalty + {controls} + {fe}"
        ),
        "A6_interact_staff": (
            f"y_log_pat ~ tone_l1 + royalty + tone_l1:high_staff + {controls} + {fe}"
        ),
        "A7_full_three_interactions": (
            "y_log_pat ~ tone_l1 + royalty + tone_l1:r1 + tone_l1:royalty + "
            f"tone_l1:high_staff + {controls} + {fe}"
        ),
    }

    key_terms = [
        "tone_l1",
        "royalty",
        "private",
        "medschool",
        "tone_l1:r1",
        "tone_l1:royalty",
        "tone_l1:high_staff",
        "patents_l1",
        "log_res_exp",
        "log_faculty",
        "log_tto_age",
        "log_tto_staff",
    ]
    rows: list[dict] = []
    for name, formula in models.items():
        model = smf.ols(formula, data=main).fit(
            cov_type="cluster", cov_kwds={"groups": main["institution"]}
        )
        for term in key_terms:
            if term in model.params.index:
                rows.append(
                    {
                        "model": name,
                        "term": term,
                        "coef": float(model.params[term]),
                        "se": float(model.bse[term]),
                        "p": float(model.pvalues[term]),
                        "nobs": int(model.nobs),
                        "r2": float(model.rsquared),
                    }
                )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "additional_variable_models.csv", index=False)


def run_additional_outcomes(panel: pd.DataFrame):
    controls = "patents_l1 + log_res_exp + log_faculty + log_tto_age + log_tto_staff"
    fe = "C(institution) + C(year)"
    rows: list[dict] = []

    # B1
    b1 = panel.dropna(
        subset=[
            "inv_disclosures",
            "tone_l1",
            "patents_l1",
            "log_res_exp",
            "log_faculty",
            "log_tto_age",
            "log_tto_staff",
        ]
    ).copy()
    b1["y"] = np.log1p(b1["inv_disclosures"])
    b1["institution"] = b1["institution"].astype("category")
    b1["year"] = b1["year"].astype(int)
    m1 = smf.ols(f"y ~ tone_l1 + {controls} + {fe}", data=b1).fit(
        cov_type="cluster", cov_kwds={"groups": b1["institution"]}
    )
    rows.append(
        {
            "model": "B1_log_inv_disclosures",
            "term": "tone_l1",
            "coef": float(m1.params["tone_l1"]),
            "se": float(m1.bse["tone_l1"]),
            "p": float(m1.pvalues["tone_l1"]),
            "nobs": int(m1.nobs),
            "r2": float(m1.rsquared),
        }
    )

    # B2
    b2 = panel.dropna(
        subset=[
            "app_per_fac",
            "tone_l1",
            "patents_l1",
            "log_res_exp",
            "log_faculty",
            "log_tto_age",
            "log_tto_staff",
        ]
    ).copy()
    b2["y"] = np.log1p(b2["app_per_fac"])
    b2["institution"] = b2["institution"].astype("category")
    b2["year"] = b2["year"].astype(int)
    m2 = smf.ols(f"y ~ tone_l1 + {controls} + {fe}", data=b2).fit(
        cov_type="cluster", cov_kwds={"groups": b2["institution"]}
    )
    rows.append(
        {
            "model": "B2_log_app_per_faculty",
            "term": "tone_l1",
            "coef": float(m2.params["tone_l1"]),
            "se": float(m2.bse["tone_l1"]),
            "p": float(m2.pvalues["tone_l1"]),
            "nobs": int(m2.nobs),
            "r2": float(m2.rsquared),
        }
    )

    # B3
    b3 = panel.dropna(
        subset=[
            "disc_per_fac",
            "tone_l1",
            "patents_l1",
            "log_res_exp",
            "log_faculty",
            "log_tto_age",
            "log_tto_staff",
        ]
    ).copy()
    b3["y"] = np.log1p(b3["disc_per_fac"])
    b3["institution"] = b3["institution"].astype("category")
    b3["year"] = b3["year"].astype(int)
    m3 = smf.ols(f"y ~ tone_l1 + {controls} + {fe}", data=b3).fit(
        cov_type="cluster", cov_kwds={"groups": b3["institution"]}
    )
    rows.append(
        {
            "model": "B3_log_disc_per_faculty",
            "term": "tone_l1",
            "coef": float(m3.params["tone_l1"]),
            "se": float(m3.bse["tone_l1"]),
            "p": float(m3.pvalues["tone_l1"]),
            "nobs": int(m3.nobs),
            "r2": float(m3.rsquared),
        }
    )

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "additional_outcomes.csv", index=False)


def run_within_between_decomposition(panel: pd.DataFrame):
    wm = panel.copy()
    wm["tone_l1_mean_i"] = wm.groupby("institution")["tone_l1"].transform("mean")
    wm["tone_l1_within"] = wm["tone_l1"] - wm["tone_l1_mean_i"]
    wm["patents_l1_mean_i"] = wm.groupby("institution")["patents_l1"].transform("mean")
    wm["patents_l1_within"] = wm["patents_l1"] - wm["patents_l1_mean_i"]

    for v in ["log_res_exp", "log_faculty", "log_tto_age", "log_tto_staff"]:
        wm[f"{v}_mean_i"] = wm.groupby("institution")[v].transform("mean")
        wm[f"{v}_within"] = wm[v] - wm[f"{v}_mean_i"]

    required = [
        "y_log_pat",
        "tone_l1_within",
        "tone_l1_mean_i",
        "patents_l1_within",
        "patents_l1_mean_i",
        "log_res_exp_within",
        "log_res_exp_mean_i",
        "log_faculty_within",
        "log_faculty_mean_i",
        "log_tto_age_within",
        "log_tto_age_mean_i",
        "log_tto_staff_within",
        "log_tto_staff_mean_i",
    ]
    wm_s = wm.dropna(subset=required).copy()
    wm_s["institution"] = wm_s["institution"].astype("category")
    wm_s["year"] = wm_s["year"].astype(int)

    f_mundlak = (
        "y_log_pat ~ tone_l1_within + tone_l1_mean_i + "
        "patents_l1_within + patents_l1_mean_i + "
        "log_res_exp_within + log_res_exp_mean_i + "
        "log_faculty_within + log_faculty_mean_i + "
        "log_tto_age_within + log_tto_age_mean_i + "
        "log_tto_staff_within + log_tto_staff_mean_i + C(year)"
    )
    m_mundlak = smf.ols(f_mundlak, data=wm_s).fit(
        cov_type="cluster", cov_kwds={"groups": wm_s["institution"]}
    )

    rows: list[dict] = []
    for term in [
        "tone_l1_within",
        "tone_l1_mean_i",
        "patents_l1_within",
        "patents_l1_mean_i",
    ]:
        rows.append(
            {
                "term": term,
                "coef": float(m_mundlak.params[term]),
                "se": float(m_mundlak.bse[term]),
                "p": float(m_mundlak.pvalues[term]),
                "nobs": int(m_mundlak.nobs),
                "r2": float(m_mundlak.rsquared),
            }
        )

    between = wm_s.groupby("institution", as_index=False).agg(
        {
            "y_log_pat": "mean",
            "tone_l1": "mean",
            "patents_l1": "mean",
            "log_res_exp": "mean",
            "log_faculty": "mean",
            "log_tto_age": "mean",
            "log_tto_staff": "mean",
        }
    )
    m_between = smf.ols(
        "y_log_pat ~ tone_l1 + patents_l1 + log_res_exp + log_faculty + log_tto_age + log_tto_staff",
        data=between,
    ).fit()
    rows.append(
        {
            "term": "between_model_tone_l1",
            "coef": float(m_between.params["tone_l1"]),
            "se": float(m_between.bse["tone_l1"]),
            "p": float(m_between.pvalues["tone_l1"]),
            "nobs": int(m_between.nobs),
            "r2": float(m_between.rsquared),
        }
    )

    fe_model = smf.ols(
        "y_log_pat ~ tone_l1 + patents_l1 + log_res_exp + log_faculty + log_tto_age + log_tto_staff + C(institution) + C(year)",
        data=wm_s,
    ).fit(cov_type="cluster", cov_kwds={"groups": wm_s["institution"]})
    rows.append(
        {
            "term": "two_way_FE_tone_l1",
            "coef": float(fe_model.params["tone_l1"]),
            "se": float(fe_model.bse["tone_l1"]),
            "p": float(fe_model.pvalues["tone_l1"]),
            "nobs": int(fe_model.nobs),
            "r2": float(fe_model.rsquared),
        }
    )

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "fe_within_between_decomposition.csv", index=False
    )


def run_across_models(panel: pd.DataFrame):
    required = [
        "y_log_pat",
        "tone_l1",
        "patents_l1",
        "log_res_exp",
        "log_faculty",
        "log_tto_age",
        "log_tto_staff",
        "royalty",
        "private",
        "medschool",
        "r1",
    ]
    s = panel.dropna(subset=required).copy()
    s["institution"] = s["institution"].astype("category")
    s["year"] = s["year"].astype(int)

    m_year = smf.ols(
        "y_log_pat ~ tone_l1 + patents_l1 + log_res_exp + log_faculty + log_tto_age + "
        "log_tto_staff + royalty + private + medschool + r1 + C(year)",
        data=s,
    ).fit(cov_type="cluster", cov_kwds={"groups": s["institution"]})

    m_pooled = smf.ols(
        "y_log_pat ~ tone_l1 + patents_l1 + log_res_exp + log_faculty + log_tto_age + "
        "log_tto_staff + royalty + private + medschool + r1",
        data=s,
    ).fit(cov_type="cluster", cov_kwds={"groups": s["institution"]})

    rows: list[dict] = []
    for name, model in [("C1_yearFE_only", m_year), ("C2_noFE_pooled", m_pooled)]:
        for term in [
            "tone_l1",
            "patents_l1",
            "log_res_exp",
            "log_faculty",
            "log_tto_age",
            "log_tto_staff",
            "royalty",
            "private",
            "medschool",
            "r1",
        ]:
            if term in model.params.index:
                rows.append(
                    {
                        "model": name,
                        "term": term,
                        "coef": float(model.params[term]),
                        "se": float(model.bse[term]),
                        "p": float(model.pvalues[term]),
                        "nobs": int(model.nobs),
                        "r2": float(model.rsquared),
                    }
                )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "across_models_year_or_pooled.csv", index=False)


def run_variation_diagnostics(panel: pd.DataFrame):
    rows: list[dict] = []
    for var in ["private", "medschool", "r1"]:
        g = panel.dropna(subset=[var]).groupby("institution")[var]
        nuniv = g.ngroups
        varying = int((g.nunique() > 1).sum())
        rows.append(
            {
                "variable": var,
                "universities_with_data": nuniv,
                "universities_within_varying": varying,
                "within_variation_share": varying / nuniv if nuniv else np.nan,
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "within_variation_diagnostics.csv", index=False)


def main():
    panel = load_panel()
    run_additional_variable_models(panel)
    run_additional_outcomes(panel)
    run_within_between_decomposition(panel)
    run_across_models(panel)
    run_variation_diagnostics(panel)
    print("Saved additional regressions and FE decomposition outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
