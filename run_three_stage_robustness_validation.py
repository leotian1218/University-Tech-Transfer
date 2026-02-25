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
    raw = pd.read_csv(INPUT_FILE)
    panel = pd.DataFrame(
        {
            "institution": raw["Institution_std"].astype(str).str.strip(),
            "year": pd.to_numeric(raw["Year"], errors="coerce"),
            "patents": pd.to_numeric(raw["New Pat App Fld"], errors="coerce"),
            "tone": pd.to_numeric(raw["Mean_Tone_Score"], errors="coerce"),
            "res_exp_m": pd.to_numeric(raw["Tot Res Exp"], errors="coerce") / 1_000_000,
            "faculty": pd.to_numeric(raw["faculty2_corrected"], errors="coerce").fillna(
                pd.to_numeric(raw["faculty2"], errors="coerce")
            ),
            "tto_age": pd.to_numeric(raw["TLO Age"], errors="coerce"),
            "tto_staff": pd.to_numeric(raw["Lic FTEs"], errors="coerce"),
            "r1": pd.to_numeric(raw["Carnegie R1"], errors="coerce"),
            "inv_disclosures": pd.to_numeric(raw["Inv Dis Rec"], errors="coerce"),
        }
    )
    panel = panel.dropna(subset=["institution", "year"]).copy()
    panel = panel.groupby(["institution", "year"], as_index=False).mean(numeric_only=True)
    panel = panel.sort_values(["institution", "year"]).reset_index(drop=True)

    panel["tone_l1"] = panel.groupby("institution")["tone"].shift(1)
    panel["tone_f1"] = panel.groupby("institution")["tone"].shift(-1)
    panel["pat_l1"] = panel.groupby("institution")["patents"].shift(1)
    panel["y_log_pat"] = np.log1p(panel["patents"])
    panel["y_log_disc"] = np.log1p(panel["inv_disclosures"])

    panel["log_res_exp"] = np.log(panel["res_exp_m"])
    panel["log_faculty"] = np.log(panel["faculty"])
    panel["log_tto_age"] = np.log1p(panel["tto_age"].clip(lower=0))
    panel["log_tto_staff"] = np.log1p(panel["tto_staff"].clip(lower=0))

    panel["res_exp_b"] = panel["res_exp_m"] / 1000.0
    panel["faculty_1k"] = panel["faculty"] / 1000.0
    panel["tto_age_10y"] = panel["tto_age"] / 10.0
    panel["tto_staff_10"] = panel["tto_staff"] / 10.0

    panel["year"] = panel["year"].astype(int)
    panel["year_c"] = panel["year"] - panel["year"].mean()
    return panel


def fit_cluster(formula: str, data: pd.DataFrame, two_way: bool = False):
    if two_way:
        groups = pd.DataFrame(
            {"g1": data["institution"].cat.codes, "g2": data["year"]}
        )
        return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": groups})
    return smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["institution"]}
    )


def add_row(
    rows: list[dict],
    section: str,
    model_name: str,
    model,
    term: str,
):
    rows.append(
        {
            "section": section,
            "model": model_name,
            "term": term,
            "coef": float(model.params.get(term, np.nan)),
            "se": float(model.bse.get(term, np.nan)),
            "p": float(model.pvalues.get(term, np.nan)),
            "nobs": int(model.nobs),
            "r2": float(getattr(model, "rsquared", np.nan)),
        }
    )


def run_stage_models(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    # Stage 1: No lag on tone, no-log controls.
    s1 = panel.dropna(
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
    s1["institution"] = s1["institution"].astype("category")
    f1 = (
        "y_log_pat ~ tone + pat_l1 + res_exp_b + faculty_1k + tto_age_10y + "
        "tto_staff_10 + C(institution) + C(year)"
    )
    m1 = fit_cluster(f1, s1)
    add_row(rows, "Stage1", "S1_NoLag_NoLogControls_FE", m1, "tone")

    # Stage 2: Lagged tone + log controls.
    s2 = panel.dropna(
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
    s2["institution"] = s2["institution"].astype("category")
    f2 = (
        "y_log_pat ~ tone_l1 + pat_l1 + log_res_exp + log_faculty + log_tto_age + "
        "log_tto_staff + C(institution) + C(year)"
    )
    m2 = fit_cluster(f2, s2)
    add_row(rows, "Stage2", "S2_Lag_LogControls_FE", m2, "tone_l1")

    # Stage 3a: Strict identification with institution-specific linear trends.
    s3a = panel.dropna(
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
    s3a["institution"] = s3a["institution"].astype("category")
    f3a = (
        "y_log_pat ~ tone + pat_l1 + log_res_exp + log_faculty + log_tto_age + "
        "log_tto_staff + C(institution) + C(year) + C(institution):year_c"
    )
    m3a = fit_cluster(f3a, s3a)
    add_row(rows, "Stage3", "S3A_NoLag_FE_InstLinearTrend", m3a, "tone")

    # Stage 3b: Strict identification via first difference.
    fd = s2.sort_values(["institution", "year"]).copy()
    for v in [
        "y_log_pat",
        "tone_l1",
        "pat_l1",
        "log_res_exp",
        "log_faculty",
        "log_tto_age",
        "log_tto_staff",
    ]:
        fd[f"d_{v}"] = fd.groupby("institution")[v].diff()
    fd = fd.dropna(
        subset=[
            "d_y_log_pat",
            "d_tone_l1",
            "d_pat_l1",
            "d_log_res_exp",
            "d_log_faculty",
            "d_log_tto_age",
            "d_log_tto_staff",
        ]
    ).copy()
    f3b = (
        "d_y_log_pat ~ d_tone_l1 + d_pat_l1 + d_log_res_exp + d_log_faculty + "
        "d_log_tto_age + d_log_tto_staff + C(year)"
    )
    m3b = fit_cluster(f3b, fd)
    add_row(rows, "Stage3", "S3B_FirstDifference_YearFE", m3b, "d_tone_l1")

    # Stage 3c: 3-year cumulative growth in patents (de-trended growth-style outcome).
    s3c = panel.dropna(
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
    s3c = s3c.sort_values(["institution", "year"])
    s3c["y_log_pat_f3"] = s3c.groupby("institution")["y_log_pat"].shift(-3)
    s3c["g3_y_log_pat"] = s3c["y_log_pat_f3"] - s3c["y_log_pat"]
    s3c = s3c.dropna(
        subset=[
            "g3_y_log_pat",
            "tone_l1",
            "pat_l1",
            "log_res_exp",
            "log_faculty",
            "log_tto_age",
            "log_tto_staff",
        ]
    ).copy()
    s3c["institution"] = s3c["institution"].astype("category")
    f3c = (
        "g3_y_log_pat ~ tone_l1 + pat_l1 + log_res_exp + log_faculty + log_tto_age + "
        "log_tto_staff + C(institution) + C(year)"
    )
    m3c = fit_cluster(f3c, s3c)
    add_row(rows, "Stage3", "S3C_ThreeYearCumGrowth_FE", m3c, "tone_l1")

    stage_df = pd.DataFrame(rows)
    stage_df.to_csv(OUTPUT_DIR / "three_stage_model_results.csv", index=False)
    return stage_df


def run_robustness_and_validity(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    # Base samples
    s1 = panel.dropna(
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
    s1["institution"] = s1["institution"].astype("category")
    s2 = panel.dropna(
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
    s2["institution"] = s2["institution"].astype("category")
    s2_placebo = panel.dropna(
        subset=[
            "y_log_pat",
            "tone_l1",
            "tone_f1",
            "pat_l1",
            "log_res_exp",
            "log_faculty",
            "log_tto_age",
            "log_tto_staff",
        ]
    ).copy()
    s2_placebo["institution"] = s2_placebo["institution"].astype("category")

    # R1: two-way clustered SE for stage1
    m_r1 = fit_cluster(
        "y_log_pat ~ tone + pat_l1 + res_exp_b + faculty_1k + tto_age_10y + tto_staff_10 + C(institution) + C(year)",
        s1,
        two_way=True,
    )
    add_row(rows, "Robustness", "R1_Stage1_TwoWayCluster", m_r1, "tone")

    # R2: two-way clustered SE for stage2
    m_r2 = fit_cluster(
        "y_log_pat ~ tone_l1 + pat_l1 + log_res_exp + log_faculty + log_tto_age + log_tto_staff + C(institution) + C(year)",
        s2,
        two_way=True,
    )
    add_row(rows, "Robustness", "R2_Stage2_TwoWayCluster", m_r2, "tone_l1")

    # R3: placebo future tone in stage2
    m_r3 = fit_cluster(
        "y_log_pat ~ tone_f1 + tone_l1 + pat_l1 + log_res_exp + log_faculty + log_tto_age + log_tto_staff + C(institution) + C(year)",
        s2_placebo,
    )
    add_row(rows, "Validity", "V1_Placebo_FutureTone", m_r3, "tone_f1")
    add_row(rows, "Validity", "V1b_CurrentTone_ControllingFuture", m_r3, "tone_l1")

    # R4: winsorized stage2 (1%-99%)
    s2w = s2.copy()
    for v in ["y_log_pat", "tone_l1"]:
        lo, hi = s2w[v].quantile([0.01, 0.99])
        s2w[v] = s2w[v].clip(lo, hi)
    m_r4 = fit_cluster(
        "y_log_pat ~ tone_l1 + pat_l1 + log_res_exp + log_faculty + log_tto_age + log_tto_staff + C(institution) + C(year)",
        s2w,
    )
    add_row(rows, "Robustness", "R3_Winsorized_1_99_Stage2", m_r4, "tone_l1")

    # R5: balanced-ish panel institutions (>=15 tone observations)
    tone_counts = (
        panel.dropna(subset=["tone"])
        .groupby("institution")
        .size()
        .rename("n_tone")
        .reset_index()
    )
    keep_inst = set(tone_counts.loc[tone_counts["n_tone"] >= 15, "institution"])
    s2b = s2[s2["institution"].astype(str).isin(keep_inst)].copy()
    if not s2b.empty:
        m_r5 = fit_cluster(
            "y_log_pat ~ tone_l1 + pat_l1 + log_res_exp + log_faculty + log_tto_age + log_tto_staff + C(institution) + C(year)",
            s2b,
        )
        add_row(rows, "Robustness", "R4_BalancedishPanel_15plus", m_r5, "tone_l1")

    # R6: R1-only and non-R1 split (stage1 no-lag spec)
    for grp, tag in [(1, "R1"), (0, "NonR1")]:
        sg = s1[s1["r1"] == grp].copy()
        if sg["institution"].nunique() < 10:
            continue
        m = fit_cluster(
            "y_log_pat ~ tone + pat_l1 + res_exp_b + faculty_1k + tto_age_10y + tto_staff_10 + C(institution) + C(year)",
            sg,
        )
        add_row(rows, "Heterogeneity", f"H1_Stage1_{tag}", m, "tone")

    # V2: reverse-direction check (does lagged patents predict tone?)
    s_rev = panel.dropna(
        subset=[
            "tone",
            "pat_l1",
            "log_res_exp",
            "log_faculty",
            "log_tto_age",
            "log_tto_staff",
        ]
    ).copy()
    s_rev["institution"] = s_rev["institution"].astype("category")
    m_v2 = fit_cluster(
        "tone ~ pat_l1 + log_res_exp + log_faculty + log_tto_age + log_tto_staff + C(institution) + C(year)",
        s_rev,
    )
    add_row(rows, "Validity", "V2_ReverseDirection_ToneOnPastPatents", m_v2, "pat_l1")

    # V3: alternative outcome validity (innovation pipeline)
    s_alt = panel.dropna(
        subset=[
            "y_log_disc",
            "tone_l1",
            "pat_l1",
            "log_res_exp",
            "log_faculty",
            "log_tto_age",
            "log_tto_staff",
        ]
    ).copy()
    s_alt["institution"] = s_alt["institution"].astype("category")
    m_v3 = fit_cluster(
        "y_log_disc ~ tone_l1 + pat_l1 + log_res_exp + log_faculty + log_tto_age + log_tto_staff + C(institution) + C(year)",
        s_alt,
    )
    add_row(rows, "Validity", "V3_AltOutcome_LogInvDisclosures", m_v3, "tone_l1")

    # R7: leave-one-university-out influence check (Stage1 and Stage2)
    def loo_summary(data: pd.DataFrame, formula: str, target: str, label: str):
        coefs = []
        pvals = []
        institutions = data["institution"].astype(str).unique().tolist()
        for inst in institutions:
            d = data[data["institution"].astype(str) != inst]
            if d["institution"].nunique() < 20:
                continue
            try:
                m = fit_cluster(formula, d)
                coefs.append(float(m.params.get(target, np.nan)))
                pvals.append(float(m.pvalues.get(target, np.nan)))
            except Exception:
                continue
        coefs = pd.Series(coefs).dropna()
        pvals = pd.Series(pvals).dropna()
        rows.append(
            {
                "section": "Robustness",
                "model": f"{label}_LOO_coef_mean",
                "term": target,
                "coef": float(coefs.mean()) if len(coefs) else np.nan,
                "se": np.nan,
                "p": np.nan,
                "nobs": int(len(coefs)),
                "r2": np.nan,
            }
        )
        rows.append(
            {
                "section": "Robustness",
                "model": f"{label}_LOO_coef_min",
                "term": target,
                "coef": float(coefs.min()) if len(coefs) else np.nan,
                "se": np.nan,
                "p": np.nan,
                "nobs": int(len(coefs)),
                "r2": np.nan,
            }
        )
        rows.append(
            {
                "section": "Robustness",
                "model": f"{label}_LOO_coef_max",
                "term": target,
                "coef": float(coefs.max()) if len(coefs) else np.nan,
                "se": np.nan,
                "p": np.nan,
                "nobs": int(len(coefs)),
                "r2": np.nan,
            }
        )
        rows.append(
            {
                "section": "Robustness",
                "model": f"{label}_LOO_share_p_lt_0.05",
                "term": target,
                "coef": float((pvals < 0.05).mean()) if len(pvals) else np.nan,
                "se": np.nan,
                "p": np.nan,
                "nobs": int(len(pvals)),
                "r2": np.nan,
            }
        )

    loo_summary(
        s1,
        "y_log_pat ~ tone + pat_l1 + res_exp_b + faculty_1k + tto_age_10y + tto_staff_10 + C(institution) + C(year)",
        "tone",
        "R5_Stage1",
    )
    loo_summary(
        s2,
        "y_log_pat ~ tone_l1 + pat_l1 + log_res_exp + log_faculty + log_tto_age + log_tto_staff + C(institution) + C(year)",
        "tone_l1",
        "R6_Stage2",
    )

    # V4: sparsity diagnostics for policy changes.
    obs = (
        panel[["institution", "year", "tone"]]
        .dropna(subset=["tone"])
        .sort_values(["institution", "year"])
        .copy()
    )
    obs["dtone"] = obs.groupby("institution")["tone"].diff()
    valid_pairs = obs["dtone"].notna()
    nonzero = (obs.loc[valid_pairs, "dtone"].abs() > 1e-12).sum()
    total_pairs = valid_pairs.sum()
    rows.append(
        {
            "section": "Validity",
            "model": "V4_ToneChangeSparsity",
            "term": "share_nonzero_dtone_pairs",
            "coef": float(nonzero / total_pairs if total_pairs else np.nan),
            "se": np.nan,
            "p": np.nan,
            "nobs": int(total_pairs),
            "r2": np.nan,
        }
    )

    robust_df = pd.DataFrame(rows)
    robust_df.to_csv(OUTPUT_DIR / "robustness_validity_suite.csv", index=False)
    return robust_df


def build_reviewer_summary(stage_df: pd.DataFrame, robust_df: pd.DataFrame):
    # Compact markdown summary for easy paper integration.
    lines = []
    lines.append("# Three-Stage Strategy + Robustness/Validity Summary")
    lines.append("")
    lines.append("## Stage Models")
    for _, row in stage_df.iterrows():
        lines.append(
            f"- `{row['model']}`: {row['term']} = {row['coef']:.3f} "
            f"(SE {row['se']:.3f}, p={row['p']:.3f}, n={int(row['nobs'])})"
        )

    lines.append("")
    lines.append("## Robustness / Validity")
    for _, row in robust_df.iterrows():
        if pd.notna(row["se"]):
            lines.append(
                f"- `{row['model']}` ({row['section']}): {row['term']} = {row['coef']:.3f} "
                f"(SE {row['se']:.3f}, p={row['p']:.3f}, n={int(row['nobs'])})"
            )
        else:
            lines.append(
                f"- `{row['model']}` ({row['section']}): {row['term']} = {row['coef']:.3f} "
                f"(n={int(row['nobs'])})"
            )

    (OUTPUT_DIR / "three_stage_reviewer_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    panel = load_panel()
    stage_df = run_stage_models(panel)
    robust_df = run_robustness_and_validity(panel)
    build_reviewer_summary(stage_df, robust_df)
    print("Saved outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
