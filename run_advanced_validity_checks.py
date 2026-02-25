from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import patsy
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
            "tone_mean": pd.to_numeric(raw["Mean_Tone_Score"], errors="coerce"),
            "tone_median": pd.to_numeric(raw["Median_Tone_Score"], errors="coerce"),
            "res_exp_m": pd.to_numeric(raw["Tot Res Exp"], errors="coerce") / 1_000_000,
            "faculty": pd.to_numeric(raw["faculty2_corrected"], errors="coerce").fillna(
                pd.to_numeric(raw["faculty2"], errors="coerce")
            ),
            "tto_age": pd.to_numeric(raw["TLO Age"], errors="coerce"),
            "tto_staff": pd.to_numeric(raw["Lic FTEs"], errors="coerce"),
        }
    )
    panel = panel.dropna(subset=["institution", "year"]).copy()
    panel = panel.groupby(["institution", "year"], as_index=False).mean(numeric_only=True)
    panel = panel.sort_values(["institution", "year"]).reset_index(drop=True)

    panel["y_log_pat"] = np.log1p(panel["patents"])
    panel["pat_l1"] = panel.groupby("institution")["patents"].shift(1)

    panel["tone_mean_l1"] = panel.groupby("institution")["tone_mean"].shift(1)
    panel["tone_median_l1"] = panel.groupby("institution")["tone_median"].shift(1)
    panel["dtone_mean"] = panel.groupby("institution")["tone_mean"].diff()
    panel["dtone_mean_l1"] = panel.groupby("institution")["dtone_mean"].shift(1)

    panel["log_res_exp"] = np.log(panel["res_exp_m"])
    panel["log_faculty"] = np.log(panel["faculty"])
    panel["log_tto_age"] = np.log1p(panel["tto_age"].clip(lower=0))
    panel["log_tto_staff"] = np.log1p(panel["tto_staff"].clip(lower=0))

    panel["year"] = panel["year"].astype(int)
    return panel


def fit_cluster(formula: str, data: pd.DataFrame):
    return smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["institution"]}
    )


def run_event_study(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Major policy updates (absolute tone change >= 0.02)
    es = panel.copy()
    es["update_major"] = (es["dtone_mean"].abs() >= 0.02).astype(float)

    # lead/lag indicators around update year
    es["lead2"] = es.groupby("institution")["update_major"].shift(-2)
    es["lead1"] = es.groupby("institution")["update_major"].shift(-1)
    es["event0"] = es.groupby("institution")["update_major"].shift(0)
    es["lag1"] = es.groupby("institution")["update_major"].shift(1)
    es["lag2"] = es.groupby("institution")["update_major"].shift(2)
    es["lag3"] = es.groupby("institution")["update_major"].shift(3)

    d = es.dropna(
        subset=[
            "y_log_pat",
            "pat_l1",
            "log_res_exp",
            "log_faculty",
            "log_tto_age",
            "log_tto_staff",
            "lead2",
            "lead1",
            "event0",
            "lag1",
            "lag2",
            "lag3",
        ]
    ).copy()
    d["institution"] = d["institution"].astype("category")

    f = (
        "y_log_pat ~ lead2 + lead1 + event0 + lag1 + lag2 + lag3 + "
        "pat_l1 + log_res_exp + log_faculty + log_tto_age + log_tto_staff + "
        "C(institution) + C(year)"
    )
    m = fit_cluster(f, d)

    terms = ["lead2", "lead1", "event0", "lag1", "lag2", "lag3"]
    rows = []
    for t in terms:
        rows.append(
            {
                "term": t,
                "coef": float(m.params.get(t, np.nan)),
                "se": float(m.bse.get(t, np.nan)),
                "p": float(m.pvalues.get(t, np.nan)),
                "nobs": int(m.nobs),
                "r2": float(m.rsquared),
            }
        )

    # Pre-trend and post-effect joint tests
    pre = m.f_test("lead2 = 0, lead1 = 0")
    post = m.f_test("event0 = 0, lag1 = 0, lag2 = 0, lag3 = 0")
    joint_rows = pd.DataFrame(
        [
            {
                "test": "pretrend_joint_leads_zero",
                "f_stat": float(pre.fvalue),
                "p": float(pre.pvalue),
                "df_denom": float(pre.df_denom),
            },
            {
                "test": "postperiod_joint_effects_zero",
                "f_stat": float(post.fvalue),
                "p": float(post.pvalue),
                "df_denom": float(post.df_denom),
            },
        ]
    )
    return pd.DataFrame(rows), joint_rows


def run_randomization_inference(panel: pd.DataFrame, n_perm: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)

    s = panel.dropna(
        subset=[
            "y_log_pat",
            "tone_mean_l1",
            "pat_l1",
            "log_res_exp",
            "log_faculty",
            "log_tto_age",
            "log_tto_staff",
        ]
    ).copy()
    s["institution"] = s["institution"].astype("category")

    formula = (
        "y_log_pat ~ tone_mean_l1 + pat_l1 + log_res_exp + log_faculty + "
        "log_tto_age + log_tto_staff + C(institution) + C(year)"
    )
    y, X = patsy.dmatrices(formula, data=s, return_type="dataframe")
    yv = y.to_numpy().ravel()
    Xv = X.to_numpy()
    tone_idx = X.columns.get_loc("tone_mean_l1")

    beta_actual = float(np.linalg.lstsq(Xv, yv, rcond=None)[0][tone_idx])

    inst_codes = s["institution"].cat.codes.to_numpy()
    tone = s["tone_mean_l1"].to_numpy()
    idx_by_inst = {}
    for code in np.unique(inst_codes):
        idx_by_inst[int(code)] = np.where(inst_codes == code)[0]

    draws = []
    Xp = Xv.copy()
    for _ in range(n_perm):
        perm_tone = np.empty_like(tone)
        for _, idx in idx_by_inst.items():
            perm_tone[idx] = rng.permutation(tone[idx])
        Xp[:, tone_idx] = perm_tone
        beta = float(np.linalg.lstsq(Xp, yv, rcond=None)[0][tone_idx])
        draws.append(beta)

    draws = np.array(draws)
    p_two_sided = float(np.mean(np.abs(draws) >= abs(beta_actual)))

    summary = pd.DataFrame(
        [
            {"metric": "beta_actual", "value": beta_actual},
            {"metric": "perm_mean", "value": float(draws.mean())},
            {"metric": "perm_std", "value": float(draws.std())},
            {"metric": "perm_p05", "value": float(np.quantile(draws, 0.05))},
            {"metric": "perm_p50", "value": float(np.quantile(draws, 0.50))},
            {"metric": "perm_p95", "value": float(np.quantile(draws, 0.95))},
            {"metric": "randomization_p_two_sided", "value": p_two_sided},
            {"metric": "n_permutations", "value": int(n_perm)},
            {"metric": "nobs", "value": int(len(s))},
        ]
    )
    draw_df = pd.DataFrame({"beta_perm_tone_l1": draws})
    return summary, draw_df


def run_alternative_tone_models(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # Build additional tone variants
    p = panel.copy()
    # within-institution z-score (for mean tone)
    g_mean = p.groupby("institution")["tone_mean"]
    mu = g_mean.transform("mean")
    sd = g_mean.transform("std")
    p["tone_mean_z"] = (p["tone_mean"] - mu) / sd.replace(0, np.nan)
    p["tone_mean_z_l1"] = p.groupby("institution")["tone_mean_z"].shift(1)

    # High-tone indicator using sample median
    med = p["tone_mean"].median(skipna=True)
    p["tone_high"] = (p["tone_mean"] >= med).astype(float)
    p["tone_high_l1"] = p.groupby("institution")["tone_high"].shift(1)

    specs = [
        (
            "AltTone_A_MedianLag",
            "tone_median_l1",
            [
                "y_log_pat",
                "tone_median_l1",
                "pat_l1",
                "log_res_exp",
                "log_faculty",
                "log_tto_age",
                "log_tto_staff",
            ],
            "y_log_pat ~ tone_median_l1 + pat_l1 + log_res_exp + log_faculty + log_tto_age + log_tto_staff + C(institution) + C(year)",
        ),
        (
            "AltTone_B_MeanWithinZ_Lag",
            "tone_mean_z_l1",
            [
                "y_log_pat",
                "tone_mean_z_l1",
                "pat_l1",
                "log_res_exp",
                "log_faculty",
                "log_tto_age",
                "log_tto_staff",
            ],
            "y_log_pat ~ tone_mean_z_l1 + pat_l1 + log_res_exp + log_faculty + log_tto_age + log_tto_staff + C(institution) + C(year)",
        ),
        (
            "AltTone_C_HighToneIndicator_Lag",
            "tone_high_l1",
            [
                "y_log_pat",
                "tone_high_l1",
                "pat_l1",
                "log_res_exp",
                "log_faculty",
                "log_tto_age",
                "log_tto_staff",
            ],
            "y_log_pat ~ tone_high_l1 + pat_l1 + log_res_exp + log_faculty + log_tto_age + log_tto_staff + C(institution) + C(year)",
        ),
        (
            "AltTone_D_MedianContemporaneous",
            "tone_median",
            [
                "y_log_pat",
                "tone_median",
                "pat_l1",
                "log_res_exp",
                "log_faculty",
                "log_tto_age",
                "log_tto_staff",
            ],
            "y_log_pat ~ tone_median + pat_l1 + log_res_exp + log_faculty + log_tto_age + log_tto_staff + C(institution) + C(year)",
        ),
    ]

    for name, term, req, formula in specs:
        d = p.dropna(subset=req).copy()
        d["institution"] = d["institution"].astype("category")
        m = fit_cluster(formula, d)
        rows.append(
            {
                "model": name,
                "term": term,
                "coef": float(m.params.get(term, np.nan)),
                "se": float(m.bse.get(term, np.nan)),
                "p": float(m.pvalues.get(term, np.nan)),
                "nobs": int(m.nobs),
                "r2": float(m.rsquared),
            }
        )
    return pd.DataFrame(rows)


def run_major_change_models(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    rows = []

    thresholds = [0.01, 0.02, 0.03]
    for thr in thresholds:
        col = f"major_{str(thr).replace('.', 'p')}"
        p[col] = (p["dtone_mean"].abs() >= thr).astype(float)
        p[f"{col}_l1"] = p.groupby("institution")[col].shift(1)

        # Event-indicator effect in FE model
        d1 = p.dropna(
            subset=[
                "y_log_pat",
                f"{col}_l1",
                "pat_l1",
                "log_res_exp",
                "log_faculty",
                "log_tto_age",
                "log_tto_staff",
            ]
        ).copy()
        d1["institution"] = d1["institution"].astype("category")
        f1 = (
            f"y_log_pat ~ {col}_l1 + pat_l1 + log_res_exp + log_faculty + "
            "log_tto_age + log_tto_staff + C(institution) + C(year)"
        )
        m1 = fit_cluster(f1, d1)
        rows.append(
            {
                "model": f"MajorChangeIndicator_thr_{thr}",
                "term": f"{col}_l1",
                "coef": float(m1.params.get(f"{col}_l1", np.nan)),
                "se": float(m1.bse.get(f"{col}_l1", np.nan)),
                "p": float(m1.pvalues.get(f"{col}_l1", np.nan)),
                "nobs": int(m1.nobs),
                "r2": float(m1.rsquared),
            }
        )

        # Signed delta effect, restricted to major-change years only
        d2 = p.dropna(
            subset=[
                "y_log_pat",
                "dtone_mean_l1",
                "pat_l1",
                "log_res_exp",
                "log_faculty",
                "log_tto_age",
                "log_tto_staff",
            ]
        ).copy()
        d2 = d2[d2["dtone_mean_l1"].abs() >= thr].copy()
        if d2.shape[0] >= 80 and d2["institution"].nunique() >= 20:
            d2["institution"] = d2["institution"].astype("category")
            f2 = (
                "y_log_pat ~ dtone_mean_l1 + pat_l1 + log_res_exp + log_faculty + "
                "log_tto_age + log_tto_staff + C(institution) + C(year)"
            )
            m2 = fit_cluster(f2, d2)
            rows.append(
                {
                    "model": f"MajorChangeOnlySignedDelta_thr_{thr}",
                    "term": "dtone_mean_l1",
                    "coef": float(m2.params.get("dtone_mean_l1", np.nan)),
                    "se": float(m2.bse.get("dtone_mean_l1", np.nan)),
                    "p": float(m2.pvalues.get("dtone_mean_l1", np.nan)),
                    "nobs": int(m2.nobs),
                    "r2": float(m2.rsquared),
                }
            )
        else:
            rows.append(
                {
                    "model": f"MajorChangeOnlySignedDelta_thr_{thr}",
                    "term": "dtone_mean_l1",
                    "coef": np.nan,
                    "se": np.nan,
                    "p": np.nan,
                    "nobs": int(d2.shape[0]),
                    "r2": np.nan,
                }
            )

    return pd.DataFrame(rows)


def write_summary(
    event_df: pd.DataFrame,
    event_tests: pd.DataFrame,
    ri_df: pd.DataFrame,
    alt_df: pd.DataFrame,
    major_df: pd.DataFrame,
):
    lines = []
    lines.append("# Advanced Validity Check Summary")
    lines.append("")
    lines.append("## Event-Study Around Major Policy Updates (|delta tone| >= 0.02)")
    for _, r in event_df.iterrows():
        lines.append(
            f"- {r['term']}: coef={r['coef']:.3f}, se={r['se']:.3f}, p={r['p']:.3f}, n={int(r['nobs'])}"
        )
    for _, r in event_tests.iterrows():
        lines.append(
            f"- {r['test']}: F={r['f_stat']:.3f}, p={r['p']:.3f}"
        )

    lines.append("")
    lines.append("## Randomization Inference (within-institution permutation)")
    for _, r in ri_df.iterrows():
        val = r["value"]
        if float(val).is_integer():
            lines.append(f"- {r['metric']}: {int(val)}")
        else:
            lines.append(f"- {r['metric']}: {val:.4f}")

    lines.append("")
    lines.append("## Alternative Tone Constructions")
    for _, r in alt_df.iterrows():
        lines.append(
            f"- {r['model']}: {r['term']}={r['coef']:.3f} (se={r['se']:.3f}, p={r['p']:.3f}, n={int(r['nobs'])})"
        )

    lines.append("")
    lines.append("## Major-Policy-Change Subsample / Indicator Checks")
    for _, r in major_df.iterrows():
        if pd.notna(r["coef"]):
            lines.append(
                f"- {r['model']}: {r['term']}={r['coef']:.3f} (se={r['se']:.3f}, p={r['p']:.3f}, n={int(r['nobs'])})"
            )
        else:
            lines.append(
                f"- {r['model']}: insufficient power/sample (n={int(r['nobs'])})"
            )

    (OUTPUT_DIR / "advanced_validity_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    panel = load_panel()

    event_df, event_tests = run_event_study(panel)
    event_df.to_csv(OUTPUT_DIR / "event_study_major_update.csv", index=False)
    event_tests.to_csv(OUTPUT_DIR / "event_study_major_update_joint_tests.csv", index=False)

    ri_summary, ri_draws = run_randomization_inference(panel, n_perm=1000, seed=42)
    ri_summary.to_csv(OUTPUT_DIR / "randomization_inference_summary.csv", index=False)
    ri_draws.to_csv(OUTPUT_DIR / "randomization_inference_draws.csv", index=False)

    alt_df = run_alternative_tone_models(panel)
    alt_df.to_csv(OUTPUT_DIR / "alternative_tone_construction_models.csv", index=False)

    major_df = run_major_change_models(panel)
    major_df.to_csv(OUTPUT_DIR / "major_policy_change_models.csv", index=False)

    write_summary(event_df, event_tests, ri_summary, alt_df, major_df)
    print("Saved advanced validity outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
