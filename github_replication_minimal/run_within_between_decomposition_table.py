from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "merged_autm_tone_data_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "regression_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def parse_num(series: pd.Series, percent: bool = False) -> pd.Series:
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


def stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def load_panel() -> pd.DataFrame:
    raw = pd.read_csv(INPUT_FILE)

    panel = pd.DataFrame(
        {
            "institution": raw["Institution_std"].astype(str).str.strip(),
            "year": pd.to_numeric(raw["Year"], errors="coerce"),
            "patents": pd.to_numeric(raw["New Pat App Fld"], errors="coerce"),
            "tone": pd.to_numeric(raw["Mean_Tone_Score"], errors="coerce"),
            "royalty": pd.to_numeric(
                raw["royalty_adjusted_numerical"], errors="coerce"
            ).fillna(parse_num(raw["Royalty Share"], percent=True)),
            "res_exp_b": pd.to_numeric(raw["Tot Res Exp"], errors="coerce")
            / 1_000_000
            / 1000.0,
            "faculty_1k": pd.to_numeric(raw["faculty2_corrected"], errors="coerce")
            .fillna(pd.to_numeric(raw["faculty2"], errors="coerce"))
            / 1000.0,
            "tto_age_10y": pd.to_numeric(raw["TLO Age"], errors="coerce") / 10.0,
            "tto_staff_10": pd.to_numeric(raw["Lic FTEs"], errors="coerce") / 10.0,
            "private": pd.to_numeric(raw["Private"], errors="coerce"),
            "medschool": pd.to_numeric(raw["MEDSCHOOL"], errors="coerce"),
            "r1": pd.to_numeric(raw["Carnegie R1"], errors="coerce"),
        }
    )

    panel = panel.dropna(subset=["institution", "year"]).copy()
    panel = panel.groupby(["institution", "year"], as_index=False).mean(numeric_only=True)
    panel = panel.sort_values(["institution", "year"]).reset_index(drop=True)

    panel["y_log_pat"] = np.log1p(panel["patents"].clip(lower=0))
    panel["tone_l1"] = panel.groupby("institution")["tone"].shift(1)
    panel["patents_l1"] = panel.groupby("institution")["patents"].shift(1)
    panel["log_res_exp"] = np.where(panel["res_exp_b"] > 0, np.log(panel["res_exp_b"]), np.nan)
    panel["log_faculty"] = np.where(
        panel["faculty_1k"] > 0, np.log(panel["faculty_1k"]), np.nan
    )
    panel["log_tto_age"] = np.log1p(panel["tto_age_10y"].clip(lower=0))
    panel["log_tto_staff"] = np.log1p(panel["tto_staff_10"].clip(lower=0))
    panel = panel.replace([np.inf, -np.inf], np.nan)

    panel["institution"] = panel["institution"].astype("category")
    panel["year"] = panel["year"].astype(int)
    return panel


def fit_year_fe(
    data: pd.DataFrame, y: str, tone_term: str, controls: list[str], cluster_col: str
) -> object:
    rhs = " + ".join([tone_term] + controls + ["C(year)"])
    f = f"{y} ~ {rhs}"
    return smf.ols(f, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data[cluster_col]}
    )


def fit_two_way_fe(
    data: pd.DataFrame, y: str, tone_term: str, controls: list[str], cluster_col: str
) -> object:
    rhs = " + ".join([tone_term] + controls + ["C(institution)", "C(year)"])
    f = f"{y} ~ {rhs}"
    return smf.ols(f, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data[cluster_col]}
    )


def run_mundlak(
    data: pd.DataFrame,
    y: str,
    tone_var: str,
    tv_controls: list[str],
    between_controls: list[str] | None = None,
) -> tuple[object, object]:
    wm = data.copy()
    decomp_vars = [tone_var] + tv_controls
    for v in decomp_vars:
        wm[f"{v}_mean_i"] = wm.groupby("institution")[v].transform("mean")
        wm[f"{v}_within"] = wm[v] - wm[f"{v}_mean_i"]

    req = [y] + [f"{v}_mean_i" for v in decomp_vars] + [f"{v}_within" for v in decomp_vars]
    wm_s = wm.dropna(subset=req).copy()

    within_terms = [f"{v}_within" for v in decomp_vars]
    mean_terms = [f"{v}_mean_i" for v in decomp_vars]
    rhs = " + ".join(within_terms + mean_terms + ["C(year)"])
    f_mundlak = f"{y} ~ {rhs}"
    m_mundlak = smf.ols(f_mundlak, data=wm_s).fit(
        cov_type="cluster", cov_kwds={"groups": wm_s["institution"]}
    )

    between = wm_s.groupby("institution", as_index=False).agg(
        {y: "mean", **{v: "mean" for v in decomp_vars}}
    )
    if between_controls:
        for bc in between_controls:
            between[bc] = wm_s.groupby("institution")[bc].mean().values
    f_between_rhs = decomp_vars.copy()
    if between_controls:
        f_between_rhs.extend(between_controls)
    f_between = f"{y} ~ " + " + ".join(f_between_rhs)
    m_between = smf.ols(f_between, data=between).fit()
    return m_mundlak, m_between


def extract_row(
    panel: str,
    model_name: str,
    term_label: str,
    term_key: str,
    model: object,
) -> dict:
    return {
        "panel": panel,
        "model": model_name,
        "term": term_label,
        "coef": float(model.params.get(term_key, np.nan)),
        "se": float(model.bse.get(term_key, np.nan)),
        "p": float(model.pvalues.get(term_key, np.nan)),
        "nobs": int(model.nobs),
        "r2": float(model.rsquared),
    }


def build_table(df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Within-Between Decomposition of Policy Tone Effects}")
    lines.append("\\label{tab:within_between_decomp}")
    lines.append("\\begin{threeparttable}")
    lines.append("\\begin{tabular}{llcccc}")
    lines.append("\\toprule")
    lines.append("Panel & Specification & Tone Component & Coef. & SE & p-value \\\\")
    lines.append("\\midrule")

    for panel_name in ["A: No-lag, No-log", "B: Lag+Log"]:
        sub = df[df["panel"] == panel_name].copy()
        lines.append(
            f"\\multicolumn{{6}}{{l}}{{\\textit{{{panel_name}}}}} \\\\"
        )
        for _, r in sub.iterrows():
            coef = f"{r['coef']:.3f}{stars(r['p'])}" if pd.notna(r["coef"]) else "--"
            se = f"{r['se']:.3f}" if pd.notna(r["se"]) else "--"
            pval = f"{r['p']:.3f}" if pd.notna(r["p"]) else "--"
            lines.append(
                f" & {r['model']} & {r['term']} & {coef} & {se} & {pval} \\\\"
            )
        n_val = int(sub["nobs"].max()) if len(sub) else 0
        lines.append(
            f"\\multicolumn{{6}}{{r}}{{\\footnotesize Max N in panel = {n_val}}} \\\\"
        )
        lines.append("\\midrule")

    if lines[-1] == "\\midrule":
        lines = lines[:-1]
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\begin{tablenotes}[flushleft]")
    lines.append("\\footnotesize")
    lines.append(
        "\\item Notes: Year FE and Two-way FE models use clustered SEs by institution. "
        "Mundlak decomposition reports within-institution and between-institution tone components in one model with year FE. "
        "Between-only model is cross-sectional on institution means."
    )
    lines.append("\\item Significance: *** p<0.01, ** p<0.05, * p<0.10.")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{threeparttable}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main() -> None:
    panel = load_panel()
    rows: list[dict] = []

    # Panel A: no lag, no log controls (aligned with Stage 1 discussion)
    controls_a = [
        "royalty",
        "res_exp_b",
        "faculty_1k",
        "tto_age_10y",
        "tto_staff_10",
        "private",
        "medschool",
        "r1",
    ]
    req_a = ["y_log_pat", "tone"] + controls_a
    a = panel.dropna(subset=req_a).copy()
    m_a_year = fit_year_fe(a, "y_log_pat", "tone", controls_a, "institution")
    m_a_tw = fit_two_way_fe(a, "y_log_pat", "tone", controls_a, "institution")

    rows.append(
        extract_row(
            "A: No-lag, No-log",
            "Year FE",
            "Tone (total mix of within+between)",
            "tone",
            m_a_year,
        )
    )
    rows.append(
        extract_row(
            "A: No-lag, No-log",
            "Two-way FE",
            "Tone (within signal)",
            "tone",
            m_a_tw,
        )
    )

    tv_controls_a = ["royalty", "res_exp_b", "faculty_1k", "tto_age_10y", "tto_staff_10", "medschool"]
    m_a_mundlak, m_a_between = run_mundlak(
        data=a,
        y="y_log_pat",
        tone_var="tone",
        tv_controls=tv_controls_a,
        between_controls=["private", "r1"],
    )
    rows.append(
        extract_row(
            "A: No-lag, No-log",
            "Mundlak",
            "Tone within component",
            "tone_within",
            m_a_mundlak,
        )
    )
    rows.append(
        extract_row(
            "A: No-lag, No-log",
            "Mundlak",
            "Tone between component",
            "tone_mean_i",
            m_a_mundlak,
        )
    )
    rows.append(
        extract_row(
            "A: No-lag, No-log",
            "Between-only",
            "Tone between component",
            "tone",
            m_a_between,
        )
    )

    # Panel B: lag + log controls (aligned with Stage 2)
    controls_b = ["patents_l1", "log_res_exp", "log_faculty", "log_tto_age", "log_tto_staff"]
    req_b = ["y_log_pat", "tone_l1"] + controls_b
    b = panel.dropna(subset=req_b).copy()
    m_b_year = fit_year_fe(b, "y_log_pat", "tone_l1", controls_b, "institution")
    m_b_tw = fit_two_way_fe(b, "y_log_pat", "tone_l1", controls_b, "institution")

    rows.append(
        extract_row(
            "B: Lag+Log",
            "Year FE",
            "Tone L1 (total mix of within+between)",
            "tone_l1",
            m_b_year,
        )
    )
    rows.append(
        extract_row(
            "B: Lag+Log",
            "Two-way FE",
            "Tone L1 (within signal)",
            "tone_l1",
            m_b_tw,
        )
    )

    m_b_mundlak, m_b_between = run_mundlak(
        data=b,
        y="y_log_pat",
        tone_var="tone_l1",
        tv_controls=["patents_l1", "log_res_exp", "log_faculty", "log_tto_age", "log_tto_staff"],
        between_controls=None,
    )
    rows.append(
        extract_row(
            "B: Lag+Log",
            "Mundlak",
            "Tone L1 within component",
            "tone_l1_within",
            m_b_mundlak,
        )
    )
    rows.append(
        extract_row(
            "B: Lag+Log",
            "Mundlak",
            "Tone L1 between component",
            "tone_l1_mean_i",
            m_b_mundlak,
        )
    )
    rows.append(
        extract_row(
            "B: Lag+Log",
            "Between-only",
            "Tone L1 between component",
            "tone_l1",
            m_b_between,
        )
    )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "within_between_decomposition_table.csv", index=False)
    tex = build_table(out)
    (OUTPUT_DIR / "within_between_decomposition_table.tex").write_text(
        tex, encoding="utf-8"
    )
    print("Saved decomposition outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
