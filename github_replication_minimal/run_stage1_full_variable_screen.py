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


def load_data() -> pd.DataFrame:
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
    panel["y_log_pat"] = np.log1p(panel["patents"])
    panel["year"] = panel["year"].astype(int)
    panel["institution"] = panel["institution"].astype("category")
    return panel


def run_models(panel: pd.DataFrame) -> tuple[object, object]:
    req = [
        "y_log_pat",
        "tone",
        "royalty",
        "res_exp_b",
        "faculty_1k",
        "tto_age_10y",
        "tto_staff_10",
        "private",
        "medschool",
        "r1",
    ]
    d = panel.dropna(subset=req).copy()

    f_year = (
        "y_log_pat ~ tone + royalty + res_exp_b + faculty_1k + tto_age_10y + "
        "tto_staff_10 + private + medschool + r1 + C(year)"
    )
    m_year = smf.ols(f_year, data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["institution"]}
    )

    f_twoway = (
        "y_log_pat ~ tone + royalty + res_exp_b + faculty_1k + tto_age_10y + "
        "tto_staff_10 + private + medschool + r1 + C(institution) + C(year)"
    )
    m_twoway = smf.ols(f_twoway, data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["institution"]}
    )
    return m_year, m_twoway


def build_output(m_year, m_twoway):
    terms = [
        ("tone", "Policy tone (current)"),
        ("royalty", "Royalty share"),
        ("res_exp_b", "Research expenditure (bn USD)"),
        ("faculty_1k", "Faculty (thousands)"),
        ("tto_age_10y", "TTO age (decades)"),
        ("tto_staff_10", "TTO staff (tens)"),
        ("private", "Private university"),
        ("medschool", "Medical school"),
        ("r1", "Carnegie R1"),
    ]

    rows = []
    for term, label in terms:
        rows.append(
            {
                "term": term,
                "label": label,
                "coef_yearFE": float(m_year.params.get(term, np.nan)),
                "se_yearFE": float(m_year.bse.get(term, np.nan)),
                "p_yearFE": float(m_year.pvalues.get(term, np.nan)),
                "coef_twowayFE": float(m_twoway.params.get(term, np.nan)),
                "se_twowayFE": float(m_twoway.bse.get(term, np.nan)),
                "p_twowayFE": float(m_twoway.pvalues.get(term, np.nan)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "stage1_full_variable_screen.csv", index=False)

    n_year = int(m_year.nobs)
    n_twoway = int(m_twoway.nobs)
    r2_year = float(m_year.rsquared)
    r2_twoway = float(m_twoway.rsquared)

    tex_lines = []
    tex_lines.append("\\begin{table}[htbp]")
    tex_lines.append("\\centering")
    tex_lines.append("\\caption{Stage 1 Full Variable Screen (No-Lag, No-Log Controls)}")
    tex_lines.append("\\label{tab:stage1_full_screen}")
    tex_lines.append("\\begin{threeparttable}")
    tex_lines.append("\\begin{tabular}{lcccccc}")
    tex_lines.append("\\toprule")
    tex_lines.append("Variable & Coef. (Year FE) & SE & p-value & Coef. (Two-way FE) & SE & p-value \\\\")
    tex_lines.append("\\midrule")
    for _, r in out.iterrows():
        c1 = (
            f"{r['coef_yearFE']:.3f}{stars(r['p_yearFE'])}"
            if pd.notna(r["coef_yearFE"])
            else "--"
        )
        s1 = f"{r['se_yearFE']:.3f}" if pd.notna(r["se_yearFE"]) else "--"
        p1 = f"{r['p_yearFE']:.3f}" if pd.notna(r["p_yearFE"]) else "--"
        c2 = (
            f"{r['coef_twowayFE']:.3f}{stars(r['p_twowayFE'])}"
            if pd.notna(r["coef_twowayFE"])
            else "--"
        )
        s2 = f"{r['se_twowayFE']:.3f}" if pd.notna(r["se_twowayFE"]) else "--"
        p2 = f"{r['p_twowayFE']:.3f}" if pd.notna(r["p_twowayFE"]) else "--"
        tex_lines.append(f"{r['label']} & {c1} & {s1} & {p1} & {c2} & {s2} & {p2} \\\\")
    tex_lines.append("\\midrule")
    tex_lines.append(
        f"Observations & \\multicolumn{{3}}{{c}}{{{n_year}}} & \\multicolumn{{3}}{{c}}{{{n_twoway}}} \\\\"
    )
    tex_lines.append(
        f"$R^2$ & \\multicolumn{{3}}{{c}}{{{r2_year:.3f}}} & \\multicolumn{{3}}{{c}}{{{r2_twoway:.3f}}} \\\\"
    )
    tex_lines.append("\\bottomrule")
    tex_lines.append("\\end{tabular}")
    tex_lines.append("\\begin{tablenotes}[flushleft]")
    tex_lines.append("\\footnotesize")
    tex_lines.append("\\item Notes: Dependent variable is $\\log(1+\\text{Patents}_{it})$. Year FE model includes year fixed effects only; Two-way FE model includes university and year fixed effects. Standard errors clustered by university. Significance: *** p<0.01, ** p<0.05, * p<0.10.")
    tex_lines.append("\\end{tablenotes}")
    tex_lines.append("\\end{threeparttable}")
    tex_lines.append("\\end{table}")

    (OUTPUT_DIR / "stage1_full_variable_screen.tex").write_text(
        "\n".join(tex_lines), encoding="utf-8"
    )


def main():
    panel = load_data()
    m_year, m_twoway = run_models(panel)
    build_output(m_year, m_twoway)
    print("Saved stage1 full screen outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
