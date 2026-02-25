from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "merged_autm_tone_data_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "regression_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def parse_num(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    s = s.str.replace("$", "", regex=False).str.replace(",", "", regex=False)
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(s, errors="coerce")


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
            "tone": pd.to_numeric(raw["Mean_Tone_Score"], errors="coerce"),
            "res_exp_m": pd.to_numeric(raw["Tot Res Exp"], errors="coerce") / 1_000_000,
            "faculty": pd.to_numeric(raw["faculty2_corrected"], errors="coerce").fillna(
                pd.to_numeric(raw["faculty2"], errors="coerce")
            ),
            "tto_age": pd.to_numeric(raw["TLO Age"], errors="coerce"),
            "tto_staff": parse_num(raw["Lic FTEs"]),
            "new_pat_app": pd.to_numeric(raw["New Pat App Fld"], errors="coerce"),
            "tot_pat_app": pd.to_numeric(raw["Tot Pat App Fld"], errors="coerce"),
            "iss_us_pat": pd.to_numeric(raw["Iss US Pat"], errors="coerce"),
            "inv_disclosures": pd.to_numeric(raw["Inv Dis Rec"], errors="coerce"),
            "tot_lic_startups": pd.to_numeric(raw["Tot Lic St-Ups"], errors="coerce"),
            "lic_iss": pd.to_numeric(raw["Lic Iss"], errors="coerce"),
            "gross_lic_inc": parse_num(raw["Gross Lic Inc"]),
            "net_lic_income": parse_num(raw["Net Licensing Income"]),
            "lic_gen_inc": parse_num(raw["Lic Gen Inc"]),
        }
    )
    panel = panel.dropna(subset=["institution", "year"]).copy()
    panel = panel.groupby(["institution", "year"], as_index=False).mean(numeric_only=True)
    panel = panel.sort_values(["institution", "year"]).reset_index(drop=True)

    for k in range(1, 6):
        panel[f"tone_l{k}"] = panel.groupby("institution")["tone"].shift(k)

    panel["log_res_exp"] = np.where(panel["res_exp_m"] > 0, np.log(panel["res_exp_m"]), np.nan)
    panel["log_faculty"] = np.where(panel["faculty"] > 0, np.log(panel["faculty"]), np.nan)
    panel["log_tto_age"] = np.log1p(panel["tto_age"].clip(lower=0))
    panel["log_tto_staff"] = np.log1p(panel["tto_staff"].clip(lower=0))
    panel = panel.replace([np.inf, -np.inf], np.nan)
    return panel


def run_models(panel: pd.DataFrame) -> pd.DataFrame:
    # Keep the same outcome order as Table 10 (tab:alternative_y_outcomes)
    outcomes = [
        ("new_pat_app", "New Pat App Fld"),
        ("tot_pat_app", "Tot Pat App Fld"),
        ("iss_us_pat", "Iss US Pat"),
        ("net_lic_income", "Net Licensing Income"),
        ("gross_lic_inc", "Gross Lic Inc"),
        ("lic_iss", "Lic Iss"),
        ("tot_lic_startups", "Tot Lic St-Ups"),
        ("lic_gen_inc", "Lic Gen Inc"),
        ("inv_disclosures", "Inv Dis Rec"),
    ]

    rows: list[dict] = []
    for order_idx, (y_key, y_label) in enumerate(outcomes):
        d = panel.copy()
        d["y"] = d[y_key]
        min_y = d["y"].min(skipna=True)
        if pd.notna(min_y) and min_y <= 0:
            d["y_shift"] = d["y"] - min_y + 1
        else:
            d["y_shift"] = d["y"]
        d["y_log"] = np.log1p(d["y_shift"].clip(lower=0))
        d["y_l1"] = d.groupby("institution")["y"].shift(1)

        for lag in range(1, 6):
            tone_term = f"tone_l{lag}"
            req = [
                "y_log",
                "y_l1",
                tone_term,
                "log_res_exp",
                "log_faculty",
                "log_tto_age",
                "log_tto_staff",
            ]
            s = d.dropna(subset=req).copy()
            if len(s) < 200:
                continue
            s["institution"] = s["institution"].astype("category")
            formula = (
                f"y_log ~ {tone_term} + y_l1 + log_res_exp + log_faculty + "
                "log_tto_age + log_tto_staff + C(institution) + C(year)"
            )
            model = smf.ols(formula, data=s).fit(
                cov_type="cluster", cov_kwds={"groups": s["institution"]}
            )
            rows.append(
                {
                    "outcome": y_label,
                    "outcome_order": order_idx,
                    "lag": lag,
                    "coef": float(model.params.get(tone_term, np.nan)),
                    "se": float(model.bse.get(tone_term, np.nan)),
                    "p": float(model.pvalues.get(tone_term, np.nan)),
                    "nobs": int(model.nobs),
                    "r2": float(model.rsquared),
                }
            )
    out = pd.DataFrame(rows).sort_values(["outcome_order", "lag"]).reset_index(drop=True)
    return out.drop(columns=["outcome_order"])


def build_tex(df: pd.DataFrame) -> str:
    lines: list[str] = []

    panel_a = [
        "New Pat App Fld",
        "Tot Pat App Fld",
        "Iss US Pat",
        "Net Licensing Income",
        "Gross Lic Inc",
    ]
    panel_b = [
        "Lic Iss",
        "Tot Lic St-Ups",
        "Lic Gen Inc",
        "Inv Dis Rec",
    ]

    def panel_rows(panel_df: pd.DataFrame) -> list[str]:
        out_lines: list[str] = []
        current = None
        for _, r in panel_df.iterrows():
            if current != r["outcome"]:
                if current is not None:
                    out_lines.append("\\midrule")
                current = r["outcome"]
            sig = stars(r["p"]) or "\\phantom{***}"
            out_lines.append(
                f"{r['outcome']} & $t-{int(r['lag'])}$ & {r['coef']:.3f} & {sig} & "
                f"{r['se']:.3f} & {r['p']:.3f} & {int(r['nobs'])} \\\\"
            )
        return out_lines

    a_df = (
        df[df["outcome"].isin(panel_a)]
        .assign(_order=lambda x: x["outcome"].map({k: i for i, k in enumerate(panel_a)}))
        .sort_values(["_order", "lag"])
        .drop(columns=["_order"])
    )
    b_df = (
        df[df["outcome"].isin(panel_b)]
        .assign(_order=lambda x: x["outcome"].map({k: i for i, k in enumerate(panel_b)}))
        .sort_values(["_order", "lag"])
        .drop(columns=["_order"])
    )

    colspec = (
        ">{\\raggedright\\arraybackslash}p{3.8cm}c"
        "S[table-format=+1.3]cS[table-format=+1.3]S[table-format=1.3]S[table-format=4.0]"
    )

    lines.append("\\begingroup")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4.5pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.08}")
    lines.append(
        "\\sisetup{table-number-alignment=center,group-separator={,},group-minimum-digits=4}"
    )

    # Panel A (with table caption + label)
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Lag-Length Sensitivity in Alternative Innovation Outcomes}")
    lines.append("\\label{tab:alternative_outcomes_lag_sensitivity}")
    lines.append("\\begin{threeparttable}")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    lines.append("Outcome & Lag & {Coef.} & Sig. & {SE} & {p-value} & {N} \\\\")
    lines.append("\\midrule")
    lines.append(
        "\\multicolumn{7}{l}{\\textit{Panel A: Patent Outcomes and Licensing Revenue}}\\\\"
    )
    lines.append("\\midrule")
    lines.extend(panel_rows(a_df))
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{threeparttable}")
    lines.append("\\end{table}")

    # Panel B (continued, without a new table number)
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(
        "\\textit{Table \\ref{tab:alternative_outcomes_lag_sensitivity} (continued).}"
    )
    lines.append("\\par\\vspace{3pt}")
    lines.append("\\begin{threeparttable}")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    lines.append("Outcome & Lag & {Coef.} & Sig. & {SE} & {p-value} & {N} \\\\")
    lines.append("\\midrule")
    lines.append(
        "\\multicolumn{7}{l}{\\textit{Panel B: Licensing Activity, Startups, and Disclosures}}\\\\"
    )
    lines.append("\\midrule")
    lines.extend(panel_rows(b_df))
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\begin{tablenotes}[flushleft]")
    lines.append("\\footnotesize")
    lines.append(
        "\\item Notes: Each row is a separate two-way FE regression with outcome transformed as "
        "$\\log(1+Y_{it})$ (or shifted-log for outcomes with non-positive values), including "
        "$Y_{i,t-1}$ and standard controls. Standard errors are clustered by institution."
    )
    lines.append("\\item Significance: *** p<0.01, ** p<0.05, * p<0.10.")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{threeparttable}")
    lines.append("\\end{table}")

    lines.append("\\endgroup")
    return "\n".join(lines)


def main() -> None:
    panel = load_panel()
    out = run_models(panel)
    out.to_csv(OUTPUT_DIR / "alternative_outcomes_lag_sensitivity.csv", index=False)
    tex = build_tex(out)
    (OUTPUT_DIR / "alternative_outcomes_lag_sensitivity.tex").write_text(
        tex, encoding="utf-8"
    )
    print("Saved lag-sensitivity outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
