# University Tech Transfer Output Replication Package

This is a GitHub package for reproducing the analysis from scripts.


## 1. Contents

- `merged_autm_tone_data_cleaned.csv`: cleaned panel data
- `Heading.xlsx`: variable/heading reference
- `run_*.py`: full analysis scripts
- `requirements.txt`: Python dependencies
- `run_all_replications.py`: one-command replication pipeline
- `regression_outputs/`: empty at start; outputs are generated after running

## 2. Environment Setup

Use Python 3.10+.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Run Full Replication

From this folder root:

```bash
python run_all_replications.py
```

This will generate all outputs into:
- `regression_outputs/`

## 4. Manual Script Order (Optional)

```bash
python run_stage1_full_variable_screen.py
python run_nolag_and_functional_form_checks.py
python run_regressions_updated_tone.py
python run_additional_regressions_and_fe_decomp.py
python run_three_stage_robustness_validation.py
python run_within_between_decomposition_table.py
python run_advanced_validity_checks.py
python run_alternative_outcomes_lag_sensitivity.py
```

## 5. Key Files Generated After Running

- `regression_outputs/three_stage_model_results.csv`
- `regression_outputs/robustness_validity_suite.csv`
- `regression_outputs/three_stage_reviewer_summary.md`
- `regression_outputs/within_between_decomposition_table.csv`
- `regression_outputs/event_study_major_update.csv`
- `regression_outputs/directional_event_study_threshold_0p02.csv`
- `regression_outputs/alternative_outcomes_lag_sensitivity.csv`
