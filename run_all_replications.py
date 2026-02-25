#!/usr/bin/env python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SCRIPTS = [
    "run_stage1_full_variable_screen.py",
    "run_nolag_and_functional_form_checks.py",
    "run_regressions_updated_tone.py",
    "run_additional_regressions_and_fe_decomp.py",
    "run_three_stage_robustness_validation.py",
    "run_within_between_decomposition_table.py",
    "run_advanced_validity_checks.py",
    "run_alternative_outcomes_lag_sensitivity.py",
]


def run_python(script_name: str) -> None:
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")
    print(f"[RUN] {script_name}")
    subprocess.run([sys.executable, str(script_path)], check=True)


def main() -> None:
    for script in SCRIPTS:
        run_python(script)
    print("[DONE] Replication pipeline finished.")


if __name__ == "__main__":
    main()
