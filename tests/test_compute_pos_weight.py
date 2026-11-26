"""End-to-end checks for the auxiliary class weight script."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compute_pos_weight.py"


def test_pos_weight_cli_reports_a_reproducible_ratio() -> None:
    command = [sys.executable, str(SCRIPT), "--episodes", "20", "--seed", "7"]

    first = subprocess.run(command, check=True, capture_output=True, text=True)
    second = subprocess.run(command, check=True, capture_output=True, text=True)
    result = json.loads(first.stdout)

    assert result == json.loads(second.stdout)
    assert result["episodes"] == result["scenario_pool_size"] == 20
    assert result["seed"] == 7
    assert result["positives"] > 0
    assert result["negatives"] > 0
    assert result["pos_weight"] == result["negatives"] / result["positives"]
