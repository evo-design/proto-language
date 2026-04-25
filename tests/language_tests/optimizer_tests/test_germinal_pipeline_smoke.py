"""Smoke test for the Germinal PD-L1 antibody design pipeline."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "examples" / "germinal" / "run_germinal_pipeline.py"
PDB_DIR = SCRIPT.parent / "pdbs"


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_germinal_vhh_smoke(tmp_path: Path) -> None:
    """Run the new Germinal VHH entrypoint end-to-end with smoke-sized overrides."""
    output_dir = tmp_path / "outputs" / "smoke_test"
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(SCRIPT),
            "--preset",
            "vhh",
            "--target-pdb",
            str(PDB_DIR / "pdl1.pdb"),
            "--target-chain",
            "A",
            "--target-hotspots",
            "A37,A39,A41,A96,A98",
            "--max-trajectories",
            "1",
            "--max-passing",
            "1",
            "--logits-steps",
            "3",
            "--softmax-steps",
            "2",
            "--search-steps",
            "3",
            "--num-seqs",
            "2",
            "--max-mpnn-sequences",
            "1",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert result.returncode == 0, f"VHH pipeline failed:\n{result.stderr[-2000:]}"
    run_dirs = sorted((output_dir / "germinal" / "pdl1").glob("run_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert any(run_dir.glob("*_binder.pdb"))

    summary_path = run_dir / "trajectory_summary.json"
    assert summary_path.exists(), "trajectory_summary.json not produced"
    summary = json.loads(summary_path.read_text())
    assert summary["num_trajectories"] == 1
    assert "trajectories" in summary
    assert len(summary["trajectories"]) == 1
    assert "stages" in summary["trajectories"][0]

    assert (run_dir / "trajectory_dynamics.png").exists(), "trajectory_dynamics.png not produced"

    variant_jsons = sorted(run_dir.glob("traj*_variant*_*.json"))
    for json_path in variant_jsons:
        assert json_path.with_suffix(".fasta").exists()
        assert json_path.with_suffix(".pdb").exists()
