"""Malinois regulatory DNA activity constraint."""

from typing import Literal

from proto_tools import (
    DEFAULT_MALINOIS_ARTIFACT_MD5,
    DEFAULT_MALINOIS_ARTIFACT_PATH,
    DEFAULT_MALINOIS_ARTIFACT_URL,
    DEFAULT_MALINOIS_DIR,
    MalinoisScoreConfig,
    MalinoisScoreInput,
    run_malinois_score,
)

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import ConstraintOutput, Sequence
from proto_language.utils import sigmoid_score

MalinoisActivityCellType = Literal["K562", "HepG2", "SKNSH"]
MalinoisActivityDirection = Literal["max", "min"]


class MalinoisActivityConfig(BaseConfig):
    """Configuration for Malinois regulatory DNA activity scoring.

    Malinois predicts MPRA activity for 200 bp DNA inserts in K562, HepG2, and
    SK-N-SH contexts. This constraint maps the requested raw cell-type score to
    a bounded lower-is-better objective:

    For ``direction="max"``:

    ``score = 1 - sigmoid((raw_score - sigmoid_center) / sigmoid_scale)``

    For ``direction="min"``:

    ``score = sigmoid((raw_score - sigmoid_center) / sigmoid_scale)``

    With the default center of 4.0, raw scores above 4.0 receive penalties below
    0.5 in maximization mode and above 0.5 in minimization mode.

    Attributes:
        cell_type (MalinoisActivityCellType): Cell-type output to optimize.
        direction (MalinoisActivityDirection): Use "max" or "min" activity.
        sigmoid_center (float): Raw score mapped to a 0.5 transformed score.
        sigmoid_scale (float): Positive scale for the sigmoid transform.
        seq_length (int): Expected DNA insert length.
        artifact_path (str): Optional local artifact tarball path.
        artifact_url (str): Download URL for the default artifact.
        artifact_md5 (str): Expected artifact checksum.
        malinois_dir (str): Optional local extracted artifact directory.
        batch_size (int): Number of sequences to score per batch.
        device (str): Device for Malinois inference.
    """

    cell_type: MalinoisActivityCellType = ConfigField(
        title="Cell Type",
        default="K562",
        description="Malinois cell-type output to optimize.",
    )
    direction: MalinoisActivityDirection = ConfigField(
        title="Optimization Direction",
        default="max",
        description="Use 'max' to encourage activity or 'min' to suppress activity in the selected cell type.",
    )
    sigmoid_center: float = ConfigField(
        title="Sigmoid Center",
        default=4.0,
        description="Raw Malinois score where the transformed constraint score is 0.5.",
    )
    sigmoid_scale: float = ConfigField(
        title="Sigmoid Scale",
        default=1.0,
        gt=0.0,
        description="Positive scale for the raw-score sigmoid transform.",
    )
    seq_length: int = ConfigField(
        title="Sequence Length",
        default=200,
        ge=1,
        description="Expected DNA insert length before Malinois MPRA flank padding.",
    )
    artifact_path: str = ConfigField(
        title="Artifact Path",
        default=DEFAULT_MALINOIS_ARTIFACT_PATH,
        description="Optional local artifact tarball path; empty uses the managed cache download.",
    )
    artifact_url: str = ConfigField(
        title="Artifact URL",
        default=DEFAULT_MALINOIS_ARTIFACT_URL,
        description="HTTPS URL used to provision the Malinois artifact.",
    )
    artifact_md5: str = ConfigField(
        title="Artifact MD5",
        default=DEFAULT_MALINOIS_ARTIFACT_MD5,
        description="Expected MD5 checksum for the downloaded Malinois artifact.",
    )
    malinois_dir: str = ConfigField(
        title="Malinois Directory",
        default=DEFAULT_MALINOIS_DIR,
        description="Optional local Malinois metadata directory; empty uses the managed cache extraction.",
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Number of sequences to score simultaneously on GPU.",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device for Malinois inference.",
    )


def malinois_activity_score(raw_score: float, config: MalinoisActivityConfig) -> tuple[float, float, float]:
    """Return ``(score, scaled_score, sigmoid_value)`` for one raw Malinois prediction."""
    scaled_score = (raw_score - config.sigmoid_center) / config.sigmoid_scale
    sigmoid_value = sigmoid_score(raw_score, config.sigmoid_center, 1.0 / config.sigmoid_scale)
    if config.direction == "max":
        score = 1.0 - sigmoid_value
    elif config.direction == "min":
        score = sigmoid_value
    else:
        raise ValueError(f"Invalid Malinois activity direction: {config.direction!r}; expected 'max' or 'min'.")
    return score, scaled_score, sigmoid_value


@constraint(
    key="malinois-activity",
    label="Malinois Activity",
    config=MalinoisActivityConfig,
    description="Score regulatory DNA activity using Malinois with max/min cell-type objectives.",
    uses_gpu=True,
    tools_called=["malinois-score", "malinois-gradient"],
    category="sequence annotation",
    supported_sequence_types=["dna"],
)
def malinois_activity_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: MalinoisActivityConfig,
) -> list[ConstraintOutput]:
    """Score DNA sequences with Malinois as a bounded lower-is-better objective."""
    if not input_sequences:
        return []

    sequences = [sequence.sequence.upper().replace(" ", "").replace("\n", "") for (sequence,) in input_sequences]
    output = run_malinois_score(
        MalinoisScoreInput(sequences=sequences),
        MalinoisScoreConfig(
            cell_types=[config.cell_type],
            seq_length=config.seq_length,
            artifact_path=config.artifact_path,
            artifact_url=config.artifact_url,
            artifact_md5=config.artifact_md5,
            malinois_dir=config.malinois_dir,
            batch_size=config.batch_size,
            device=config.device,
        ),
    )

    results: list[ConstraintOutput] = []
    for result in output.results:
        raw_score = float(result.scores[config.cell_type])
        score, scaled_score, sigmoid_value = malinois_activity_score(raw_score, config)
        results.append(
            ConstraintOutput(
                score=score,
                metadata={
                    "malinois_cell_type": config.cell_type,
                    "malinois_direction": config.direction,
                    "malinois_raw_score": raw_score,
                    "malinois_scaled_score": scaled_score,
                    "malinois_sigmoid_value": sigmoid_value,
                    "malinois_activity_score": score,
                    "sigmoid_center": config.sigmoid_center,
                    "sigmoid_scale": config.sigmoid_scale,
                },
            )
        )
    return results
