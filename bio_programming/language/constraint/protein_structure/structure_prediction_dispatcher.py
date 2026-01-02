"""
Not a constraint---implements a switch statement that runs a structure
prediction tool based on a user's input.

Used by multiple structure prediction constraints.
"""

from typing import Any, Dict, List

from proto_language.tools.structure_prediction.schemas import (
    StructurePredictionComplex,
    StructurePredictionOutput,
)


def predict_structures(
    complexes: List[StructurePredictionComplex],
    tool_name: str,
    tool_config: Dict[str, Any],
) -> StructurePredictionOutput:
    """
    Dispatch structure prediction to the specified tool.

    Dynamically imports tools to avoid circular dependencies.

    Args:
        complexes: List of complexes to predict structures for.
        tool_name: Name of the structure prediction tool.
        tool_config: Tool-specific configuration dictionary.

    Returns:
        StructurePredictionOutput containing predicted structures and metrics.

    Raises:
        ValueError: If tool_name is not recognized.
    """
    if tool_name == "esmfold":
        from proto_language.tools.structure_prediction.esmfold import (
            run_esmfold,
            ESMFoldInput,
            ESMFoldConfig,
        )
        cfg = ESMFoldConfig(**tool_config)
        return run_esmfold(ESMFoldInput(complexes=complexes), cfg)

    elif tool_name == "alphafold3":
        from proto_language.tools.structure_prediction.af3 import (
            run_af3,
            AlphaFold3Input,
            AlphaFold3Config,
        )
        cfg = AlphaFold3Config(**tool_config)
        return run_af3(AlphaFold3Input(complexes=complexes), cfg)

    elif tool_name == "boltz":
        from proto_language.tools.structure_prediction.boltz import (
            run_boltz,
            BoltzInput,
            BoltzConfig,
        )
        cfg = BoltzConfig(**tool_config)
        return run_boltz(BoltzInput(complexes=complexes), cfg)

    elif tool_name == "chai":
        from proto_language.tools.structure_prediction.chai import (
            run_chai,
            ChaiInput,
            ChaiConfig,
        )
        cfg = ChaiConfig(**tool_config)
        return run_chai(ChaiInput(complexes=complexes), cfg)

    else:
        raise ValueError(
            f"Unknown structure prediction tool: '{tool_name}'. "
            "Supported tools: esmfold, alphafold3, boltz, chai"
        )
