from proto_language.language.constraint import ConstraintRegistry
from proto_language.language.optimizer.constraint_compiler import gradient_support_for_constraint_spec


def test_structure_plddt_reports_config_dependent_compiled_gradient_support() -> None:
    support = gradient_support_for_constraint_spec(ConstraintRegistry.get("structure-plddt"))

    assert support == {
        "rules": [
            {
                "source": "compiled",
                "label": "ESMFold gradient",
                "when": [{"config_path": "structure_tool", "equals": "esmfold"}],
                "requires_scoring": True,
                "target_segment": {"kind": "any_input"},
                "required_segments": [{"kind": "all_inputs", "sequence_types": ["protein"]}],
            },
            {
                "source": "compiled",
                "label": "AF2 multimer gradient",
                "when": [{"config_path": "structure_tool", "equals": "alphafold2_multimer"}],
                "requires_scoring": True,
                "target_segment": {
                    "kind": "input_index_from_config",
                    "config_path": "alphafold2_multimer_config.binder_input_index",
                },
                "required_segments": [
                    {
                        "kind": "input_index_from_config",
                        "config_path": "alphafold2_multimer_config.binder_input_index",
                        "sequence_types": ["protein"],
                    },
                    {
                        "kind": "input_indices_from_config",
                        "config_path": "alphafold2_multimer_config.target_input_indices",
                        "sequence_types": ["protein"],
                    },
                ],
            },
        ]
    }


def test_structure_distogram_reports_af2_compiled_gradient_support() -> None:
    support = gradient_support_for_constraint_spec(ConstraintRegistry.get("structure-distogram-cce"))

    assert support == {
        "rules": [
            {
                "source": "compiled",
                "label": "AF2 multimer gradient",
                "when": [{"config_path": "structure_tool", "equals": "alphafold2_multimer"}],
                "requires_scoring": True,
                "target_segment": {
                    "kind": "input_index_from_config",
                    "config_path": "alphafold2_multimer_config.binder_input_index",
                },
                "required_segments": [
                    {
                        "kind": "input_index_from_config",
                        "config_path": "alphafold2_multimer_config.binder_input_index",
                        "sequence_types": ["protein"],
                    },
                    {
                        "kind": "input_indices_from_config",
                        "config_path": "alphafold2_multimer_config.target_input_indices",
                        "sequence_types": ["protein"],
                    },
                ],
            }
        ]
    }


def test_plain_discrete_constraint_has_no_compiled_gradient_metadata() -> None:
    assert gradient_support_for_constraint_spec(ConstraintRegistry.get("gc-content")) is None
