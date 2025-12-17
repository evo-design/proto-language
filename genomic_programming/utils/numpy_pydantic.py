from __future__ import annotations
import numpy as np
from pydantic_core import core_schema
from typing import Any


class NumpyArray(np.ndarray):
    """
    Custom pydantic type that wraps a numpy ndarray.
    """
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: Any,
    ) -> core_schema.CoreSchema:
        """
        Defines the Pydantic core schema for the NumpyArray type.

        This method specifies how Pydantic should handle this type:
        1.  Validation: It accepts any input and tries to convert it to a NumPy array.
            If conversion fails, it raises a ValueError.
        2.  Serialization: It tells Pydantic to convert the np.ndarray back to a list
            when serializing the model (e.g., to JSON).
        """
        def validate_from_any(value: Any) -> np.ndarray:
            try:
                return np.asarray(value)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Could not convert {value} to a NumPy array: {e}")

        # Create a schema that validates from any Python object using our function.
        from_python_schema = core_schema.no_info_plain_validator_function(validate_from_any)

        return core_schema.json_or_python_schema(
            json_schema=from_python_schema,
            python_schema=core_schema.union_schema(
                [
                    # Allow instances of np.ndarray to pass through without modification.
                    core_schema.is_instance_schema(np.ndarray),
                    # Otherwise, use our validation function.
                    from_python_schema,
                ]
            ),
            # Define how to serialize the object.
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: instance.tolist(),
                when_used="json",
            ),
        )
