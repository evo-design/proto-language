"""Tensor serialization utilities for ML model outputs.

This module provides standardized functions for serializing PyTorch tensors
to nested Python lists for JSON transport and cloud RPC communication.
"""
from __future__ import annotations

def serialize_logits(logits, move_to_cpu: bool = True):
    """Serialize torch tensors to nested lists for JSON/transport.

    Handles:
    - None: returns None
    - Single Tensor: returns List[List[float]] via .tolist()
    - List[Tensor]: returns List[List[List[float]]] via [t.tolist() for t in logits]
    - Already serialized (nested lists): returns as-is
    """
    if logits is None:
        return None

    # Check if already serialized (nested list of numbers)
    # A tensor would have .tolist() method, a list wouldn't unless it's a list of tensors
    if isinstance(logits, list):
        # Check if it's a list of tensors or already serialized
        if len(logits) == 0:
            return logits
        first_elem = logits[0]
        # If first element has tolist, it's a list of tensors
        if hasattr(first_elem, "tolist"):
            # List of tensors
            if move_to_cpu:
                return [t.cpu().tolist() for t in logits]
            return [t.tolist() for t in logits]
        # Already serialized (list of lists/numbers)
        return logits

    # Single tensor
    if hasattr(logits, "tolist"):
        if move_to_cpu:
            return logits.cpu().tolist()
        return logits.tolist()

    # Fallback: return as-is (shouldn't happen with proper types)
    return logits
