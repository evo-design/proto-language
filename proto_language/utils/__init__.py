"""Shared utilities for the language runtime."""

from proto_language.utils.gradients import (
    MERGERS,
    GradientMerger,
    GradientMergerName,
    MGDAMerger,
    PCGradMerger,
    WeightedSumMerger,
    align_norms,
    normalize_gradient,
)
from proto_language.utils.io import (
    build_results,
    flatten_constraints,
    flatten_constructs,
    flatten_optimization,
    flatten_sequences,
    load_fasta,
    to_csv,
    to_json,
    to_tsv,
    to_xlsx,
    to_xlsx_workbook,
    write_export,
    write_results_folder,
)
from proto_language.utils.ml_optimizers import (
    ML_OPTIMIZERS,
    SGD,
    Adam,
    AdamConfig,
    MLOptimizer,
    MLOptimizerType,
)
from proto_language.utils.scheduling import (
    SCHEDULES,
    Schedule,
    Scheduler,
    constant_schedule,
    cosine_anneal,
    exponential_decay,
    linear_decay,
    progress,
    quadratic_decay,
)
from proto_language.utils.scoring import (
    LOG_BASE,
    MAX_ENERGY,
    MAX_GC_CONTENT,
    MIN_ENERGY,
    MIN_GC_CONTENT,
    calculate_gc_content,
    calculate_normalized_deviation,
    calculate_percentage_range_deviation,
    calculate_range_deviation,
    inverse_sigmoid_score,
    sigmoid_score,
    softmax,
    validate_range,
)
from proto_language.utils.sequence_matrices import (
    mean_peak_probability,
    one_hot_protein_matrix,
)
from proto_language.utils.serialization import (
    format_pydantic_error,
    is_plain_int,
    make_json_safe,
)
from proto_language.utils.spinner import (
    has_active_progress_bar,
    progress_bar,
    set_substatus,
)

__all__ = [
    # Scoring math
    "MIN_ENERGY",
    "MAX_ENERGY",
    "LOG_BASE",
    "MIN_GC_CONTENT",
    "MAX_GC_CONTENT",
    "validate_range",
    "calculate_range_deviation",
    "calculate_percentage_range_deviation",
    "calculate_gc_content",
    "calculate_normalized_deviation",
    "sigmoid_score",
    "inverse_sigmoid_score",
    "softmax",
    # Serialization
    "format_pydantic_error",
    "make_json_safe",
    "is_plain_int",
    # Sequence matrices
    "one_hot_protein_matrix",
    "mean_peak_probability",
    # I/O
    "load_fasta",
    "build_results",
    "flatten_sequences",
    "flatten_constraints",
    "flatten_constructs",
    "flatten_optimization",
    "to_csv",
    "to_tsv",
    "to_json",
    "to_xlsx",
    "to_xlsx_workbook",
    "write_export",
    "write_results_folder",
    # Gradient utilities
    "GradientMerger",
    "GradientMergerName",
    "WeightedSumMerger",
    "PCGradMerger",
    "MGDAMerger",
    "MERGERS",
    "align_norms",
    "normalize_gradient",
    # ML optimizers
    "MLOptimizer",
    "MLOptimizerType",
    "SGD",
    "Adam",
    "AdamConfig",
    "ML_OPTIMIZERS",
    # Scheduling
    "Schedule",
    "Scheduler",
    "SCHEDULES",
    "progress",
    "constant_schedule",
    "linear_decay",
    "cosine_anneal",
    "exponential_decay",
    "quadratic_decay",
    # Spinner / progress (reused from proto-tools)
    "progress_bar",
    "set_substatus",
    "has_active_progress_bar",
]
