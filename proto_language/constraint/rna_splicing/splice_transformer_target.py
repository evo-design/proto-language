"""Window concatenated three-segment targets to the SpliceTransformer length.

SpliceTransformer requires a fixed ``TARGET_LENGTH`` (1000 bp) target window, but
some constructs concatenate real gene fragments (e.g. the HSV-TK intron-boundary
flanks) whose combined length exceeds that. This helper extracts a
``TARGET_LENGTH`` window centred on the requested splice positions and remaps
those positions into the window so donor/acceptor indices stay aligned with the
trimmed sequence.

Examples:
    >>> start = splice_target_window_start(1436, [564], [867])  # 215
    >>> remap_positions([564], start)  # [349]
"""

import logging

from proto_tools import TARGET_LENGTH as SPLICE_TRANSFORMER_TARGET_LENGTH

logger = logging.getLogger(__name__)


def splice_target_window_start(target_length: int, *position_lists: list[int]) -> int:
    """Compute the start offset of the SpliceTransformer target window.

    The window has width ``TARGET_LENGTH`` and is centred on the requested splice
    positions, clamped so it stays within ``[0, target_length]``. When the target
    already matches ``TARGET_LENGTH`` the offset is ``0`` (identity window).

    Args:
        target_length (int): Length of the concatenated target sequence(s).
        position_lists (list[int]): One or more variadic lists of zero-indexed
            positions (e.g. donor and acceptor positions) that must all fall
            inside the returned window.

    Returns:
        int: Zero-indexed start offset of the ``TARGET_LENGTH`` window.

    Raises:
        ValueError: If no positions are supplied, if ``target_length`` is shorter
            than ``TARGET_LENGTH``, or if the positions span more than
            ``TARGET_LENGTH`` and cannot all fit in a single window.
    """
    positions = [pos for position_list in position_lists for pos in position_list]
    if not positions:
        raise ValueError("At least one splice position is required to window the SpliceTransformer target.")

    if target_length < SPLICE_TRANSFORMER_TARGET_LENGTH:
        raise ValueError(
            f"SpliceTransformer target length {target_length} is shorter than the required "
            f"{SPLICE_TRANSFORMER_TARGET_LENGTH} bp; cannot window."
        )
    if target_length == SPLICE_TRANSFORMER_TARGET_LENGTH:
        return 0

    lo, hi = min(positions), max(positions)
    if not (lo >= 0 and hi < target_length):
        raise ValueError(
            f"Splice positions must lie within the target sequence [0, {target_length}); got [{lo}, {hi}]."
        )
    if hi - lo >= SPLICE_TRANSFORMER_TARGET_LENGTH:
        raise ValueError(
            f"Splice positions span {hi - lo + 1} bp, which exceeds the SpliceTransformer "
            f"target window of {SPLICE_TRANSFORMER_TARGET_LENGTH} bp; cannot fit them in one window."
        )

    center = (lo + hi) // 2
    start = center - SPLICE_TRANSFORMER_TARGET_LENGTH // 2
    start = max(0, min(start, target_length - SPLICE_TRANSFORMER_TARGET_LENGTH))

    if lo < start or hi >= start + SPLICE_TRANSFORMER_TARGET_LENGTH:
        raise ValueError(
            f"Could not place a {SPLICE_TRANSFORMER_TARGET_LENGTH} bp window covering positions "
            f"[{lo}, {hi}] within a target of length {target_length}."
        )
    if start:
        logger.debug(
            "Windowing SpliceTransformer target of %d bp to %d bp starting at offset %d.",
            target_length,
            SPLICE_TRANSFORMER_TARGET_LENGTH,
            start,
        )
    return int(start)


def apply_target_window(target_seqs: list[str], start: int) -> list[str]:
    """Slice each target to the ``TARGET_LENGTH`` window beginning at ``start``.

    Args:
        target_seqs (list[str]): Equal-length concatenated target sequences.
        start (int): Zero-indexed window start from :func:`splice_target_window_start`.

    Returns:
        list[str]: Windowed target sequences, each exactly ``TARGET_LENGTH`` bp.
    """
    end = start + SPLICE_TRANSFORMER_TARGET_LENGTH
    return [target_seq[start:end] for target_seq in target_seqs]


def remap_positions(positions: list[int], start: int) -> list[int]:
    """Shift positions into the windowed coordinate frame.

    Args:
        positions (list[int]): Positions in the original target coordinate frame.
        start (int): Zero-indexed window start from :func:`splice_target_window_start`.

    Returns:
        list[int]: Positions relative to the start of the windowed target.
    """
    return [pos - start for pos in positions]
