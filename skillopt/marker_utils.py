"""Helpers for balanced regions delimited by paired text markers."""
from __future__ import annotations


def find_marked_region(
    text: str, start_marker: str, end_marker: str, start_at: int = 0
) -> tuple[int, int] | None:
    """Return ``(start, end_exclusive)`` for a balanced marker region.

    Repeated/nested start markers are treated as nesting, so the matching end
    is the end marker that returns the nesting depth to zero.
    """
    start = text.find(start_marker, start_at)
    if start == -1:
        return None

    depth = 1
    cursor = start + len(start_marker)
    while cursor < len(text):
        next_start = text.find(start_marker, cursor)
        next_end = text.find(end_marker, cursor)
        if next_end == -1:
            return None
        if next_start != -1 and next_start < next_end:
            depth += 1
            cursor = next_start + len(start_marker)
        else:
            depth -= 1
            cursor = next_end + len(end_marker)
            if depth == 0:
                return start, cursor
    return None


def is_within_marked_region(
    text: str, target: str, start_marker: str, end_marker: str
) -> bool:
    """Return whether the first occurrence of *target* is inside a region."""
    if not target:
        return False
    target_at = text.find(target)
    if target_at == -1:
        return False
    cursor = 0
    while True:
        bounds = find_marked_region(text, start_marker, end_marker, cursor)
        if bounds is None:
            return False
        start, end = bounds
        if start <= target_at < end:
            return True
        cursor = end
