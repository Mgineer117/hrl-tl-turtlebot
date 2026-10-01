"""Device utility functions for compute device normalization."""

from __future__ import annotations


def normalize_device(device: str) -> str:
    """Normalizes device string, mapping digit strings like '1' to 'cuda:1'.

    Args:
        device: Device name or index string.

    Returns:
        Normalized device string (e.g., 'cuda:1', 'cpu').
    """
    cleaned = device.strip()
    if cleaned.isdigit():
        return f"cuda:{cleaned}"
    return cleaned
