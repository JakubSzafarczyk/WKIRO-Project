import numpy as np


def create_windows(sequence: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
    """
    Dzieli sekwencję czasową na okna.

    Parameters
    ----------
    sequence:
        Tablica o kształcie (n_frames, n_features).
    window_size:
        Liczba klatek w jednym oknie.
    step_size:
        Przesunięcie okna.

    Returns
    -------
    np.ndarray
        Tablica o kształcie (n_windows, window_size, n_features).
    """
    if sequence.ndim != 2:
        raise ValueError("Sekwencja musi mieć kształt (n_frames, n_features).")
    if window_size <= 0:
        raise ValueError("window_size musi być dodatnie.")
    if step_size <= 0:
        raise ValueError("step_size musi być dodatnie.")

    n_frames = sequence.shape[0]

    if n_frames < window_size:
        return np.empty((0, window_size, sequence.shape[1]), dtype=sequence.dtype)

    windows = []

    for start in range(0, n_frames - window_size + 1, step_size):
        end = start + window_size
        windows.append(sequence[start:end])

    return np.stack(windows).astype(sequence.dtype, copy=False)


def create_labeled_windows(
    sequence: np.ndarray,
    label: int,
    window_size: int,
    step_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create windows and a matching vector of repeated labels."""
    windows = create_windows(sequence, window_size=window_size, step_size=step_size)
    labels = np.full((windows.shape[0],), label, dtype=np.int64)
    return windows, labels


def window_start_indices(n_frames: int, window_size: int, step_size: int) -> np.ndarray:
    """Return frame indices where complete windows start."""
    if window_size <= 0:
        raise ValueError("window_size musi być dodatnie.")
    if step_size <= 0:
        raise ValueError("step_size musi być dodatnie.")
    if n_frames < window_size:
        return np.empty((0,), dtype=np.int64)
    return np.arange(0, n_frames - window_size + 1, step_size, dtype=np.int64)


def create_window_metadata(
    n_frames: int,
    window_size: int,
    step_size: int,
    **metadata: object,
) -> list[dict]:
    """Create one metadata dictionary per complete window."""
    starts = window_start_indices(n_frames, window_size=window_size, step_size=step_size)
    rows = []
    for start in starts:
        rows.append(
            {
                **metadata,
                "start_frame": int(start),
                "end_frame": int(start + window_size),
            }
        )
    return rows
