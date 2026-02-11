import numpy as np


def fill_short_gaps(binary: np.ndarray, max_gap: int) -> np.ndarray:
    """
    Fill gaps of 0s with length <= max_gap that are between 1s.

    Example:
      [1, 1, 0, 0, 1, 1] with max_gap=2 -> [1, 1, 1, 1, 1, 1]
      [1, 1, 0, 0, 0, 1] with max_gap=2 -> [1, 1, 0, 0, 0, 1] (gap too long)
    """
    if max_gap <= 0 or binary.size == 0:
        return binary.copy()

    x = binary.astype(np.int32).copy()
    n = len(x)

    i = 0
    while i < n:
        if x[i] == 0:
            start = i
            while i < n and x[i] == 0:
                i += 1
            end = i  # [start, end) is the zero-run
            gap_len = end - start

            left_is_one = start - 1 >= 0 and x[start - 1] == 1
            right_is_one = end < n and x[end] == 1

            if left_is_one and right_is_one and gap_len <= max_gap:
                x[start:end] = 1
        else:
            i += 1

    return x
