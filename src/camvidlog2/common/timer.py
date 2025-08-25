import time


class CumulativeTimer:
    """
    Utility class to cumulatively time code execution using a context manager.
    Example usage:
        timer = CumulativeTimer()
        with timer:
            # code block 1
        with timer:
            # code block 2
        print(timer.elapsed)  # Total elapsed time across both blocks
    """

    def __init__(self):
        self._start = None
        self._elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.perf_counter()
        if self._start is not None:
            self._elapsed += end - self._start
            self._start = None

    @property
    def elapsed(self):
        """Total accumulated time in seconds."""
        return self._elapsed

    def reset(self):
        """Reset the accumulated timer to zero."""
        self._elapsed = 0.0
        self._start = None
