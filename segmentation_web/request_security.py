"""In-process request guards for the private inference service."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable


class SlidingWindowRateLimiter:
    """Apply one process-wide fixed quota over a rolling time window."""

    def __init__(
        self,
        *,
        request_limit: int,
        window_seconds: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if request_limit <= 0 or window_seconds <= 0:
            raise ValueError("rate limit and window must be positive")
        self.request_limit = request_limit
        self.window_seconds = window_seconds
        self._clock = clock
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def admit(self) -> tuple[bool, int]:
        """Return admission and whole seconds until another slot is available."""

        now = self._clock()
        cutoff = now - self.window_seconds
        with self._lock:
            while self._timestamps and self._timestamps[0] <= cutoff:
                self._timestamps.popleft()
            if len(self._timestamps) >= self.request_limit:
                retry_after = max(
                    1,
                    math.ceil(
                        self._timestamps[0] + self.window_seconds - now
                    ),
                )
                return False, retry_after
            self._timestamps.append(now)
            return True, 0
