import pytest

from request_security import SlidingWindowRateLimiter


def test_sliding_window_limiter_recovers_after_window() -> None:
    now = [100.0]
    limiter = SlidingWindowRateLimiter(
        request_limit=2,
        window_seconds=10,
        clock=lambda: now[0],
    )

    assert limiter.admit() == (True, 0)
    assert limiter.admit() == (True, 0)
    assert limiter.admit() == (False, 10)

    now[0] = 110.0

    assert limiter.admit() == (True, 0)


@pytest.mark.parametrize(
    ("request_limit", "window_seconds"),
    [(0, 10), (10, 0), (-1, 10)],
)
def test_sliding_window_limiter_rejects_invalid_configuration(
    request_limit: int,
    window_seconds: int,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        SlidingWindowRateLimiter(
            request_limit=request_limit,
            window_seconds=window_seconds,
        )
