from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class TailRepetition:
    """A detected end-of-string periodic loop."""

    period: int
    reps: float
    span: int
    rem: int
    unit: str
    loop_start: int
    cover_ratio: float


def _match_len_from_end(text: str, p: int) -> int:
    """Count how far ``text[i] == text[i - p]`` holds walking back from the end."""
    n = len(text)
    i = 0
    lim = n - p
    while i < lim and text[n - 1 - i] == text[n - 1 - i - p]:
        i += 1
    return i


def detect_tail_repetition(
    text: str,
    *,
    max_p: int = 512,
    min_reps: float = 3.0,
    min_span: int = 80,
    tail_n: int = 4096,
) -> TailRepetition | None:
    """Find the strongest end-aligned period at the end of *text*.

    From the last character, count how far ``s[i] == s[i - p]`` holds.  An
    incomplete final cycle is allowed.  Returns ``None`` when no candidate
    meets ``min_reps`` and ``min_span``.

    ``tail_n`` bounds only the *period search* (cost control).  Once the period
    is picked, the loop is extended left over the **whole** text, so a loop far
    longer than ``tail_n`` still reports its true start.
    """
    if not text:
        return None
    n_full = len(text)
    t = text[max(0, n_full - tail_n) :]
    n = len(t)
    if n < min_span:
        return None

    best: tuple[int, int] | None = None  # (span_in_window, period)
    upper = min(max_p, max(1, n // 3))
    for p in range(1, upper + 1):
        span = _match_len_from_end(t, p) + p
        if span / p >= min_reps and span >= min_span:
            # Ties keep the smallest period (the true unit, not a multiple).
            if best is None or span > best[0]:
                best = (span, p)

    if best is None:
        return None
    p = best[1]

    # Extend over the full text: the window can truncate a much longer loop.
    span = min(_match_len_from_end(text, p) + p, n_full)
    rem = span % p
    loop_start = n_full - span
    unit = text[n_full - rem - p : n_full - rem] if rem else text[n_full - p :]
    return TailRepetition(
        period=p,
        reps=span / p,
        span=span,
        rem=rem,
        unit=unit,
        loop_start=loop_start,
        cover_ratio=span / n_full,
    )


def trim_tail_repetition(
    text: str,
    rep: TailRepetition,
    *,
    keep_reps: int = 1,
) -> str:
    """Drop the looping suffix, keeping ``keep_reps`` full periods.

    Anything after the loop start is replaced by ``unit * keep_reps``.
    Closing JSON braces (if the cut leaves the object open) is left to the
    existing JSON repair path.
    """
    keep = max(0, int(keep_reps))
    prefix = text[: rep.loop_start]
    if keep == 0 or not rep.unit:
        return prefix
    return prefix + (rep.unit * keep)
