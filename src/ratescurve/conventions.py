"""Calendars, date arithmetic and day-count conventions.

Only what the project needs, implemented explicitly so every convention can
be audited:

* TARGET2 business days (weekends plus New Year, Good Friday, Easter Monday,
  1 May, 25 and 26 December). Used for euro bond settlement (T+2) and for
  Eurex delivery and last trading days.
* ACT/365F and ACT/360 year fractions.
* ACT/ACT ICMA accrual fractions on a regular quasi-coupon schedule
  (ICMA Rule 251), which also cover short and long first coupons.

References: ICMA Primary Market Handbook, Rule 251; ECB TARGET2 closing days.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta


def easter_sunday(year: int) -> date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm, Meeus/Jones/Butcher)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month, day = divmod(h + ell - 7 * m + 114, 31)
    return date(year, month, day + 1)


def is_target_holiday(d: date) -> bool:
    """True on TARGET2 closing days that fall on a weekday."""
    if (d.month, d.day) in {(1, 1), (5, 1), (12, 25), (12, 26)}:
        return True
    easter = easter_sunday(d.year)
    return d in (easter - timedelta(days=2), easter + timedelta(days=1))


def is_business_day(d: date) -> bool:
    """TARGET2 business day: not a weekend and not a TARGET2 holiday."""
    return d.weekday() < 5 and not is_target_holiday(d)


def add_business_days(d: date, n: int) -> date:
    """Move ``n`` TARGET2 business days forward (``n > 0``) or backward (``n < 0``)."""
    step = 1 if n >= 0 else -1
    remaining = abs(n)
    current = d
    while remaining:
        current += timedelta(days=step)
        if is_business_day(current):
            remaining -= 1
    return current


def following(d: date) -> date:
    """Following business-day convention."""
    while not is_business_day(d):
        d += timedelta(days=1)
    return d


def add_months(d: date, months: int) -> date:
    """Calendar month shift; the day is clipped to the end of the target month."""
    index = d.year * 12 + (d.month - 1) + months
    year, month0 = divmod(index, 12)
    last = calendar.monthrange(year, month0 + 1)[1]
    return date(year, month0 + 1, min(d.day, last))


def yf_act365f(d0: date, d1: date) -> float:
    """ACT/365 Fixed year fraction."""
    return (d1 - d0).days / 365.0


def yf_act360(d0: date, d1: date) -> float:
    """ACT/360 year fraction (money-market convention, e.g. €STR, repo)."""
    return (d1 - d0).days / 360.0


def quasi_coupon_schedule(maturity: date, frequency: int, before: date) -> list[date]:
    """Regular quasi-coupon dates generated backwards from ``maturity``.

    Returns the ascending list of dates ``maturity - k * 12/frequency months``
    (k = 0, 1, ...) down to and including the first date ``<= before``.
    """
    if frequency not in (1, 2, 4, 12):
        raise ValueError(f"Unsupported coupon frequency {frequency}")
    step = 12 // frequency
    dates = [maturity]
    k = 0
    while dates[-1] > before:
        k += 1
        dates.append(add_months(maturity, -step * k))
    return sorted(dates)


def icma_fraction(start: date, end: date, schedule: list[date]) -> float:
    """ACT/ACT ICMA accrual between ``start`` and ``end``, in coupon periods.

    The interval is split along the quasi-coupon ``schedule``; each piece counts
    ``days in piece / days in its quasi-coupon period``. A regular full period
    therefore counts exactly 1, and short or long first periods follow ICMA
    Rule 251. ``schedule`` must cover ``[start, end]``.
    """
    if end < start:
        raise ValueError("end before start")
    if start < schedule[0] or end > schedule[-1]:
        raise ValueError("schedule does not cover the accrual interval")
    total = 0.0
    for q0, q1 in zip(schedule[:-1], schedule[1:]):
        lo, hi = max(start, q0), min(end, q1)
        if hi > lo:
            total += (hi - lo).days / (q1 - q0).days
    return total
