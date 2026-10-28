"""Calendar and day-count conventions."""

from datetime import date

import pytest

from ratescurve.conventions import (
    add_business_days,
    add_months,
    easter_sunday,
    following,
    icma_fraction,
    is_business_day,
    quasi_coupon_schedule,
    yf_act360,
    yf_act365f,
)


@pytest.mark.parametrize(
    ("year", "expected"),
    [(2024, date(2024, 3, 31)), (2025, date(2025, 4, 20)), (2026, date(2026, 4, 5)), (2027, date(2027, 3, 28))],
)
def test_easter(year, expected):
    assert easter_sunday(year) == expected


def test_target_holidays_and_weekends():
    assert not is_business_day(date(2026, 4, 3))  # Good Friday
    assert not is_business_day(date(2026, 4, 6))  # Easter Monday
    assert not is_business_day(date(2026, 5, 1))
    assert not is_business_day(date(2026, 12, 25))
    assert not is_business_day(date(2026, 12, 26))
    assert not is_business_day(date(2026, 10, 31))  # Saturday
    assert is_business_day(date(2026, 10, 27))


def test_settlement_t_plus_2():
    assert add_business_days(date(2026, 10, 27), 2) == date(2026, 10, 29)
    assert add_business_days(date(2026, 10, 1), 2) == date(2026, 10, 5)  # over a weekend
    assert add_business_days(date(2026, 4, 2), 2) == date(2026, 4, 8)  # over Easter
    assert add_business_days(date(2026, 12, 10), -2) == date(2026, 12, 8)
    assert following(date(2026, 1, 10)) == date(2026, 1, 12)


def test_add_months_clips_end_of_month():
    assert add_months(date(2026, 1, 31), 1) == date(2026, 2, 28)
    assert add_months(date(2028, 1, 31), 1) == date(2028, 2, 29)
    assert add_months(date(2026, 8, 15), -12) == date(2025, 8, 15)
    assert add_months(date(2026, 3, 31), -1) == date(2026, 2, 28)


def test_year_fractions():
    assert yf_act365f(date(2026, 1, 1), date(2027, 1, 1)) == 1.0
    assert yf_act360(date(2026, 10, 29), date(2026, 12, 10)) == pytest.approx(42 / 360)


def test_icma_regular_short_long():
    sched = quasi_coupon_schedule(date(2036, 2, 15), 1, date(2025, 1, 1))
    assert icma_fraction(date(2026, 2, 15), date(2027, 2, 15), sched) == pytest.approx(1.0)
    # short: 37 days of a 365-day period
    assert icma_fraction(date(2026, 1, 9), date(2026, 2, 15), sched) == pytest.approx(37 / 365)
    # long first coupon: 37/365 + 1
    assert icma_fraction(date(2026, 1, 9), date(2027, 2, 15), sched) == pytest.approx(1 + 37 / 365)
    with pytest.raises(ValueError):
        icma_fraction(date(2027, 1, 1), date(2026, 1, 1), sched)
    with pytest.raises(ValueError):
        icma_fraction(date(2020, 1, 1), date(2026, 1, 1), sched)


def test_quasi_schedule_semiannual():
    sched = quasi_coupon_schedule(date(2030, 11, 15), 2, date(2026, 10, 1))
    assert sched[0] == date(2026, 5, 15)
    assert sched[-1] == date(2030, 11, 15)
    assert all((b - a).days in range(181, 185) for a, b in zip(sched[:-1], sched[1:]))
    with pytest.raises(ValueError):
        quasi_coupon_schedule(date(2030, 1, 1), 3, date(2026, 1, 1))
