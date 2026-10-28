"""Instrument definitions: dated fixed-rate bonds and constant-maturity par instruments.

``FixedRateBond`` models a real government bond (Bund, Bobl, Schatz) with an
annual or semiannual coupon, ACT/ACT ICMA accrual, unadjusted payment dates,
no ex-coupon period, and optional irregular (short or long) first coupon via
the interest commencement date.

``ParInstrument`` models one point of a constant-maturity par yield curve
(US Treasury par curve): a theoretical security priced at par on a regular
year-fraction grid, with no calendar.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .conventions import add_months, icma_fraction, quasi_coupon_schedule


@dataclass(frozen=True)
class FixedRateBond:
    """Fixed-coupon bullet bond.

    Attributes:
        isin: identifier.
        coupon: annual coupon rate as a decimal (0.029 for 2.9 %).
        maturity: redemption date (unadjusted).
        frequency: coupons per year (1 for Bunds, 2 for US Treasuries).
        interest_start: interest commencement date. ``None`` means the bond is
            assumed to have only regular coupon periods.
        first_coupon: first payment date; must lie on the quasi-coupon schedule.
            ``None`` with an ``interest_start`` means the first quasi-coupon date
            after ``interest_start`` (short or regular first coupon); a later
            date gives a long first coupon.
        face: redemption amount per unit of nominal (100 = prices per 100).
    """

    isin: str
    coupon: float
    maturity: date
    frequency: int = 1
    interest_start: date | None = None
    first_coupon: date | None = None
    face: float = 100.0

    def __post_init__(self) -> None:
        if self.first_coupon is not None:
            if self.interest_start is None:
                raise ValueError("first_coupon requires interest_start")
            sched = quasi_coupon_schedule(self.maturity, self.frequency, self.interest_start)
            if self.first_coupon not in sched or self.first_coupon <= self.interest_start:
                raise ValueError("first_coupon must be a quasi-coupon date after interest_start")

    def schedule(self, before: date) -> list[date]:
        """Quasi-coupon dates from the last one ``<= before`` up to maturity."""
        bound = before if self.interest_start is None else min(before, self.interest_start)
        return quasi_coupon_schedule(self.maturity, self.frequency, bound)

    def first_coupon_date(self) -> date | None:
        """First payment date after the interest commencement date (if known)."""
        if self.interest_start is None:
            return None
        if self.first_coupon is not None:
            return self.first_coupon
        sched = quasi_coupon_schedule(self.maturity, self.frequency, self.interest_start)
        return next(d for d in sched if d > self.interest_start)

    def _accrual_start(self, settle: date, sched: list[date]) -> date:
        """Start of the coupon period running at ``settle``."""
        last_quasi = max(d for d in sched if d <= settle)
        if self.interest_start is not None and settle < self.first_coupon_date():  # type: ignore[operator]
            return self.interest_start
        return last_quasi

    def accrued(self, settle: date) -> float:
        """Accrued interest per ``face`` at ``settle`` (ACT/ACT ICMA)."""
        if settle >= self.maturity:
            return 0.0
        if self.interest_start is not None and settle < self.interest_start:
            raise ValueError("settlement before interest commencement")
        sched = self.schedule(settle)
        start = self._accrual_start(settle, sched)
        return self.face * self.coupon / self.frequency * icma_fraction(start, settle, sched)

    def cashflows(self, settle: date) -> list[tuple[date, float]]:
        """Payments strictly after ``settle`` as ``(date, amount per face)``.

        A coupon falling on ``settle`` belongs to the seller and is excluded.
        """
        if settle >= self.maturity:
            return []
        sched = self.schedule(settle)
        regular = self.face * self.coupon / self.frequency
        first_pay = self.first_coupon_date()
        flows: list[tuple[date, float]] = []
        for d in sched:
            if d <= settle:
                continue
            if first_pay is not None and d < first_pay:
                continue
            if first_pay is not None and d == first_pay:
                amount = regular * icma_fraction(self.interest_start, d, sched)  # type: ignore[arg-type]
            else:
                amount = regular
            flows.append((d, amount))
        last_date, last_amount = flows[-1]
        flows[-1] = (last_date, last_amount + self.face)
        return flows

    def coupons_between(self, start: date, end: date) -> list[tuple[date, float]]:
        """Coupon payments (excluding redemption) with ``start < date <= end``."""
        flows = []
        for d, amount in self.cashflows(start):
            if d > end:
                break
            flows.append((d, amount - (self.face if d == self.maturity else 0.0)))
        return flows

    def period_exponents(self, settle: date, dates: list[date]) -> list[float]:
        """ACT/ACT ICMA time from ``settle`` to each date, in coupon periods."""
        sched = quasi_coupon_schedule(self.maturity, self.frequency, settle)
        return [icma_fraction(settle, d, sched) for d in dates]


@dataclass(frozen=True)
class ParInstrument:
    """Constant-maturity par instrument (one point of a par yield curve).

    For ``tenor <= 1/frequency`` the instrument is a single payment of
    ``1 + rate * tenor`` at ``tenor`` (money-market, bond-equivalent basis);
    otherwise it pays ``rate / frequency`` every ``1/frequency`` years and 1 at
    maturity. In both cases its price is 1 (par).
    """

    tenor: float
    rate: float
    frequency: int = 2

    def cashflow_times(self) -> list[tuple[float, float]]:
        """``(time in years, amount)`` per unit face."""
        step = 1.0 / self.frequency
        if self.tenor <= step + 1e-12:
            return [(self.tenor, 1.0 + self.rate * self.tenor)]
        n = round(self.tenor / step)
        if abs(n * step - self.tenor) > 1e-9:
            raise ValueError(f"Tenor {self.tenor} is not a multiple of the coupon period")
        flows = [((k + 1) * step, self.rate * step) for k in range(n)]
        flows[-1] = (flows[-1][0], flows[-1][1] + 1.0)
        return flows


def months_to_years(months: float) -> float:
    """Tenor in months to years (exact twelfths)."""
    return months / 12.0


__all__ = ["FixedRateBond", "ParInstrument", "add_months", "months_to_years"]
