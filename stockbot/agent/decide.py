"""Turn a policy action (target exposure) into a human-readable trading decision."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Decision:
    ticker: str
    action: str              # BUY | SELL | SHORT | COVER | HOLD
    target_exposure: float   # policy output, fraction of the capital slice, negative = short
    current_exposure: float
    delta_exposure: float
    amount_usd: float        # notional to trade (positive = buy shares, negative = sell shares)
    shares: float
    price: float
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (f"{self.ticker:6s} {self.action:5s} target={self.target_exposure:+.2f} "
                f"(now {self.current_exposure:+.2f}) ${self.amount_usd:+,.0f} = {self.shares:+.3f} sh @ {self.price:.2f} {self.note}")


def decide(ticker: str, target: float, current: float, capital: float, price: float, deadband: float = 0.05,
           allow_short: bool = True, min_trade_usd: float = 50.0) -> Decision:
    """``target``/``current`` are exposures relative to ``capital`` (the ticker's capital slice)."""
    target = float(max(-1.0, min(1.0, target)))
    if not allow_short:
        target = max(0.0, target)
    delta = target - current
    amount = delta * capital
    # closing a position is never blocked by the band or the minimum order: a remnant under the minimum (META, Nov 2026:
    # 0.8 of a share left by the fractional days) would otherwise stay for good while the plan "sold" it every day
    full_exit = abs(target) < 1e-9 and abs(current * capital) >= 1.0
    if price <= 0 or (not full_exit and (abs(delta) < deadband or abs(amount) < min_trade_usd)):
        return Decision(ticker, "HOLD", target, current, 0.0, 0.0, 0.0, price, "within deadband" if abs(delta) < deadband else "below min trade")
    if delta > 0:
        action = "COVER" if current < -1e-6 and target <= 1e-6 else ("BUY" if target > 0 else "COVER")
        if current < 0 < target:
            action = "BUY"  # cover the short and go long
    else:
        action = "SELL" if current > 1e-6 and target >= -1e-6 else "SHORT"
        if current > 0 > target:
            action = "SHORT"  # sell the long and go short
    return Decision(ticker, action, target, current, delta, amount, amount / price, price)
