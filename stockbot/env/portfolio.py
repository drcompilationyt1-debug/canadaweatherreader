"""Single-asset portfolio accounting shared by the simulator and the paper broker."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TradeResult:
    delta_shares: float
    traded_value: float
    cost: float


class Portfolio:
    """``fees`` (a ``FeeSchedule``) replaces the proportional ``commission`` with the broker's real
    per-share / per-order schedule; ``fee_scale`` is the fraction of real equity one simulated
    position stands for (the per-ticker capital slice), so per-order minimums bite realistically."""

    def __init__(self, cash: float, commission: float = 0.0005, slippage: float = 0.0005,
                 allow_short: bool = True, borrow_rate_annual: float = 0.03, max_leverage: float = 1.0,
                 fees=None, fee_scale: float = 1.0):
        self.initial_cash = float(cash)
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.allow_short = bool(allow_short)
        self.borrow_daily = float(borrow_rate_annual) / 252.0
        self.max_leverage = float(max_leverage)
        self.fees = fees
        self.fee_scale = float(fee_scale) if fee_scale else 1.0
        self.reset()

    def trade_cost(self, delta_shares: float, price: float) -> float:
        traded = abs(delta_shares) * price
        if self.fees is None:
            return traded * (self.commission + self.slippage)
        side = "buy" if delta_shares > 0 else "sell"
        return self.fees.cost_scaled(abs(delta_shares), price, side, self.fee_scale) + traded * self.slippage

    def reset(self, cash: float | None = None) -> None:
        if cash is not None:
            self.initial_cash = float(cash)
        self.cash = self.initial_cash
        self.shares = 0.0
        self.total_costs = 0.0
        self.n_trades = 0

    # ------------------------------------------------------------------ state
    def equity(self, price: float) -> float:
        return self.cash + self.shares * price

    def exposure(self, price: float) -> float:
        eq = self.equity(price)
        return (self.shares * price / eq) if eq > 1e-9 else 0.0

    def clip_target(self, target: float) -> float:
        lo = -self.max_leverage if self.allow_short else 0.0
        return float(min(max(target, lo), self.max_leverage))

    # ------------------------------------------------------------------ actions
    def rebalance(self, target_exposure: float, price: float) -> TradeResult:
        target = self.clip_target(target_exposure)
        eq = self.equity(price)
        target_shares = target * eq / price if price > 0 else 0.0
        delta = target_shares - self.shares
        traded = abs(delta) * price
        cost = self.trade_cost(delta, price) if traded > 0 else 0.0
        self.cash -= delta * price + cost
        self.shares += delta
        self.total_costs += cost
        if traded > 0:
            self.n_trades += 1
        return TradeResult(delta, traded, cost)

    def accrue(self, price: float) -> float:
        """Daily borrow fee on short positions."""
        if self.shares < 0:
            fee = -self.shares * price * self.borrow_daily
            self.cash -= fee
            self.total_costs += fee
            return fee
        return 0.0
