"""Trading fees, charged everywhere the bot trades: the simulator, the built-in paper broker and -
virtually, since Alpaca's paper account is free - the Alpaca runs.  The policy therefore learns
with the same costs it will pay for real.

Default preset ``moomoo``: moomoo's per-share schedule for US stocks (commission $0.0049/share
min $0.99 per order + platform fee $0.005/share min $1.00 per order, SEC fee and FINRA trading
activity fee on sells), i.e. the moomoo Canada / international schedule, which is the most
expensive one - a bot that is profitable under it is profitable under the cheaper US-resident
schedule (``moomoo_us``: commission-free, regulatory fees only) too.
"""
from __future__ import annotations

from dataclasses import dataclass, field

PRESETS: dict[str, dict[str, float]] = {
    # moomoo (Canada / non-US residents): per-share commission + platform fee with per-order minimums
    "moomoo": {"commission_per_share": 0.0049, "commission_min": 0.99, "platform_per_share": 0.005, "platform_min": 1.00,
               "per_order": 0.0, "sec_fee_rate": 0.0000206, "sec_min": 0.01, "finra_taf_per_share": 0.000195,
               "fractional_pct": 0.0099, "fractional_max": 0.99},
    # moomoo Financial Inc. for US residents: commission-free, regulatory fees on sells only
    "moomoo_us": {"commission_per_share": 0.0, "commission_min": 0.0, "platform_per_share": 0.0, "platform_min": 0.0,
                  "per_order": 0.0, "sec_fee_rate": 0.0000206, "sec_min": 0.01, "finra_taf_per_share": 0.000195,
                  "fractional_pct": 0.0, "fractional_max": 0.0},
    # moomoo flat $0.99 per order (non-US residents, promotional schedule)
    "moomoo_intl": {"commission_per_share": 0.0, "commission_min": 0.0, "platform_per_share": 0.0, "platform_min": 0.0,
                    "per_order": 0.99, "sec_fee_rate": 0.0000206, "sec_min": 0.01, "finra_taf_per_share": 0.000195,
                    "fractional_pct": 0.0099, "fractional_max": 0.99},
    "none": {"commission_per_share": 0.0, "commission_min": 0.0, "platform_per_share": 0.0, "platform_min": 0.0, "per_order": 0.0,
             "sec_fee_rate": 0.0, "sec_min": 0.0, "finra_taf_per_share": 0.0, "fractional_pct": 0.0, "fractional_max": 0.0},
    # moomoo Canada, Canadian stocks (TSX / TSX-V, CAD): commission C$0.0049/sh min C$0.49 + platform C$0.01/sh min C$1.00,
    # no SEC / FINRA fees (ECN fees depend on the venue and are ignored); whole shares only
    "moomoo_ca": {"commission_per_share": 0.0049, "commission_min": 0.49, "platform_per_share": 0.01, "platform_min": 1.00,
                  "per_order": 0.0, "sec_fee_rate": 0.0, "sec_min": 0.0, "finra_taf_per_share": 0.0,
                  "fractional_pct": 0.0, "fractional_max": 0.0},
}


@dataclass
class FeeSchedule:
    preset: str = "moomoo"
    commission_per_share: float = 0.0049
    commission_min: float = 0.99
    platform_per_share: float = 0.005
    platform_min: float = 1.00
    per_order: float = 0.0
    sec_fee_rate: float = 0.0000206     # sells: fraction of the notional
    sec_min: float = 0.01
    finra_taf_per_share: float = 0.000195   # sells
    fractional_pct: float = 0.0099      # orders below one share: percentage of the notional ...
    fractional_max: float = 0.99        # ... capped here
    max_cost_bps: float = 25.0          # orders whose fees would exceed this share of their value are not worth placing
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_preset(cls, name: str = "moomoo", **overrides) -> "FeeSchedule":
        if name == "bps":  # legacy proportional model handled by the callers (env.commission)
            name = "none"
        if name not in PRESETS:
            raise ValueError(f"unknown fee preset {name!r}; choose from {sorted(PRESETS)} or bps")
        params = {**PRESETS[name], **{k: float(v) for k, v in overrides.items() if v is not None}}
        return cls(preset=name, **params)

    @classmethod
    def from_config(cls, cfg) -> "FeeSchedule | None":
        """``None`` when the config asks for the legacy proportional commission (``fees.preset: bps``)."""
        f = dict(cfg.section("fees")) if hasattr(cfg, "section") else dict(cfg or {})
        preset = str(f.pop("preset", "moomoo") or "moomoo").lower()
        if preset == "bps":
            return None
        return cls.from_preset(preset, **{k: v for k, v in f.items() if k in cls.__dataclass_fields__ and k not in ("preset", "extra")})

    # ------------------------------------------------------------------ pricing
    def cost(self, shares: float, price: float, side: str = "buy") -> float:
        """Total fee of one order of ``shares`` at ``price`` (USD)."""
        shares = abs(float(shares))
        if shares <= 0 or price <= 0:
            return 0.0
        notional = shares * price
        if shares < 1.0 and self.fractional_pct > 0:
            fee = min(notional * self.fractional_pct, self.fractional_max)
        else:
            fee = self.per_order
            if self.commission_per_share > 0 or self.commission_min > 0:
                fee += max(shares * self.commission_per_share, self.commission_min)
            if self.platform_per_share > 0 or self.platform_min > 0:
                fee += max(shares * self.platform_per_share, self.platform_min)
        if str(side).lower().startswith("sell"):
            if self.sec_fee_rate > 0:
                fee += max(notional * self.sec_fee_rate, self.sec_min)
            fee += shares * self.finra_taf_per_share
        return float(fee)

    def cost_scaled(self, shares: float, price: float, side: str, scale: float) -> float:
        """Fee for a simulated trade whose real-world size is ``scale`` x the simulated one (per-order
        minimums then bite the way they will on a capital slice), expressed back in simulated dollars."""
        scale = float(scale) if scale and scale > 0 else 1.0
        return self.cost(shares * scale, price, side) / scale

    def fixed_min(self) -> float:
        """Smallest possible fee of a whole-share order."""
        return self.per_order + self.commission_min + self.platform_min

    def min_trade_usd(self, max_cost_bps: float | None = None) -> float:
        """Order value below which the fixed minimums alone exceed ``max_cost_bps`` of the order."""
        bps = float(max_cost_bps if max_cost_bps is not None else self.max_cost_bps)
        fixed = self.fixed_min()
        return fixed / (bps / 1e4) if fixed > 0 and bps > 0 else 0.0

    def describe(self) -> str:
        if self.per_order and not self.commission_per_share:
            base = f"${self.per_order:.2f}/order"
        elif self.commission_per_share or self.platform_per_share:
            base = (f"${self.commission_per_share:.4f}/sh (min ${self.commission_min:.2f}) + platform ${self.platform_per_share:.3f}/sh "
                    f"(min ${self.platform_min:.2f})")
        else:
            base = "commission-free"
        reg = f"; sells + SEC {1e6 * self.sec_fee_rate:.1f}/M + TAF ${self.finra_taf_per_share:.6f}/sh" if (self.sec_fee_rate or self.finra_taf_per_share) else ""
        return f"{self.preset}: {base}{reg}"


class FeeBook:
    """One ``FeeSchedule`` per market (``us`` / ``ca``), looked up by ticker.

    ``fees.preset`` is the US / default schedule; ``fees.by_market`` overrides per market
    (default ``{ca: moomoo_ca}``).  ``FeeBook.for_ticker`` is what the simulator, the paper broker
    and the runner use, so every market is charged its own broker fees."""

    def __init__(self, default: "FeeSchedule | None", by_market: dict | None = None):
        self.default = default
        self.by_market: dict[str, FeeSchedule | None] = dict(by_market or {})

    @classmethod
    def from_config(cls, cfg) -> "FeeBook":
        from .markets import MARKETS

        default = FeeSchedule.from_config(cfg)
        f = dict(cfg.section("fees")) if hasattr(cfg, "section") else dict(cfg or {})
        overrides = dict(f.get("by_market") or {})
        by_market: dict[str, FeeSchedule | None] = {}
        for market, spec in MARKETS.items():
            preset = overrides.get(market, spec["fees"] if market != "us" else None)
            if preset is None:
                continue
            preset = str(preset).lower()
            if preset == "bps":
                by_market[market] = None
            elif default is not None and preset == default.preset:
                by_market[market] = default
            else:
                by_market[market] = FeeSchedule.from_preset(preset, max_cost_bps=f.get("max_cost_bps"))
        return cls(default, by_market)

    @classmethod
    def from_names(cls, default: str | None, by_market: dict[str, str] | None = None) -> "FeeBook":
        d = None if default in (None, "bps") else FeeSchedule.from_preset(str(default))
        return cls(d, {m: (None if p == "bps" else FeeSchedule.from_preset(p)) for m, p in (by_market or {}).items()})

    def for_market(self, market: str) -> "FeeSchedule | None":
        return self.by_market.get(market, self.default)

    def for_ticker(self, ticker: str) -> "FeeSchedule | None":
        from .markets import market_of

        return self.for_market(market_of(ticker))

    def names(self) -> dict[str, str]:
        out = {"default": self.default.preset if self.default else "bps"}
        out.update({m: (s.preset if s else "bps") for m, s in self.by_market.items()})
        return out
