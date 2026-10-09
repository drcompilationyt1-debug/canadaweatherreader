"""The daily book: look every day, trade rarely, let winners run (replaces the frozen holding period).

Research behind it (reports kept off the repo, 2026-10-09): professional systematic traders re-check positions daily but
trade only when a change clearly pays for its cost; they keep a name while it stays inside a wider rank band than the
one it was bought in (Novy-Marx & Velikov 2016: the single most effective cost control), let winners run under a wide
trailing stop scaled to the name's own volatility instead of a fixed profit target (momentum persists; fixed targets cap
the few trades that pay for everything), sell into abnormal spikes that look like overreaction (Chan 2003; Bali, Cakici &
Whitelaw 2011), and sell losers on evidence that the fall will continue - company-specific bad news with a sharp drop, an
extreme stock-specific drop, the ranking collapsing - not because of the purchase price (Lo & Remorov 2017: tight price
stops lose money on single stocks).

The owner's rules on top: never sell at a loss unless it is really bad; days with more and days with less activity
instead of a holding period (a quiet day still takes an urgent exit or a very good deal); never spend everything at once;
and every order must be worth its fee - moomoo's US$1.99 minimum makes a small order pure waste, so every trade here has
to beat a multiple of its own round-trip cost (``hurdle``) and nothing smaller than the minimum order is ever sent.

Everything is a pure function of the day's numbers, so the live runner and a historical replay share it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from math import sqrt

import numpy as np
from scipy.stats import norm

# what a day's activity level allows: the share of the book's stock capacity it may fill, new names a day, and the
# multiplier on the cost hurdle a trade must clear (quiet days demand more, active days a little less)
LEVELS = {
    "quiet": {"invest": 0.60, "new": 0, "hurdle": 1.5},
    "careful": {"invest": 0.80, "new": 1, "hurdle": 1.25},
    "normal": {"invest": 1.00, "new": 2, "hurdle": 1.0},
    "active": {"invest": 1.00, "new": 3, "hurdle": 0.8},
}


@dataclass
class DailyRules:
    k: int = 20                  # names the book aims to hold
    buy_rank: int = 20           # a new name must rank within this
    hold_rank: int = 40          # a held name keeps its place while it ranks within this
    min_hold_days: int = 5       # no ordinary exit before this (urgent exits excepted)
    hurdle: float = 2.0          # a trade's expected 20-day gain must beat this many round-trip costs
    max_new: int = 3             # new names a day at most (the activity level may allow fewer)
    max_swaps: int = 1           # swaps (sell the weakest holding for a clearly better name) a day at most
    starter: float = 1.0         # a first buy as a share of a slot (< 1: staged entry, the rest after starter_days)
    starter_days: int = 10
    spike_z: float = 3.0         # a rise this many of the name's own volatilities within spike_window days is a spike
    spike_window: int = 10
    spike_sell: float = 0.5      # share sold into a spike that looks like overreaction (never an order under the minimum)
    arm_z: float = 2.5           # the trailing stop arms once the gain since entry is this many volatilities (or on a spike)
    trail_atr: float = 2.5       # trailing stop: the highest close since entry minus this many average daily moves
    bad_drop_z: float = 2.5      # really bad: a 10-day stock-specific drop of this many volatilities
    bad_rank_pct: float = 0.3    # really bad: the ranking puts it in the bottom 30% of the universe
    news_drop_z: float = 2.0     # really bad: bad news confirmed by a stock-specific drop of this many volatilities
    chase_z: float = 2.0         # a candidate that rose this many volatilities in 5 days waits for a pullback (no chasing)
    rebuy_z: float = 1.0         # a name sold high is bought back only after falling this many 10-day volatilities under the sale
    rebuy_days: int = 30         # ... within this many days
    super_rank: int = 3          # a very good deal: rank within this, most models agree, gain over super_hurdle round trips
    super_agree: float = 0.6
    super_hurdle: float = 3.0
    stress_pct: float = 0.9      # market volatility above this percentile of its last three years = stress
    agree_hi: float = 0.6        # most models agree on the top names = an active day
    agree_lo: float = 0.35
    spread: float = 0.0005       # half-spread per side added to the fees
    premium: float = 0.005       # what 20 days in stocks instead of idle cash is worth (~6.5%/yr): counted for a buy into an
                                 # open slot, not for a swap (both sides are stocks)

    @classmethod
    def from_config(cls, d: dict | None) -> "DailyRules":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


@dataclass
class Stats:
    price: float
    vol: float                   # daily log-return volatility (60 days)
    atr: float                   # average absolute daily move (22 days), in price units
    ret5: float
    resid10: float               # 10-day return minus the benchmark's
    run126: float                # six-month run
    closes: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))


def name_stats(close, bench=None) -> Stats | None:
    c = np.asarray(close, dtype=float)
    c = c[np.isfinite(c) & (c > 0)]
    if len(c) < 12:
        return None
    r = np.diff(np.log(c))
    vol = float(np.std(r[-60:], ddof=1)) if len(r) > 5 else 0.02
    vol = vol if np.isfinite(vol) and vol > 1e-4 else 0.02
    atr = float(np.mean(np.abs(np.diff(c[-23:]))))
    b10 = 0.0
    if bench is not None:
        b = np.asarray(bench, dtype=float)
        b = b[np.isfinite(b) & (b > 0)]
        if len(b) > 11:
            b10 = b[-1] / b[-11] - 1.0
    return Stats(price=float(c[-1]), vol=vol, atr=atr if atr > 0 else vol * c[-1], ret5=float(c[-1] / c[-6] - 1.0),
                 resid10=float(c[-1] / c[-11] - 1.0 - b10), run126=float(c[-1] / c[max(0, len(c) - 127)] - 1.0), closes=c)


def expected_return(pct: float, ic: float = 0.05, disp: float = 0.08) -> float:
    """The 20-day excess return a score percentile is worth: information coefficient x cross-sectional dispersion x the
    normal quantile of the percentile (Grinold's 'alpha = IC x volatility x score')."""
    if pct is None or not np.isfinite(pct):
        return 0.0
    return float(ic * disp * norm.ppf(np.clip(pct, 0.005, 0.995)))


def market_activity(bench, agreement: float | None, rules: DailyRules) -> tuple[str, dict]:
    """The day's activity level from the market (volatility percentile, the 200-day trend) and how far the models agree."""
    b = np.asarray(bench if bench is not None else [], dtype=float)
    b = b[np.isfinite(b) & (b > 0)]
    ag = 0.5 if agreement is None or not np.isfinite(agreement) else float(agreement)
    if len(b) < 260:
        return "normal", {"agreement": ag}
    r = np.diff(np.log(b))
    v = np.array([np.std(r[i - 20:i], ddof=1) for i in range(max(20, len(r) - 756), len(r) + 1)])
    vpct = float((v <= v[-1]).mean())
    down = bool(b[-1] < b[-200:].mean())
    stress = vpct >= rules.stress_pct
    if stress and down:
        level = "quiet"
    elif stress or (down and ag < rules.agree_lo):
        level = "careful"
    elif ag >= rules.agree_hi and not down:
        level = "active"
    else:
        level = "normal"
    return level, {"vol_pct": vpct, "below_200d": down, "agreement": ag}


def plan_day(*, weights: dict[str, float], book: dict[str, dict], age: dict[str, int], pct: dict[str, float],
             stats: dict[str, Stats], agree: dict[str, float], level: str, slot: float, capacity: float, cash_w: float,
             cost_rt: dict[str, float], min_w: dict[str, float], ic: float, disp: float, bad_news: set[str] | None = None,
             sold_high: dict[str, dict] | None = None, max_w: float = 1.0, rules: DailyRules
             ) -> tuple[dict[str, float], list[tuple], dict[str, dict]]:
    """One day's plan.

    ``weights`` the current weights (fraction of equity) of the held names; ``book`` their records (entry price, date, highest
    close since entry, trailing stop armed, spike sold, percentile at entry, starter); ``age`` days held; ``pct`` today's score
    percentile per name (smoothed; 1 = best); ``stats`` price statistics; ``agree`` the share of independent models that
    put a name in their top fifth; ``slot`` a full slot's weight; ``capacity`` the weight the stock slots may fill; ``cash_w``
    idle cash above the reserve as a weight (today's sale proceeds settle tomorrow and are not counted); ``cost_rt`` round-trip
    cost of a full slot as a fraction; ``min_w`` the smallest order worth its fee as a weight; ``ic``/``disp`` turn a percentile
    into an expected return; ``bad_news`` names whose news verdict is clearly negative; ``sold_high`` names recently sold into
    strength ({price, left: days to keep watching, w: the weight sold}) - sell high, buy back low with ALL of it (more shares
    for the same money, so every round compounds); ``max_w`` the largest weight one name may reach.

    Returns (new weights for the names that change, events, the updated book)."""
    bad_news = bad_news or set()
    sold_high = sold_high if sold_high is not None else {}
    lv = LEVELS.get(level, LEVELS["normal"])
    hurdle = rules.hurdle * lv["hurdle"]
    order = sorted((t for t in pct if np.isfinite(pct[t])), key=lambda t: -pct[t])
    rank = {t: i + 1 for i, t in enumerate(order)}
    exp = {t: expected_return(pct.get(t), ic, disp) for t in pct}
    out: dict[str, float] = {}
    events: list[tuple] = []
    book = {t: dict(r) for t, r in book.items() if t in weights}
    held = [t for t, w in weights.items() if w > 1e-6]

    def w_now(t: str) -> float:
        return out.get(t, weights.get(t, 0.0))

    # 1) exits, any day: really bad (even at a loss), the trailing stop, spikes, leaving the rank band (never at a loss)
    for t in held:
        s, rec = stats.get(t), book.get(t)
        if s is None or rec is None:
            continue
        rec["high"] = max(float(rec.get("high", s.price)), s.price)
        entry = float(rec["entry"])
        gain = s.price / entry - 1.0
        a = int(age.get(t, 0))
        p = pct.get(t, np.nan)
        z10 = s.resid10 / (s.vol * sqrt(10.0))
        why = None
        if np.isfinite(p) and p < rules.bad_rank_pct:
            why = f"the ranking collapsed (bottom {100 * (1 - p):.0f}%)"
        elif z10 <= -rules.bad_drop_z:
            why = f"a stock-specific drop of {abs(z10):.1f} volatilities in 10 days"
        elif t in bad_news and z10 <= -rules.news_drop_z:
            why = f"bad news confirmed by a {abs(z10):.1f}-volatility drop"
        if why:
            out[t] = 0.0
            events.append(("bad", t, gain, why))
            continue
        n = int(np.clip(min(a, rules.spike_window), 3, rules.spike_window))
        c = s.closes
        rise = float(np.log(c[-1] / c[-1 - n])) if len(c) > n else 0.0
        spike_z = rise / (s.vol * sqrt(n))
        if not rec.get("armed") and (spike_z >= rules.spike_z or np.log(max(1.0 + gain, 1e-9)) >= rules.arm_z * s.vol * sqrt(max(a, 5))):
            rec["armed"] = True
            events.append(("armed", t, gain, f"a {spike_z:.1f}-volatility rise: trailing stop on"))
        if rec.get("armed"):
            stop = max(rec["high"] - rules.trail_atr * s.atr, entry * (1.0 + cost_rt.get(t, 0.0)))
            if s.price <= stop and gain > 0:
                out[t] = 0.0
                prev = float((sold_high.get(t) or {}).get("w", 0.0))
                sold_high[t] = {"price": s.price, "left": rules.rebuy_days, "w": prev + weights[t]}
                events.append(("trail", t, gain, f"fell to the trailing stop {stop:.2f} from a high of {rec['high']:.2f}"))
                continue
        if spike_z >= rules.spike_z and not rec.get("spiked") and gain > 0:
            one_day = float(np.max(np.diff(np.log(c[-1 - n:])))) / rise if rise > 0 else 0.0
            p0 = float(rec.get("pct0", p if np.isfinite(p) else 0.5))
            overreaction = one_day >= 0.6 or (np.isfinite(p) and p <= p0) or s.run126 >= 0.5
            sale = weights[t] * rules.spike_sell
            if overreaction and sale >= min_w.get(t, 0.0) and weights[t] - sale >= min_w.get(t, 0.0):
                out[t] = weights[t] - sale
                rec["spiked"] = True
                sold_high[t] = {"price": s.price, "left": rules.rebuy_days, "w": sale}
                events.append(("spike", t, gain, f"sold {100 * rules.spike_sell:.0f}% into a {spike_z:.1f}-volatility spike"))
                continue
        if a >= rules.min_hold_days and rank.get(t, 10 ** 6) > rules.hold_rank:
            if gain > cost_rt.get(t, 0.0):
                out[t] = 0.0
                events.append(("band", t, gain, f"left the top {rules.hold_rank} (now #{rank.get(t, '?')})"))
            else:
                events.append(("kept_loss", t, gain, f"ranks #{rank.get(t, '?')} but is under its buy price - held (not really bad)"))
    # 2) buy back what was sold high once it is clearly cheaper and still ranks inside the band - with ALL the money the sale
    #    raised (at the lower price that is more shares: each sell-high / buy-low round compounds), and complete staged entries
    invested = sum(w_now(t) for t in weights)
    room = min(capacity * lv["invest"] - invested, cash_w)
    for t, rec in list(sold_high.items()):
        s = stats.get(t)
        if s is None or rank.get(t, 10 ** 6) > rules.hold_rank or t in out:
            continue
        if s.price <= float(rec["price"]) * (1.0 - rules.rebuy_z * s.vol * sqrt(10.0)):
            w_old = w_now(t)
            add = min(max(float(rec.get("w", 0.0)), slot - w_old), max_w - w_old, max(0.0, cash_w))
            if add >= min_w.get(t, 0.0):
                out[t] = w_old + add
                room -= add
                sold_high.pop(t)
                old = book.get(t)
                entry = (w_old * float(old["entry"]) + add * s.price) / (w_old + add) if old and w_old > 0 else s.price
                book[t] = {"entry": entry, "high": s.price, "pct0": pct.get(t, 0.5), "armed": False, "spiked": False,
                           **({} if old and w_old > 0 else {"date_new": True})}
                events.append(("rebuy", t, s.price / float(rec["price"]) - 1.0,
                               f"bought back {100 * (1 - s.price / float(rec['price'])):.0f}% under the sale with all of its money"))
    for t in held:
        rec = book.get(t)
        if rec is None or not rec.get("starter") or w_now(t) <= 0 or int(age.get(t, 0)) < rules.starter_days:
            continue
        add = min(slot - w_now(t), room)
        if rank.get(t, 10 ** 6) <= rules.buy_rank and add >= min_w.get(t, 0.0):
            out[t] = w_now(t) + add
            room -= add
            rec["starter"] = False
            events.append(("complete", t, 0.0, "staged entry completed (still ranks well)"))
    # 3) new names into open slots - only the ones worth their fees, never one that just spiked, at most what the day allows
    held_after = [t for t in weights if w_now(t) > 1e-6] + [t for t in out if t not in weights and out[t] > 1e-6]
    open_slots = rules.k - len(held_after)

    def buyable(t: str) -> bool:
        s = stats[t]
        if t in sold_high and s.price > float(sold_high[t]["price"]) * (1.0 - rules.rebuy_z * s.vol * sqrt(10.0)):
            return False                                          # sold high recently: only back in cheaper
        z5 = np.log(1.0 + s.ret5) / (s.vol * sqrt(5.0))
        if z5 >= rules.chase_z:
            events.append(("wait", t, s.ret5, f"rose {z5:.1f} volatilities in 5 days - waiting for a pullback"))
            return False
        return True

    cands = [t for t in order if t not in held_after and rank[t] <= rules.buy_rank and t in stats and t not in out]
    cands = [t for t in cands if buyable(t)]
    allowed = min(lv["new"], rules.max_new)
    bought, super_used = 0, False

    def is_super(t: str, gain: float) -> bool:
        return rank[t] <= rules.super_rank and agree.get(t, 0.0) >= rules.super_agree and gain >= rules.super_hurdle * cost_rt.get(t, 0.0)

    for t in list(cands):
        if open_slots <= 0:
            break
        gain = exp[t] + rules.premium                                 # cash put to work earns the market too
        superdeal = is_super(t, gain)
        extra = bought >= allowed
        if extra and not (superdeal and not super_used):
            continue
        if gain < hurdle * cost_rt.get(t, 0.0) and not superdeal:
            continue
        size = slot * rules.starter if rules.starter < 1.0 else slot
        if size < min_w.get(t, 0.0):
            size = slot
        if size > room + 1e-9 or size < min_w.get(t, 0.0):
            break
        s = stats[t]
        out[t] = size
        room -= size
        open_slots -= 1
        bought += 1
        super_used = super_used or extra
        cands.remove(t)
        sold_high.pop(t, None)
        book[t] = {"entry": s.price, "high": s.price, "pct0": pct.get(t, 0.5), "starter": size < slot - 1e-9, "date_new": True}
        events.append(("buy", t, gain, ("a very good deal, " if extra else "") + f"#{rank[t]}, expected {100 * gain:+.1f}% "
                       f"in 20 days vs {100 * cost_rt.get(t, 0.0):.2f}% round trip"))
    # 4) swaps when the slots are full: the weakest holding (never at a loss) for a clearly better name
    if open_slots <= 0 and cands:
        swaps = rules.max_swaps if lv["new"] > 0 else 0
        weakest = sorted((t for t in held if w_now(t) > 1e-6 and t not in out and int(age.get(t, 0)) >= rules.min_hold_days
                          and t in stats and t in book and stats[t].price / float(book[t]["entry"]) - 1.0 > cost_rt.get(t, 0.0)),
                         key=lambda t: pct.get(t, 0.0))
        for t_out in weakest:
            if not cands:
                break
            t_in = cands[0]
            gap = exp[t_in] - exp.get(t_out, 0.0)
            superdeal = is_super(t_in, gap) and not super_used
            if swaps <= 0 and not superdeal:
                break
            if gap < hurdle * cost_rt.get(t_in, 0.0) and not superdeal:
                break
            out[t_out] = 0.0
            out[t_in] = slot
            cands.pop(0)
            if swaps <= 0:
                super_used = True
            swaps -= 1
            book[t_in] = {"entry": stats[t_in].price, "high": stats[t_in].price, "pct0": pct.get(t_in, 0.5), "date_new": True}
            events.append(("swap", t_in, gap, f"replaces {t_out}: {100 * gap:+.1f}% better expected 20-day return vs "
                                              f"{100 * cost_rt.get(t_in, 0.0):.2f}% round trip"))
    for t in list(sold_high):
        sold_high[t]["left"] = int(sold_high[t].get("left", rules.rebuy_days)) - 1
        if sold_high[t]["left"] <= 0:
            sold_high.pop(t)
    book = {t: r for t, r in book.items() if w_now(t) > 1e-6}
    return out, events, book


def describe(events: list[tuple]) -> str:
    """A short note for the session log."""
    words = {"bad": "sold (really bad)", "trail": "trailing stop sold", "spike": "sold into a spike", "band": "sold (left the band)",
             "rebuy": "bought back", "complete": "completed", "buy": "bought", "swap": "swapped in", "wait": "waiting",
             "armed": "armed", "kept_loss": "kept"}
    parts = []
    for kind, t, val, why in events:
        if kind in ("buy", "swap"):
            parts.append(f"{words[kind]} {t} ({why})")
        else:
            parts.append(f"{words[kind]} {t} {100 * val:+.1f}% ({why})")
    return "; ".join(parts)


def shadow_step(sh: dict, scores: dict[str, float], prices: dict[str, float], bars_since_last: int | None, k: int, every: int,
                hysteresis: int, fee, reserve: float = 0.10) -> dict:
    """The previous rule (top-k re-ranked every ``every`` bars, frozen in between) run on the same live prices, so the paper
    accounts compare the two strategies on real data.  ``sh``: {"cash", "units": {t: shares}}; ``fee(t, dollars)`` = one
    order's fee in dollars.  Returns the updated shadow with its equity."""
    from .ranking import select_top

    units = {t: float(u) for t, u in (sh.get("units") or {}).items()}
    cash = float(sh.get("cash", 0.0))
    value = cash + sum(u * prices.get(t, 0.0) for t, u in units.items())
    if bars_since_last is None or bars_since_last >= every:
        held = [t for t, u in units.items() if u > 0]
        chosen = select_top(scores, held, k, hysteresis)
        target = value * (1.0 - reserve) / max(k, 1)
        for t in [t for t in held if t not in chosen]:
            proceeds = units.pop(t) * prices.get(t, 0.0)
            cash += proceeds - fee(t, proceeds)
        for t in chosen:
            if t in units or prices.get(t, 0.0) <= 0:
                continue
            spend = min(target, cash - fee(t, target))
            if spend <= 0:
                continue
            units[t] = spend / prices[t]
            cash -= spend + fee(t, spend)
        sh["rebalanced"] = True
    else:
        sh["rebalanced"] = False
    sh["units"], sh["cash"] = units, cash
    sh["equity"] = cash + sum(u * prices.get(t, 0.0) for t, u in units.items())
    return sh
