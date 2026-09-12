"""Direction scorecard: which of the models actually predict whether a stock goes up?

Every trading cycle records, per ticker, the directional vote of each predictive block (+1 up,
-1 down, 0 no opinion) read straight from the observation vector, plus the policy's own lean.
The record is settled twice: with the move over the trading session (open -> end of the session,
``horizon="session"``) and with the next daily close (``horizon="daily"``).  ``stockbot forecast``
prints today's votes next to each model's running hit rate, so after a few weeks it is obvious
which opinions are worth listening to (and the dumb ``momentum`` baseline shows what "no skill"
looks like).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from ..signals.layout import ObservationLayout

log = get_logger(__name__)

# voter name -> (block, feature): the signed feature whose sign is the block's opinion about the next move
VOTERS: dict[str, tuple[str, str]] = {
    "dl_forecast": ("dl_forecast", "dl_pred"),          # GRU predicted 5-day log return
    "alpha_factors": ("alpha_factors", "af_pred"),      # qlib-lite factor model
    "qlib": ("qlib", "qlib_score"),                     # qlib model score
    "es_agent": ("es_agent", "es_action"),              # evolution-strategy agent: buy / sell / hold
    "dqn_agent": ("dqn_agent", "dqn_action"),           # DQN agent
    "ta_keras": ("ta_keras", "akm_action"),             # akurgat's Keras classifiers
    "llm_trader": ("llm_trader", "nofx_direction"),     # nofx-style LLM decision
    "news_llm": ("news_llm", "llm_direction"),          # LLM news reader: bullish / bearish
    "sentiment": ("sentiment", "vader_mean"),           # VADER headline sentiment
    "candles": ("candles", "direction_score"),          # candlestick patterns weighted by their historical edge
    "talib_candles": ("talib_candles", "direction_score"),
    "strategy_zoo": ("strategy_zoo", "vote"),           # majority of the ported rule-based strategies
    "trend": ("trend", "slope_30"),                     # 30-day regression slope
    "trading_agents": ("trading_agents", "ta_decision"),
    "ai_hedge_fund": ("ai_hedge_fund", "ahf_conviction"),
    "momentum": ("technical", "ret_20"),                # baseline: last 20-day return, no model at all
}
VOTE_EPS = 1e-6


def votes_from_vector(layout: ObservationLayout, signal_vec: np.ndarray, policy_delta: float | None = None,
                      deadband: float = 0.0) -> dict[str, dict[str, float]]:
    """``{voter: {"vote": -1|0|1, "score": float}}`` for every voter whose block is present and available.

    ``policy_delta`` is the policy's target exposure minus the current one: adding to a position is
    its "up" vote, cutting is "down", a change inside the deadband is no opinion."""
    out: dict[str, dict[str, float]] = {}
    for voter, (block, feature) in VOTERS.items():
        try:
            b = layout.block(block)
        except KeyError:
            continue
        if signal_vec[b.offset] < 0.5 or feature not in b.feature_names:
            continue
        score = float(signal_vec[b.start + b.feature_names.index(feature)])
        out[voter] = {"vote": float(np.sign(score)) if abs(score) > VOTE_EPS else 0.0, "score": round(score, 5)}
    if policy_delta is not None:
        score = float(policy_delta)
        out["policy"] = {"vote": float(np.sign(score)) if abs(score) >= max(float(deadband), VOTE_EPS) else 0.0, "score": round(score, 5)}
    return out


def consensus(votes: dict[str, dict[str, float]], exclude: tuple[str, ...] = ("momentum", "policy")) -> float:
    """Mean vote of the model voters in -1..1 (0 = no opinion / split)."""
    vals = [v["vote"] for k, v in votes.items() if k not in exclude and v["vote"] != 0]
    return float(np.mean(vals)) if vals else 0.0


class DirectionBoard:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _append(self, rec: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def record(self, *, ticker: str, date: str, price: float, votes: dict[str, dict[str, float]], mode: str = "") -> None:
        self._append({"type": "votes", "ts": datetime.now(timezone.utc).isoformat(), "mode": mode, "ticker": ticker,
                      "date": date, "price": float(price), "votes": votes, "consensus": consensus(votes)})

    def load(self) -> pd.DataFrame | None:
        if not self.path.exists():
            return None
        rows = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return pd.DataFrame(rows) if rows else None

    def settle(self, ticker: str, horizon: str, price: float, date: str, decision_date: str | None = None) -> dict | None:
        """Attach the realised move to the latest (or the given) votes record of ``ticker`` for ``horizon``."""
        df = self.load()
        if df is None or len(df) == 0 or price <= 0:
            return None
        votes = df[(df["type"] == "votes") & (df["ticker"] == ticker)]
        if decision_date is not None:
            votes = votes[votes["date"] == decision_date]
        elif horizon == "daily":
            votes = votes[votes["date"] < date]        # the daily outcome needs a later close
        if len(votes) == 0:
            return None
        rec = votes.iloc[-1]
        if "horizon" in df.columns:
            done = df[(df["type"] == "outcome") & (df["ticker"] == ticker) & (df["decision_date"] == rec["date"]) & (df["horizon"] == horizon)]
            if len(done):
                return None
        if float(rec["price"]) <= 0:
            return None
        ret = float(np.log(price / float(rec["price"])))
        out = {"type": "outcome", "ts": datetime.now(timezone.utc).isoformat(), "horizon": horizon, "ticker": ticker,
               "decision_date": rec["date"], "date": date, "price_then": float(rec["price"]), "price_now": float(price),
               "log_return": ret, "votes": rec["votes"]}
        self._append(out)
        return out

    # ------------------------------------------------------------------ reporting
    def latest(self, date: str | None = None) -> pd.DataFrame | None:
        """Votes of the most recent date as a ticker x voter table (consensus and policy last)."""
        df = self.load()
        if df is None:
            return None
        votes = df[df["type"] == "votes"]
        if len(votes) == 0:
            return None
        date = date or str(votes["date"].max())
        rows = {}
        for _, r in votes[votes["date"] == date].iterrows():
            rows[r["ticker"]] = {k: v["vote"] for k, v in r["votes"].items()} | {"consensus": round(float(r["consensus"]), 2)}
        tbl = pd.DataFrame(rows).T
        order = [c for c in list(VOTERS) + ["policy", "consensus"] if c in tbl.columns]
        tbl = tbl[order]
        tbl.index.name = date
        return tbl

    def scorecard(self, horizon: str | None = None) -> pd.DataFrame | None:
        """Per voter: settled votes, hit rate, and the return earned by following the vote (bps per outcome)."""
        df = self.load()
        if df is None or "horizon" not in df.columns:
            return None
        outc = df[df["type"] == "outcome"]
        if horizon:
            outc = outc[outc["horizon"] == horizon]
        if len(outc) == 0:
            return None
        stats: dict[str, dict] = {}
        for _, r in outc.iterrows():
            ret = float(r["log_return"])
            for voter, v in (r["votes"] or {}).items():
                vote = float(v.get("vote", 0.0))
                s = stats.setdefault(voter, {"n": 0, "hits": 0, "up_votes": 0, "edge": 0.0, "abstain": 0})
                if vote == 0:
                    s["abstain"] += 1
                    continue
                s["n"] += 1
                s["up_votes"] += int(vote > 0)
                s["hits"] += int(np.sign(ret) == vote)
                s["edge"] += ret * vote
        rows = []
        for voter, s in stats.items():
            n = s["n"]
            rows.append({"model": voter, "n": n, "hit_rate": s["hits"] / n if n else np.nan, "edge_bps": 1e4 * s["edge"] / n if n else np.nan,
                         "up_share": s["up_votes"] / n if n else np.nan, "abstained": s["abstain"]})
        tbl = pd.DataFrame(rows).sort_values(["hit_rate", "n"], ascending=[False, False]).reset_index(drop=True)
        return tbl

    def summary(self) -> dict:
        df = self.load()
        if df is None:
            return {"votes": 0, "outcomes": 0}
        out = {"votes": int((df["type"] == "votes").sum()), "outcomes": int((df["type"] == "outcome").sum())}
        for h in ("session", "daily"):
            sc = self.scorecard(h)
            if sc is not None and len(sc):
                best = sc.iloc[0]
                out[f"best_{h}"] = {"model": best["model"], "hit_rate": round(float(best["hit_rate"]), 3), "n": int(best["n"])}
        return out
