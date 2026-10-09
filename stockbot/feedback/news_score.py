"""Does the news predict anything?  The scorecard of the logged news / LLM verdicts (data/experience/news_verdicts.jsonl).

Historical backtests of LLM verdicts are worthless (the models have read the future), so the only honest test is forward:
every morning the runner logs each name's LLM-trader verdict, news-LLM sentiment and FinBERT score; here each verdict is
settled against what the stock then did relative to the rest of the universe over 5 and 20 trading days.  Research bar
(Lopez-Lira & Tang 2023; Ke, Kelly & Xiu 2019): in large caps the news effect is small (~0.1%) and short, so a verdict is
trusted with more than a confirming vote only after about 400 independent calls with a t-statistic of at least 2 - and the
stated confidence must actually line up with being right.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (5, 20)


def _t(values) -> tuple[float, float, int]:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) < 3:
        return float("nan"), float("nan"), int(len(v))
    sd = v.std(ddof=1)
    return float(v.mean()), float(v.mean() / sd * np.sqrt(len(v))) if sd > 0 else float("nan"), int(len(v))


def score_news(path: str | Path, closes: pd.DataFrame, min_calls: int = 400) -> dict:
    """``closes``: dates x tickers.  Returns the scorecard (also what `stockbot news-score` prints)."""
    f = Path(path)
    if not f.exists():
        return {"verdicts": 0, "note": f"no verdicts logged yet ({f})"}
    rows = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    if df.empty:
        return {"verdicts": 0}
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(["date", "ticker"], keep="last")
    closes = closes.sort_index()
    pos = {d: i for i, d in enumerate(closes.index)}
    out: dict = {"verdicts": int(len(df)), "days": int(df["date"].nunique()), "first": str(df["date"].min().date()),
                 "last": str(df["date"].max().date())}
    for h in HORIZONS:
        fwd = closes.shift(-h) / closes - 1.0
        excess = fwd.sub(fwd.median(axis=1), axis=0)                    # against the universe that day
        vals = []
        for d, t in zip(df["date"], df["ticker"]):
            i = pos.get(d)
            vals.append(excess.iat[i, closes.columns.get_loc(t)] if i is not None and t in closes.columns else np.nan)
        df[f"x{h}"] = vals
    num = lambda c: pd.to_numeric(df.get(c), errors="coerce") if c in df else pd.Series(np.nan, index=df.index)  # noqa: E731
    close_call = (num("nofx_close") > 0.5) & (num("nofx_direction") <= -0.9)
    open_call = num("nofx_open") > 0.5
    conf = num("nofx_confidence")
    groups = {"llm_full_close": close_call, "llm_open_or_add": open_call,
              "news_llm_negative": (num("llm_sentiment") <= -0.4) & (num("llm_confidence") >= 0.6),
              "finbert_negative": (num("fb_net") <= -0.3) & (num("fb_n") >= 3)}
    for name, mask in groups.items():
        g = df[mask.fillna(False)]
        res = {"calls": int(len(g))}
        for h in HORIZONS:
            per_day = g.groupby("date")[f"x{h}"].mean()                 # one number per day: calls on a day are not independent
            m, t, n = _t(per_day.to_numpy())
            res[f"excess_{h}d"], res[f"t_{h}d"], res[f"days_{h}d"] = m, t, n
        out[name] = res
    buckets = {}
    for lo, hi in ((0.0, 0.7), (0.7, 0.85), (0.85, 1.01)):
        g = df[close_call.fillna(False) & (conf >= lo) & (conf < hi)]
        buckets[f"{lo:.2f}-{min(hi, 1.0):.2f}"] = {"calls": int(len(g)),
                                                    "hit_rate_20d": float((g["x20"] < 0).mean()) if g["x20"].notna().any() else float("nan")}
    out["llm_full_close_by_confidence"] = buckets
    ics = {}
    signed = num("nofx_direction") * conf
    for name, col in (("llm_trader", signed), ("news_llm", num("llm_sentiment")), ("finbert", num("fb_net"))):
        daily = []
        for d, g in df.assign(s=col).groupby("date"):
            g = g[["s", "x20"]].dropna()
            if len(g) >= 10 and g["s"].nunique() > 1:
                daily.append(g["s"].corr(g["x20"], method="spearman"))
        m, t, n = _t(daily)
        ics[name] = {"ic_20d": m, "t": t, "days": n}
    out["rank_ic"] = ics
    c = out["llm_full_close"]
    out["llm_close_proven"] = bool(c["calls"] >= min_calls and np.isfinite(c.get("t_20d", np.nan)) and c["t_20d"] <= -2.0)
    return out
