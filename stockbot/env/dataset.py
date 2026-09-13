"""MarketDataset: per-ticker prices + the assembled signal matrix the simulator steps through."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from ..signals.base import SignalContext, SignalProvider
from ..signals.layout import ObservationLayout

log = get_logger(__name__)


@dataclass
class TickerData:
    ticker: str
    dates: np.ndarray            # datetime64[ns]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    signals: np.ndarray          # (T, signal_dim) float32, availability flags included
    min_start: int = 0           # first index where the core blocks are available

    def __len__(self) -> int:
        return len(self.close)

    def slice(self, start: int, end: int, reset_min_start: bool = True) -> "TickerData":
        return TickerData(self.ticker, self.dates[start:end], self.open[start:end], self.high[start:end],
                          self.low[start:end], self.close[start:end], self.volume[start:end],
                          self.signals[start:end], 0 if reset_min_start else max(0, self.min_start - start))


def compute_signal_arrays(frames: dict[str, pd.DataFrame], providers: list[SignalProvider], ctx: SignalContext,
                          latest_only: bool = False) -> dict[str, dict[str, np.ndarray | None]]:
    """Run every provider over every ticker -> {ticker: {block: array | None}}."""
    ctx.extra["frames"] = frames
    per_block: dict[str, dict[str, np.ndarray | None]] = {}
    ctx.extra["per_block"] = per_block        # filled as providers run: later blocks (xs_rank) can read earlier ones
    ctx.extra.pop("latest_vectors", None)
    for p in providers:
        if not p.enabled:
            continue
        if p.live_only and not latest_only:
            per_block[p.name] = {t: None for t in frames}
            continue
        ok, why = p.availability()
        if not ok:
            log.info("signal %-16s unavailable: %s", p.name, why)
            per_block[p.name] = {t: None for t in frames}
            continue
        if p.needs_universe:
            try:
                per_block[p.name] = p.compute_history_all(frames)
            except Exception as e:  # noqa: BLE001
                log.warning("signal %s failed (universe): %s", p.name, e)
                per_block[p.name] = {t: None for t in frames}
        else:
            per_block[p.name] = {t: p.safe_history(t, df) for t, df in frames.items()}
        n_ok = sum(v is not None for v in per_block[p.name].values())
        log.info("signal %-16s computed for %d/%d tickers", p.name, n_ok, len(frames))
    ctx.extra.pop("per_block", None)
    return {t: {b: per_block[b][t] for b in per_block} for t in frames}


class MarketDataset:
    def __init__(self, layout: ObservationLayout, data: dict[str, TickerData], meta: dict | None = None):
        self.layout = layout
        self.data = data
        self.meta = meta or {}

    @property
    def tickers(self) -> list[str]:
        return list(self.data)

    def __len__(self) -> int:
        return len(self.data)

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, frames: dict[str, pd.DataFrame], providers: list[SignalProvider], layout: ObservationLayout,
              ctx: SignalContext, fit: bool = True, train_end: str | None = None) -> "MarketDataset":
        ctx.extra["frames"] = frames
        for p in providers:
            if not p.enabled or p.live_only:
                continue
            if fit:
                try:
                    p.fit(frames, train_end)
                    p.save_state()
                except Exception as e:  # noqa: BLE001
                    log.warning("fitting signal %s failed: %s", p.name, e)
            else:
                p.load_state()
        arrays = compute_signal_arrays(frames, providers, ctx)
        data: dict[str, TickerData] = {}
        core = [b.name for b in layout.blocks if b.name in ("technical", "trend", "candles")]
        for t, df in frames.items():
            sig = layout.assemble(len(df), arrays[t])
            avail = np.ones(len(df), dtype=bool)
            for name in core:
                avail &= sig[:, layout.block(name).offset] > 0.5
            first = int(np.argmax(avail)) if avail.any() else len(df) - 1
            data[t] = TickerData(
                t, df.index.to_numpy(dtype="datetime64[ns]"),
                df["open"].to_numpy(float), df["high"].to_numpy(float), df["low"].to_numpy(float),
                df["close"].to_numpy(float), df["volume"].to_numpy(float), sig, first,
            )
        meta = {"train_end": train_end, "tickers": list(frames), "signature": layout.signature()}
        return cls(layout, data, meta)

    # ------------------------------------------------------------------ split / filter
    def split(self, train_end: str | pd.Timestamp) -> tuple["MarketDataset", "MarketDataset"]:
        cut = np.datetime64(pd.Timestamp(train_end), "ns")
        train, test = {}, {}
        for t, td in self.data.items():
            k = int(np.searchsorted(td.dates, cut, side="right"))
            if k > td.min_start + 30:
                train[t] = td.slice(0, k, reset_min_start=False)
            if len(td) - k > 30:
                test[t] = td.slice(k, len(td))
        return cls_like(self, train), cls_like(self, test)

    def subset(self, tickers: list[str]) -> "MarketDataset":
        return cls_like(self, {t: self.data[t] for t in tickers if t in self.data})

    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        lo = min(pd.Timestamp(td.dates[0]) for td in self.data.values())
        hi = max(pd.Timestamp(td.dates[-1]) for td in self.data.values())
        return lo, hi

    # ------------------------------------------------------------------ persistence
    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.layout.save(folder / "layout.json")
        (folder / "meta.json").write_text(json.dumps(self.meta, default=str), encoding="utf-8")
        for t, td in self.data.items():
            np.savez_compressed(folder / f"{t.replace('/', '_')}.npz", dates=td.dates, open=td.open, high=td.high,
                                low=td.low, close=td.close, volume=td.volume, signals=td.signals,
                                min_start=np.array(td.min_start))

    @classmethod
    def load(cls, folder: str | Path) -> "MarketDataset":
        folder = Path(folder)
        layout = ObservationLayout.load(folder / "layout.json")
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8")) if (folder / "meta.json").exists() else {}
        data = {}
        for f in sorted(folder.glob("*.npz")):
            z = np.load(f, allow_pickle=False)
            t = f.stem
            data[t] = TickerData(t, z["dates"], z["open"], z["high"], z["low"], z["close"], z["volume"],
                                 z["signals"], int(z["min_start"]))
        return cls(layout, data, meta)


def cls_like(ds: MarketDataset, data: dict[str, TickerData]) -> MarketDataset:
    return MarketDataset(ds.layout, data, dict(ds.meta))
