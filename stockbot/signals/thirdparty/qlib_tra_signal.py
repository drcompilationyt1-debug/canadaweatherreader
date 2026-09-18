"""qlib's TRA ("Temporal Routing Adaptor", Lin et al., KDD 2021) as a second qlib block.

The same Alpha158 features the LightGBM qlib block uses, scored by a sequence model that routes each stock-day to one of
several predictors depending on the regime it looks like - the qlib benchmark where it reports a higher information
coefficient than LightGBM on Chinese A-shares.  ``scripts/qlib_train.py --model tra`` trains it on the weekend (it is far
slower than the trees) and writes ``models/qlib/pred_tra.parquet``; the block reads that table and is masked without it.
"""
from __future__ import annotations

from ...paths import resolve
from .qlib_signal import QlibSignal


class QlibTRASignal(QlibSignal):
    name = "qlib_tra"
    feature_names = ["tra_score", "tra_rank"]
    tier = "B"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.pred_path = resolve(self.cfg.get("predictions", "models/qlib/pred_tra.parquet"))
        self.max_age_bars = int(self.cfg.get("max_age_bars", 7) or 0)     # refreshed on the weekend: a score lasts the week
