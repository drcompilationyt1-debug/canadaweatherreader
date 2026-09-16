import numpy as np
import pandas as pd

import stockbot.agent  # noqa: F401  (import order: the agent package first, as everywhere else)
from stockbot.env.dataset import MarketDataset, TickerData
from stockbot.signals.layout import ObservationLayout
from stockbot.signals.pruning import block_values, load_block_mask, prune_blocks, with_block_mask


def _ds(n=400, names=("A", "B", "C", "D", "E")):
    layout = ObservationLayout([("technical", ["ret_20"]), ("noise", ["n1", "n2"]), ("good", ["g1"])])
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2024-01-01", periods=n).to_numpy(dtype="datetime64[ns]")
    data = {}
    for t in names:
        c = 100 * np.cumprod(1 + rng.normal(0.0, 0.02, n))
        logc = np.log(c)
        fwd = np.full(n, np.nan)
        fwd[:-20] = logc[20:] - logc[:-20]
        arrays = {"technical": rng.normal(size=(n, 1)).astype(np.float32), "noise": rng.normal(size=(n, 2)).astype(np.float32),
                  "good": (np.nan_to_num(fwd, nan=0.0) + rng.normal(0, 0.01, n)).reshape(-1, 1).astype(np.float32)}
        sig = layout.assemble(n, arrays)
        data[t] = TickerData(t, dates, c, c, c, c, np.ones(n), sig, 0)
    return MarketDataset(layout, data, {})


def test_pruning_masks_only_blocks_without_value_and_keeps_hysteresis(cfg, tmp_path):
    ds = _ds()
    cfg.set_path("models_dir", str(tmp_path))
    vals = block_values(ds, days=126, min_names=3)
    assert vals["good"]["max_abs_t"] > 5 and vals["good"]["best_feature"] == "good.g1"     # foresight of the forward return: huge IC
    assert vals["noise"]["max_abs_t"] < 3
    out = tmp_path / "block_mask.json"
    rep = prune_blocks(cfg, ds, out_path=out, days=126, mask_t=3.0, unmask_t=4.0, min_names=3)
    assert "noise" in rep["masked"] and "good" not in rep["masked"] and "technical" not in rep["masked"]   # core blocks are protected
    assert rep["newly_masked"] == ["noise"] and load_block_mask(tmp_path) == rep["masked"]

    m = with_block_mask(cfg, ds)
    b = ds.layout.block("noise")
    assert (m.data["A"].signals[:, b.offset] == 0).all() and (m.data["A"].signals[:, b.start:b.end] == 0).all()
    g = ds.layout.block("good")
    assert (m.data["A"].signals[:, g.offset] == 1).all() and (ds.data["A"].signals[:, b.offset] == 1).all()   # the raw dataset is untouched
    assert m.meta["masked_blocks"] == ["noise"]

    rep2 = prune_blocks(cfg, ds, out_path=out, days=126, mask_t=0.0, unmask_t=99.0, min_names=3)    # masked blocks need the higher bar
    assert "noise" in rep2["masked"] and rep2["masked"]["noise"]["since"] == rep["masked"]["noise"]["since"] and rep2["newly_masked"] == []
    rep3 = prune_blocks(cfg, ds, out_path=out, days=126, mask_t=0.0, unmask_t=0.0, min_names=3)     # any value brings it back
    assert rep3["masked"] == {} and rep3["unmasked"] == ["noise"] and load_block_mask(tmp_path) == {}
    rep4 = prune_blocks(cfg, ds, out_path=out, days=126, mask_t=3.0, unmask_t=4.0, min_names=3, protect=["noise"])   # the owner's call
    assert rep4["masked"] == {} and "noise" in rep4["protected_without_value"]              # reported, never masked
    cfg.set_path("signals.pruning.enabled", False)
    assert with_block_mask(cfg, ds) is ds


def test_apply_mask_on_a_single_vector():
    layout = ObservationLayout([("technical", ["a"]), ("noise", ["n1", "n2"])])
    v = layout.assemble_latest({"technical": np.array([0.5]), "noise": np.array([1.0, 2.0])})
    m = layout.apply_mask(v, ["noise", "not_a_block"])
    assert m.tolist() == [1.0, 0.5, 0.0, 0.0, 0.0] and v.tolist() == [1.0, 0.5, 1.0, 1.0, 2.0]
