"""Checks for the finishing-order models (f1elo/rank_models.py).

    py test_rank_models.py            # about 1 minute
    py test_rank_models.py --trees    # also the CatBoost / XGBoost leakage check

1. The Plackett-Luce gradient matches finite differences.
2. The likelihood of ELO's strengths matches calibration.pl_order_loglik.
3. Walk-forward leakage: scrambling the results of season s (and every
   later season) leaves the walk-forward predictions for season s
   unchanged, so a season's predictions use only earlier seasons' results.
4. A simulated table has exactly the columns of cached_predictions, its
   retirement chances are ELO's, and every row sums to 1.
"""
from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np

from f1elo import calibration as cal
from f1elo import rank_models as rm


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")
    return ok


def scrambled(data: rm.StageData, season: int, seed: int = 1) -> rm.StageData:
    """A copy with the classified order of every race from ``season`` on
    shuffled (labels only; features untouched)."""
    rng = np.random.default_rng(seed)
    fields = data.fields.copy()
    fin = fields["finish"].to_numpy().copy()
    rn = fields["race_no"].to_numpy()
    for r in np.unique(rn[fields["year"].to_numpy() >= season]):
        i = np.flatnonzero((rn == r) & (fin > 0))
        fin[i] = rng.permutation(fin[i])
    fields["finish"] = fin
    return rm.StageData(data.stage, fields, data.elo, data.X, data.p_dnf, data.cfg)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trees", action="store_true")
    args = p.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ok = True
    data = rm.load_stage("post_quali")

    print("1. gradient")
    g = rm.Groups(data.race_no, data.finish, data.rows(2015, 2022))
    x = rm.elo_offset(data)[g.rows] + np.random.default_rng(0).normal(0, 0.3, len(g.rows))
    nll, gr, _ = rm.pl_grad_hess(x, g)
    err = 0.0
    for k in (0, 7, 500, len(x) - 1):
        e = 1e-5
        xp, xm = x.copy(), x.copy()
        xp[k] += e
        xm[k] -= e
        num = (rm.pl_grad_hess(xp, g)[0] - rm.pl_grad_hess(xm, g)[0]) / (2 * e)
        err = max(err, abs(num - gr[k]))
    ok &= check("analytic vs numeric gradient", err < 1e-5, f"max error {err:.2e}")

    print("2. ELO likelihood")
    rows = data.rows(1990, 2022)
    g = rm.Groups(data.race_no, data.finish, rows)
    mine = rm.pl_metrics(rm.elo_offset(data)[g.rows], g)["order_nll"]
    ref = cal.pl_order_loglik(data.fields, data.cfg.finish_car_weight, data.cfg.finish_scale,
                              cal.race_mask(data.fields, (1990, 2022)))
    ok &= check("order loss = calibration.pl_order_loglik", abs(mine - ref) < 1e-8,
                f"{mine:.6f} vs {ref:.6f}")

    print("3. walk-forward leakage")
    s = 2010
    makers = {"logit": lambda: rm.LogitModel(rm.LOGIT_SETS["post_quali"]["form"])}
    if args.trees:
        cols = rm.feature_columns(data.X, "post_quali")
        makers["catboost"] = lambda: rm.TreeModel("catboost", cols, 100, checkpoints=[100])
        makers["xgb"] = lambda: rm.TreeModel("xgb", cols, 100, depth=2, offset=None,
                                             checkpoints=[100])
        makers["xgb_pl"] = lambda: rm.TreeModel("xgb_pl", cols, 100, checkpoints=[100])
    bad = scrambled(data, s)
    for name, make in makers.items():
        a = rm.walk_forward(data, make, s, s)
        b = rm.walk_forward(bad, make, s, s)
        ka, kb = list(a.scores)[-1], list(b.scores)[-1]
        r = data.year == s
        diff = float(np.nanmax(np.abs(a.scores[ka][r] - b.scores[kb][r])))
        ok &= check(f"{name}: season {s} predictions unchanged when {s}+ results are scrambled",
                    diff < 1e-9, f"max change {diff:.2e}")
        c = rm.walk_forward(scrambled(data, s - 1), make, s, s)
        diff2 = float(np.nanmax(np.abs(a.scores[ka][r] - c.scores[list(c.scores)[-1]][r])))
        ok &= check(f"{name}: ... but they do change when {s - 1} is scrambled (the test can fail)",
                    diff2 > 1e-6, f"max change {diff2:.2e}")

    print("4. prediction table")
    wf = rm.walk_forward(data, makers["logit"], 2019, 2023)
    score = wf.scores[None]
    temps = rm.season_temperatures(data, score, wf.seasons, 3)
    xm, _ = rm.scaled(data, score, temps)
    rows = data.rows(min(temps), max(temps))
    table, pred = rm.simulate_model(data, xm, temps, rows)
    ok &= check("same columns as cached_predictions", list(table.columns) == list(data.elo.columns))
    ok &= check("p_dnf is ELO's", np.allclose(table["p_dnf"], data.p_dnf[rows]))
    pc = [c for c in table.columns if c.startswith("P") and c[1:].isdigit()] + ["DNF"]
    ok &= check("rows sum to 1", np.allclose(table[pc].sum(axis=1), 1.0))
    elo_dnf = data.elo.loc[rows, "DNF"].to_numpy()
    diff = float(np.abs(table["DNF"].to_numpy() - elo_dnf).max())
    ok &= check("the same simulated retirements as ELO's table (same seeds and draws)",
                diff == 0.0, f"max |DNF - ELO's DNF| {diff:.2e}")
    print("all checks passed" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
