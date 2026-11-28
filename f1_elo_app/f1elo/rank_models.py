"""Finishing-order models (step 6b): who finishes ahead of whom, among the
cars that finish.

Every model here gives each car a score ``x`` (natural-log Plackett-Luce
units: P(A ahead of B | both finish) = 1 / (1 + exp(x_B - x_A)) at one
scale). The scores replace the ELO strengths in the race simulation
(f1elo/simulate.py); retirements stay with the ELO retirement model
(``calibration.stage_dnf``: P(DNF), the shared race factor and, after
qualifying, the grid-slot step).

Models
------
  logit      rank-ordered logit (Plackett-Luce with features):
             x = sum_k b_k * feature_k, fitted by maximum likelihood over
             each race's classified finishing order, with an L2 penalty on
             the standardised coefficients. ELO's strength is one of the
             features, so the other coefficients are what ELO misses.
  catboost   CatBoost ranker (YetiRank or PairLogit), grouped by race
  xgb        XGBoost ``rank:pairwise``
  xgb_pl     XGBoost with the exact Plackett-Luce loss as a custom
             objective (gradient and diagonal Hessian below)
  The tree models can start from ELO's strength (``offset="elo"``: the
  trees only learn the correction) and handle missing values natively.

From scores to simulation strengths (``fit_temperature``)
---------------------------------------------------------
A model's raw scores are multiplied by a temperature ``t`` fitted on
out-of-sample predictions (the Plackett-Luce likelihood of the classified
order), with optionally a second temperature ``t_top`` for the first
``top_places`` places, like ELO's finish_top_scale / finish_top_places.
How predictable a place is falls steadily down the order (the winner most,
the back markers least), so ``t`` is fitted on the places just behind the
top ones (``max_places``, e.g. P4-P5) rather than the whole order: fitted
on the whole order it comes out too flat for the points places.
In the simulation's rating-point units: strength S = t * x * scale / ln 10
with scale = finish_scale (525), and top_scale = scale * t / t_top.

Validation
----------
Season-by-season walk-forward (``walk_forward``): for each season s from
``first_oof``, train on ``train_start`` .. s-1, predict season s. The
temperature for season s is fitted on the walk-forward predictions of the
seasons before s (at least ``MIN_TEMP_SEASONS``), so every prediction from
then on is fully out of sample. Settings (penalty, trees, iterations) are
chosen on the tuning seasons (``SCORE_PERIOD``, 2005-2022); the holdout
(2023 on) is predicted the same way, season by season, with those settings
frozen.

The prediction tables (``prediction_table``) have exactly the columns of
``calibration.cached_predictions`` (``strength`` is the model's S), for the
races from the first fully out-of-sample season on, and are cached as
output/race_predictions_<model>_<stage>.parquet.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import calibration as cal
from . import config
from . import features as F
from .config import DEFAULT_CONFIG, EloConfig
from .data_loader import build_race_frame
from .simulate import N_SIMS, SEED, pre_race_order, race_seed, simulate_race

LN10 = math.log(10.0)
STAGES = ("pre", "post_quali")
TUNE = (1990, 2022)
HOLDOUT_START = 2023
SCORE_PERIOD = (2005, 2022)
TRAIN_START = 1990
FIRST_OOF = 2000
MIN_TEMP_SEASONS = 3
MODEL_DIR = config.OUTPUT_DIR / "models"
CACHE_VERSION = 1

# Feature sets for the rank-ordered logit. "a*b" is an interaction.
_Q = ["q_gap_pole_rel", "q_grid_slot", "q_log_grid", "q_pit_lane", "q_gap_tm_rel"]
_FORM = ["drv_pace_rel", "team_pace_rel", "team_deficit_rel", "drv_fastlap_rel"]
LOGIT_SETS = {
    "post_quali": {
        "elo": ["elo_strength_rel"],
        "quali": ["elo_strength_rel"] + _Q,
        # the default: ELO + qualifying + race-pace form
        "form": ["elo_strength_rel"] + _Q + _FORM,
        # without the log grid rank (the straight-line grid slot only)
        "linear": ["elo_strength_rel", "q_gap_pole_rel", "q_grid_slot", "q_pit_lane",
                   "q_gap_tm_rel"] + _FORM,
        "full": ["elo_strength_rel"] + _Q + _FORM
                + ["street*q_grid_slot", "circ_grid_corr_rel*q_grid_slot",
                   "circ_pole_win_rel*q_pole"],
        "split": ["elo_driver_rel", "elo_car_rel"] + _Q + _FORM,
    },
    "pre": {
        "elo": ["elo_strength_rel"],
        "form": ["elo_strength_rel"] + _FORM,
        "full": ["elo_strength_rel"] + _FORM
                + ["team_fastlap_rel", "drv_deficit_rel", "street*elo_strength_rel"],
        # the default: driver and car ELO as separate terms + form
        "split": ["elo_driver_rel", "elo_car_rel"] + _FORM,
    },
}
DEFAULT_SET = {"post_quali": "form", "pre": "split"}

# Context columns the trees may use (race-level or already relative); every
# other stage feature is used only in its relative (``_rel``) version when
# it has one.
TREE_EXCLUDE = ("elo_rank",)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class StageData:
    """Everything a stage needs, all in the same rows (race order, then
    finishing order): the calibration fields, ELO's prediction table, the
    feature table and the retirement chance the simulation uses."""
    stage: str
    fields: pd.DataFrame
    elo: pd.DataFrame
    X: pd.DataFrame
    p_dnf: np.ndarray
    cfg: EloConfig = field(default_factory=lambda: DEFAULT_CONFIG)
    fingerprint: str = ""

    @property
    def year(self) -> np.ndarray:
        return self.fields["year"].to_numpy()

    @property
    def race_no(self) -> np.ndarray:
        return self.fields["race_no"].to_numpy()

    @property
    def finish(self) -> np.ndarray:
        return self.fields["finish"].to_numpy()

    def rows(self, first: int, last: int) -> np.ndarray:
        y = self.year
        return (y >= first) & (y <= last)

    @property
    def latest_year(self) -> int:
        return int(self.year.max())


def load_stage(stage: str, cfg: EloConfig = DEFAULT_CONFIG, progress=None) -> StageData:
    """Fields, ELO predictions and features of a stage (all cached by the
    earlier steps; the first run builds them)."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    say = progress or (lambda *_: None)
    df = build_race_frame()
    fields = cal.build_fields(stage, cfg, df, progress=bool(progress))
    elo = cal.cached_predictions(stage, cfg, df=df, fields=fields, progress=bool(progress))
    say(f"  loading the {stage} features ...")
    X = F.cached_features(stage, cfg=cfg, progress=progress).reset_index(drop=True)
    for c in ("raceId", "driverId"):
        if not (np.array_equal(X[c].to_numpy(), fields[c].to_numpy())
                and np.array_equal(elo[c].to_numpy(), fields[c].to_numpy())):
            raise ValueError(f"features / predictions / fields rows differ ({c})")
    if not np.array_equal(X["y_finish"].to_numpy(), fields["finish"].to_numpy()):
        raise ValueError("features' y_finish differs from the fields' finish")
    X = X.copy()
    if "q_grid_rank" in X:
        # The front of the grid counts more than a straight line in the
        # slot allows: pole, front row and the log of the grid rank.
        X["q_pole"] = (X["q_grid_rank"] == 1).astype(float)
        X["q_front_row"] = (X["q_grid_rank"] <= 2).astype(float)
        X["q_log_grid"] = np.log(X["q_grid_rank"].astype(float))
    X["elo_log_rank"] = np.log(X["elo_rank"].astype(float))
    p_dnf = cal.stage_dnf(fields, stage, cfg)
    return StageData(stage, fields, elo, X, p_dnf, cfg, cal.data_fingerprint(df))


def feature_columns(X: pd.DataFrame, stage: str) -> list[str]:
    """The broad feature set for the trees: every stage feature, in its
    relative version when it has one."""
    feats, _ = F.stage_columns(X, stage)
    have = set(feats)
    out = []
    for c in feats:
        if c in TREE_EXCLUDE or c.startswith("y_"):
            continue
        if not c.endswith("_rel") and f"{c}_rel" in have:
            continue
        out.append(c)
    return out


def elo_offset(data: StageData) -> np.ndarray:
    """ELO's strength in Plackett-Luce units at the finish scale (relative
    to the race; the level doesn't matter)."""
    return data.X["elo_strength_rel"].to_numpy(float) * LN10 / data.cfg.finish_scale


# ---------------------------------------------------------------------------
# Plackett-Luce over the classified cars
# ---------------------------------------------------------------------------
class Groups:
    """The classified rows of some races, in finishing order, as a padded
    (races, places) index into ``rows`` (-1 = no car)."""

    def __init__(self, race_no, finish, rows):
        race_no, finish = np.asarray(race_no), np.asarray(finish)
        rows = np.flatnonzero(rows) if np.asarray(rows).dtype == bool else np.asarray(rows)
        rows = rows[finish[rows] > 0]
        rows = rows[np.lexsort((finish[rows], race_no[rows]))]
        rn = race_no[rows]
        new = np.r_[True, rn[1:] != rn[:-1]] if len(rn) else np.zeros(0, bool)
        gid = np.cumsum(new) - 1
        starts = np.flatnonzero(new)
        pos = np.arange(len(rows)) - starts[gid]
        self.R = int(len(starts))
        self.M = int(pos.max()) + 1 if len(pos) else 0
        self.idx = np.full((self.R, self.M), -1)
        self.idx[gid, pos] = np.arange(len(rows))
        self.valid = self.idx >= 0
        self.rows = rows
        self.race_no = rn[starts]


def _stage_terms(x: np.ndarray, g: Groups):
    """U (scores in place order), L (log-sum-exp of each place's remaining
    cars) and the log chance of each place's actual car."""
    U = np.where(g.valid, x[np.maximum(g.idx, 0)], -np.inf)
    with np.errstate(invalid="ignore"):
        L = np.logaddexp.accumulate(U[:, ::-1], axis=1)[:, ::-1]
        T = np.where(g.valid, U - L, 0.0)
    return U, L, T


def pl_terms(x_main, g: Groups, x_top=None, top_places: int = 0) -> np.ndarray:
    """(races, places) log chance of every place's actual car: the first
    ``top_places`` places at ``x_top``, the rest at ``x_main`` (both
    indexed like ``g.rows``)."""
    T = _stage_terms(np.asarray(x_main, float), g)[2]
    if top_places and x_top is not None:
        Tt = _stage_terms(np.asarray(x_top, float), g)[2]
        T = np.where(np.arange(g.M)[None, :] < top_places, Tt, T)
    return T


def pl_metrics(x_main, g: Groups, x_top=None, top_places: int = 0) -> dict:
    """Mean loss per race: the whole classified order, the first three
    places, and the winner (-ln P(the winner wins | who finished))."""
    T = pl_terms(x_main, g, x_top, top_places)
    return {"order_nll": float(-T.sum() / g.R), "top3_nll": float(-T[:, :3].sum() / g.R),
            "win_nll": float(-T[:, 0].sum() / g.R), "races": g.R}


def pl_grad_hess(x, g: Groups):
    """Negative log-likelihood (sum over races) of one scale's scores and
    its gradient and diagonal Hessian per row of ``g.rows``.

    With P_jk = exp(x_k - L_j) the chance that car k takes place j (k at or
    behind j): d(-ll)/dx_k = sum_{j<=k} P_jk - 1 and
    d2(-ll)/dx_k^2 = sum_{j<=k} P_jk (1 - P_jk)."""
    x = np.asarray(x, float)
    U, L, T = _stage_terms(x, g)
    m = np.where(g.valid, U, -np.inf).max(axis=1, keepdims=True)
    with np.errstate(over="ignore", invalid="ignore"):
        a = np.where(g.valid, np.exp(np.minimum(m - L, 300.0)), 0.0)
        e = np.where(g.valid, np.exp(U - m), 0.0)
    s1 = e * np.cumsum(a, axis=1)
    s2 = e ** 2 * np.cumsum(a ** 2, axis=1)
    grad = np.zeros(len(x))
    hess = np.zeros(len(x))
    v = g.valid
    grad[g.idx[v]] = (s1 - 1.0)[v]
    hess[g.idx[v]] = np.maximum(s1 - s2, 1e-9)[v]
    return float(-T.sum()), grad, hess


# ---------------------------------------------------------------------------
# Temperature: raw scores -> simulation strengths
# ---------------------------------------------------------------------------
@dataclass
class Temperature:
    """``t`` for the main scale, ``t_top`` for the first ``top_places``
    places; ``fit_places`` = the places the main scale was fitted on (0 =
    the whole classified order). Only t, t_top and top_places reach the
    simulation."""
    t: float = 1.0
    t_top: float = 1.0
    top_places: int = 0
    fit_places: int = 0

    def main(self, score):
        return self.t * np.asarray(score, float)

    def top(self, score):
        return self.t_top * np.asarray(score, float)

    def scales(self, finish_scale: float) -> tuple[float, float | None]:
        """(scale, top_scale) for simulate_race with S = t * x * scale / ln 10."""
        if not self.top_places:
            return finish_scale, None
        return finish_scale, finish_scale * self.t / self.t_top


def fit_temperature(score, g: Groups, top_places: int = 0,
                    max_places: int | None = None) -> Temperature:
    """t (and t_top for the first ``top_places`` places) maximising the
    Plackett-Luce likelihood of the classified order of ``g``'s races.
    ``max_places``: only the first places count (who finished where among
    the front of the field), so the very predictable order of the back
    markers doesn't set the scale of the midfield."""
    s = np.asarray(score, float)
    keep = (np.arange(g.M)[None, :] < max_places) if max_places else 1.0

    def f(p):
        t = math.exp(p[0])
        tt = math.exp(p[1]) if top_places else t
        return -(pl_terms(t * s, g, tt * s, top_places) * keep).sum() / g.R

    x0 = [0.0, 0.0] if top_places else [0.0]
    r = minimize(f, x0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7})
    t = math.exp(r.x[0])
    return Temperature(t, math.exp(r.x[1]) if top_places else t, top_places,
                       int(max_places or 0))


# ---------------------------------------------------------------------------
# Rank-ordered logit
# ---------------------------------------------------------------------------
@dataclass
class Design:
    """Terms -> a standardised matrix. A term is a column or "a*b". Missing
    values are filled with the race's mean of the cars that have it (0 for
    the ``_rel`` features, i.e. an average car), with a missing-value
    indicator per term that was missing in training (``indicators``).
    Continuous terms are clipped to the training ``clip`` / 1 - ``clip``
    quantiles (wet qualifying makes gaps of 10-30%). A suffix picks the
    tail: "q_gap_pole:hi" clips only the slow end (the front of the grid
    keeps its full gap), ":lo" only the low end, ":none" neither."""
    terms: list
    indicators: bool = True
    clip: float = 0.02
    names: list = field(default_factory=list)
    lo: dict = field(default_factory=dict)
    hi: dict = field(default_factory=dict)
    mean: list = field(default_factory=list)
    std: list = field(default_factory=list)
    with_ind: list = field(default_factory=list)

    @staticmethod
    def _raw(X: pd.DataFrame, term: str) -> np.ndarray:
        term = term.split(":")[0]
        if "*" in term:
            a, b = term.split("*")
            return X[a].to_numpy(float) * X[b].to_numpy(float)
        return X[term].to_numpy(float)

    def _columns(self, X: pd.DataFrame, race_no) -> tuple[list, list]:
        cols, inds = [], []
        for t in self.terms:
            v = self._raw(X, t)
            miss = np.isnan(v)
            if miss.any():
                fill = pd.Series(v).groupby(race_no).transform("mean").to_numpy()
                v = np.where(miss, np.nan_to_num(fill, nan=0.0), v)
            if t in self.lo:
                v = np.clip(v, self.lo[t], self.hi[t])
            cols.append(v)
            inds.append(miss.astype(float))
        return cols, inds

    def fit(self, X: pd.DataFrame, race_no, rows) -> "Design":
        self.lo, self.hi = {}, {}
        raw = [self._raw(X, t)[rows] for t in self.terms]
        for t, v in zip(self.terms, raw):
            v = v[~np.isnan(v)]
            tail = t.split(":")[1] if ":" in t else "both"
            if self.clip and len(np.unique(v)) > 2 and tail != "none":
                lo, hi = (float(q) for q in np.quantile(v, [self.clip, 1 - self.clip]))
                self.lo[t] = lo if tail in ("both", "lo") else -np.inf
                self.hi[t] = hi if tail in ("both", "hi") else np.inf
        cols, inds = self._columns(X, race_no)
        self.with_ind = [bool(self.indicators and ind[rows].any()) for ind in inds]
        mats = [c[rows] for c in cols] + [i[rows] for i, w in zip(inds, self.with_ind) if w]
        M = np.column_stack(mats)
        self.mean = M.mean(axis=0).tolist()
        sd = M.std(axis=0)
        self.std = np.where(sd > 0, sd, 1.0).tolist()
        self.names = list(self.terms) + [f"{t} missing" for t, w in zip(self.terms, self.with_ind) if w]
        return self

    def transform(self, X: pd.DataFrame, race_no) -> np.ndarray:
        cols, inds = self._columns(X, race_no)
        M = np.column_stack(cols + [i for i, w in zip(inds, self.with_ind) if w])
        return (M - np.array(self.mean)) / np.array(self.std)

    def to_dict(self) -> dict:
        return asdict(self)


class LogitModel:
    """Rank-ordered logit: x = Z b, Z the standardised design."""
    kind = "logit"

    def __init__(self, terms, lam: float = 1.0, indicators: bool = True,
                 clip: float = 0.02):
        self.terms, self.lam = list(terms), float(lam)
        self.indicators, self.clip = indicators, clip
        self.design: Design | None = None
        self.beta: np.ndarray | None = None
        self.converged = None

    def params(self) -> dict:
        return {"terms": self.terms, "lam": self.lam, "indicators": self.indicators,
                "clip": self.clip}

    def fit(self, data: StageData, rows) -> "LogitModel":
        rn = data.race_no
        self.design = Design(self.terms, self.indicators, self.clip).fit(data.X, rn, rows)
        g = Groups(rn, data.finish, rows)
        Z = self.design.transform(data.X, rn)[g.rows]
        lam, R = self.lam, g.R

        def f(b):
            nll, gr, _ = pl_grad_hess(Z @ b, g)
            return nll / R + 0.5 * lam * b @ b, Z.T @ gr / R + lam * b

        b0 = np.zeros(Z.shape[1]) if self.beta is None or len(self.beta) != Z.shape[1] else self.beta
        r = minimize(f, b0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
        self.beta, self.converged = r.x, bool(r.success)
        return self

    def predict(self, data: StageData, n_iter=None) -> np.ndarray:
        return self.design.transform(data.X, data.race_no) @ self.beta

    def coefficients(self) -> pd.DataFrame:
        """Per term: the coefficient per raw unit (natural PL units), per
        standard deviation, and in ELO points (how many points of ELO
        strength the same raw unit is worth, at the fitted ELO coefficient
        when the model has elo_strength_rel)."""
        d = self.design
        raw = self.beta / np.array(d.std)
        out = pd.DataFrame({"term": d.names, "per_unit": raw, "per_sd": self.beta,
                            "sd": d.std})
        if "elo_strength_rel" in d.names:
            b_elo = raw[d.names.index("elo_strength_rel")]
            out["elo_points_per_unit"] = raw / b_elo
        return out

    def to_dict(self) -> dict:
        return {"kind": self.kind, "params": self.params(), "design": self.design.to_dict(),
                "beta": self.beta.tolist()}


# ---------------------------------------------------------------------------
# Tree rankers
# ---------------------------------------------------------------------------
def _relevance(finish: np.ndarray, race_no: np.ndarray) -> np.ndarray:
    """Higher = better: classified cars in a race minus the place (the
    last classified car 0)."""
    ncls = pd.Series(finish > 0).groupby(race_no).transform("sum").to_numpy()
    return (ncls - finish).astype(float)


class TreeModel:
    """CatBoost (``catboost``: YetiRank / PairLogit) or XGBoost (``xgb``:
    rank:pairwise; ``xgb_pl``: the Plackett-Luce loss) on classified cars,
    grouped by race. ``offset="elo"`` starts every car from ELO's strength
    (PL units at the finish scale) so the trees fit the correction.
    ``checkpoints`` are the iteration counts predictions are made at (the
    count is chosen by walk-forward validation)."""

    def __init__(self, kind: str, columns, iterations: int = 600, depth: int = 3,
                 lr: float = 0.05, l2: float = 10.0, loss: str = "YetiRank",
                 offset: str | None = "elo", checkpoints=None, min_child: float = 1.0,
                 subsample: float = 0.8, colsample: float = 0.8, seed: int = 0,
                 threads: int = -1):
        self.kind, self.columns = kind, list(columns)
        self.iterations, self.depth, self.lr, self.l2 = int(iterations), int(depth), lr, l2
        self.loss, self.offset, self.min_child = loss, offset, min_child
        self.subsample, self.colsample, self.seed, self.threads = subsample, colsample, seed, threads
        self.checkpoints = sorted(set(int(c) for c in (checkpoints or [iterations])
                                      if c <= iterations))
        self.model = None

    def params(self) -> dict:
        return {"kind": self.kind, "columns": self.columns, "iterations": self.iterations,
                "depth": self.depth, "lr": self.lr, "l2": self.l2, "loss": self.loss,
                "offset": self.offset, "min_child": self.min_child,
                "subsample": self.subsample, "colsample": self.colsample, "seed": self.seed}

    def _offset(self, data: StageData) -> np.ndarray:
        return elo_offset(data) if self.offset == "elo" else np.zeros(len(data.X))

    def fit(self, data: StageData, rows) -> "TreeModel":
        g = Groups(data.race_no, data.finish, rows)
        r = g.rows                                     # race order, finishing order
        X = data.X[self.columns].iloc[r].astype(float)
        y = _relevance(data.finish, data.race_no)[r]
        base = self._offset(data)[r]
        qid = data.race_no[r]
        if self.kind == "catboost":
            from catboost import CatBoost, Pool
            pool = Pool(X, label=y, group_id=qid, baseline=base)
            self.model = CatBoost({
                "loss_function": self.loss, "iterations": self.iterations,
                "depth": self.depth, "learning_rate": self.lr, "l2_leaf_reg": self.l2,
                "random_seed": self.seed, "verbose": False, "thread_count": self.threads,
                "allow_writing_files": False, "bootstrap_type": "Bernoulli",
                "subsample": self.subsample, "rsm": self.colsample})
            self.model.fit(pool)
        else:
            import xgboost as xgb
            dm = xgb.DMatrix(X, label=y, base_margin=base, qid=qid)
            p = {"max_depth": self.depth, "eta": self.lr, "lambda": self.l2,
                 "min_child_weight": self.min_child,
                 "subsample": self.subsample, "colsample_bytree": self.colsample,
                 "seed": self.seed, "nthread": self.threads if self.threads > 0 else 0,
                 "tree_method": "hist"}
            if self.kind == "xgb":
                p.update(objective="rank:pairwise", lambdarank_pair_method="mean",
                         lambdarank_num_pair_per_sample=8)
                self.model = xgb.train(p, dm, self.iterations)
            else:
                # r is already in race and finishing order, so gl.rows is
                # 0 .. len(r) - 1 in the DMatrix's row order.
                gl = Groups(qid, data.finish[r], np.arange(len(r)))

                def obj(pred, dtrain):
                    _, gr, h = pl_grad_hess(pred, gl)
                    return gr, h
                p["disable_default_eval_metric"] = 1
                self.model = xgb.train(p, dm, self.iterations, obj=obj)
        return self

    def predict(self, data: StageData, n_iter: int | None = None) -> np.ndarray:
        X = data.X[self.columns].astype(float)
        base = self._offset(data)
        k = n_iter or self.iterations
        if self.kind == "catboost":
            return self.model.predict(X, prediction_type="RawFormulaVal", ntree_end=k) + base
        import xgboost as xgb
        return self.model.predict(xgb.DMatrix(X, base_margin=base), iteration_range=(0, k))

    def predict_all(self, data: StageData) -> dict:
        return {k: self.predict(data, k) for k in self.checkpoints}

    def importance(self, data: StageData, rows) -> pd.DataFrame:
        """Mean |SHAP| per feature on ``rows`` (and CatBoost's own
        PredictionValuesChange, XGBoost's total gain)."""
        X = data.X[self.columns].iloc[np.flatnonzero(rows) if np.asarray(rows).dtype == bool
                                      else rows].astype(float)
        if self.kind == "catboost":
            from catboost import Pool
            gid = data.race_no[np.flatnonzero(rows) if np.asarray(rows).dtype == bool else rows]
            pool = Pool(X, group_id=gid)
            shap = self.model.get_feature_importance(pool, type="ShapValues")[:, :-1]
            own = self.model.get_feature_importance(type="PredictionValuesChange")
            own_name = "catboost_pvc"
        else:
            import xgboost as xgb
            shap = self.model.predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
            gain = self.model.get_score(importance_type="total_gain")
            own = np.array([gain.get(c, 0.0) for c in self.columns])
            own = 100 * own / own.sum() if own.sum() > 0 else own
            own_name = "xgb_total_gain_pct"
        out = pd.DataFrame({"feature": self.columns,
                            "mean_abs_shap": np.abs(shap).mean(axis=0), own_name: own})
        return out.sort_values("mean_abs_shap", ascending=False, ignore_index=True)

    def save(self, path_stem) -> list:
        if self.kind == "catboost":
            p = f"{path_stem}.cbm"
            self.model.save_model(p)
        else:
            p = f"{path_stem}.ubj"
            self.model.save_model(p)
        return [p]


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
@dataclass
class WalkForward:
    """Season-by-season out-of-sample scores: ``scores[k]`` per row (NaN
    outside the predicted seasons) for every iteration checkpoint k (one
    key, None, for the logit)."""
    scores: dict
    seasons: list
    seconds: float
    fits: list = field(default_factory=list)


def walk_forward(data: StageData, make_model, first: int = FIRST_OOF, last: int | None = None,
                 train_start: int = TRAIN_START, progress=None) -> WalkForward:
    """For each season s in first .. last: train ``make_model()`` on the
    classified cars of train_start .. s-1, predict every car of season s."""
    say = progress or (lambda *_: None)
    last = data.latest_year if last is None else last
    y = data.year
    scores: dict = {}
    t0 = time.perf_counter()
    seasons = [s for s in range(first, last + 1)
               if (y == s).any() and ((y >= train_start) & (y < s)).any()]
    fits = []
    for s in seasons:
        m = make_model()
        m.fit(data, (y >= train_start) & (y < s))
        rows = y == s
        preds = m.predict_all(data) if hasattr(m, "predict_all") else {None: m.predict(data)}
        for k, v in preds.items():
            arr = scores.setdefault(k, np.full(len(y), np.nan))
            arr[rows] = v[rows]
        fits.append({"season": s, "converged": getattr(m, "converged", None)})
        say(f"    {s} [{time.perf_counter() - t0:.0f} s]")
    return WalkForward(scores, seasons, time.perf_counter() - t0, fits)


def season_temperatures(data: StageData, score, seasons, top_places: int = 0,
                        min_seasons: int = MIN_TEMP_SEASONS,
                        max_places: int | None = None) -> dict:
    """{season: Temperature} fitted on the out-of-sample scores of the
    seasons before it (from the first predicted season), for the seasons
    with at least ``min_seasons`` such seasons."""
    y = data.year
    out = {}
    for i, s in enumerate(seasons):
        if i < min_seasons:
            continue
        rows = np.isin(y, seasons[:i])
        g = Groups(data.race_no, data.finish, rows)
        out[s] = fit_temperature(score[g.rows], g, top_places, max_places)
    return out


def scaled(data: StageData, score, temps: dict) -> tuple[np.ndarray, np.ndarray]:
    """(x_main, x_top) per row with each season's temperature (NaN where
    a season has none)."""
    y = data.year
    xm = np.full(len(y), np.nan)
    xt = np.full(len(y), np.nan)
    for s, t in temps.items():
        r = y == s
        xm[r], xt[r] = t.main(score[r]), t.top(score[r])
    return xm, xt


def oof_metrics(data: StageData, x_main, x_top, top_places, period) -> dict:
    """Plackett-Luce losses of out-of-sample strengths on a period."""
    rows = data.rows(*period) & ~np.isnan(x_main)
    g = Groups(data.race_no, data.finish, rows)
    return pl_metrics(x_main[g.rows], g, x_top[g.rows], top_places)


# ---------------------------------------------------------------------------
# Simulation and prediction tables
# ---------------------------------------------------------------------------
def subset(fields: pd.DataFrame, rows) -> pd.DataFrame:
    """The fields of some races with race_no renumbered 0, 1, ... (what
    calibration's scoring functions expect)."""
    f = fields[rows].reset_index(drop=True).copy()
    r = f["race_no"].to_numpy()
    f["race_no"] = np.r_[0, np.cumsum(r[1:] != r[:-1])] if len(r) else r
    return f


def simulate_model(data: StageData, x_main, temps: dict, rows, n_sims: int = N_SIMS,
                   seed: int | None = SEED) -> tuple[pd.DataFrame, cal.Pred]:
    """Simulate the races in ``rows`` with the model's strengths and the ELO
    retirement model: every race with its own seed (race_seed) and its cars
    in ELO's pre-race order, so the retirement draws are exactly those of
    the ELO table and only the finishing order differs. Returns the
    prediction table (cached_predictions' columns, ``strength`` = the
    model's S in rating points) and the Pred for scoring on
    ``subset(fields, rows)``."""
    cfg = data.cfg
    f = subset(data.fields, rows)
    idx = np.flatnonzero(rows)
    S = x_main[idx] * cfg.finish_scale / LN10
    q = data.p_dnf[idx]
    elo_s = cal.strength_of(f, cfg.finish_car_weight)
    years = f["year"].to_numpy()
    ids, drv = f["raceId"].to_numpy(), f["driverId"].to_numpy()
    out = cal._empty(f)
    out.sims = n_sims
    starts, ends = cal.race_bounds(f)
    for r in range(len(starts)):
        a, b = starts[r], ends[r]
        n = b - a
        t = temps[int(years[a])]
        scale, top_scale = t.scales(cfg.finish_scale)
        o = a + pre_race_order(elo_s[a:b], drv[a:b])
        pos, win = simulate_race(S[o], q[o], n_sims, race_seed(seed, ids[a]), scale,
                                 cfg.dnf_race_corr, None, top_scale, t.top_places)
        ix = (np.arange(n)[None, :] * (n + 1) + pos).ravel()
        probs = np.bincount(ix, minlength=n * (n + 1)).reshape(n, n + 1) / n_sims
        out.P[o, :n] = probs[:, :n]
        out.dnf[o] = probs[:, n]
        out.win[o] = win
        out.fin[r, :n + 1] = np.bincount((pos < n).sum(axis=1), minlength=n + 1) / n_sims
    table = cal.prediction_table(f, out, cfg.finish_car_weight)
    table["strength"] = S
    table["p_dnf"] = q
    table["race_no"] = data.fields["race_no"].to_numpy()[idx]
    # Exactly ELO's columns (P1 ... P<largest field ever>, 0 past a race's size).
    pcols = [c for c in data.elo.columns if c.startswith("P") and c[1:].isdigit()]
    for c in pcols:
        if c not in table:
            table[c] = 0.0
    table = table[list(data.elo.columns)]
    out.fin = cal.exact_finishers(f, q, cfg.dnf_race_corr)
    return table, out


def table_path(name: str, stage: str):
    return config.OUTPUT_DIR / f"race_predictions_{name}_{stage}.parquet"


def write_table(table: pd.DataFrame, name: str, stage: str, info: dict) -> None:
    path = table_path(name, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    key = hashlib.sha1(json.dumps(info, sort_keys=True, default=str).encode()).hexdigest()[:16]
    path.with_suffix(".json").write_text(json.dumps({"key": key, "version": CACHE_VERSION,
                                                     **info}, indent=1, default=str))


def read_table(name: str, stage: str) -> pd.DataFrame | None:
    """A model's cached prediction table (cached_predictions' columns), or
    None if it hasn't been built (``py train_rank_models.py``)."""
    path = table_path(name, stage)
    return pd.read_parquet(path) if path.exists() else None


# ---------------------------------------------------------------------------
# Scoring against ELO and the grid slot
# ---------------------------------------------------------------------------
def elo_pred(data: StageData, rows, n_sims: int = N_SIMS) -> cal.Pred:
    f = subset(data.fields, rows)
    t = data.elo[rows].reset_index(drop=True)
    return cal.pred_from_table(f, t, cal.exact_finishers(f, data.p_dnf[rows],
                                                         data.cfg.dnf_race_corr), n_sims)


def grid_pred(data: StageData, rows, walk_forward_fit: bool = False) -> cal.Pred:
    """The grid-slot table (step 4's baseline): the historical results of
    each grid slot over 1990-2022, or (``walk_forward_fit``) over
    TRAIN_START .. the season before each race."""
    full = data.fields
    gr = cal.grid_rank(full)
    f = subset(full, rows)
    if not walk_forward_fit:
        counts = cal.rank_table(full, gr, cal.race_mask(full, TUNE))
        return cal.rank_table_pred(f, gr[rows], counts)
    out = cal._empty(f, with_fin=False)
    years_f = f["year"].to_numpy()
    rf = cal.race_frame(full)["year"].to_numpy()
    for s in np.unique(years_f):
        counts = cal.rank_table(full, gr, (rf >= TRAIN_START) & (rf < s))
        m = years_f == s
        races = np.unique(f["race_no"].to_numpy()[m])
        p = cal.rank_table_pred(f, gr[rows], counts, races=races)
        out.P[m], out.dnf[m], out.win[m] = p.P[m], p.dnf[m], p.win[m]
    return out


SHOW = ["win_ll", "podium_ll", "podium_brier", "points_ll", "points_brier", "rps"]


def compare(fields_sub: pd.DataFrame, preds: dict, masks: dict, reference: str,
            boot: int = 2000) -> pd.DataFrame:
    """Every predictor x period x score with 95% bootstrap intervals (races
    resampled), and each predictor's difference from ``reference`` (paired)."""
    rows = []
    for label, m in masks.items():
        per = {k: cal.race_scores(fields_sub, p, races=m) for k, p in preds.items()}
        base = per[reference]
        bsum = cal.summarise_scores(base)
        for name, r in per.items():
            s = cal.summarise_scores(r)
            for k in SHOW:
                row = {"predictor": name, "period": label, "metric": k, "value": s[k],
                       "races": s["races"], "entries": s["entries"]}
                if boot:
                    row["ci_lo"], row["ci_hi"] = cal.bootstrap(r, k, reps=boot, level=0.95)
                    if name != reference:
                        row["diff"] = s[k] - bsum[k]
                        row["diff_lo"], row["diff_hi"] = cal.bootstrap(r, k, base, reps=boot,
                                                                       level=0.95)
                rows.append(row)
    return pd.DataFrame(rows)


def bands(fields_sub: pd.DataFrame, preds: dict, masks: dict) -> pd.DataFrame:
    """Calibration bands (win / podium / points per points era) for every
    predictor and period, as calibration_bands_<stage>.csv."""
    out = []
    for name, p in preds.items():
        ch = cal.entry_chances(fields_sub, p)
        for label, m in masks.items():
            c = ch[m[fields_sub["race_no"].to_numpy()]]
            for o in ("win", "podium", "points"):
                groups = [("all", np.ones(len(c), bool))]
                if o == "points":
                    eras = cal.points_era(c["points_positions"])
                    groups = [(g, eras == g) for g in sorted(set(eras),
                                                             key=lambda s: int(s.split()[1]))]
                for gname, gm in groups:
                    if not gm.any():
                        continue
                    t = cal.calibration_bands(c[o][gm], c[f"y_{o}"][gm])
                    t.insert(0, "group", gname)
                    t.insert(0, "outcome", o)
                    t.insert(0, "period", label)
                    t.insert(0, "predictor", name)
                    out.append(t)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# Saved models
# ---------------------------------------------------------------------------
def save_model(model, name: str, stage: str, temp: Temperature, info: dict) -> list:
    """The final model (trained on every season up to the latest) and its
    temperature, in output/models/."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stem = MODEL_DIR / f"{name}_{stage}"
    meta = {"name": name, "stage": stage, "temperature": asdict(temp), **info}
    files = []
    if isinstance(model, LogitModel):
        meta["model"] = model.to_dict()
    else:
        meta["model"] = model.params()
        files += model.save(stem)
    p = f"{stem}.json"
    with open(p, "w") as fh:
        json.dump(meta, fh, indent=1, default=str)
    return [p] + files
