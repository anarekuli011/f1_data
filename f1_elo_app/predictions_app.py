"""The app's Predictions and Calibration pages (drawn by app.py).

Predictions: one race's chances of every finishing position, from the race
simulation (f1elo/simulate.py). Past races come from step 4's cached tables
(f1elo/calibration.py, output/race_predictions_<stage>.parquet); the next
race is simulated here (simulate.predict_next_race).

Every page reads predictions through ``load_predictions``, which returns the
standard table (one row per car: the inputs, P1 ... Pn, DNF, win, podium,
points, exp_pos, actual). The model behind it is looked up in ``SOURCES``,
so another model that returns the same table (a step 6 gradient-boosted
model, say) only needs an entry there and a "Model" option.

Calibration: how far the predictions can be trusted, from the CSVs that
backtest_calibration.py writes to output/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_common import PALETTE, choice
from f1elo import calibration, config
from f1elo.backtest import prepare_races
from f1elo.data_loader import build_race_frame, load_data
from f1elo.reliability import entry_history, features
from f1elo.simulate import DNF, KEYS, N_SIMS, predict_next_race
from f1elo.strength import add_names

CFG = config.DEFAULT_CONFIG
state = st.session_state

# Everything the predictions are built from; a change (the sidebar update)
# gives every cache below a new key.
DATA_FILES = [config.DATA_DIR / f for f in (config.RACES_CSV, config.RESULTS_CSV,
                                            config.DRIVERS_CSV, config.CONSTRUCTORS_CSV,
                                            config.QUALIFYING_CSV)]
STAGES = {"Before qualifying": "pre", "After qualifying": "post_quali"}
STAGE_TEXT = {"pre": "before qualifying (ratings before the weekend)",
              "post_quali": "after qualifying (ratings after that weekend's qualifying "
                            "pass, and the grid slot in the retirement chance)"}
# Retirement counts behind P(DNF) (f1elo/reliability.py).
COUNT_COLUMNS = ["base_rate", "team_dnfs", "team_entries", "driver_dnfs",
                 "driver_entries", "team_rate", "driver_rate", "p_blend"]
# The standard table: inputs, then the chances, then the result.
INPUT_COLUMNS = ["driver", "team", "driverId", "constructorId", "driver_elo",
                 "constructor_elo", "strength", "p_dnf", "grid", *COUNT_COLUMNS]
OUTCOME_COLUMNS = [DNF, "win", "podium", "points", "exp_pos", "actual"]
# Widget keys this module draws (app.py re-saves them every run so they
# survive while another view is shown).
STATE_KEYS = ("pred:mode", "pred:year", "pred:race", "pred:stage", "pred:driver",
              "pred:seat", "pred:new_driver", "cal:stage", "cal:period", "cal:predictor")
HIGHLIGHT = "crimson"     # the actual result, in the heatmap and the bar chart


def data_mtimes() -> tuple:
    return tuple(p.stat().st_mtime if p.exists() else 0.0 for p in DATA_FILES)


# -- shared data (read-only, one copy for every session) ----------------------
@st.cache_resource(max_entries=1, show_spinner="Loading the race data…")
def race_data(mtimes: tuple) -> dict:
    df = build_race_frame()
    hist = entry_history(df, CFG)
    return {"df": df, "raw": load_data(), "hist": hist, "rel": features(hist, CFG)}


@st.cache_resource(max_entries=1, show_spinner="Preparing every race for the engine replay…")
def prepared_races(mtimes: tuple) -> list:
    return prepare_races(race_data(mtimes)["df"], CFG)


@st.cache_resource(max_entries=2, show_spinner="Loading the predictions for every past race "
                   "(after new data this rebuilds them once: about 75 s)…")
def all_predictions(mtimes: tuple, stage: str) -> pd.DataFrame:
    return calibration.cached_predictions(stage, CFG, df=race_data(mtimes)["df"])


@st.cache_data(max_entries=4, show_spinner=False)
def past_races(mtimes: tuple) -> pd.DataFrame:
    """Races with predictions: raceId, year, round, name (Indy 500 excluded)."""
    p = all_predictions(mtimes, "pre")
    return p.drop_duplicates("raceId")[["raceId", "year", "round", "name"]].reset_index(drop=True)


@st.cache_data(max_entries=4, show_spinner=False)
def latest_entries(mtimes: tuple) -> list:
    """The latest race's entry list, (driverId, constructorId) per seat: the
    next race's default field (as in strength.next_race_strengths)."""
    result = prepared_races(mtimes)[-1][4]
    seats = result[["driverId", "constructorId"]].drop_duplicates("driverId")
    return [(int(d), int(c)) for d, c in zip(seats["driverId"], seats["constructorId"])]


# -- the standard prediction table ---------------------------------------------
def _standard(table: pd.DataFrame, n: int) -> pd.DataFrame:
    pcols = [f"P{i + 1}" for i in range(n)]
    out = table.reindex(columns=INPUT_COLUMNS + pcols + OUTCOME_COLUMNS).reset_index(drop=True)
    out["actual"] = [None if pd.isna(a) else str(a) for a in out["actual"]]
    return out


def _with_counts(table: pd.DataFrame, rel: pd.DataFrame) -> pd.DataFrame:
    """Add the retirement counts behind P(DNF), car by car (a 1950s driver in
    two cars of one make is matched in order)."""
    left = table.drop(columns=[c for c in COUNT_COLUMNS if c in table]).copy()
    right = rel[KEYS + COUNT_COLUMNS].copy()
    left["_n"] = left.groupby(KEYS, dropna=False).cumcount()
    right["_n"] = right.groupby(KEYS, dropna=False).cumcount()
    return left.merge(right, on=KEYS + ["_n"], how="left").drop(columns="_n")


def elo_simulation(mtimes: tuple, stage: str, race_id, entries) -> dict:
    """The ELO race simulation (steps 3-4). Past race: step 4's cached table.
    Next race (race_id None): simulated now, on ``entries`` or the latest
    race's entry list."""
    data = race_data(mtimes)
    if race_id is None:
        ent = (None if entries is None
               else pd.DataFrame(list(entries), columns=["driverId", "constructorId"]))
        prepared = prepared_races(mtimes)
        t = predict_next_race(ent, df=data["df"], prepared=prepared, hist=data["hist"],
                              cfg=CFG, raw=data["raw"])
        latest_id, year, rnd, _, _ = prepared[-1]
        latest_name = data["df"].loc[data["df"]["raceId"] == latest_id, "name"].iloc[0]
        info = dict(t.attrs["race"], points_positions=t.attrs["points_positions"],
                    next=True, entries_from=f"{year} R{rnd} {latest_name}")
        return {"table": _standard(t, len(t)), "info": info}
    preds = all_predictions(mtimes, stage)
    t = calibration.race_table(preds, race_id)
    t = _with_counts(t, data["rel"][data["rel"]["raceId"] == race_id])
    t = add_names(t, data["raw"])
    races = data["raw"]["races"].set_index("raceId")
    first = t.iloc[0]
    info = {"raceId": int(race_id), "year": int(first["year"]), "round": int(first["round"]),
            "name": first["name"], "date": races["date"].get(race_id),
            "points_positions": int(first["points_positions"]), "next": False}
    return {"table": _standard(t, int(first["n"])), "info": info}


# Model sources: each returns {"table": the standard table, "info": the race}.
SOURCES = {"ELO simulation": elo_simulation}


@st.cache_data(max_entries=64, show_spinner="Simulating the race (10,000 times)…")
def load_predictions(mtimes: tuple, stage: str, race_id=None, entries=None,
                     source: str = "ELO simulation") -> dict:
    """One race's standard table and race info. ``race_id`` None = the next
    race (before qualifying only); ``entries``: a changed entry list for it,
    as a tuple of (driverId, constructorId)."""
    table = SOURCES[source](mtimes, stage, race_id, entries)
    table["info"].update(stage=stage, n_sims=N_SIMS, source=source)
    return table


# -- formatting ----------------------------------------------------------------
def pct(x) -> str:
    """A chance as a percentage, never a misleading 0.0% or 100.0%."""
    if x is None or pd.isna(x):
        return "–"
    if x < 0.0005:
        return "<0.1%"
    if x > 0.9995:
        return ">99.9%"
    return f"{np.round(100 * x, 1):.1f}%"      # rounded as predict_race.py does


def full_height(rows: int) -> int:
    """Table height in pixels that shows every row without scrolling."""
    return 35 * (rows + 1) + 3


def dedupe(names) -> list:
    """Unique labels (a 1950s driver could drive two cars in one race)."""
    seen: dict = {}
    out = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n} ({seen[n]})")
    return out


# -- Predictions: sidebar --------------------------------------------------------
def sidebar() -> dict:
    """Race and stage pickers. Returns the selection."""
    mtimes = data_mtimes()
    st.header("Race")
    mode = choice("Race", ["Next race", "Past race"], "pred:mode",
                  label_visibility="collapsed")
    race_id = None
    if mode == "Past race":
        races = past_races(mtimes)
        years = sorted(races["year"].unique(), reverse=True)
        if state.get("pred:year") not in years:
            state["pred:year"] = years[0]
        year = st.selectbox("Season", years, key="pred:year")
        in_year = races[races["year"] == year]
        label = dict(zip(in_year["raceId"], "R" + in_year["round"].astype(str)
                         + " · " + in_year["name"]))
        options = in_year["raceId"].tolist()
        if state.get("pred:race") not in options:
            state["pred:race"] = options[-1]          # the season's latest race
        race_id = st.selectbox("Round", options, format_func=label.get, key="pred:race")

    st.header("Stage")
    is_next = race_id is None
    stage_help = ("Before qualifying: the ratings going into the weekend. After "
                  "qualifying: the ratings after that weekend's qualifying pass, and "
                  "the grid slot in each car's retirement chance.")
    if is_next:
        # A fixed, disabled copy; the past-race choice is kept for later.
        st.segmented_control("Stage", list(STAGES), default=list(STAGES)[0], disabled=True,
                             key="pred:stage_next", label_visibility="collapsed",
                             help=stage_help)
        st.caption("The next race is predicted before qualifying only: the app has "
                   "no qualifying results for it yet.")
        stage = "pre"
    else:
        stage = STAGES[choice("Stage", list(STAGES), "pred:stage",
                              label_visibility="collapsed", help=stage_help)]
    return {"race_id": race_id, "stage": stage, "mtimes": mtimes}


# -- Predictions: page -------------------------------------------------------------
def _swap_seat(seats, driver_labels):
    seat, new = state.get("pred:seat"), state.get("pred:new_driver")
    if seat is None or new is None:
        return
    swaps = dict(state.get("pred:swaps", {}))
    current = [swaps.get(i, d) for i, (d, _) in enumerate(seats)]
    if new in current and current.index(new) != seat:
        state["pred:swap_msg"] = (f"{driver_labels.get(new, new)} already has a seat; "
                                  "change that seat first.")
        return
    if new == seats[seat][0]:
        swaps.pop(seat, None)
    else:
        swaps[seat] = new
    state["pred:swaps"] = swaps
    state["pred:new_driver"] = None
    state.pop("pred:swap_msg", None)


def _reset_seats():
    state["pred:swaps"] = {}
    state.pop("pred:swap_msg", None)


def entry_list_editor(mtimes, driver_labels, team_names) -> tuple | None:
    """The next race's seats, with any swaps. Returns the entries to pass to
    load_predictions (None = the latest race's list, unchanged)."""
    seats = latest_entries(mtimes)
    swaps = {s: d for s, d in state.get("pred:swaps", {}).items() if s < len(seats)}
    current = [(swaps.get(i, d), c) for i, (d, c) in enumerate(seats)]
    with st.popover(f"Change a seat{f' ({len(swaps)} changed)' if swaps else ''}"):
        st.caption("The field is copied from the latest race. Put a different driver "
                   "in a seat: they race with that car's rating and their own driver "
                   "rating and retirement record (a driver new to the data can't be "
                   "picked).")
        st.selectbox("Seat", range(len(seats)), key="pred:seat",
                     format_func=lambda i: f"{team_names.get(current[i][1], current[i][1])} · "
                                           f"{driver_labels.get(current[i][0], current[i][0])}")
        st.selectbox("Driver", list(driver_labels), format_func=driver_labels.get,
                     index=None, placeholder="Type a name…", key="pred:new_driver")
        left, right = st.columns(2)
        left.button("Put in this seat", on_click=_swap_seat, args=(seats, driver_labels),
                    disabled=state.get("pred:new_driver") is None, width="stretch")
        right.button("Reset all seats", on_click=_reset_seats, disabled=not swaps,
                     width="stretch")
        if "pred:swap_msg" in state:
            st.warning(state["pred:swap_msg"])
        if swaps:
            st.caption("Changed: " + "; ".join(
                f"{team_names.get(seats[s][1], seats[s][1])}: "
                f"{driver_labels.get(seats[s][0], seats[s][0])} → "
                f"{driver_labels.get(d, d)}" for s, d in sorted(swaps.items())))
    return tuple(current) if swaps else None


def main_table(t: pd.DataFrame, info: dict, past: bool) -> None:
    pts = info["points_positions"]
    points_col = f"Points (top {pts})"
    exp_col = "Exp. position (if classified)"
    show = pd.DataFrame({"Driver": t["driver"], "Team": t["team"], "Win": t["win"],
                         "Podium": t["podium"], points_col: t["points"], "DNF": t[DNF],
                         exp_col: t["exp_pos"]})
    if past:
        show["Actual"] = t["actual"]
    chances = ["Win", "Podium", points_col, "DNF"]
    styler = (show.style.format(pct, subset=chances)
              .format("{:.1f}", subset=[exp_col], na_rep="–"))
    if past:
        styler = styler.map(lambda a: "background-color: rgba(220, 20, 60, 0.14); "
                                      "font-weight: 600", subset=["Actual"])
    sims = f"{info['n_sims']:,}"
    help_ = {
        "Win": "The chance of winning, computed exactly for each simulated set of "
               "finishers (less noise than counting the simulated wins, which the "
               "heatmap's P1 shows).",
        "Podium": f"Share of the {sims} simulated races finishing in the top 3.",
        points_col: f"Share of the simulated races finishing in the top {pts}: this "
                    "season's scoring places (5 in the 1950s, 6 in 1960-2002, 8 in "
                    "2003-09, 10 since 2010).",
        "DNF": "Share of the simulated races not finishing. The chance that went in "
               "(P(DNF)) is under Inputs.",
        exp_col: "Average finishing position over the simulated races where the car "
                 "is classified. Retirements are left out, so a car that often retires "
                 "can still show a good number here.",
        "Actual": "What happened: the classified place (counted among the classified "
                  "cars) or DNF.",
    }
    st.dataframe(styler, hide_index=True, width="stretch", height=full_height(len(show)),
                 column_config={c: st.column_config.Column(help=h) for c, h in help_.items()})


def inputs_table(t: pd.DataFrame, stage: str) -> None:
    w = CFG.finish_car_weight
    show = pd.DataFrame({
        "Driver": t["driver"], "Team": t["team"],
        "D (driver)": t["driver_elo"].round(), "C (car)": t["constructor_elo"].round(),
        f"Strength (D + {w:g}·C)": t["strength"].round(), "P(DNF)": t["p_dnf"],
        "Team DNFs / entries": [f"{d:.0f} / {n:.0f}" for d, n in
                                zip(t["team_dnfs"], t["team_entries"])],
        "Driver DNFs / entries": [f"{d:.0f} / {n:.0f}" for d, n in
                                  zip(t["driver_dnfs"], t["driver_entries"])],
        "Era DNF rate": t["base_rate"],
    })
    if stage == "post_quali":
        show.insert(2, "Grid", t["grid"].map(lambda g: "pit lane" if g == 0 else f"P{g:.0f}"))
    styler = show.style.format(pct, subset=["P(DNF)", "Era DNF rate"]).format(
        "{:.0f}", subset=["D (driver)", "C (car)", f"Strength (D + {w:g}·C)"])
    col_help = {
        "D (driver)": st.column_config.Column(help="The driver's ELO rating going into "
                                                   "the race (after qualifying, at that stage)."),
        "C (car)": st.column_config.Column(help="The constructor's ELO rating at the same moment."),
        f"Strength (D + {w:g}·C)": st.column_config.Column(
            help="What the finishing order is drawn from (Plackett-Luce): a higher "
                 "strength finishes ahead more often."),
        "P(DNF)": st.column_config.Column(help="The chance of not finishing that went "
                                               "into the simulation."),
        "Team DNFs / entries": st.column_config.Column(
            help=f"Retirements over the team's last {CFG.dnf_team_races} races, all its cars."),
        "Driver DNFs / entries": st.column_config.Column(
            help=f"Retirements over the driver's last {CFG.dnf_driver_races} races."),
        "Era DNF rate": st.column_config.Column(
            help=f"Share of all entries that retired over the last {CFG.dnf_base_races} races."),
    }
    st.dataframe(styler, hide_index=True, width="stretch", height=full_height(len(show)),
                 column_config=col_help)


def heatmap(t: pd.DataFrame, past: bool) -> None:
    n = sum(c.startswith("P") and c[1:].isdigit() for c in t.columns)
    cols = [f"P{i + 1}" for i in range(n)] + [DNF]
    z = t[cols].to_numpy(float)
    names = dedupe(t["driver"])
    whole = np.floor(100 * z + 0.5).astype(int)         # rounded half up
    text = np.where(whole >= 1, whole.astype(str), "")
    hover = np.vectorize(pct)(z)
    fig = go.Figure(go.Heatmap(
        z=100 * z, x=cols, y=names, text=text, texttemplate="%{text}",
        customdata=hover, colorscale="Blues", zmin=0,
        hovertemplate="<b>%{y}</b> · %{x}: %{customdata}<extra></extra>",
        colorbar=dict(title="%", thickness=12), xgap=1, ygap=1))
    if past:
        for i, a in enumerate(t["actual"]):
            if a is None:
                continue
            j = n if a == DNF else int(a) - 1
            fig.add_shape(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
                          line=dict(color=HIGHLIGHT, width=2.5))
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(side="top", title=None)
    fig.update_layout(height=120 + 24 * len(t), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")


def driver_chart(row: pd.Series, n: int, past: bool) -> None:
    cols = [f"P{i + 1}" for i in range(n)] + [DNF]
    vals = row[cols].to_numpy(float)
    colors = [PALETTE[0]] * n + ["#8c8c8c"]
    line_w = [0] * (n + 1)
    actual = row["actual"] if past else None
    if actual is not None:
        j = n if actual == DNF else int(actual) - 1
        line_w[j] = 3
    fig = go.Figure(go.Bar(
        x=cols, y=100 * vals, marker=dict(color=colors, line=dict(color=HIGHLIGHT, width=line_w)),
        text=[pct(v) if v >= 0.01 else "" for v in vals], textposition="outside",
        textfont=dict(size=11), cliponaxis=False,
        customdata=[pct(v) for v in vals],
        hovertemplate="%{x}: %{customdata}<extra></extra>"))
    if actual is not None:
        fig.add_annotation(x=cols[j], y=100 * vals[j], text="actual", showarrow=True,
                           arrowhead=0, ay=-48, font=dict(color=HIGHLIGHT))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_title="chance (%)", xaxis_title=None, bargap=0.25,
                      xaxis=dict(type="category"))
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, width="stretch")


def dnf_story(row: pd.Series, stage: str) -> None:
    """Why this P(DNF): the counts it was built from, next to the era rate."""
    base = row["base_rate"]
    td, tn, dd, dn = (row[c] for c in ("team_dnfs", "team_entries",
                                       "driver_dnfs", "driver_entries"))
    team = (f"**{td:.0f} DNFs in the team's last {tn:.0f} entries** ({pct(td / tn)}; "
            f"its last {CFG.dnf_team_races} races, all its cars)" if tn
            else "no earlier entries for this team (the era rate is used)")
    driver = (f"**{dd:.0f} in the driver's last {dn:.0f} entries**, blended with "
              f"{CFG.dnf_driver_shrink:g} entries' worth of the era rate → "
              f"{pct(row['driver_rate'])}" if dn
              else "no earlier entries for this driver (the era rate is used)")
    wd = CFG.dnf_driver_weight
    last = ("then the ratings (a stronger car and driver than the field's average "
            "retire a little less)")
    if stage == "post_quali":
        grid = row["grid"]
        slot = "the pit lane" if grid == 0 else f"P{grid:.0f}"
        last += f" and the grid slot ({slot}; the front of the grid retires less)"
    st.markdown(
        f"**Why P(DNF) = {pct(row['p_dnf'])}**\n"
        f"- Era rate: **{pct(base)}** of all entries retired over the last "
        f"{CFG.dnf_base_races} races\n"
        f"- Team: {team}\n"
        f"- Driver: {driver}\n"
        f"- Blend: {1 - wd:g} × team + {wd:g} × driver = {pct(row['p_blend'])}, "
        f"{last} → **{pct(row['p_dnf'])}**")


def page(sel: dict, driver_labels: dict) -> None:
    mtimes, stage, race_id = sel["mtimes"], sel["stage"], sel["race_id"]
    entries = None
    team_names = (race_data(mtimes)["raw"]["constructors"]
                  .set_index("constructorId")["name"].to_dict())
    st.title("Race predictions")
    if race_id is None:
        entries = entry_list_editor(mtimes, driver_labels, team_names)
    try:
        res = load_predictions(mtimes, stage, race_id, entries)
    except Exception as exc:          # missing data, a cache that can't be built
        st.error(f"Couldn't build the predictions: {exc}")
        st.stop()
    t, info = res["table"], res["info"]
    past = not info["next"]

    when = f" · {info['date']}" if info.get("date") else ""
    name = info.get("name") or f"round {info['round']}"
    st.subheader(f"{info['year']} · R{info['round']} · {name}{when}"
                 + ("" if past else " (next race)"))
    caveats = [f"Stage: {STAGE_TEXT[stage]}",
               f"{info['n_sims']:,} simulated races",
               "no weather, sprints or grid penalties in the model"]
    if not past:
        caveats.append(f"entry list copied from the latest race ({info['entries_from']})"
                       + (", with the seats changed above" if entries else ""))
    st.caption(" · ".join(caveats) + ". Points % is the most trustworthy number, then "
               "podium %, then win % (see the Calibration view).")

    main_table(t, info, past)
    with st.expander("Inputs: ratings, strength and retirement chance"):
        inputs_table(t, stage)
        st.caption(f"Strength = D + {CFG.finish_car_weight:g}·C. The finishing order of "
                   "the cars that finish is drawn from the strengths; the first "
                   f"{CFG.finish_top_places} places on a sharper scale "
                   f"({CFG.finish_top_scale:g}) than the rest ({CFG.finish_scale:g}). "
                   "Retirements are drawn first, each car with its P(DNF) and a shared "
                   f"race factor ({CFG.dnf_race_corr:g}).")

    st.subheader("Every position")
    heatmap(t, past)
    st.caption("Chance of each finishing position (%), plus DNF; blank = under 0.5%. "
               "Hover for the exact share." + (" Outlined: what happened." if past else ""))

    st.subheader("One driver")
    ids = t["driverId"].tolist()
    if state.get("pred:driver") not in ids:
        state["pred:driver"] = ids[0]
    names = dict(zip(ids, t["driver"]))
    did = st.selectbox("Driver", ids, format_func=names.get, key="pred:driver")
    row = t.iloc[ids.index(did)]
    left, right = st.columns([3, 2], gap="large")
    with left:
        driver_chart(row, len(t), past)
    with right:
        st.markdown(f"**{row['driver']}** · {row['team']}: win {pct(row['win'])}, "
                    f"podium {pct(row['podium'])}, points {pct(row['points'])}, "
                    f"expected position if classified {row['exp_pos']:.1f}"
                    + (f"; actual **{row['actual']}**" if past else ""))
        dnf_story(row, stage)


# -- Calibration ---------------------------------------------------------------------
METRICS = {
    "win_ll": "Winner log loss (per race)",
    "podium_ll": "Podium log loss",
    "podium_brier": "Podium Brier score",
    "points_ll": "Points log loss",
    "points_brier": "Points Brier score",
    "rps": "Ranked probability score",
    "dnf_ll": "DNF log loss",
    "fin_ll": "Number of finishers, log loss (per race)",
    "fin_var_ratio": "Number of finishers, variance ratio (1 = right spread)",
}
OUTCOME_TITLES = {"win": "Win", "podium": "Podium", "points": "Points", "dnf": "DNF"}
CAL_KINDS = ("bands", "scores", "finishers")


def cal_path(kind: str, stage: str):
    return config.OUTPUT_DIR / f"calibration_{kind}_{stage}.csv"


@st.cache_data(max_entries=16, show_spinner=False)
def read_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def cal_data(stage: str) -> dict | None:
    paths = [cal_path(k, stage) for k in CAL_KINDS]
    if not all(p.exists() for p in paths):
        return None
    return {k: read_csv(str(p), p.stat().st_mtime) for k, p in zip(CAL_KINDS, paths)}


def cal_sidebar() -> dict:
    st.header("Backtest")
    stage = STAGES[choice("Stage", list(STAGES), "cal:stage")]
    data = cal_data(stage)
    if data is None:
        return {"stage": stage, "data": None}
    periods = list(dict.fromkeys(data["scores"]["period"]))
    period = choice("Period", periods, "cal:period",
                    help="1990-2022 is where the settings were tuned; 2023-2026 was "
                         "held out (fewer races, so wider intervals).")
    predictors = list(dict.fromkeys(data["bands"]["predictor"]))
    if state.get("cal:predictor") not in predictors:
        state["cal:predictor"] = predictors[0]
    predictor = st.selectbox("Predictor in the diagrams", predictors, key="cal:predictor",
                             help="The model, or one of the baselines it is compared with.")
    return {"stage": stage, "data": data, "period": period, "predictor": predictor}


def _band(bands, stage_period, outcome, lo, hi):
    b = bands[(bands["predictor"] == "model") & (bands["period"] == stage_period)
              & (bands["outcome"] == outcome) & np.isclose(bands["band_lo"], lo)
              & np.isclose(bands["band_hi"], hi)]
    return None if b.empty else b.iloc[0]


def trust_notes() -> None:
    """Plain-language notes, with the numbers read from the CSVs when present."""
    pre, post = cal_data("pre"), cal_data("post_quali")
    tune = "1990-2022 (tune)"
    podium = "Podium favourites above 80% are overconfident"
    longshot = "Win long shots under 2% are too generous"
    if pre is not None:
        b = _band(pre["bands"], tune, "podium", 0.8, 1.0)
        if b is not None:
            podium += (f": {b['predicted']:.2f} predicted vs {b['actual']:.2f} actual "
                       f"({b['entries']:,} cases, 1990-2022)")
        b = _band(pre["bands"], tune, "win", 0.0, 0.02)
        if b is not None:
            longshot += (f": {pct(b['predicted'])} predicted vs {pct(b['actual'])} "
                         f"actual (1990-2022)")
    grid = ("After qualifying, the grid-slot baseline predicts the winner better than "
            "the model")
    if post is not None:
        s = post["scores"]
        s = s[(s["metric"] == "win_ll") & (s["period"] == tune)].set_index("predictor")
        if {"model", "grid slot"} <= set(s.index):
            g = s.loc["grid slot"]
            grid += (f" (winner log loss {g['value']:.3f} vs {s.loc['model', 'value']:.3f}, "
                     f"1990-2022; difference interval {g['diff_lo']:+.3f} to "
                     f"{g['diff_hi']:+.3f})")
    grid += ("; the model is better on points and the whole order. Step 6 (a model "
             "that uses the grid) is meant to address this.")
    st.markdown(
        "**How far to trust the numbers**\n"
        "- **Points % is the most reliable, then podium %, then win %.**\n"
        f"- {podium}.\n"
        f"- {longshot}.\n"
        "- Retirements are under-predicted in the first races of a new rules era "
        "(first 7 races: 25.5% retired vs 20.3% predicted over 1998-2026; 23% vs 13% "
        "in 2026).\n"
        f"- {grid}")


def reliability_diagram(bands: pd.DataFrame, outcome: str) -> None:
    b = bands[bands["outcome"] == outcome]
    groups = list(dict.fromkeys(b["group"]))
    fig = go.Figure()
    top = float(max(b["predicted"].max(), b["actual"].max(), 0.05))
    top = min(1.0, top * 1.08)
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="perfect",
                             line=dict(color="#9e9e9e", dash="dash", width=1),
                             hoverinfo="skip", showlegend=False))
    for i, g in enumerate(groups):
        d = b[b["group"] == g]
        hover = [f"band {lo:.0%}–{hi:.0%}<br>{n:,} cases<br>predicted {pct(p)}"
                 f"<br>actual {pct(a)}<br>z = {z:+.1f}"
                 for lo, hi, n, p, a, z in zip(d["band_lo"], d["band_hi"], d["entries"],
                                               d["predicted"], d["actual"], d["z"])]
        fig.add_trace(go.Scatter(
            x=d["predicted"], y=d["actual"], mode="lines+markers",
            name=g if g != "all" else OUTCOME_TITLES[outcome],
            showlegend=len(groups) > 1, line=dict(color=PALETTE[i], width=2),
            marker=dict(size=8), text=hover, hovertemplate="%{text}<extra></extra>",
            error_y=dict(type="data", array=2 * d["se"], thickness=1, width=0)))
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10),
                      title=dict(text=OUTCOME_TITLES[outcome], x=0.02, font=dict(size=15)),
                      xaxis_title="predicted", yaxis_title="actual",
                      legend=dict(orientation="h", y=-0.25))
    fig.update_xaxes(range=[0, top], tickformat=".0%")
    fig.update_yaxes(range=[0, top], tickformat=".0%", scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, width="stretch")


def scores_tables(scores: pd.DataFrame) -> None:
    s = scores.copy()
    s["label"] = s["metric"].map(METRICS)
    order = [m for m in METRICS if m in set(s["metric"])]
    preds = list(dict.fromkeys(s["predictor"]))
    s["cell"] = [f"{v:.4f} ({lo:.4f}–{hi:.4f})" if pd.notna(lo) else f"{v:.4f}"
                 for v, lo, hi in zip(s["value"], s["ci_lo"], s["ci_hi"])]
    wide = (s.pivot_table(index="metric", columns="predictor", values="cell", aggfunc="first")
             .reindex(index=order, columns=preds))
    wide.index = wide.index.map(METRICS)
    st.dataframe(wide.rename_axis("Score").fillna("–"), width="stretch")
    st.caption("Lower is better (except the variance ratio: 1 is right). In brackets: "
               "95% interval from resampling races.")

    d = s[(s["predictor"] != "model") & s["diff"].notna() & (s["metric"] != "fin_var_ratio")]
    if d.empty:
        return
    d = d.assign(
        cell=[f"{x:+.4f} ({lo:+.4f} to {hi:+.4f})"
              for x, lo, hi in zip(d["diff"], d["diff_lo"], d["diff_hi"])],
        verdict=np.where(d["diff_lo"] > 0, "model",
                         np.where(d["diff_hi"] < 0, "baseline", "")))
    rows = [m for m in order if m in set(d["metric"])]
    cols = [p for p in preds if p in set(d["predictor"])]
    cells = d.pivot_table(index="metric", columns="predictor", values="cell",
                          aggfunc="first").reindex(index=rows, columns=cols).fillna("–")
    verdict = d.pivot_table(index="metric", columns="predictor", values="verdict",
                            aggfunc="first").reindex(index=rows, columns=cols)
    colour = {"model": "background-color: rgba(46, 145, 229, 0.18)",
              "baseline": "background-color: rgba(220, 20, 60, 0.18)"}
    cells.index = cells.index.map(METRICS)
    verdict.index = cells.index
    styler = cells.rename_axis("Score").style.apply(
        lambda _: verdict.map(lambda v: colour.get(v, "")), axis=None)
    st.dataframe(styler, width="stretch")
    st.caption("Each baseline's score minus the model's, on the same races, with its 95% "
               "interval: positive = the model is better. Blue: the model is clearly "
               "better (the whole interval above 0); red: the baseline is clearly better; "
               "no colour: no clear difference.")


def finishers_chart(fin: pd.DataFrame, scores: pd.DataFrame) -> None:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=fin["retirements"], y=fin["actual_races"], name="actual races",
                         marker_color=PALETTE[0],
                         hovertemplate="%{x} retirements: %{y} races<extra></extra>"))
    fig.add_trace(go.Scatter(x=fin["retirements"], y=fin["predicted_races"],
                             name="predicted races", mode="lines+markers",
                             line=dict(color=PALETTE[1], width=2), marker=dict(size=8),
                             hovertemplate="%{x} retirements: %{y:.1f} predicted<extra></extra>"))
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="retirements in the race", yaxis_title="races",
                      legend=dict(orientation="h", y=1.1), bargap=0.2)
    st.plotly_chart(fig, width="stretch")
    v = scores[(scores["metric"] == "fin_var_ratio") & (scores["predictor"] == "model")]
    if len(v):
        st.caption(f"How many cars retire per race, real races vs the model's prediction "
                   f"(summed over races). Variance ratio {v['value'].iloc[0]:.2f} "
                   "(1 = the model's spread of finishers matches real races).")


def cal_page(sel: dict) -> None:
    st.title("How far to trust the predictions")
    trust_notes()
    stage, data = sel["stage"], sel["data"]
    if data is None:
        flag = "" if stage == "pre" else " --stage post_quali"
        st.info("The calibration results aren't in `output/` yet. Create them from the "
                "f1_elo_app folder (a few minutes; add `--experiments none` to skip the "
                "experiment grids):")
        st.code(f"py backtest_calibration.py{flag}", language="powershell")
        return
    period, predictor = sel["period"], sel["predictor"]
    st.caption(f"Stage: {STAGE_TEXT[stage]} · period {period}. Every race scored from "
               "information available before it (Indy 500 excluded).")

    st.subheader("Predicted vs actual")
    bands = data["bands"]
    bands = bands[(bands["period"] == period) & (bands["predictor"] == predictor)]
    outcomes = [o for o in OUTCOME_TITLES if o in set(bands["outcome"])]
    for i in range(0, len(outcomes), 2):
        cols = st.columns(2)
        for col, outcome in zip(cols, outcomes[i:i + 2]):
            with col:
                reliability_diagram(bands, outcome)
    st.caption(f"{predictor}: cars grouped into bands by their predicted chance; each "
               "point is a band's average prediction against how often it happened "
               "(bars: ±2 standard errors). On the dashed line = right on average. "
               "Points is split by the season's scoring places (top 6 / 8 / 10).")

    st.subheader("Model vs baselines")
    sc = data["scores"]
    scores_tables(sc[sc["period"] == period])

    st.subheader("Number of finishers")
    fin = data["finishers"]
    finishers_chart(fin[fin["period"] == period], sc[sc["period"] == period])
