"""F1 ELO viewer: driver and constructor ratings over time.

Run from the f1_elo_app folder (after `py run_elo.py`):

    py -m streamlit run app.py
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.express as px
import streamlit as st

from f1elo import config, expected_score, updater

RESULTS = config.OUTPUT_DIR / "results_with_elo.csv"
PALETTE = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24

# Everything that differs between the two views.
VIEWS = {
    "drivers": {"title": "F1 driver ELO", "one": "driver", "many": "drivers",
                "column": "Driver", "id": "driverId", "rating": "DriverEloRating",
                "names": config.DATA_DIR / "drivers.csv"},
    "constructors": {"title": "F1 constructor ELO", "one": "team", "many": "teams",
                     "column": "Constructor", "id": "constructorId", "rating": "EloRating",
                     "names": config.DATA_DIR / "constructors.csv"},
}
STATE_KEYS = ("sel", "seasons", "from", "to", "columns",
              "min_on", "min_races", "top_on", "top_n")
# Filter defaults: both off; the numbers are what each box starts at.
FILTER_DEFAULTS = {"min_on": False, "min_races": 10, "top_on": False, "top_n": 20}

st.set_page_config(page_title="F1 ELO", layout="wide")


# -- data -------------------------------------------------------------------
def _with_years(names: pd.DataFrame, frame: pd.DataFrame, name_col: pd.Series) -> pd.DataFrame:
    span = frame.groupby("id")["year"].agg(first="min", last="max")
    names = names.join(span, on="id", how="inner")
    years = names["first"].astype(str).where(
        names["first"] == names["last"],
        names["first"].astype(str) + "–" + names["last"].astype(str))
    names["label"] = name_col.loc[names.index] + " (" + years + ")"
    return names


@st.cache_data
def load(mtimes: tuple):
    """Ratings for both views plus the race calendar. The mtimes refresh the cache."""
    res = pd.read_csv(RESULTS, low_memory=False, parse_dates=["date"],
                      usecols=["raceId", "driverId", "constructorId", "date", "year",
                               "round", "name", "DriverEloRating", "EloRating"])
    res = res.dropna(subset=["date"]).sort_values(["date", "round"])

    races = (res.drop_duplicates("raceId")[["raceId", "date", "year", "round", "name"]]
                .reset_index(drop=True))
    races["pos"] = races.index  # chronological position
    races["label"] = (races["year"].astype(str) + " · R" + races["round"].astype(str)
                      + " · " + races["name"])
    races = races.set_index("raceId")

    # The grid's typical team in each race: the median team rating after the
    # race, each team counted once (a 15-car make is still one team). Median
    # rather than mean: ratings are right-skewed (a dominant team or two sits
    # far above the rest), which pulls the mean up by 8-38 points on average
    # and makes the dominant car itself look less dominant. Being above the
    # median means being expected to beat more than half the teams.
    # The same for drivers: the median driver rating after each race, each
    # driver counted once. grid_medians[view] is the view's own grid.
    grid_medians = {
        view: (res.drop_duplicates(["raceId", cfg["id"]])
                  .groupby("raceId")[cfg["rating"]].median())
        for view, cfg in VIEWS.items()
    }
    team_median = grid_medians["constructors"]

    out = {}
    for view, cfg in VIEWS.items():
        # One row per driver (or team) per race: teammates share the team rating,
        # and 1950s shared drives duplicate drivers.
        frame = res.drop_duplicates(["raceId", cfg["id"]])
        frame = frame.assign(grid_median=frame["raceId"].map(grid_medians[view]))
        cols = ["raceId", "id", "date", "year", "round", "name", "rating", "grid_median"]
        if view == "drivers":
            # The team the driver raced for in each race, its rating then, and
            # the grid's median team rating in that race.
            frame = frame.assign(team=frame["constructorId"], team_rating=frame["EloRating"],
                                 team_grid_median=frame["raceId"].map(team_median))
            cols += ["team", "team_rating", "team_grid_median"]
        frame = frame.rename(columns={cfg["id"]: "id", cfg["rating"]: "rating"})[cols]
        frame["pos"] = frame["raceId"].map(races["pos"])
        names = pd.read_csv(cfg["names"]).rename(columns={cfg["id"]: "id"})
        if view == "drivers":
            names = _with_years(names, frame, names["forename"] + " " + names["surname"])
            names = names.sort_values(["surname", "forename"])
        else:
            names = _with_years(names, frame, names["name"]).sort_values("name")
        out[view] = (frame, dict(zip(names["id"], names["label"])))
    team_names = (pd.read_csv(VIEWS["constructors"]["names"])
                    .set_index("constructorId")["name"].to_dict())
    latest_medians = {view: m.loc[races.index[-1]] for view, m in grid_medians.items()}
    return out, races, team_names, latest_medians


# Table columns that can be shown or hidden, in display order. Only
# DEFAULT_COLUMNS start shown (and come back with "Default columns"); the
# grid comparisons and the team ones (drivers view only) start hidden.
BASE_COLUMNS = ["Start", "End", "Grid median at end", "End vs grid", "End vs grid %",
                "End pos score", "Change", "Peak", "Grid median at peak", "Peak vs grid",
                "Peak vs grid %", "Peak pos score", "Peak at", "Races"]

# Positional scores: the average of a row's positions (1 = highest) by three
# columns, within the list as shown (after the filters), so they change with
# the list's boundary. Lower = better. The first column is the rating the
# top-N cut uses when ranking by the score.
POS_SCORES = {
    "End pos score": ("End", "Grid median at end", "End vs grid"),
    "Peak pos score": ("Peak", "Grid median at peak", "Peak vs grid"),
    "Peak team pos score": ("Peak team rating", "Team grid median at peak",
                            "Peak team vs grid"),
}
POS_HELP = ("The average of this row's positions in the list shown (1 = highest) "
            "by {a}, {b} and {c}. Lower = better; 1.0 = first in all three. It's "
            "relative to the list, so it changes with the filters.")
RANK_OPTIONS = ["End", "Peak", "Change"]
DEFAULT_COLUMNS = ["Start", "End", "Change", "Peak", "Peak at", "Races"]
BASE_HELP = {
    "Grid median at end": "The median rating of the grid after the race where End "
                          "was taken (the row's last race in the period), each "
                          "{one} counted once.",
    "End vs grid": "End minus Grid median at end: how far above (+) or below (-) "
                   "the typical {one} on the grid at that moment.",
    "Grid median at peak": "The median rating of the grid after the race where the "
                           "peak was reached, each {one} counted once.",
    "Peak vs grid": "Peak minus Grid median at peak: how far above the typical "
                    "{one} they stood at their best.",
}
# The "%" columns turn a gap to the grid median into the model's own win
# chance: expected_score on the engine's 400 scale. For drivers the cars
# cancel (same car), so it reads as head-to-head driver skill.
WIN_HELP = {
    "drivers": "The chance of beating the grid's median driver in the same car, "
               "from the rating gap {when} (the model's own win chance: 50% = "
               "typical, 76% = wins about 3 in 4, 90% = 9 in 10).",
    "constructors": "The chance of beating the grid's median team, from the rating "
                    "gap {when} (the model's own win chance: 50% = typical, 76% = "
                    "wins about 3 in 4, 90% = 9 in 10).",
}
WIN_WHEN = {"End vs grid %": "at the end", "Peak vs grid %": "at the peak"}


def win_chance(gap: float) -> int:
    """Chance, in whole percent, of beating someone rated `gap` points lower."""
    return round(100 * expected_score(gap, 0.0, config.DEFAULT_CONFIG.elo_scale))


TEAM_COLUMNS = ["Current team", "Current team: peak", "Current team: end",
                "Current team: change", "Peak team", "Peak team rating",
                "Team grid median at peak", "Peak team vs grid", "Peak team vs grid %",
                "Peak team pos score"]
TEAM_HELP = {
    "Current team": "The team in the driver's last race of the period.",
    "Current team: peak": "That team's highest rating during the driver's current "
                          "stint with it (within the period).",
    "Current team: end": "That team's rating after the driver's last race of the period.",
    "Current team: change": "How much that team's rating moved during the driver's "
                            "current stint (within the period).",
    "Peak team": "The team the driver was racing for when they reached their peak.",
    "Peak team rating": "That team's rating at the moment the driver peaked "
                        "(after the same race).",
    "Team grid median at peak": "The median team rating of the grid after the race "
                                "where the driver peaked, each team counted once.",
    "Peak team vs grid": "Peak team rating minus Team grid median at peak: how far "
                         "above (+) or below (-) the typical team that car was. "
                         "Positive = a better car than the typical one.",
    "Peak team vs grid %": "The chance of that car beating the grid's median car "
                           "at the driver's peak, from the rating gap (the model's "
                           "own win chance: 50% = typical, 90% = 9 in 10).",
}


def _stint_figures(stint: pd.DataFrame) -> tuple:
    """Team rating peak, end and change over one stint (rows in race order)."""
    r = stint["team_rating"]
    return round(r.max()), round(r.iloc[-1]), round(r.iloc[-1] - r.iloc[0])


def summary(view: pd.DataFrame, labels: dict, column: str,
            team_names: dict | None = None) -> pd.DataFrame:
    rows = []
    for eid, g in view.groupby("id"):
        peak = g.loc[g["rating"].idxmax()]
        start, end = g["rating"].iloc[0], g["rating"].iloc[-1]
        grid = g["grid_median"].iloc[-1]  # the grid when End was taken
        peak_gap = peak["rating"] - peak["grid_median"]
        row = {column: labels[eid], "Start": round(start), "End": round(end),
               "Grid median at end": round(grid), "End vs grid": round(end - grid),
               "End vs grid %": win_chance(end - grid),
               "Change": round(end - start), "Peak": round(peak["rating"]),
               "Grid median at peak": round(peak["grid_median"]),
               "Peak vs grid": round(peak_gap), "Peak vs grid %": win_chance(peak_gap),
               "Peak at": f"{peak['name']} {peak['year']}", "Races": len(g)}
        if "team" in g.columns:
            # Stints: runs of consecutive races (in this period) with one team.
            g = g.sort_values("pos")
            stint = (g["team"] != g["team"].shift()).cumsum()
            current = g[stint == stint.iloc[-1]]
            top, last, change = _stint_figures(current)
            row["Current team"] = team_names.get(current["team"].iloc[0],
                                                 current["team"].iloc[0])
            row["Current team: peak"] = top
            row["Current team: end"] = last
            row["Current team: change"] = change
            # The team at the driver's peak race, and its rating after that race.
            row["Peak team"] = team_names.get(peak["team"], peak["team"])
            row["Peak team rating"] = round(peak["team_rating"])
            team_gap = peak["team_rating"] - peak["team_grid_median"]
            row["Team grid median at peak"] = round(peak["team_grid_median"])
            row["Peak team vs grid"] = round(team_gap)
            row["Peak team vs grid %"] = win_chance(team_gap)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("End", ascending=False)


# -- setup ------------------------------------------------------------------
missing = [p.name for p in [RESULTS, *(c["names"] for c in VIEWS.values())] if not p.exists()]
if missing:
    st.error(f"Missing {', '.join(missing)}. Run `py run_elo.py` in this folder first.")
    st.stop()

data, races, team_names, latest_medians = load(tuple(p.stat().st_mtime for p in
                         [RESULTS, *(c["names"] for c in VIEWS.values())]))
all_seasons = sorted(races["year"].unique(), reverse=True)
state = st.session_state

# Streamlit forgets a widget's value when the widget isn't drawn, which would wipe
# the hidden view's selection. Re-saving the values each run keeps both views.
for _key in [f"{v}:{k}" for v in VIEWS for k in STATE_KEYS] + ["line_style", "rank_by"]:
    if _key in state:
        state[_key] = state[_key]


# -- period helpers ---------------------------------------------------------
def race_options(v) -> list:
    """Races offered in the From/To pickers: the picked seasons, or all of them."""
    seasons = state[f"{v}:seasons"]
    pool = races[races["year"].isin(seasons)] if seasons else races
    return pool.index.tolist()


def span_races(v, ids) -> tuple:
    """First and last race of the selection (whole history if nothing selected)."""
    frame = data[v][0]
    pos = frame.loc[frame["id"].isin(ids), "pos"]
    lo, hi = (pos.min(), pos.max()) if len(pos) else (0, len(races) - 1)
    return races.index[lo], races.index[hi]


def visible_races(v) -> pd.DataFrame:
    """Races between From and To, limited to the picked seasons if any."""
    lo, hi = races.loc[state[f"{v}:from"], "pos"], races.loc[state[f"{v}:to"], "pos"]
    out = races[(races["pos"] >= lo) & (races["pos"] <= hi)]
    seasons = state[f"{v}:seasons"]
    return out[out["year"].isin(seasons)] if seasons else out


def set_period(v, first, last):
    state[f"{v}:from"], state[f"{v}:to"] = first, last


# -- callbacks: each control decides what happens to the period --------------
def on_selection_change(v):
    if not state[f"{v}:seasons"]:
        set_period(v, *span_races(v, state[f"{v}:sel"]))
    # Picked seasons stay put while the selection changes.


def on_seasons_change(v):
    seasons = state[f"{v}:seasons"]
    if seasons:
        pool = races[races["year"].isin(seasons)]
        set_period(v, pool.index[0], pool.index[-1])
    else:
        set_period(v, *span_races(v, state[f"{v}:sel"]))


def on_race_change(v):
    first, last = state[f"{v}:from"], state[f"{v}:to"]
    if races.loc[first, "pos"] > races.loc[last, "pos"]:
        set_period(v, last, first)


def add_from_period(v):
    frame = data[v][0]
    ids = frame.loc[frame["raceId"].isin(visible_races(v).index), "id"].unique().tolist()
    sel = state[f"{v}:sel"]
    state[f"{v}:sel"] = sel + [e for e in ids if e not in sel]


def clear_selection(v):
    state[f"{v}:sel"] = []


def reset_period(v):
    state[f"{v}:seasons"] = []
    set_period(v, *span_races(v, state[f"{v}:sel"]))


# First visit: open each view on the latest season's top 3, over their full span.
for v in VIEWS:
    if f"{v}:sel" not in state:
        frame = data[v][0]
        latest = frame[frame["year"] == frame["year"].max()]
        state[f"{v}:sel"] = latest.groupby("id")["rating"].last().nlargest(3).index.tolist()
        state[f"{v}:seasons"] = []
        set_period(v, *span_races(v, state[f"{v}:sel"]))

# After an update, stretch each view's period to take in any new races.
if state.pop("refresh_period", False):
    for v in VIEWS:
        on_seasons_change(v)

# -- sidebar -----------------------------------------------------------------
with st.sidebar:
    v = st.radio("View", list(VIEWS), format_func=str.capitalize, horizontal=True,
                 key="view")
    cfg = VIEWS[v]
    frame, labels = data[v]

    st.header(cfg["many"].capitalize())
    st.multiselect("Choose one or more", options=list(labels), format_func=labels.get,
                   placeholder="Type a name…", key=f"{v}:sel",
                   on_change=on_selection_change, args=(v,))

    st.header("Period")
    st.multiselect("Seasons", options=all_seasons, key=f"{v}:seasons",
                   placeholder="All seasons", on_change=on_seasons_change, args=(v,))
    options = race_options(v)
    race_label = races["label"].to_dict()
    st.selectbox("From race", options=options, format_func=race_label.get,
                 key=f"{v}:from", on_change=on_race_change, args=(v,))
    st.selectbox("To race", options=options, format_func=race_label.get,
                 key=f"{v}:to", on_change=on_race_change, args=(v,))

    shown_races = visible_races(v)
    sel = state[f"{v}:sel"]
    in_period = frame.loc[frame["raceId"].isin(shown_races.index), "id"].unique()
    to_add = len(set(in_period) - set(sel))
    add_label = (f"All {cfg['many']} in this period are selected" if not to_add else
                 f"Add 1 more {cfg['one']} from this period" if to_add == 1 else
                 f"Add all {to_add} {cfg['many']} from this period")
    st.button(add_label, on_click=add_from_period, args=(v,), disabled=not to_add,
              key=f"{v}:add_all", width="stretch")
    left, right = st.columns(2)
    left.button("Reset period", on_click=reset_period, args=(v,), key=f"{v}:reset",
                width="stretch")
    right.button(f"Clear {cfg['many']}", on_click=clear_selection, args=(v,),
                 key=f"{v}:clear", width="stretch", disabled=not sel)
    st.caption(f"{len(shown_races)} races in this period.")

    # Filters on the chart and the table, both off by default.
    st.header("Filters")
    for k, default in FILTER_DEFAULTS.items():
        state.setdefault(f"{v}:{k}", default)
    st.toggle("Minimum races", key=f"{v}:min_on",
              help=f"Hide {cfg['many']} with fewer races than this in the selected "
                   "period (the table's Races column).")
    st.number_input("At least this many races in the period", min_value=1, step=1,
                    key=f"{v}:min_races", disabled=not state[f"{v}:min_on"])
    st.toggle("Top positions only", key=f"{v}:top_on",
              help="Keep only the top positions by the 'Rank by' choice above the "
                   "table (End, Peak or Change). Ranking by a positional score cuts "
                   "by its rating (Peak pos score: the top N by Peak), then orders "
                   "that list by the score. Equal values share a position, so a "
                   "tie at the cut-off can show a few more.")
    st.number_input("Show the top", min_value=1, step=1, key=f"{v}:top_n",
                    disabled=not state[f"{v}:top_on"])

    st.header("Data")
    st.caption(f"Latest race: {races['label'].iloc[-1]}")
    if st.button("Check for new races", key="update", width="stretch",
                 help="Fetches this season from the Jolpica API, then re-exports "
                      "and recomputes all ratings if anything changed."):
        with st.status("Updating…", expanded=True) as status:
            try:
                report = updater.update(progress=status.write)
            except updater.UpdateError as exc:
                state["update_error"] = str(exc)
                state.pop("update_report", None)
            else:
                state["update_report"] = report
                state.pop("update_error", None)
                state["refresh_period"] = True
        st.rerun()


def dismiss_update():
    state.pop("update_report", None)
    state.pop("update_error", None)


# -- main area ---------------------------------------------------------------
st.title(cfg["title"])
st.caption(f"Latest race: {races['label'].iloc[-1]} · grid median after it: "
           f"**{latest_medians['drivers']:.0f}** for drivers, "
           f"**{latest_medians['constructors']:.0f}** for teams "
           "(each driver or team counted once).")

if "update_error" in state:
    st.error(f"**Update failed.** {state['update_error']}")
    st.button("Dismiss", on_click=dismiss_update, key="dismiss_error")
elif "update_report" in state:
    report = state["update_report"]
    lines = "\n".join(f"- {line}" for line in updater.describe(report))
    footer = (f"{report['requests']} API requests · {report['seconds']}s"
              + (" · ratings recomputed" if report["ratings_recomputed"] else ""))
    box = st.success if report["new"] or report["corrected"] else st.info
    box(f"**Data check for {report['seasons']}**\n\n{lines}\n\n{footer}")
    st.button("Dismiss", on_click=dismiss_update, key="dismiss_report")
if not sel:
    st.info(f"Pick {cfg['many']} in the sidebar, or choose seasons and press "
            f"“Add all {cfg['many']} from this period”.")
    st.stop()

view = frame[frame["id"].isin(sel) & frame["raceId"].isin(shown_races.index)].copy()
if view.empty:
    st.warning(f"None of the selected {cfg['many']} raced in this period.")
    st.stop()
view["label"] = view["id"].map(labels)

# Columns this view offers, and the ones picked (drawn below the chart).
optional = BASE_COLUMNS + (TEAM_COLUMNS if v == "drivers" else [])
state.setdefault(f"{v}:columns", list(DEFAULT_COLUMNS))
# Drop column names that no longer exist (a session from an older version).
state[f"{v}:columns"] = [c for c in state[f"{v}:columns"] if c in optional]

# Rank by offers a positional score only while its column is shown; if the
# chosen one was hidden, fall back to End.
rank_options = RANK_OPTIONS + [c for c in POS_SCORES if c in state[f"{v}:columns"]]
state.setdefault("rank_by", "End")
if state["rank_by"] not in rank_options:
    state["rank_by"] = state["_rank_by"] = "End"
rank_by = state["rank_by"]

# The table, ranked and filtered (the Rank by control is drawn under the
# chart, but its value is known now). The chart shows the same rows.
table = summary(view, labels, cfg["column"], team_names)
in_period = len(table)
if state[f"{v}:min_on"]:
    table = table[table["Races"] >= state[f"{v}:min_races"]]
if state[f"{v}:top_on"]:
    # The list's boundary: the top N by a rating. Ranking by a positional
    # score cuts by its rating (Peak pos score -> Peak), then reorders.
    cut_by = POS_SCORES[rank_by][0] if rank_by in POS_SCORES else rank_by
    table = table[table[cut_by].rank(method="min", ascending=False) <= state[f"{v}:top_n"]]
# Positional scores, relative to the list as it now stands.
for score, cols in POS_SCORES.items():
    if all(c in table.columns for c in cols):
        table[score] = sum(table[c].rank(method="min", ascending=False)
                           for c in cols) / len(cols)
low_first = rank_by in POS_SCORES  # a lower score is better
table = table.sort_values(rank_by, ascending=low_first)
table.insert(0, "#", table[rank_by].rank(method="min", ascending=low_first).astype(int))
filtered_out = in_period - len(table)
if table.empty:
    st.warning(f"The filters leave no {cfg['many']} to show. Lower the minimum "
               "races or turn the filters off in the sidebar.")
    st.stop()
view = view[view["label"].isin(table[cfg["column"]])]

# Legend in leaderboard order: highest rating at the end of the period first.
order = view.groupby("label")["rating"].last().sort_values(ascending=False).index.tolist()

def keep_choice(key):
    # Clicking the active option would deselect it; keep the last choice instead.
    if state[key] is None:
        state[key] = state[f"_{key}"]
    state[f"_{key}"] = state[key]


def choice(label, options, key, **kwargs) -> str:
    """A segmented control that always has exactly one option picked."""
    state.setdefault(key, options[0])
    state.setdefault(f"_{key}", state[key])
    st.segmented_control(label, options, key=key, on_change=keep_choice, args=(key,),
                         **kwargs)
    return state[key]


smooth = choice("Line style", ["Steps", "Smooth"], "line_style",
                label_visibility="collapsed",
                help="Steps hold each rating until the next race; "
                     "Smooth curves through the post-race ratings.") == "Smooth"

fig = px.line(view, x="date", y="rating", color="label",
              line_shape="spline" if smooth else "hv",
              custom_data=["name", "year", "round"], category_orders={"label": order},
              color_discrete_sequence=PALETTE)
fig.update_traces(hovertemplate="<b>%{fullData.name}</b><br>%{customdata[1]} R%{customdata[2]}"
                                " · %{customdata[0]}<br>ELO %{y:.0f}<extra></extra>")
if smooth:
    # Gentler than Plotly's default (1.0) so curves overshoot the real ratings less.
    fig.update_traces(line_smoothing=0.6)
fig.update_layout(height=560, hovermode="closest", legend_title_text="",
                  xaxis_title=None, yaxis_title="ELO rating",
                  margin=dict(l=10, r=10, t=10, b=10))

# Cut out the years between non-consecutive picked seasons.
day = dt.timedelta(days=1)
dates, years = shown_races["date"].tolist(), shown_races["year"].tolist()
breaks = [dict(bounds=[dates[i] + day, dates[i + 1] - day])
          for i in range(len(dates) - 1) if years[i + 1] - years[i] > 1
          and state[f"{v}:seasons"]]
pad = dt.timedelta(days=7)
fig.update_xaxes(range=[dates[0] - pad, dates[-1] + pad], rangebreaks=breaks)
st.plotly_chart(fig, width="stretch")

shown = len(table)
notes = []
if in_period < len(sel):
    notes.append(f"{len(sel) - in_period} selected {cfg['many']} didn't race in this period")
if filtered_out:
    notes.append(f"{filtered_out} hidden by the filters")
if breaks:
    notes.append("years between the picked seasons are cut out")
st.caption(f"Showing {shown} {cfg['many']}" + (f" ({'; '.join(notes)})" if notes else "")
           + ". Ratings are recorded after each race. Drag across the chart to zoom, "
           "double-click to zoom back out.")

# Columns sits next to Rank by, on the left. A horizontal container sizes
# both to their content and wraps Columns onto the next line when space
# runs out, so they never overlap (fixed-ratio columns squeezed it on
# narrow pages).
with st.container(horizontal=True, vertical_alignment="bottom", gap="medium"):
    rank_by = choice("Rank by", rank_options, "rank_by",
                     help="Orders the table and numbers each row by this column. "
                          "Equal values share a position. A positional score "
                          "appears here while its column is shown; it ranks "
                          "lowest first.")
    with st.popover("Columns", width="content"):
        st.pills("Shown columns", optional, selection_mode="multi", key=f"{v}:columns",
                 help="Pick the columns to show. The rank and name are always shown.")
        st.button("Default columns", key=f"{v}:columns_default", width="stretch",
                  on_click=lambda key=f"{v}:columns": state.update({key: list(DEFAULT_COLUMNS)}))
# The table's hover toolbar floats just above its top edge; keep it clear of
# the controls when they span the full width (phones).
st.space("small")

shown_cols = [c for c in optional if c in state[f"{v}:columns"]]
col_config = {"#": st.column_config.NumberColumn(width="small", pinned=True)}
col_config.update({c: st.column_config.Column(help=h) for c, h in TEAM_HELP.items()})
for c in ("Grid median at end", "Grid median at peak"):
    col_config[c] = st.column_config.NumberColumn(help=BASE_HELP[c].format(one=cfg["one"]))
# Above or below the grid: show the sign (+191 / -42).
for c in ("End vs grid", "Peak vs grid"):
    col_config[c] = st.column_config.NumberColumn(
        help=BASE_HELP[c].format(one=cfg["one"]), format="%+d")
col_config["Peak team vs grid"] = st.column_config.NumberColumn(
    help=TEAM_HELP["Peak team vs grid"], format="%+d")
# Win chances, stored as whole percentages (0-100).
for c, when in WIN_WHEN.items():
    col_config[c] = st.column_config.NumberColumn(
        help=WIN_HELP[v].format(when=when), format="%d%%")
col_config["Peak team vs grid %"] = st.column_config.NumberColumn(
    help=TEAM_HELP["Peak team vs grid %"], format="%d%%")
for c, (a, b, c3) in POS_SCORES.items():
    col_config[c] = st.column_config.NumberColumn(
        help=POS_HELP.format(a=a, b=b, c=c3), format="%.2f")
st.dataframe(table, hide_index=True, width="stretch",
             column_order=["#", cfg["column"], *shown_cols], column_config=col_config)