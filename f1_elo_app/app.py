"""F1 ELO viewer: driver and constructor ratings over time.

Run from the f1_elo_app folder (after `py run_elo.py`):

    py -m streamlit run app.py
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.express as px
import streamlit as st

from f1elo import config, updater

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
STATE_KEYS = ("sel", "seasons", "from", "to")

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

    out = {}
    for view, cfg in VIEWS.items():
        # One row per driver (or team) per race: teammates share the team rating,
        # and 1950s shared drives duplicate drivers.
        frame = (res.drop_duplicates(["raceId", cfg["id"]])
                    .rename(columns={cfg["id"]: "id", cfg["rating"]: "rating"})
                    [["raceId", "id", "date", "year", "round", "name", "rating"]])
        frame["pos"] = frame["raceId"].map(races["pos"])
        names = pd.read_csv(cfg["names"]).rename(columns={cfg["id"]: "id"})
        if view == "drivers":
            names = _with_years(names, frame, names["forename"] + " " + names["surname"])
            names = names.sort_values(["surname", "forename"])
        else:
            names = _with_years(names, frame, names["name"]).sort_values("name")
        out[view] = (frame, dict(zip(names["id"], names["label"])))
    return out, races


def summary(view: pd.DataFrame, labels: dict, column: str) -> pd.DataFrame:
    rows = []
    for eid, g in view.groupby("id"):
        peak = g.loc[g["rating"].idxmax()]
        start, end = g["rating"].iloc[0], g["rating"].iloc[-1]
        rows.append({column: labels[eid], "Start": round(start), "End": round(end),
                     "Change": round(end - start), "Peak": round(peak["rating"]),
                     "Peak at": f"{peak['name']} {peak['year']}", "Races": len(g)})
    return pd.DataFrame(rows).sort_values("End", ascending=False)


# -- setup ------------------------------------------------------------------
missing = [p.name for p in [RESULTS, *(c["names"] for c in VIEWS.values())] if not p.exists()]
if missing:
    st.error(f"Missing {', '.join(missing)}. Run `py run_elo.py` in this folder first.")
    st.stop()

data, races = load(tuple(p.stat().st_mtime for p in
                         [RESULTS, *(c["names"] for c in VIEWS.values())]))
all_seasons = sorted(races["year"].unique(), reverse=True)
state = st.session_state

# Streamlit forgets a widget's value when the widget isn't drawn, which would wipe
# the hidden view's selection. Re-saving the values each run keeps both views.
for _key in [f"{v}:{k}" for v in VIEWS for k in STATE_KEYS] + ["line_style"]:
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

# Legend in leaderboard order: highest rating at the end of the period first.
order = view.groupby("label")["rating"].last().sort_values(ascending=False).index.tolist()

def keep_line_style():
    # Clicking the active option would deselect it; keep the last choice instead.
    if state["line_style"] is None:
        state["line_style"] = state["_line_style"]
    state["_line_style"] = state["line_style"]


state.setdefault("line_style", "Steps")
state.setdefault("_line_style", state["line_style"])
st.segmented_control("Line style", ["Steps", "Smooth"], key="line_style",
                     on_change=keep_line_style, label_visibility="collapsed",
                     help="Steps hold each rating until the next race; "
                          "Smooth curves through the post-race ratings.")
smooth = state["line_style"] == "Smooth"

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

shown = view["id"].nunique()
notes = []
if shown < len(sel):
    notes.append(f"{len(sel) - shown} selected {cfg['many']} didn't race in this period")
if breaks:
    notes.append("years between the picked seasons are cut out")
st.caption(f"Showing {shown} {cfg['many']}" + (f" ({'; '.join(notes)})" if notes else "")
           + ". Ratings are recorded after each race. Drag across the chart to zoom, "
           "double-click to zoom back out.")
st.dataframe(summary(view, labels, cfg["column"]), hide_index=True, width="stretch")