"""Helpers shared by the app's pages (app.py and predictions_app.py)."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

PALETTE = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24


def keep_choice(key):
    # Clicking the active option would deselect it; keep the last choice instead.
    state = st.session_state
    if state[key] is None:
        state[key] = state[f"_{key}"]
    state[f"_{key}"] = state[key]


def choice(label, options, key, **kwargs) -> str:
    """A segmented control that always has exactly one option picked."""
    state = st.session_state
    state.setdefault(key, options[0])
    if state[key] not in options:      # options changed since last time
        state[key] = state[f"_{key}"] = options[0]
    state.setdefault(f"_{key}", state[key])
    st.segmented_control(label, options, key=key, on_change=keep_choice, args=(key,),
                         **kwargs)
    return state[key]
