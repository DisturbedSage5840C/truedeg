"""Tiny Streamlit demo -- the live slider is the whole point.

    streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.features import load_features
from src.model import fit_full, inchident_curve

st.set_page_config(page_title="Inchident", layout="centered")
st.title("Inchident -- counterfactual tyre degradation")


@st.cache_resource
def _load():
    df = load_features()
    model, X = fit_full(df)
    return df, model, X


df, model, X = _load()

c1, c2 = st.columns(2)
event = c1.selectbox("Circuit", sorted(df.Event.unique()))
comp = c2.selectbox("Compound", ["SOFT", "MEDIUM", "HARD"])
temp = st.slider("Track temperature (deg C)", 20, 55, 35)
age = st.slider("Tyre age (laps)", 0, 25, 15)

mask = (df.Event == event) & (df.Compound == comp)
if not mask.any():
    st.warning("No laps for that circuit / compound combination in the dataset.")
    st.stop()

row = X[mask].iloc[0].to_dict()
row["TrackTemp"] = temp
ages, deg = inchident_curve(model, X, row)

st.metric(
    "Tyre penalty at this age", f"{deg[age]:+.2f} s",
    help="Pace lost purely to tyre condition vs. a fresh tyre in "
         "identical conditions",
)

fig = go.Figure()
fig.add_scatter(x=ages, y=deg, mode="lines", name="Degradation")
fig.add_scatter(x=[age], y=[deg[age]], mode="markers",
                marker=dict(size=14), name="Selected")
fig.update_layout(xaxis_title="Tyre age (laps)",
                  yaxis_title="Pace loss vs fresh tyre (s)", height=420)
st.plotly_chart(fig, use_container_width=True)
