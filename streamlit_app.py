# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
from datetime import timedelta

st.set_page_config(page_title="Electricity Demand Forecast in GB", page_icon="⚡", layout="wide")

DATA_PATH = "historic_demand_2009_2024_noNaN.csv"

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df.columns = [c.strip() for c in df.columns]
        return df
    else:
        st.error("Dataset not found. Make sure the CSV is committed to the GitHub repo!")
        st.stop()

df = load_data()


# ------------------ PREPROCESS (automated, no raw dump) ------------------
# basic cleaning + detection
df.columns = [c.strip() for c in df.columns]

# detect date & tsd columns heuristically
date_col = next((c for c in df.columns if "date" in c.lower() or "time" in c.lower()), df.columns[0])
tsd_col = next((c for c in df.columns if any(k in c.lower() for k in ["tsd","demand","total_system_demand","total demand"])), None)
if tsd_col is None:
    # fallback: second column
    tsd_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

# convert types
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df[tsd_col] = pd.to_numeric(df[tsd_col], errors="coerce")
df = df.dropna(subset=[date_col, tsd_col]).sort_values(date_col).reset_index(drop=True)

# aggregated series (daily)
ts_daily = df.set_index(date_col).resample("D")[tsd_col].mean().fillna(method="ffill")

# ------------------ UI (hidden-data pattern) ------------------
# Header
st.title("⚡ Electricity Demand Forecast in Great Britain")
st.markdown("Project overview: focusing only on Transmission System Demand (TSD).")

# Sidebar filters (country/time not necessary if single-country)
st.sidebar.header("View options")
min_date, max_date = int(ts_daily.index.min().year), int(ts_daily.index.max().year)
year_range = st.sidebar.slider("Year range", min_value=min_date, max_value=max_date,
                               value=(min_date, max_date), step=1)

# apply year filter to aggregated series
mask = (ts_daily.index.year >= year_range[0]) & (ts_daily.index.year <= year_range[1])
ts_view = ts_daily[mask]

# ---------- KPI (visible preview & full date range) ----------
# compute full date range strings (no truncation)
date_min = ts_daily.index.min()
date_max = ts_daily.index.max()
date_range_full = f"{date_min.strftime('%Y-%m-%d')} → {date_max.strftime('%Y-%m-%d')}"

total_records = len(df)
avg_daily = ts_view.mean()

# show metrics (visible)
k1, k2, k3 = st.columns([1, 2, 1])
k1.metric("Records (rows)", f"{total_records:,}")
k2.metric("Date range", date_range_full)
k3.metric("Avg daily TSD", f"{avg_daily:,.2f}")

# small helper note
st.markdown("**Note:** Raw dataset is kept for backend processing. Below you can preview a limited sample for transparency.")

st.markdown("---")

# ---------- Data preview (show 20 rows of core columns) ----------
st.subheader("Data preview (first 20 rows)")
# show only important columns (date + tsd) to keep privacy
preview_cols = [date_col, tsd_col]
preview_df = df.loc[:, preview_cols].head(20).copy()
preview_df[date_col] = preview_df[date_col].dt.strftime('%Y-%m-%d %H:%M:%S')  # readable format
st.dataframe(preview_df, use_container_width=True)

# optional: allow user to toggle a larger preview (safe)
if st.checkbox("Show more rows (100) — for demo only"):
    more_df = df.loc[:, preview_cols].head(100).copy()
    more_df[date_col] = more_df[date_col].dt.strftime('%Y-%m-%d %H:%M:%S')
    st.dataframe(more_df)


# Time series overview (visual only)
st.subheader("Time series overview (aggregated)")
chart = alt.Chart(ts_view.reset_index()).mark_line().encode(
    x=alt.X(f"{ts_view.index.name}:T", title="Date"),
    y=alt.Y(f"{ts_view.name}:Q", title="TSD"),
    tooltip=[alt.Tooltip(f"{ts_view.index.name}:T", title="Date"), alt.Tooltip(f"{ts_view.name}:Q", format=".2f", title="TSD")]
).properties(height=350)
st.altair_chart(chart, use_container_width=True)

# Seasonal summary (boxplot by month)
st.subheader("Seasonality — monthly distribution (aggregated)")
monthly = ts_view.groupby(ts_view.index.month).agg(list)
# prepare DataFrame for boxplot
box_df = pd.DataFrame({
    "month": ts_view.index.month,
    "tsd": ts_view.values
})
box = alt.Chart(box_df).mark_boxplot().encode(
    x=alt.X("month:O", title="Month"),
    y=alt.Y("tsd:Q", title="TSD")
).properties(height=300)
st.altair_chart(box, use_container_width=True)

st.markdown("---")

# Quick baseline + forecast placeholder (no heavy model run by default)
st.subheader("Quick baseline & forecast preview")
horizon = st.slider("Forecast horizon (days)", min_value=30, max_value=365, value=90, step=1)
if len(ts_view) > horizon + 10:
    train = ts_view.iloc[:-horizon]
    test = ts_view.iloc[-horizon:]
    baseline = train.shift(365).reindex(test.index).fillna(method="ffill").fillna(train.mean())
    # show baseline vs actual (aggregated)
    df_plot = pd.DataFrame({"Date": test.index, "Actual": test.values, "Baseline": baseline.values}).melt(id_vars="Date", var_name="Type", value_name="Value")
    st.altair_chart(alt.Chart(df_plot).mark_line().encode(x="Date:T", y="Value:Q", color="Type:N").properties(height=350), use_container_width=True)
    # show metrics
    mape = np.mean(np.abs((test.values - baseline.values)/(test.values + 1e-9)))*100
    st.metric("Baseline MAPE", f"{mape:.2f}%")
else:
    st.info("Not enough history for selected horizon. Reduce horizon or expand date range.")

st.markdown("---")
st.subheader("Model Comparison — Baseline vs Final XGBoost Model")

# --- Baseline info (current baseline computed earlier)
baseline_mape = mape  # from baseline section
baseline_rmse = np.sqrt(np.mean((test.values - baseline.values)**2))

# --- Final model scores (from your PPT)
xgb_mape = 9.623
xgb_rmse = 3.32  # adjust if needed

# Metric cards for quick comparison
c1, c2 = st.columns(2)
c1.metric("Baseline MAPE", f"{baseline_mape:.2f}%")
c1.metric("Baseline RMSE", f"{baseline_rmse:.2f} GW")

c2.metric("XGBoost Final MAPE", f"{xgb_mape:.2f}%", delta=f"{baseline_mape - xgb_mape:.2f}% ↓")
c2.metric("XGBoost Final RMSE", f"{xgb_rmse:.2f} GW", delta=f"{baseline_rmse - xgb_rmse:.2f} ↓")

# Comparison table
comp_df = pd.DataFrame({
    "Metric": ["MAPE (%)", "RMSE (GW)"],
    "Baseline (Seasonal-Naive)": [f"{baseline_mape:.2f}", f"{baseline_rmse:.2f}"],
    "XGBoost Final Model": [f"{xgb_mape:.2f}", f"{xgb_rmse:.2f}"]
})
st.table(comp_df)

st.success("XGBoost improved forecasting accuracy significantly compared to the baseline model.")


# Insights & PPT highlights (text pulled from your presentation)
st.subheader("Key insights (from dataset & analysis)")
st.write("""
- Strong seasonal pattern: higher demand in winter months (Dec–Feb).  
- Daily peaks morning & evening, weekdays > weekends.  
- Use forecasts to support reserve capacity planning, storage scheduling, and demand-response programs.
""")

