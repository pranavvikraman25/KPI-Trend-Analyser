# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(page_title="CKPI Trend Explorer (Door Friction)", layout="wide")
st.title("CKPI Trend Explorer — Door Friction (Trend + Peaks/Lows)")

st.markdown("""
Upload a file (xlsx / xls / csv / json) with columns:
`eq | ckpi | ckpi_statistics_date | side | floor | ave | min | max | stddev | cnt`.

This tool focuses on one CKPI (default: `doorFriction`).  
Filters: CKPI, floor(s), date range.  
Points inside the *No Corrective Action* range are shown green; outside are yellow (Corrective Action).
""")

# ---------------- Thresholds (from the image)
# Keep keys using the raw ckpi text you'll have (case-insensitive comparison)
KPI_THRESHOLDS = {
    "doorfriction": (30.0, 50.0),                  # no corrective: [30,50], corrective: <30 or >50
    "doorspeederror": (0.05, 0.08),                # [0.05,0.08]
    "landing door lock hook closing time": (0.2, 0.6),
    "landing door lock hook open time": (0.3, None), # no corrective: >0.3s (so lower bound only)
    "maximum force during coupler compress": (5.0, 28.0),
    "landing door lock roller clearance": (None, 0.029) # no corrective: <0.029 (upper bound only)
}

# ---------------- Helpers ----------------
def read_file(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded, engine="openpyxl")
    if name.endswith(".xls"):
        return pd.read_excel(uploaded, engine="xlrd")
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith(".json"):
        return pd.read_json(uploaded)
    # fallback attempt
    return pd.read_csv(uploaded)

def parse_dates(df, col):
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    return df

def point_status(value, thresh):
    """Return 'ok' if within no-corrective range, else 'corrective'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nodata"
    low, high = thresh
    # handle one-sided thresholds
    if (low is not None) and (high is not None):
        return "ok" if (low <= value <= high) else "corrective"
    if (low is None) and (high is not None):
        return "ok" if value <= high else "corrective"
    if (low is not None) and (high is None):
        return "ok" if value >= low else "corrective"
    return "corrective"

def detect_peaks_lows(values, low_thresh, high_thresh):
    """
    Local neighbor method:
      - peak: value > neighbors and > high_thresh OR > mean+std
      - low: value < neighbors and < low_thresh OR < mean-std
    Returns indices lists (peaks_idx, lows_idx).
    """
    arr = np.array(values, dtype=float)
    n = len(arr)
    peaks = []
    lows = []
    if n < 3:
        return peaks, lows
    mean = np.nanmean(arr)
    std = np.nanstd(arr) if not np.isnan(arr).all() else 0.0
    for i in range(1, n-1):
        a, b, c = arr[i-1], arr[i], arr[i+1]
        if np.isnan(b): 
            continue
        # local max
        if (not np.isnan(a)) and (not np.isnan(c)) and (b > a) and (b > c):
            cond_high = False
            if high_thresh is not None and b > high_thresh:
                cond_high = True
            elif b > (mean + std):
                cond_high = True
            if cond_high:
                peaks.append(i)
        # local min
        if (not np.isnan(a)) and (not np.isnan(c)) and (b < a) and (b < c):
            cond_low = False
            if low_thresh is not None and b < low_thresh:
                cond_low = True
            elif b < (mean - std):
                cond_low = True
            if cond_low:
                lows.append(i)
    return peaks, lows

def color_cycle(i):
    palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f"]
    return palette[i % len(palette)]

# ---------------- Upload & Detect columns ----------------
uploaded = st.file_uploader("Upload KPI file", type=["xlsx","xls","csv","json"])
if not uploaded:
    st.info("Upload a KPI file to begin.")
    st.stop()

try:
    df = read_file(uploaded)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

if df.empty:
    st.error("Uploaded file is empty.")
    st.stop()

# normalize column names lookup (case-insensitive)
cols_lower = {c.lower(): c for c in df.columns}

# required columns present?
required = ["ckpi_statistics_date", "ave", "ckpi", "floor"]
for req in required:
    if req not in cols_lower:
        st.error(f"Required column '{req}' not found in your file (case-insensitive). Found: {list(df.columns)}")
        st.stop()

date_col = cols_lower["ckpi_statistics_date"]
ave_col = cols_lower["ave"]
ckpi_col = cols_lower["ckpi"]
floor_col = cols_lower["floor"]

# parse dates
df = parse_dates(df, date_col)
if df[date_col].isna().all():
    st.error("Could not parse any dates in the date column. Ensure format like DD-MM-YYYY.")
    st.stop()

# ---------------- Filters
st.sidebar.header("Filters")
# CKPI choices
ckpi_choices = sorted(df[ckpi_col].dropna().astype(str).unique(), key=str.lower)
default_ckpi = "doorFriction" if "doorFriction" in ckpi_choices else ckpi_choices[0]
ckpi_choice = st.sidebar.selectbox("CKPI to analyze", ckpi_choices, index=ckpi_choices.index(default_ckpi))

# filter data to that CKPI (case-insensitive)
df_ckpi = df[df[ckpi_col].astype(str).str.lower() == str(ckpi_choice).lower()].copy()
if df_ckpi.empty:
    st.error(f"No rows for CKPI '{ckpi_choice}'.")
    st.stop()

# floor multi-select
floors_avail = sorted(df_ckpi[floor_col].dropna().unique(), key=lambda x: str(x))
selected_floors = st.sidebar.multiselect("Select floor(s)", options=floors_avail, default=floors_avail[:2] if floors_avail else [])

if not selected_floors:
    st.sidebar.warning("Select at least one floor.")
    st.stop()

# date range
min_dt = df_ckpi[date_col].min().date()
max_dt = df_ckpi[date_col].max().date()
start_date, end_date = st.sidebar.date_input("Date range", [min_dt, max_dt])

# detection sensitivity
std_factor = st.sidebar.slider("Peak/low sensitivity (higher = fewer flags)", 0.5, 3.0, 1.0, 0.1)

# filter by date and floors
mask = (df_ckpi[date_col].dt.date >= start_date) & (df_ckpi[date_col].dt.date <= end_date) & (df_ckpi[floor_col].isin(selected_floors))
plot_df = df_ckpi[mask].copy()
if plot_df.empty:
    st.error("No data after applying filters.")
    st.stop()

# ensure numeric ave
plot_df[ave_col] = pd.to_numeric(plot_df[ave_col], errors="coerce")

# ---------------- Build interactive plot ----------------
st.subheader(f"Trend — CKPI: {ckpi_choice} — Floors: {', '.join(map(str, selected_floors))}")
fig = go.Figure()
floor_colors = {}
summary = []

for i, floor in enumerate(selected_floors):
    s = plot_df[plot_df[floor_col] == floor].sort_values(by=date_col)
    if s.empty:
        continue
    color = color_cycle(i)
    floor_colors[floor] = color

    # add line
    fig.add_trace(go.Scatter(
        x=s[date_col],
        y=s[ave_col],
        mode="lines+markers",
        name=f"Floor {floor}",
        line=dict(color=color, width=2),
        marker=dict(size=6, color=color),
        hovertemplate=(
            "Date: %{x|%Y-%m-%d}<br>"
            f"Floor: {floor}<br>"
            "ave: %{y:.4f}<extra></extra>"
        )
    ))

    # compute statuses for marker coloring
    key = str(ckpi_choice).strip().lower()
    thresh = KPI_THRESHOLDS.get(key, (None, None))
    low_thresh, high_thresh = thresh

    statuses = s[ave_col].apply(lambda v: point_status(v, (low_thresh, high_thresh))).values

    # markers colored by status (overlay)
    status_colors = []
    for stt in statuses:
        if stt == "ok":
            status_colors.append("#2ca02c")  # green
        elif stt == "corrective":
            status_colors.append("#ffcc00")  # yellow
        else:
            status_colors.append("#B0B0B0")  # grey for nodata

    fig.add_trace(go.Scatter(
        x=s[date_col],
        y=s[ave_col],
        mode="markers",
        name=f"Status (Floor {floor})",
        marker=dict(size=9, color=status_colors, symbol="circle", line=dict(color="#000000", width=1)),
        hovertemplate=(
            "Date: %{x|%Y-%m-%d}<br>"
            f"Floor: {floor}<br>"
            "ave: %{y:.4f}<br>"
            f"min: %{customdata[0]:.4f}<br>"
            f"max: %{customdata[1]:.4f}<br>"
            f"stddev: %{customdata[2]:.4f}<br>"
            f"cnt: %{customdata[3]}<extra></extra>"
        ),
        customdata=np.stack([
            pd.to_numeric(s.get("min", np.nan), errors="coerce").fillna(np.nan),
            pd.to_numeric(s.get("max", np.nan), errors="coerce").fillna(np.nan),
            pd.to_numeric(s.get("stddev", np.nan), errors="coerce").fillna(np.nan),
            pd.to_numeric(s.get("cnt", np.nan), errors="coerce").fillna(np.nan),
        ], axis=-1),
        showlegend=False
    ))

    # detect peaks/lows relative to thresholds
    peaks_idx, lows_idx = detect_peaks_lows(s[ave_col].values, low_thresh, high_thresh)
    if peaks_idx:
        fig.add_trace(go.Scatter(
            x=s[date_col].values[peaks_idx],
            y=s[ave_col].values[peaks_idx],
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="#d62728"),
            name=f"Peaks (Floor {floor})",
            hovertext=[f"Peak<br>{pd.to_datetime(d).date()}<br>ave={v:.3f}" for d, v in zip(s[date_col].values[peaks_idx], s[ave_col].values[peaks_idx])]
        ))
    if lows_idx:
        fig.add_trace(go.Scatter(
            x=s[date_col].values[lows_idx],
            y=s[ave_col].values[lows_idx],
            mode="markers",
            marker=dict(symbol="triangle-down", size=12, color="#1f77b4"),
            name=f"Lows (Floor {floor})",
            hovertext=[f"Low<br>{pd.to_datetime(d).date()}<br>ave={v:.3f}" for d, v in zip(s[date_col].values[lows_idx], s[ave_col].values[lows_idx])]
        ))

    # summary per floor
    peaks_dates = [pd.to_datetime(s[date_col].values[idx]).date() for idx in peaks_idx]
    lows_dates = [pd.to_datetime(s[date_col].values[idx]).date() for idx in lows_idx]
    summary.append({
        "floor": floor,
        "rows": len(s),
        "avg": float(np.nanmean(s[ave_col])),
        "std": float(np.nanstd(s[ave_col])),
        "peaks_count": len(peaks_idx),
        "peaks_dates": peaks_dates,
        "lows_count": len(lows_idx),
        "lows_dates": lows_dates
    })

# layout
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="ave",
    hovermode="closest",
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# ---------------- Analysis Summary (text)
st.subheader("Trend Analysis Summary")
for s in summary:
    st.markdown(f"**Floor {s['floor']}** — rows: {s['rows']}, avg: {s['avg']:.3f}, std: {s['std']:.3f}")
    st.markdown(f"- Peaks: **{s['peaks_count']}** — dates: {', '.join(d.isoformat() for d in s['peaks_dates']) if s['peaks_dates'] else 'None'}")
    st.markdown(f"- Lows: **{s['lows_count']}** — dates: {', '.join(d.isoformat() for d in s['lows_dates']) if s['lows_dates'] else 'None'}")
    st.write("---")

# --------------- Downloads
st.subheader("Download results")
def df_to_excel_bytes(df_):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df_.to_excel(writer, index=False, sheet_name="filtered")
    out.seek(0)
    return out

st.download_button("Download filtered rows (Excel)", data=df_to_excel_bytes(plot_df), file_name="filtered_kpi_rows.xlsx")

# peaks/lows aggregated download
peaks_rows = []
for s in summary:
    for d in s['peaks_dates']:
        peaks_rows.append({"floor": s["floor"], "type": "peak", "date": d.isoformat()})
    for d in s['lows_dates']:
        peaks_rows.append({"floor": s["floor"], "type": "low", "date": d.isoformat()})
if peaks_rows:
    peaks_df = pd.DataFrame(peaks_rows)
    st.download_button("Download peaks/lows (Excel)", data=df_to_excel_bytes(peaks_df), file_name="kpi_peaks_lows.xlsx")
else:
    st.info("No peaks/lows detected for selected settings.")

st.success("Trend analysis ready. Adjust date/floor or sensitivity for different results.")
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(page_title="CKPI Trend Explorer (Door Friction)", layout="wide")
st.title("CKPI Trend Explorer — Door Friction (Trend + Peaks/Lows)")

st.markdown("""
Upload a file (xlsx / xls / csv / json) with columns:
`eq | ckpi | ckpi_statistics_date | side | floor | ave | min | max | stddev | cnt`.

This tool focuses on one CKPI (default: `doorFriction`).  
Filters: CKPI, floor(s), date range.  
Points inside the *No Corrective Action* range are shown green; outside are yellow (Corrective Action).
""")

# ---------------- Thresholds (from the image)
# Keep keys using the raw ckpi text you'll have (case-insensitive comparison)
KPI_THRESHOLDS = {
    "doorfriction": (30.0, 50.0),                  # no corrective: [30,50], corrective: <30 or >50
    "doorspeederror": (0.05, 0.08),                # [0.05,0.08]
    "landing door lock hook closing time": (0.2, 0.6),
    "landing door lock hook open time": (0.3, None), # no corrective: >0.3s (so lower bound only)
    "maximum force during coupler compress": (5.0, 28.0),
    "landing door lock roller clearance": (None, 0.029) # no corrective: <0.029 (upper bound only)
}

# ---------------- Helpers ----------------
def read_file(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded, engine="openpyxl")
    if name.endswith(".xls"):
        return pd.read_excel(uploaded, engine="xlrd")
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith(".json"):
        return pd.read_json(uploaded)
    # fallback attempt
    return pd.read_csv(uploaded)

def parse_dates(df, col):
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    return df

def point_status(value, thresh):
    """Return 'ok' if within no-corrective range, else 'corrective'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nodata"
    low, high = thresh
    # handle one-sided thresholds
    if (low is not None) and (high is not None):
        return "ok" if (low <= value <= high) else "corrective"
    if (low is None) and (high is not None):
        return "ok" if value <= high else "corrective"
    if (low is not None) and (high is None):
        return "ok" if value >= low else "corrective"
    return "corrective"

def detect_peaks_lows(values, low_thresh, high_thresh):
    """
    Local neighbor method:
      - peak: value > neighbors and > high_thresh OR > mean+std
      - low: value < neighbors and < low_thresh OR < mean-std
    Returns indices lists (peaks_idx, lows_idx).
    """
    arr = np.array(values, dtype=float)
    n = len(arr)
    peaks = []
    lows = []
    if n < 3:
        return peaks, lows
    mean = np.nanmean(arr)
    std = np.nanstd(arr) if not np.isnan(arr).all() else 0.0
    for i in range(1, n-1):
        a, b, c = arr[i-1], arr[i], arr[i+1]
        if np.isnan(b): 
            continue
        # local max
        if (not np.isnan(a)) and (not np.isnan(c)) and (b > a) and (b > c):
            cond_high = False
            if high_thresh is not None and b > high_thresh:
                cond_high = True
            elif b > (mean + std):
                cond_high = True
            if cond_high:
                peaks.append(i)
        # local min
        if (not np.isnan(a)) and (not np.isnan(c)) and (b < a) and (b < c):
            cond_low = False
            if low_thresh is not None and b < low_thresh:
                cond_low = True
            elif b < (mean - std):
                cond_low = True
            if cond_low:
                lows.append(i)
    return peaks, lows

def color_cycle(i):
    palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f"]
    return palette[i % len(palette)]

# ---------------- Upload & Detect columns ----------------
uploaded = st.file_uploader("Upload KPI file", type=["xlsx","xls","csv","json"])
if not uploaded:
    st.info("Upload a KPI file to begin.")
    st.stop()

try:
    df = read_file(uploaded)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

if df.empty:
    st.error("Uploaded file is empty.")
    st.stop()

# normalize column names lookup (case-insensitive)
cols_lower = {c.lower(): c for c in df.columns}

# required columns present?
required = ["ckpi_statistics_date", "ave", "ckpi", "floor"]
for req in required:
    if req not in cols_lower:
        st.error(f"Required column '{req}' not found in your file (case-insensitive). Found: {list(df.columns)}")
        st.stop()

date_col = cols_lower["ckpi_statistics_date"]
ave_col = cols_lower["ave"]
ckpi_col = cols_lower["ckpi"]
floor_col = cols_lower["floor"]

# parse dates
df = parse_dates(df, date_col)
if df[date_col].isna().all():
    st.error("Could not parse any dates in the date column. Ensure format like DD-MM-YYYY.")
    st.stop()

# ---------------- Filters
st.sidebar.header("Filters")
# CKPI choices
ckpi_choices = sorted(df[ckpi_col].dropna().astype(str).unique(), key=str.lower)
default_ckpi = "doorFriction" if "doorFriction" in ckpi_choices else ckpi_choices[0]
ckpi_choice = st.sidebar.selectbox("CKPI to analyze", ckpi_choices, index=ckpi_choices.index(default_ckpi))

# filter data to that CKPI (case-insensitive)
df_ckpi = df[df[ckpi_col].astype(str).str.lower() == str(ckpi_choice).lower()].copy()
if df_ckpi.empty:
    st.error(f"No rows for CKPI '{ckpi_choice}'.")
    st.stop()

# floor multi-select
floors_avail = sorted(df_ckpi[floor_col].dropna().unique(), key=lambda x: str(x))
selected_floors = st.sidebar.multiselect("Select floor(s)", options=floors_avail, default=floors_avail[:2] if floors_avail else [])

if not selected_floors:
    st.sidebar.warning("Select at least one floor.")
    st.stop()

# date range
min_dt = df_ckpi[date_col].min().date()
max_dt = df_ckpi[date_col].max().date()
start_date, end_date = st.sidebar.date_input("Date range", [min_dt, max_dt])

# detection sensitivity
std_factor = st.sidebar.slider("Peak/low sensitivity (higher = fewer flags)", 0.5, 3.0, 1.0, 0.1)

# filter by date and floors
mask = (df_ckpi[date_col].dt.date >= start_date) & (df_ckpi[date_col].dt.date <= end_date) & (df_ckpi[floor_col].isin(selected_floors))
plot_df = df_ckpi[mask].copy()
if plot_df.empty:
    st.error("No data after applying filters.")
    st.stop()

# ensure numeric ave
plot_df[ave_col] = pd.to_numeric(plot_df[ave_col], errors="coerce")

# ---------------- Build interactive plot ----------------
st.subheader(f"Trend — CKPI: {ckpi_choice} — Floors: {', '.join(map(str, selected_floors))}")
fig = go.Figure()
floor_colors = {}
summary = []

for i, floor in enumerate(selected_floors):
    s = plot_df[plot_df[floor_col] == floor].sort_values(by=date_col)
    if s.empty:
        continue
    color = color_cycle(i)
    floor_colors[floor] = color

    # add line
    fig.add_trace(go.Scatter(
        x=s[date_col],
        y=s[ave_col],
        mode="lines+markers",
        name=f"Floor {floor}",
        line=dict(color=color, width=2),
        marker=dict(size=6, color=color),
        hovertemplate=(
            "Date: %{x|%Y-%m-%d}<br>"
            f"Floor: {floor}<br>"
            "ave: %{y:.4f}<extra></extra>"
        )
    ))

    # compute statuses for marker coloring
    key = str(ckpi_choice).strip().lower()
    thresh = KPI_THRESHOLDS.get(key, (None, None))
    low_thresh, high_thresh = thresh

    statuses = s[ave_col].apply(lambda v: point_status(v, (low_thresh, high_thresh))).values

    # markers colored by status (overlay)
    status_colors = []
    for stt in statuses:
        if stt == "ok":
            status_colors.append("#2ca02c")  # green
        elif stt == "corrective":
            status_colors.append("#ffcc00")  # yellow
        else:
            status_colors.append("#B0B0B0")  # grey for nodata

    fig.add_trace(go.Scatter(
        x=s[date_col],
        y=s[ave_col],
        mode="markers",
        name=f"Status (Floor {floor})",
        marker=dict(size=9, color=status_colors, symbol="circle", line=dict(color="#000000", width=1)),
        hovertemplate=(
            "Date: %{x|%Y-%m-%d}<br>"
            f"Floor: {floor}<br>"
            "ave: %{y:.4f}<br>"
            f"min: %{customdata[0]:.4f}<br>"
            f"max: %{customdata[1]:.4f}<br>"
            f"stddev: %{customdata[2]:.4f}<br>"
            f"cnt: %{customdata[3]}<extra></extra>"
        ),
        customdata=np.stack([
            pd.to_numeric(s.get("min", np.nan), errors="coerce").fillna(np.nan),
            pd.to_numeric(s.get("max", np.nan), errors="coerce").fillna(np.nan),
            pd.to_numeric(s.get("stddev", np.nan), errors="coerce").fillna(np.nan),
            pd.to_numeric(s.get("cnt", np.nan), errors="coerce").fillna(np.nan),
        ], axis=-1),
        showlegend=False
    ))

    # detect peaks/lows relative to thresholds
    peaks_idx, lows_idx = detect_peaks_lows(s[ave_col].values, low_thresh, high_thresh)
    if peaks_idx:
        fig.add_trace(go.Scatter(
            x=s[date_col].values[peaks_idx],
            y=s[ave_col].values[peaks_idx],
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="#d62728"),
            name=f"Peaks (Floor {floor})",
            hovertext=[f"Peak<br>{pd.to_datetime(d).date()}<br>ave={v:.3f}" for d, v in zip(s[date_col].values[peaks_idx], s[ave_col].values[peaks_idx])]
        ))
    if lows_idx:
        fig.add_trace(go.Scatter(
            x=s[date_col].values[lows_idx],
            y=s[ave_col].values[lows_idx],
            mode="markers",
            marker=dict(symbol="triangle-down", size=12, color="#1f77b4"),
            name=f"Lows (Floor {floor})",
            hovertext=[f"Low<br>{pd.to_datetime(d).date()}<br>ave={v:.3f}" for d, v in zip(s[date_col].values[lows_idx], s[ave_col].values[lows_idx])]
        ))

    # summary per floor
    peaks_dates = [pd.to_datetime(s[date_col].values[idx]).date() for idx in peaks_idx]
    lows_dates = [pd.to_datetime(s[date_col].values[idx]).date() for idx in lows_idx]
    summary.append({
        "floor": floor,
        "rows": len(s),
        "avg": float(np.nanmean(s[ave_col])),
        "std": float(np.nanstd(s[ave_col])),
        "peaks_count": len(peaks_idx),
        "peaks_dates": peaks_dates,
        "lows_count": len(lows_idx),
        "lows_dates": lows_dates
    })

# layout
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="ave",
    hovermode="closest",
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# ---------------- Analysis Summary (text)
st.subheader("Trend Analysis Summary")
for s in summary:
    st.markdown(f"**Floor {s['floor']}** — rows: {s['rows']}, avg: {s['avg']:.3f}, std: {s['std']:.3f}")
    st.markdown(f"- Peaks: **{s['peaks_count']}** — dates: {', '.join(d.isoformat() for d in s['peaks_dates']) if s['peaks_dates'] else 'None'}")
    st.markdown(f"- Lows: **{s['lows_count']}** — dates: {', '.join(d.isoformat() for d in s['lows_dates']) if s['lows_dates'] else 'None'}")
    st.write("---")

# --------------- Downloads
st.subheader("Download results")
def df_to_excel_bytes(df_):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df_.to_excel(writer, index=False, sheet_name="filtered")
    out.seek(0)
    return out

st.download_button("Download filtered rows (Excel)", data=df_to_excel_bytes(plot_df), file_name="filtered_kpi_rows.xlsx")

# peaks/lows aggregated download
peaks_rows = []
for s in summary:
    for d in s['peaks_dates']:
        peaks_rows.append({"floor": s["floor"], "type": "peak", "date": d.isoformat()})
    for d in s['lows_dates']:
        peaks_rows.append({"floor": s["floor"], "type": "low", "date": d.isoformat()})
if peaks_rows:
    peaks_df = pd.DataFrame(peaks_rows)
    st.download_button("Download peaks/lows (Excel)", data=df_to_excel_bytes(peaks_df), file_name="kpi_peaks_lows.xlsx")
else:
    st.info("No peaks/lows detected for selected settings.")

st.success("Trend analysis ready. Adjust date/floor or sensitivity for different results.")
