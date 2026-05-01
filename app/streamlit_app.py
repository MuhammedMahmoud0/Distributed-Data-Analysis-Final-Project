import streamlit as st
import requests
import pandas as pd
import datetime

API = "http://localhost:8000"

st.set_page_config(
    page_title="Revenue Forecast",
    page_icon="📈",
    layout="wide",
)

# ── Sidebar navigation ─────────────────────────────────────────────────────
page = st.sidebar.radio("Navigation", ["📈 Predict", "📊 Monitor", "📋 Logs"])
st.sidebar.markdown("---")
st.sidebar.caption("Revenue Forecast App")

# Check API health
try:
    health = requests.get(f"{API}/health", timeout=3).json()
    st.sidebar.success(f"API: {health.get('status', 'unknown')}")
except Exception:
    st.sidebar.error("API unreachable")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — Predict
# ══════════════════════════════════════════════════════════════════════════
if page == "📈 Predict":
    st.title("📈 Revenue Forecast")
    st.write(
        "Fill in the features below and click **Predict** to get a revenue forecast."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Date")
        event_date = st.date_input(
            "Event Date",
            value=datetime.date.today(),
            min_value=datetime.date(2009, 1, 1),
            max_value=datetime.date(2026, 12, 31),
        )

        st.subheader("Lag features")
        lag_1 = st.number_input("lag_1", value=30000.0, step=500.0)
        lag_7 = st.number_input("lag_7", value=28000.0, step=500.0)
        lag_14 = st.number_input("lag_14", value=26000.0, step=500.0)
        lag_28 = st.number_input("lag_28", value=25000.0, step=500.0)

    with col2:
        st.subheader("Rolling features")
        rolling_mean_7 = st.number_input("rolling_mean_7", value=29000.0, step=500.0)
        rolling_std_7 = st.number_input("rolling_std_7", value=5000.0, step=100.0)
        rolling_mean_28 = st.number_input("rolling_mean_28", value=27000.0, step=500.0)

        st.subheader("Calendar features")
        day_of_week = st.number_input("day_of_week", min_value=1, max_value=7, value=5)
        week_of_year = st.number_input(
            "week_of_year", min_value=1, max_value=53, value=35
        )
        month = st.number_input("month", min_value=1, max_value=12, value=9)
        quarter = st.number_input("quarter", min_value=1, max_value=4, value=3)
        is_weekend = st.selectbox("is_weekend", [0, 1])

    st.markdown("---")

    if st.button("🔮 Predict", use_container_width=True):
        payload = [
            {
                "event_date": str(event_date),
                "lag_1": lag_1,
                "lag_7": lag_7,
                "lag_14": lag_14,
                "lag_28": lag_28,
                "rolling_mean_7": rolling_mean_7,
                "rolling_std_7": rolling_std_7,
                "rolling_mean_28": rolling_mean_28,
                "day_of_week": int(day_of_week),
                "week_of_year": int(week_of_year),
                "month": int(month),
                "quarter": int(quarter),
                "is_weekend": int(is_weekend),
            }
        ]

        with st.spinner("Running prediction..."):
            try:
                response = requests.post(f"{API}/predict", json=payload, timeout=60)
            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to the API. Make sure Flask is running on port 8000."
                )
                st.stop()

        if response.status_code == 200:
            result = response.json()
            pred = result[0]["prediction"]

            st.success(f"### Predicted Revenue: £{pred:,.2f}")

            st.info(
                f"**Date:** {result[0]['event_date']}  \n"
                f"**Prediction:** £{pred:,.2f}"
            )
        else:
            st.error(f"API error ({response.status_code}): {response.text}")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — Monitor
# ══════════════════════════════════════════════════════════════════════════
elif page == "📊 Monitor":
    st.title("📊 Model Performance Monitor")

    if st.button("🔄 Refresh metrics"):
        st.rerun()

    try:
        metrics = requests.get(f"{API}/metrics", timeout=10).json()
    except Exception as e:
        st.error(f"Could not fetch metrics: {e}")
        st.stop()

    if "error" in metrics:
        st.warning(f"No metrics yet: {metrics['error']}")
        st.stop()

    # ── KPI cards ──────────────────────────────────────────────────────
    st.subheader("Prediction volume")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total predictions", metrics.get("total_predictions", "—"))
    col2.metric("Last prediction", metrics.get("last_prediction_at", "—")[:10])
    col3.metric("Date from", metrics.get("date_range", {}).get("from", "—"))
    col4.metric("Date to", metrics.get("date_range", {}).get("to", "—"))

    # ── Prediction stats ───────────────────────────────────────────────
    st.subheader("Prediction distribution")
    stats = metrics.get("prediction_stats", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean", f"£{stats.get('mean', 0):,.2f}")
    c2.metric("Min", f"£{stats.get('min', 0):,.2f}")
    c3.metric("Max", f"£{stats.get('max', 0):,.2f}")
    c4.metric("Std", f"£{stats.get('std', 0):,.2f}")

    # ── Error metrics (only if actuals exist) ─────────────────────────
    if "error_metrics" in metrics:
        st.subheader("Accuracy vs actuals")
        em = metrics["error_metrics"]
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Compared rows", em.get("n_compared"))
        e2.metric("RMSE", f"£{em.get('RMSE', 0):,.2f}")
        e3.metric("MAE", f"£{em.get('MAE', 0):,.2f}")
        e4.metric("MAPE", f"{em.get('MAPE', 0):.2f}%")
    else:
        st.info(
            "Actuals not available yet — run the pipeline to generate `/tmp/actuals.parquet`."
        )


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — Logs
# ══════════════════════════════════════════════════════════════════════════
elif page == "📋 Logs":
    st.title("📋 Prediction Logs")

    n = st.slider(
        "Show last N predictions", min_value=10, max_value=500, value=50, step=10
    )

    if st.button("🔄 Refresh logs"):
        st.rerun()

    try:
        logs = requests.get(f"{API}/logs?n={n}", timeout=10).json()
    except Exception as e:
        st.error(f"Could not fetch logs: {e}")
        st.stop()

    if not logs:
        st.info("No prediction logs yet.")
        st.stop()

    df_logs = pd.DataFrame(logs)

    # Show table
    st.dataframe(
        df_logs[["timestamp", "event_date", "prediction"]].rename(
            columns={
                "timestamp": "Timestamp",
                "event_date": "Date",
                "prediction": "Predicted Revenue (£)",
            }
        ),
        use_container_width=True,
    )

    # Chart predictions over time
    if "event_date" in df_logs.columns and "prediction" in df_logs.columns:
        st.subheader("Predictions over time")
        chart_df = (
            df_logs[["event_date", "prediction"]]
            .sort_values("event_date")
            .rename(columns={"event_date": "Date", "prediction": "Predicted Revenue"})
        )
        st.line_chart(chart_df.set_index("Date"))
