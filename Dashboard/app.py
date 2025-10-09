# Dashboard/app.py
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import time
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

API_URL = "http://localhost:5000"
st.set_page_config(page_title="AIOps Dashboard", layout="wide")

# ---------------- LOGIN ---------------- #
# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None

# Login UI
if not st.session_state.authenticated:
    st.title("🔐 AIOps Dashboard Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        try:
            response = requests.post(f"{API_URL}/login", json={"username": username, "password": password})
            if response.status_code == 200:
                st.session_state.authenticated = True
                st.session_state.username = response.json().get("username")
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
        except Exception as e:
            st.error(f"⚠️ Unable to reach backend: {e}")

    st.stop()

# Sidebar logout
st.sidebar.write(f"👤 Logged in as **{st.session_state.username}**")
if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.username = None
    st.rerun()

# ---------------- HELPERS ---------------- #
def get_data(endpoint, params=None):
    try:
        r = requests.get(f"{API_URL}{endpoint}", params=params, timeout=5)
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}

def post_data(endpoint, json_payload):
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=json_payload, timeout=10)
        return r
    except Exception as e:
        return None

# ---------------- NAVIGATION ---------------- #
tabs = st.tabs(["Overview", "System Monitoring", "ML Model Monitoring", "Anomalies", "Events & Alerts", "Users & Teams"])

# --- Overview ---
with tabs[0]:
    st.title("AIOps Dashboard")
    overview = get_data("/overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Logs Processed", f"{overview.get('total_logs', 0):,}")
    col2.metric("Anomalies Detected", f"{overview.get('anomalies', 0):,}")
    col3.metric("Models Deployed", f"{overview.get('models_deployed', 0):,}")
    col4.metric("Active Users", f"{overview.get('active_users', 0):,}")

    st.markdown("---")
    st.subheader("Log File Location")
    if "log_path" not in st.session_state:
        # fetch persisted path if exists
        st.session_state.log_path = ""
        try:
            import os
            if os.path.exists("log_path.txt"):
                st.session_state.log_path = open("log_path.txt").read().strip()
        except Exception:
            pass

    new_path = st.text_input("Log file path", st.session_state.log_path or "/var/log/jboss/server.log")
    if st.button("Save Log Path"):
        resp = post_data("/set-log-path", {"path": new_path})
        if resp is not None and resp.status_code == 200:
            st.session_state.log_path = new_path
            st.success("Saved log path")
        else:
            st.error("Failed to save log path")

    st.markdown("---")
    st.subheader("Quick Analyzer")
    log_entry = st.text_area("Paste log to analyze", value="[[2025-09-22 14:35:17] ERROR [org.jboss.ejb3] – Transaction timeout for UserSessionBean")
    if st.button("Analyze Log"):
        if log_entry.strip():
            r = post_data("/analyze", {"log": log_entry, "file_path": st.session_state.log_path})
            if r is not None and r.status_code == 200:
                res = r.json()
                if res.get("is_anomaly"):
                    st.error("⚠️ Anomaly detected")
                else:
                    st.success("✅ No anomaly")
                st.json(res)
            else:
                st.error("Analyze failed")
        else:
            st.warning("Enter a log first")

# --- System Monitoring ---
with tabs[1]:
    st.title("System Monitoring")
    st_autorefresh(interval=60000, key="sys-refresh")

    # Initialize session state for storing history
    if "stats_history" not in st.session_state:
        st.session_state.stats_history = {
            "time": [],
            "cpu": [],
            "memory": [],
            "disk": [],
            "net_sent": [],
            "net_recv": []
        }

    sys_stats = get_data("/system-stats")

    if "error" not in sys_stats:
        # Append new stats
        st.session_state.stats_history["time"].append(time.strftime("%H:%M:%S"))
        st.session_state.stats_history["cpu"].append(sys_stats["cpu"])
        st.session_state.stats_history["memory"].append(sys_stats["memory"])
        st.session_state.stats_history["disk"].append(sys_stats["disk"])
        st.session_state.stats_history["net_sent"].append(sys_stats["network_sent"])
        st.session_state.stats_history["net_recv"].append(sys_stats["network_recv"])

        # Keep only the last 30 records
        for k in st.session_state.stats_history:
            st.session_state.stats_history[k] = st.session_state.stats_history[k][-30:]

        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("CPU (%)", f"{sys_stats['cpu']:.1f}")
        col2.metric("Memory (%)", f"{sys_stats['memory']:.1f}")
        col3.metric("Disk (%)", f"{sys_stats['disk']:.1f}")
        col4.metric("Net Sent", f"{sys_stats['network_sent']:,}")
        col5.metric("Net Recv", f"{sys_stats['network_recv']:,}")

        # Convert to DataFrame
        df_stats = pd.DataFrame(st.session_state.stats_history)

        # Create Plotly figures
        fig_usage = px.line(
            df_stats, x="time", y=["cpu", "memory", "disk"], title="Usage (%)"
        )
        fig_network = px.line(
            df_stats, x="time", y=["net_sent", "net_recv"], title="Network I/O"
        )

        # Display charts (fixed deprecation warning)
        st.plotly_chart(fig_usage, config={"responsive": True}, use_container_width=True)
        st.plotly_chart(fig_network, config={"responsive": True}, use_container_width=True)

    else:
        st.error(sys_stats.get("error"))

# --- ML Model Monitoring ---
with tabs[2]:
    st.title("ML Model Monitoring")
    if st.button("Get Model Info"):
        r = get_data("/model-info")
        if "error" not in r:
            st.json(r)
        else:
            st.error(r.get("error"))

    st.markdown("### Retrain")
    uploaded_files = st.file_uploader("Upload CSV logs (multiple allowed)", type=["csv"], accept_multiple_files=True)
    if uploaded_files:
        all_logs = []
        for f in uploaded_files:
            try:
                df = pd.read_csv(f)
            except Exception as e:
                st.error(f"Failed to read {f.name}: {e}")
                continue
            if "log" not in df.columns:
                poss = [c for c in df.columns if "log" in c.lower() or "message" in c.lower()]
                if poss:
                    df.rename(columns={poss[0]: "log"}, inplace=True)
                else:
                    st.error(f"{f.name} has no log column")
                    continue
            all_logs.extend(df[["log"]].to_dict("records"))
        if all_logs:
            st.dataframe(pd.DataFrame(all_logs).head(), width="stretch")
            if st.button("Retrain with uploaded files"):
                r = requests.post(f"{API_URL}/retrain", json=all_logs)
                if r.status_code == 200:
                    st.success("Retrain started")
                    st.json(r.json())
                else:
                    st.error(r.text)

# --- Anomalies ---
with tabs[3]:
    st.title("Anomalies")

    # --- Filters (Feature 6) ---
    col_start, col_end = st.columns(2)
    start_date = col_start.date_input("Start date", value=None)
    end_date = col_end.date_input("End date", value=None)
    params = {}
    if start_date:
        params["start"] = pd.to_datetime(start_date).isoformat()
    if end_date:
        params["end"] = (pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).isoformat()

    # --- Fetch data when user clicks button ---
    if st.button("Load Anomalies"):
        history = get_data("/anomaly-history", params=params)
        live = get_data("/live-logs")

        if "error" in history:
            st.error(history["error"])
            history = []
        if "error" in live:
            st.error(live["error"])
            live = []

        # Merge both datasets (remove duplicates)
        all_anomalies = []
        if isinstance(history, list):
            all_anomalies.extend(history)
        if isinstance(live, list):
            all_anomalies.extend(live)

        if all_anomalies:
            df = (
                pd.DataFrame(all_anomalies)
                .drop_duplicates(subset=["timestamp"])
                .sort_values(by="timestamp", ascending=False)
            )

            # Add tag column if missing
            if "tag" not in df.columns:
                df["tag"] = ""

            # --- Auto Tagging (Feature 5) ---
            def suggest_tag(log):
                text = str(log).lower()
                if "error" in text or "fatal" in text or "critical" in text:
                    return "critical"
                elif "warn" in text:
                    return "warning"
                else:
                    return "info"

            if "suggested_tag" not in df.columns:
                df["suggested_tag"] = df["log"].apply(suggest_tag)

            # --- Display unified anomaly table ---
            st.dataframe(df, width='stretch')

            # --- Tagging Section ---
            st.markdown("### Tag Management")
            timestamps = df["timestamp"].astype(str).tolist()
            sel = st.selectbox("Select anomaly to tag", timestamps)
            new_tag = st.text_input("Tag to assign", value="")

            if st.button("Save Tag"):
                if sel and new_tag:
                    r = requests.post(f"{API_URL}/anomaly/{sel}/tag", json={"tag": new_tag})
                    if r and r.status_code == 200:
                        st.success("✅ Tag saved successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save tag")
        else:
            st.info("No anomalies found (neither live nor historical).")
    else:
        st.info("Click 'Load Anomalies' to fetch data.")

# --- Events & Alerts ---
with tabs[4]:
    st.title("Events & Alerts")
    events = get_data("/events")
    if "error" in events:
        st.error(events["error"])
    elif isinstance(events, list) and events:
        df = pd.DataFrame(events)
        # grouping control
        group_by = st.selectbox("Group by", ["none", "severity", "type"])
        if group_by != "none" and group_by in df.columns:
            grouped = df.groupby(group_by).size().reset_index(name="count")
            st.bar_chart(grouped.set_index(group_by))
        st.dataframe(df, width="stretch")
        # acknowledge
        active = df[df["status"] == "active"]["timestamp"].tolist()
        if active:
            sel = st.selectbox("Acknowledge alert (select)", active)
            if st.button("Acknowledge"):
                r = requests.post(f"{API_URL}/events/ack/{sel}")
                if r.status_code == 200:
                    st.success("Acknowledged")
                    st.rerun()
                else:
                    st.error("Ack failed")
    else:
        st.info("No events available")

# --- Users & Teams ---
with tabs[5]:
    st.title("Users & Teams")
    st.subheader("Add user")
    with st.form("add_user"):
        name = st.text_input("Full name")
        email = st.text_input("Email")
        role = st.selectbox("Role", ["admin", "analyst", "viewer"])
        team = st.text_input("Team")
        submitted = st.form_submit_button("Add")
    if submitted and name and email and team:
        r = requests.post(f"{API_URL}/users", json={"name": name, "email": email, "role": role, "team": team})
        if r.status_code == 200:
            st.success("User added")
        else:
            st.error(r.text)

    st.markdown("### Current users")
    r = get_data("/users")
    if "error" in r:
        st.error(r["error"])
    else:
        users = r
        if users:
            df_users = pd.DataFrame(users)
            teams = ["All"] + sorted(df_users["team"].dropna().unique().tolist())
            sel_team = st.selectbox("Filter by team", teams)
            if sel_team != "All":
                df_users = df_users[df_users["team"] == sel_team]
            st.dataframe(df_users, width="stretch")
        else:
            st.info("No users yet")