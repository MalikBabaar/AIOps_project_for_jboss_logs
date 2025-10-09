# app.py
import os
import joblib
import pandas as pd
import psutil
import yaml
import traceback
from dotenv import load_dotenv
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fastapi.responses import JSONResponse, Response
from prometheus_client import Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
from sklearn.ensemble import IsolationForest
from typing import List, Optional
from celery.result import AsyncResult

# ---------------- CONFIG ---------------- #
load_dotenv()
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

app = FastAPI()

# Files / paths
ANOMALY_HISTORY_FILE = "anomaly_history.csv"
MODEL_PATH_FILE = "latest_model.txt"
VECTORIZER_PATH_FILE = "latest_vectorizer.txt"
EVENTS_FILE = "events.csv"
USERS_FILE = "users.csv"
LOG_PATH_FILE = "log_path.txt"

# Temporary hardcoded credentials (later you can move to DB)
USERS = {
    "admin": "admin",
    "fahad": "fahad123",
}

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/login")
async def login_user(credentials: LoginRequest):
    if credentials.username in USERS and USERS[credentials.username] == credentials.password:
        return {"status": "success", "username": credentials.username}
    else:
        raise HTTPException(status_code=401, detail="Invalid username or password")

def get_latest_model_and_vectorizer():
    try:
        with open(MODEL_PATH_FILE, "r") as f:
            model_path = f.read().strip()
        with open(VECTORIZER_PATH_FILE, "r") as f:
            vectorizer_path = f.read().strip()
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer, model_path, vectorizer_path
    except Exception:
        return None, None, None, None

model, vectorizer, MODEL_PATH, VECTORIZER_PATH = get_latest_model_and_vectorizer()
API_PORT = int(os.getenv("API_PORT", config.get("api_port", 5000)))
ANOMALY_THRESHOLD = float(config.get("anomaly_threshold", 0))

# ---------------- INPUT MODEL ---------------- #
class LogInput(BaseModel):
    log: str
    label: Optional[int] = None
    file_path: Optional[str] = None

# ---------------- EVENTS & ALERTS ---------------- #
def save_event(source, event_type, severity, message, status="active"):
    """Helper to store alerts in CSV"""
    record = pd.DataFrame([{
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "type": event_type,
        "severity": severity,
        "message": message,
        "status": status
    }])
    if os.path.exists(EVENTS_FILE):
        record.to_csv(EVENTS_FILE, mode="a", header=False, index=False)
    else:
        record.to_csv(EVENTS_FILE, index=False)

@app.get("/events")
def get_events():
    """Fetch all events"""
    if not os.path.exists(EVENTS_FILE):
        return []
    try:
        df = pd.read_csv(EVENTS_FILE, on_bad_lines="skip")
        df = df.where(pd.notnull(df), None)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading events: {e}")

@app.post("/events/ack/{timestamp}")
def acknowledge_event(timestamp: str):
    """Mark an event as acknowledged"""
    if not os.path.exists(EVENTS_FILE):
        raise HTTPException(status_code=404, detail="No events file found")
    try:
        df = pd.read_csv(EVENTS_FILE, on_bad_lines="skip")
        if timestamp not in df["timestamp"].values:
            raise HTTPException(status_code=404, detail="Event not found")
        df.loc[df["timestamp"] == timestamp, "status"] = "acknowledged"
        df.to_csv(EVENTS_FILE, index=False)
        return {"message": f"Event {timestamp} acknowledged successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating event: {e}")

# ---------------- ANALYZE LOGS ---------------- #
@app.post("/analyze")
async def analyze(payload: LogInput):
    if model is None:
        raise HTTPException(status_code=500, detail="No trained model found. Retrain first.")
    try:
        log_text = payload.log
        file_path = payload.file_path or (open(LOG_PATH_FILE).read().strip() if os.path.exists(LOG_PATH_FILE) else "unknown")

        # Handle dict input (sometimes JSON can nest "log")
        if isinstance(log_text, dict):
            log_text = log_text.get("log", str(log_text))

        # Compute anomaly score
        X = vectorizer.transform([log_text])
        score = float(model.decision_function(X)[0])
        is_anomaly = bool(score < ANOMALY_THRESHOLD)

        # Force mark critical keywords as anomalies
        critical_keywords = ["ERROR", "FATAL", "CRITICAL", "OUT OF MEMORY", "KERNEL PANIC", "SHUTDOWN"]
        if any(keyword.lower() in log_text.lower() for keyword in critical_keywords):
            is_anomaly = True

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

    # -------------------------------------------------------------------
    # Save result
    # -------------------------------------------------------------------
    current_timestamp = datetime.now(timezone.utc).isoformat()

    # Define consistent column order
    columns = ["timestamp", "log", "file_path", "anomaly_score", "is_anomaly"]
    record = pd.DataFrame(
        [[current_timestamp, log_text, file_path, round(score, 5), is_anomaly]],
        columns=columns
    )

    # Append safely to CSV (add header only once)
    record.to_csv(ANOMALY_HISTORY_FILE, mode="a", header=not os.path.exists(ANOMALY_HISTORY_FILE), index=False)

    # -------------------------------------------------------------------
    # Save event if anomaly detected
    # -------------------------------------------------------------------
    if is_anomaly:
        severity = "critical" if any(k.lower() in log_text.lower() for k in critical_keywords) else "warning"
        save_event(file_path, "Anomaly", severity, log_text[:200], "active")

    # Return clean JSON response
    return JSONResponse(content={
        "timestamp": current_timestamp,
        "log": log_text,
        "file_path": file_path,
        "anomaly_score": round(score, 5),
        "is_anomaly": is_anomaly
    })

# ---------------- RETRAIN ---------------- #
@app.post("/retrain")
async def retrain(new_logs: List[LogInput]):
    try:
        if not new_logs:
            raise HTTPException(status_code=400, detail="No logs provided for retraining.")

        df_features = pd.DataFrame([item.dict() for item in new_logs])
        if "log" not in df_features.columns:
            raise HTTPException(status_code=400, detail="Logs must contain a 'log' field.")

        if len(df_features) > 10000:
            df_features = df_features.sample(10000, random_state=42)

        # Vectorize
        new_vectorizer = TfidfVectorizer(max_features=5000)
        X = new_vectorizer.fit_transform(df_features["log"].astype(str))

        # Train Isolation Forest
        contamination = 0.05
        new_model = IsolationForest(contamination=contamination, random_state=42)
        new_model.fit(X)

        # Save model + vectorizer
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("models", exist_ok=True)
        model_filename = f"models/isolation_forest_{version}.joblib"
        vectorizer_filename = f"models/vectorizer_{version}.joblib"

        joblib.dump(new_model, model_filename)
        joblib.dump(new_vectorizer, vectorizer_filename)

        with open(MODEL_PATH_FILE, "w") as f:
            f.write(model_filename)
        with open(VECTORIZER_PATH_FILE, "w") as f:
            f.write(vectorizer_filename)

        # Update globals
        global model, vectorizer, MODEL_PATH, VECTORIZER_PATH
        model, vectorizer = new_model, new_vectorizer
        MODEL_PATH, VECTORIZER_PATH = model_filename, vectorizer_filename

        # Predict anomalies on training set
        y_pred = new_model.predict(X)
        y_pred = [0 if p == 1 else 1 for p in y_pred]
        anomalies = sum(y_pred)
        anomaly_ratio = anomalies / len(y_pred)

        acc = prec = rec = f1 = None
        if "label" in df_features.columns and df_features["label"].notna().any():
            try:
                y_true = df_features["label"].astype(int).tolist()
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
            except Exception:
                pass

        return JSONResponse(content={
            "status": "success",
            "new_model_version": version,
            "contamination_used": contamination,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "training_time": datetime.now(timezone.utc).isoformat(),
            "detected_anomalies": anomalies,
            "anomaly_ratio": anomaly_ratio
        })

    except Exception as e:
        error_detail = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Retrain failed: {e}\n{error_detail}")

# ---------------- ASYNC retrain placeholders (if you use Celery) ---------------- #
# Keep as-is if celery configured; otherwise these routes can remain but won't be used.
try:
    from celery_app import celery_app, retrain_model_task
    @app.post("/retrain-async")
    async def retrain_async(new_logs: List[LogInput]):
        df_data = [item.dict() for item in new_logs]
        task = retrain_model_task.delay(df_data)
        return {"task_id": task.id, "status": "queued"}

    @app.get("/task-status/{task_id}")
    def get_task_status(task_id: str):
        result = AsyncResult(task_id, app=celery_app)
        return {"task_id": task_id, "status": result.status, "result": result.result}
except Exception:
    # Celery not configured — ignore
    pass

# ---------------- Anomaly History / Live Logs ---------------- #
def load_anomaly_history(start: str = None, end: str = None, limit: int = None):
    """Shared function to read anomaly history data with proper timezone handling."""
    if not os.path.exists(ANOMALY_HISTORY_FILE):
        return []
    try:
        df = pd.read_csv(ANOMALY_HISTORY_FILE, on_bad_lines="skip")
        df.dropna(subset=["timestamp", "log"], inplace=True)

        # Make all timestamps UTC-aware
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

        # Apply start/end filters (convert them to UTC too)
        if start:
            start_dt = pd.to_datetime(start, utc=True)
            df = df[df["timestamp"] >= start_dt]
        if end:
            end_dt = pd.to_datetime(end, utc=True)
            df = df[df["timestamp"] <= end_dt]

        # Apply limit and sort
        if limit:
            df = df.tail(limit)
        df = df.sort_values(by="timestamp", ascending=False)

        # Replace NaN with None for JSON safety
        df = df.where(pd.notnull(df), None)
        return df.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load anomaly history: {e}")


@app.get("/anomaly-history")
async def anomaly_history(
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    limit: Optional[int] = Query(None)
):
    """Return anomaly history with optional filters."""
    return load_anomaly_history(start=start, end=end, limit=limit)


@app.get("/live-logs")
async def live_logs(limit: int = 10):
    """Return the latest anomalies for live dashboard view."""
    return load_anomaly_history(limit=limit)

@app.post("/anomaly/{timestamp}/tag")
def tag_anomaly(timestamp: str, payload: dict):
    """Add or update a tag for an anomaly identified by timestamp."""
    if not os.path.exists(ANOMALY_HISTORY_FILE):
        raise HTTPException(status_code=404, detail="No anomaly history file found")
    try:
        df = pd.read_csv(ANOMALY_HISTORY_FILE, on_bad_lines="skip")
        if timestamp not in df["timestamp"].astype(str).values:
            raise HTTPException(status_code=404, detail="Anomaly not found")
        if "tag" not in df.columns:
            df["tag"] = None
        df.loc[df["timestamp"].astype(str) == timestamp, "tag"] = payload.get("tag")
        df.to_csv(ANOMALY_HISTORY_FILE, index=False)
        return {"message": "Tag updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- Overview ---------------- #
@app.get("/overview")
def overview():
    total_logs = anomalies = 0
    try:
        if os.path.exists(ANOMALY_HISTORY_FILE):
            df = pd.read_csv(ANOMALY_HISTORY_FILE, on_bad_lines="skip")
            total_logs = len(df)

            if "is_anomaly" in df.columns:
                df["is_anomaly"] = df["is_anomaly"].astype(str).str.strip().str.lower()
                df["is_anomaly"] = df["is_anomaly"].isin(["true", "1", "yes", "y", "t"])
                anomalies = int(df["is_anomaly"].sum())
    except Exception as e:
        print("Error in overview():", e)
        total_logs = 0
        anomalies = 0

    active_users = 0
    if os.path.exists(USERS_FILE):
        try:
            active_users = pd.read_csv(USERS_FILE).shape[0]
        except Exception:
            pass

    return {
        "total_logs": int(total_logs),
        "anomalies": int(anomalies),
        "models_deployed": 1 if model is not None else 0,
        "active_users": int(active_users)
    }

# ---------------- Users & Teams ---------------- #
@app.post("/users")
def add_user(user: dict):
    required = ["name", "email", "role", "team"]
    if not all(k in user for k in required):
        raise HTTPException(status_code=400, detail="Missing user fields")
    df = pd.DataFrame([{
        "name": user["name"],
        "email": user["email"],
        "role": user["role"],
        "team": user["team"],
        "created_at": datetime.now(timezone.utc).isoformat()
    }])
    if os.path.exists(USERS_FILE):
        df.to_csv(USERS_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(USERS_FILE, index=False)
    return {"status": "success", "message": f"User {user['name']} added."}

@app.get("/users")
def list_users():
    if not os.path.exists(USERS_FILE):
        return []
    df = pd.read_csv(USERS_FILE, on_bad_lines="skip")
    return df.to_dict(orient="records")

@app.delete("/users/{email}")
def delete_user(email: str):
    if not os.path.exists(USERS_FILE):
        raise HTTPException(status_code=404, detail="No users found")
    df = pd.read_csv(USERS_FILE, on_bad_lines="skip")
    if email not in df["email"].values:
        raise HTTPException(status_code=404, detail="User not found")
    df = df[df["email"] != email]
    df.to_csv(USERS_FILE, index=False)
    return {"status": "success", "message": f"User {email} deleted"}

@app.get("/teams")
def list_teams():
    if not os.path.exists(USERS_FILE):
        return []
    df = pd.read_csv(USERS_FILE, on_bad_lines="skip")
    teams = df["team"].dropna().unique().tolist()
    return teams

# ---------------- System Stats ---------------- #
registry = CollectorRegistry()
cpu_gauge = Gauge("system_cpu_usage_percent", "CPU usage percentage", registry=registry)
mem_gauge = Gauge("system_memory_usage_percent", "Memory usage percentage", registry=registry)
disk_gauge = Gauge("system_disk_usage_percent", "Disk usage percentage", registry=registry)
net_sent_gauge = Gauge("system_network_sent_bytes", "Network sent in bytes", registry=registry)
net_recv_gauge = Gauge("system_network_received_bytes", "Network received in bytes", registry=registry)

@app.get("/system-stats")
def get_system_stats():
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent
    disk = psutil.disk_usage('/').percent
    net_io = psutil.net_io_counters()
    if cpu > 90:
        save_event("System", "Resource Alert", "critical", f"High CPU usage detected: {cpu}%", "active")
    if mem > 80:
        save_event("System", "Resource Alert", "warning", f"High memory usage detected: {mem}%", "active")
    if disk > 85:
        save_event("System", "Resource Alert", "warning", f"High disk usage detected: {disk}%", "active")
    cpu_gauge.set(cpu)
    mem_gauge.set(mem)
    disk_gauge.set(disk)
    net_sent_gauge.set(net_io.bytes_sent)
    net_recv_gauge.set(net_io.bytes_recv)
    return {
        "cpu": cpu,
        "memory": mem,
        "disk": disk,
        "network_sent": net_io.bytes_sent,
        "network_recv": net_io.bytes_recv,
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "vectorizer_path": VECTORIZER_PATH
    }

@app.get("/model-info")
def model_info():
    if not os.path.exists(MODEL_PATH_FILE):
        raise HTTPException(status_code=404, detail="No model found")
    with open(MODEL_PATH_FILE, "r") as f:
        model_path = f.read().strip()
    return {
        "status": "ok",
        "latest_model": model_path,
        "file_size": os.path.getsize(model_path) if os.path.exists(model_path) else 0,
        "last_updated": datetime.fromtimestamp(os.path.getmtime(model_path), tz=timezone.utc).isoformat() if os.path.exists(model_path) else None
    }

# ---------------- Utilities ---------------- #
@app.post("/set-log-path")
def set_log_path(payload: dict):
    path = payload.get("path")
    if not path:
        raise HTTPException(status_code=400, detail="Missing 'path' field")
    with open(LOG_PATH_FILE, "w") as f:
        f.write(path)
    return {"message": f"log path set to {path}"}