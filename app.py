"""
Flask Application — AI-Based Data Visualization Assistant
"""
import io
import json
import os
import uuid
from pathlib import Path

import numpy as np # type: ignore
import pandas as pd
from flask import Flask, jsonify, render_template, request, session # type: ignore

from data_processor import (
    build_chart_data,
    classify_columns,
    generate_summary,
    get_all_meta,
    init_db,
    load_and_clean,
    save_meta,
)
from nlp_processor import parse_query

# ── App setup ─────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", "viz-assistant-secret-2024")

ALLOWED_EXT = {".csv", ".xls", ".xlsx"}
# In-memory dataset store: session_id -> DataFrame
_DATASTORE: dict[str, pd.DataFrame] = {}
_COLTYPE_STORE: dict[str, dict] = {}


def _allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXT


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


app.json_encoder = NpEncoder  # type: ignore


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400
    if not _allowed_file(file.filename):
        return jsonify({"error": "Only CSV and Excel files are supported."}), 400

    # Save file
    ext      = Path(file.filename).suffix.lower()
    safe_name = f"{uuid.uuid4().hex}{ext}"
    filepath  = UPLOAD_DIR / safe_name
    file.save(str(filepath))

    try:
        df, info = load_and_clean(str(filepath))
    except Exception as e:
        filepath.unlink(missing_ok=True)
        return jsonify({"error": f"Failed to read file: {e}"}), 422

    # Store in session
    sid = session.get("sid") or uuid.uuid4().hex
    session["sid"]      = sid
    session["filepath"] = str(filepath)
    _DATASTORE[sid]     = df
    _COLTYPE_STORE[sid] = info["col_types"]

    # DB metadata
    meta_id = save_meta(file.filename, df.shape[0], df.shape[1], info["col_types"])

    # Preview (first 5 rows as list of dicts)
    preview_df = df.head(5).copy()
    for col in preview_df.columns:
        if pd.api.types.is_datetime64_any_dtype(preview_df[col]):
            preview_df[col] = preview_df[col].dt.strftime("%Y-%m-%d")
    preview = preview_df.to_dict(orient="records")

    return jsonify({
        "success":  True,
        "meta_id":  meta_id,
        "filename": file.filename,
        "rows":     int(df.shape[0]),
        "cols":     int(df.shape[1]),
        "columns":  list(df.columns),
        "col_types": info["col_types"],
        "missing_removed": info["missing_removed"],
        "preview":  preview,
    })


@app.route("/api/summary", methods=["GET"])
def summary():
    sid = session.get("sid")
    if not sid or sid not in _DATASTORE:
        return jsonify({"error": "No dataset loaded. Please upload first."}), 400

    df        = _DATASTORE[sid]
    col_types = _COLTYPE_STORE[sid]
    data      = generate_summary(df, col_types)
    return jsonify(data)


@app.route("/api/query", methods=["POST"])
def query_endpoint():
    sid = session.get("sid")
    if not sid or sid not in _DATASTORE:
        return jsonify({"error": "No dataset loaded. Please upload first."}), 400

    body  = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    df        = _DATASTORE[sid]
    col_types = _COLTYPE_STORE[sid]
    columns   = list(df.columns)

    parsed     = parse_query(query, columns, col_types)
    chart_data = build_chart_data(df, parsed, col_types)

    return jsonify({
        "parsed":     parsed,
        "chart_data": chart_data,
    })


@app.route("/api/columns", methods=["GET"])
def get_columns():
    sid = session.get("sid")
    if not sid or sid not in _DATASTORE:
        return jsonify({"error": "No dataset loaded."}), 400
    df        = _DATASTORE[sid]
    col_types = _COLTYPE_STORE[sid]
    return jsonify({"columns": list(df.columns), "col_types": col_types})


@app.route("/api/history", methods=["GET"])
def history():
    return jsonify({"datasets": get_all_meta()})


@app.route("/api/custom_chart", methods=["POST"])
def custom_chart():
    """Manual chart configuration endpoint."""
    sid = session.get("sid")
    if not sid or sid not in _DATASTORE:
        return jsonify({"error": "No dataset loaded."}), 400

    body       = request.get_json(silent=True) or {}
    x_col      = body.get("x_col")
    y_col      = body.get("y_col")
    chart_type = body.get("chart_type", "bar")
    agg        = body.get("aggregation", "sum")

    df        = _DATASTORE[sid]
    col_types = _COLTYPE_STORE[sid]

    if x_col not in df.columns:
        return jsonify({"error": f"Column '{x_col}' not found."}), 400

    parsed = {
        "x_col":      x_col,
        "y_col":      y_col,
        "chart_type": chart_type,
        "aggregation": agg,
        "intent":     "compare",
        "title":      f"{(y_col or x_col).replace('_',' ').title()} by {x_col.replace('_',' ').title()}",
    }
    chart_data = build_chart_data(df, parsed, col_types)
    return jsonify({"chart_data": chart_data})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)
