"""
Data Processing Module — reads, cleans, and summarises datasets.
"""
import io
import json
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent / "instance" / "datasets.db"


# ── DB helpers ────────────────────────────────────────────────────────────────

def _get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dataset_meta (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT    NOT NULL,
            rows     INTEGER,
            cols     INTEGER,
            col_info TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_meta(filename: str, rows: int, cols: int, col_info: dict) -> int:
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO dataset_meta (filename, rows, cols, col_info) VALUES (?,?,?,?)",
        (filename, rows, cols, json.dumps(col_info)),
    )
    conn.commit()
    meta_id = cur.lastrowid
    conn.close()
    return meta_id


def get_all_meta() -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM dataset_meta ORDER BY uploaded_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Column typing ─────────────────────────────────────────────────────────────

def classify_columns(df: pd.DataFrame) -> dict[str, str]:
    """Return {col_name: 'numeric' | 'categorical' | 'datetime' | 'text'}."""
    types: dict[str, str] = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            types[col] = "datetime"
        else:
            # Try parsing as datetime
            try:
                parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                if parsed.notna().mean() > 0.7:
                    types[col] = "datetime"
                    continue
            except Exception:
                pass
            nunique = df[col].nunique()
            total   = len(df[col].dropna())
            ratio   = nunique / total if total > 0 else 0
            types[col] = "categorical" if (ratio < 0.5 or nunique <= 30) else "text"
    return types


# ── Main processing ───────────────────────────────────────────────────────────

def load_and_clean(filepath: str) -> tuple[pd.DataFrame, dict]:
    """Load, clean, and return (df, info_dict)."""
    path = Path(filepath)
    if path.suffix.lower() in (".xls", ".xlsx"):
        df = pd.read_excel(filepath)
    else:
        # Try common encodings / separators
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                df = pd.read_csv(filepath, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            df = pd.read_csv(filepath, encoding="utf-8", errors="replace")

    original_shape = df.shape

    # Standardise column names
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]

    # Drop fully-empty rows/cols
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    # Identify types
    col_types = classify_columns(df)

    # Convert detected datetime columns
    for col, ctype in col_types.items():
        if ctype == "datetime" and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")

    # Fill missing values
    for col in df.columns:
        if col_types.get(col) == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
        elif col_types.get(col) in ("categorical", "text"):
            fill_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(fill_val)

    info = {
        "original_shape": list(original_shape),
        "cleaned_shape":  list(df.shape),
        "col_types":      col_types,
        "columns":        list(df.columns),
        "missing_removed": int(original_shape[0] - df.shape[0]),
    }
    return df, info


def generate_summary(df: pd.DataFrame, col_types: dict) -> dict:
    """Generate dataset summary statistics and insights."""
    summary: dict = {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_stats": {},
        "categorical_stats": {},
        "insights": [],
    }

    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols     = [c for c, t in col_types.items() if t == "categorical"]

    # Numeric stats
    for col in numeric_cols:
        s = df[col].describe()
        summary["numeric_stats"][col] = {
            "min":    round(float(s["min"]), 2),
            "max":    round(float(s["max"]), 2),
            "mean":   round(float(s["mean"]), 2),
            "median": round(float(df[col].median()), 2),
            "std":    round(float(s["std"]), 2),
        }

    # Categorical stats
    for col in cat_cols[:5]:  # limit to 5
        vc = df[col].value_counts()
        summary["categorical_stats"][col] = {
            "unique_values": int(df[col].nunique()),
            "top_value":     str(vc.index[0]) if len(vc) > 0 else "N/A",
            "top_count":     int(vc.iloc[0])  if len(vc) > 0 else 0,
        }

    # Auto-insights
    for col in numeric_cols[:3]:
        stats = summary["numeric_stats"][col]
        col_label = col.replace("_", " ").title()
        summary["insights"].append(
            f"📊 {col_label}: ranges from {stats['min']:,} to {stats['max']:,} "
            f"with a mean of {stats['mean']:,}."
        )

    for col in cat_cols[:2]:
        s = summary["categorical_stats"][col]
        col_label = col.replace("_", " ").title()
        summary["insights"].append(
            f"🏷️ {col_label}: {s['unique_values']} unique values. "
            f"Most common: '{s['top_value']}' ({s['top_count']} times)."
        )

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        # Find highest off-diagonal correlation
        mask = np.ones(corr.shape, dtype=bool)
        np.fill_diagonal(mask, False)
        abs_corr = corr.where(mask).abs()
        if not abs_corr.empty:
            max_idx = abs_corr.stack().idxmax()
            max_val = corr.loc[max_idx[0], max_idx[1]]
            a = max_idx[0].replace("_", " ").title()
            b = max_idx[1].replace("_", " ").title()
            direction = "positively" if max_val > 0 else "negatively"
            summary["insights"].append(
                f"🔗 Strongest correlation: {a} and {b} are {direction} "
                f"correlated (r={max_val:.2f})."
            )

    return summary


def build_chart_data(df: pd.DataFrame, parsed: dict, col_types: dict) -> dict:
    """
    Execute the parsed query on df and return chart-ready data + insights.
    """
    x_col     = parsed.get("x_col")
    y_col     = parsed.get("y_col")
    agg       = parsed.get("aggregation", "sum")
    chart_type = parsed.get("chart_type", "bar")
    title     = parsed.get("title", "Chart")

    result: dict = {
        "chart_type": chart_type,
        "title": title,
        "labels": [],
        "datasets": [],
        "insights": [],
        "error": None,
    }

    try:
        # ── Histogram ──────────────────────────────────────────────────────
        if chart_type == "histogram":
            col = x_col if x_col and col_types.get(x_col) == "numeric" else (
                next((c for c in df.columns if col_types.get(c) == "numeric"), None)
            )
            if col is None:
                raise ValueError("No numeric column found for histogram.")
            values = df[col].dropna().tolist()
            result["labels"] = [col]
            result["datasets"] = [{"label": col, "data": values}]
            result["insights"] = [
                f"Distribution of {col.replace('_',' ').title()}:",
                f"  • Min: {min(values):,.2f}",
                f"  • Max: {max(values):,.2f}",
                f"  • Mean: {sum(values)/len(values):,.2f}",
                f"  • Std Dev: {pd.Series(values).std():,.2f}",
            ]
            return result

        # ── Scatter ────────────────────────────────────────────────────────
        if chart_type == "scatter":
            numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
            xc = x_col if x_col and col_types.get(x_col) == "numeric" else (
                numeric_cols[0] if numeric_cols else None
            )
            yc = y_col if y_col and col_types.get(y_col) == "numeric" else (
                numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0] if numeric_cols else None
            )
            if not xc or not yc:
                raise ValueError("Need two numeric columns for scatter plot.")
            points = df[[xc, yc]].dropna()
            result["labels"] = [xc, yc]
            result["datasets"] = [{
                "label": f"{xc} vs {yc}",
                "data": [{"x": row[xc], "y": row[yc]} for _, row in points.iterrows()],
            }]
            corr_val = points[xc].corr(points[yc])
            result["insights"] = [
                f"Scatter plot of {xc.replace('_',' ').title()} vs {yc.replace('_',' ').title()}.",
                f"  • Pearson correlation: {corr_val:.3f}",
                f"  • {abs(corr_val)*100:.0f}% strength " +
                ("(positive)" if corr_val > 0 else "(negative)" if corr_val < 0 else "(no)") +
                " linear relationship.",
            ]
            return result

        # ── Heatmap ────────────────────────────────────────────────────────
        if chart_type == "heatmap":
            numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
            if len(numeric_cols) < 2:
                raise ValueError("Need at least 2 numeric columns for a heatmap.")
            corr_df = df[numeric_cols].corr().round(2)
            result["labels"] = numeric_cols
            result["datasets"] = [
                {
                    "label": col,
                    "data": corr_df[col].tolist(),
                }
                for col in numeric_cols
            ]
            result["insights"] = ["Correlation matrix of all numeric columns."]
            return result

        # ── Aggregated charts (bar / line / pie / box) ─────────────────────
        if not x_col:
            raise ValueError("No suitable x column found.")

        if y_col and col_types.get(y_col) == "numeric":
            # Group by x, aggregate y
            if col_types.get(x_col) == "datetime":
                # Resample by month
                tmp = df.set_index(x_col).sort_index()
                grouped = getattr(tmp[y_col].resample("ME"), agg)()
                labels = [str(d.strftime("%b %Y")) for d in grouped.index]
                values = grouped.tolist()
            else:
                grp = df.groupby(x_col)[y_col]
                agg_fn = getattr(grp, agg)
                grouped = agg_fn().sort_values(ascending=False)
                if chart_type == "line":
                    grouped = agg_fn().sort_index()
                labels = [str(l) for l in grouped.index]
                values = [round(v, 2) for v in grouped.tolist()]

            result["labels"] = labels[:50]  # cap at 50 labels
            result["datasets"] = [{"label": y_col.replace("_", " ").title(), "data": values[:50]}]

            # Insights
            if values:
                max_idx = values.index(max(values))
                min_idx = values.index(min(values))
                col_label = y_col.replace("_", " ").title()
                x_label   = x_col.replace("_", " ").title()
                result["insights"] = [
                    f"📈 Highest {col_label}: {labels[max_idx]} "
                    f"({agg.title()} = {max(values):,.2f})",
                    f"📉 Lowest {col_label}: {labels[min_idx]} "
                    f"({agg.title()} = {min(values):,.2f})",
                    f"📊 Overall {agg}: {sum(values):,.2f}",
                    f"📐 Average across {len(labels)} {x_label} groups: "
                    f"{sum(values)/len(values):,.2f}",
                ]
        else:
            # Just count occurrences of x_col categories
            vc = df[x_col].value_counts().head(20)
            result["labels"]   = [str(l) for l in vc.index]
            result["datasets"] = [{"label": "Count", "data": vc.tolist()}]
            result["insights"] = [
                f"Top category: '{vc.index[0]}' with {vc.iloc[0]:,} records.",
            ]

    except Exception as e:
        result["error"] = str(e)
        result["insights"] = [f"⚠️ Could not process query: {e}"]

    return result
