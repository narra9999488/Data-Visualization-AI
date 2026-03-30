"""
NLP Query Processor - Interprets natural language queries about data
Uses keyword/pattern matching to understand user intent without external NLP libraries
"""
import re


# ── Intent patterns ──────────────────────────────────────────────────────────
INTENT_PATTERNS = {
    "trend": [
        r"\btrend\b", r"\bover time\b", r"\bby (month|year|date|day|week|quarter)\b",
        r"\btime series\b", r"\bgrowth\b", r"\bprogress\b", r"\btimeline\b",
        r"\bhistorical\b", r"\bmonthly\b", r"\bannual\b", r"\bweekly\b",
    ],
    "compare": [
        r"\bcompar(e|ison|ing)\b", r"\bvs\b", r"\bversus\b", r"\bdifference\b",
        r"\bwhich (is|has|are) (higher|lower|best|worst|most|least)\b",
        r"\brank\b", r"\btop\b", r"\bbottom\b", r"\bhighest\b", r"\blowest\b",
    ],
    "distribution": [
        r"\bdistribut(e|ion)\b", r"\bspread\b", r"\brange\b", r"\bhistogram\b",
        r"\bfrequency\b", r"\bvariation\b", r"\boutlier\b",
    ],
    "proportion": [
        r"\bproportion\b", r"\bpercentage\b", r"\bshare\b", r"\bbreakdown\b",
        r"\bcomposition\b", r"\bpie\b", r"\bportion\b", r"\bcontribution\b",
    ],
    "correlation": [
        r"\bcorrelat(e|ion)\b", r"\brelationship\b", r"\bscatter\b",
        r"\brelat(e|ed|ion)\b", r"\bdepend\b", r"\bimpact\b", r"\beffect\b",
    ],
    "summary": [
        r"\bsummar(y|ize)\b", r"\bstatistic\b", r"\boverview\b", r"\bdescribe\b",
        r"\banalyze\b", r"\binsight\b", r"\bshow (me )?(the )?data\b",
        r"\bwhat (is|are)\b",
    ],
}

AGGREGATION_PATTERNS = {
    "sum":   [r"\btotal\b", r"\bsum\b", r"\badd\b", r"\boverall\b"],
    "mean":  [r"\baverage\b", r"\bmean\b", r"\bavg\b"],
    "count": [r"\bcount\b", r"\bnumber of\b", r"\bhow many\b", r"\bfrequency\b"],
    "max":   [r"\bmaximum\b", r"\bmax\b", r"\bhighest\b", r"\bbest\b", r"\btop\b"],
    "min":   [r"\bminimum\b", r"\bmin\b", r"\blowest\b", r"\bworst\b", r"\bbottom\b"],
}

CHART_OVERRIDES = {
    "bar":       [r"\bbar (chart|graph)\b", r"\bbar\b"],
    "line":      [r"\bline (chart|graph)\b", r"\bline\b"],
    "pie":       [r"\bpie (chart|graph)\b", r"\bpie\b"],
    "scatter":   [r"\bscatter (plot|chart|graph)\b", r"\bscatter\b"],
    "histogram": [r"\bhistogram\b"],
    "heatmap":   [r"\bheatmap\b", r"\bheat map\b", r"\bcorrelation matrix\b"],
    "box":       [r"\bbox plot\b", r"\bboxplot\b", r"\bwhisker\b"],
}


def _match_patterns(text: str, patterns: list[str]) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


def _detect_intent(query: str) -> str:
    for intent, patterns in INTENT_PATTERNS.items():
        if _match_patterns(query, patterns):
            return intent
    return "summary"


def _detect_aggregation(query: str) -> str:
    for agg, patterns in AGGREGATION_PATTERNS.items():
        if _match_patterns(query, patterns):
            return agg
    return "sum"


def _detect_chart_override(query: str) -> str | None:
    for chart_type, patterns in CHART_OVERRIDES.items():
        if _match_patterns(query, patterns):
            return chart_type
    return None


def _find_column_mentions(query: str, columns: list[str]) -> list[str]:
    """Find which dataset columns are mentioned in the query."""
    query_lower = query.lower()
    mentioned = []
    for col in columns:
        col_lower = col.lower()
        col_words = re.sub(r"[_\-]", " ", col_lower)
        if col_lower in query_lower or col_words in query_lower:
            mentioned.append(col)
    return mentioned


def recommend_chart(intent: str, col_types: dict, x_col: str, y_col: str) -> str:
    """Recommend chart type based on intent and column types."""
    x_type = col_types.get(x_col, "unknown")
    y_type = col_types.get(y_col, "unknown")

    if intent == "trend":
        return "line"
    if intent == "distribution":
        return "histogram"
    if intent == "proportion":
        return "pie"
    if intent == "correlation":
        return "scatter"
    if intent == "compare":
        if x_type == "categorical":
            return "bar"
        return "scatter"
    # summary
    if x_type == "datetime":
        return "line"
    if x_type == "categorical" and y_type == "numeric":
        return "bar"
    if x_type == "numeric" and y_type == "numeric":
        return "scatter"
    return "bar"


def parse_query(query: str, columns: list[str], col_types: dict) -> dict:
    """
    Parse a natural language query and return structured instructions.

    Returns:
        {
          intent: str,
          aggregation: str,
          chart_type: str,
          x_col: str | None,
          y_col: str | None,
          filter: str | None,
          mentioned_cols: list[str],
          title: str,
        }
    """
    intent = _detect_intent(query)
    aggregation = _detect_aggregation(query)
    chart_override = _detect_chart_override(query)
    mentioned_cols = _find_column_mentions(query, columns)

    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    datetime_cols = [c for c, t in col_types.items() if t == "datetime"]
    categorical_cols = [c for c, t in col_types.items() if t == "categorical"]

    # Decide x and y columns
    x_col = y_col = None

    if mentioned_cols:
        # Use mentioned columns when possible
        num_mentioned = [c for c in mentioned_cols if col_types.get(c) == "numeric"]
        cat_mentioned = [c for c in mentioned_cols if col_types.get(c) == "categorical"]
        dt_mentioned  = [c for c in mentioned_cols if col_types.get(c) == "datetime"]

        if dt_mentioned:
            x_col = dt_mentioned[0]
        elif cat_mentioned:
            x_col = cat_mentioned[0]
        elif num_mentioned:
            x_col = num_mentioned[0]

        remaining_num = [c for c in num_mentioned if c != x_col]
        if remaining_num:
            y_col = remaining_num[0]
        elif num_mentioned and x_col not in num_mentioned:
            y_col = num_mentioned[0]
        elif numeric_cols:
            y_col = next((c for c in numeric_cols if c != x_col), numeric_cols[0])
    else:
        # Fall back to heuristics
        if intent == "trend" and datetime_cols:
            x_col = datetime_cols[0]
            y_col = numeric_cols[0] if numeric_cols else None
        elif intent in ("compare", "proportion") and categorical_cols:
            x_col = categorical_cols[0]
            y_col = numeric_cols[0] if numeric_cols else None
        elif intent == "distribution" and numeric_cols:
            x_col = numeric_cols[0]
        elif intent == "correlation" and len(numeric_cols) >= 2:
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
        else:
            if categorical_cols:
                x_col = categorical_cols[0]
            elif datetime_cols:
                x_col = datetime_cols[0]
            elif numeric_cols:
                x_col = numeric_cols[0]
            y_col = numeric_cols[0] if numeric_cols and x_col not in numeric_cols else (
                numeric_cols[1] if len(numeric_cols) > 1 else None
            )

    chart_type = chart_override or recommend_chart(
        intent, col_types, x_col or "", y_col or ""
    )

    # Build human-readable title
    parts = []
    if y_col:
        agg_label = aggregation.capitalize() if aggregation != "sum" else "Total"
        parts.append(f"{agg_label} {y_col.replace('_', ' ').title()}")
    if x_col:
        parts.append(f"by {x_col.replace('_', ' ').title()}")
    title = " ".join(parts) if parts else "Data Overview"

    return {
        "intent": intent,
        "aggregation": aggregation,
        "chart_type": chart_type,
        "x_col": x_col,
        "y_col": y_col,
        "filter": None,
        "mentioned_cols": mentioned_cols,
        "title": title,
    }
