# DataLens AI — Data Visualization Assistant

## Architecture

```
ai_viz_assistant/
├── app.py               # Flask application & REST API
├── data_processor.py    # Data loading, cleaning, analysis
├── nlp_processor.py     # NL query understanding (keyword NLP)
├── run.py               # Entry point
├── requirements.txt
├── templates/
│   └── index.html       # Full single-page frontend
├── static/              # Static assets (if any)
├── uploads/             # Uploaded & sample datasets
│   ├── sample_sales.csv
│   └── sample_students.csv
└── instance/
    └── datasets.db      # SQLite metadata store
```

## Quick Start

```bash
pip install -r requirements.txt
python run.py
# Open http://127.0.0.1:5000
```

## API Endpoints

| Method | Endpoint          | Description                        |
|--------|-------------------|------------------------------------|
| POST   | /api/upload       | Upload CSV/Excel dataset           |
| GET    | /api/summary      | Dataset statistics & auto-insights |
| POST   | /api/query        | Natural language query → chart     |
| GET    | /api/columns      | Column names & types               |
| POST   | /api/custom_chart | Manual chart builder               |
| GET    | /api/history      | Upload history (SQLite)            |

## NLP Query Examples

- "Show sales trend by month" → Line chart
- "Compare profit by region" → Bar chart
- "Which category has the highest revenue?" → Bar chart
- "Show distribution of GPA" → Histogram
- "Proportion of sales by category" → Pie chart
- "Correlation between sales and profit" → Scatter plot
- "Average score by department" → Bar chart

## Chart Types

| Data Pattern          | Chart      |
|-----------------------|------------|
| Time + Numeric        | Line       |
| Category + Numeric    | Bar        |
| Category proportion   | Pie        |
| Numeric distribution  | Histogram  |
| Numeric vs Numeric    | Scatter    |
| Multi-numeric         | Heatmap    |

## Technologies

- **Backend**: Python 3.11, Flask 3.x
- **Data**: Pandas, NumPy
- **Database**: SQLite (via sqlite3)
- **NLP**: Custom keyword/regex pattern matching
- **Frontend**: Vanilla HTML/CSS/JS + Chart.js 4
- **Fonts**: Google Fonts (Syne, DM Mono, DM Sans)
