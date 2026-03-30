"""Entry point — initialise DB and start Flask dev server."""
from data_processor import init_db
from app import app

if __name__ == "__main__":
    init_db()
    print("\n" + "="*55)
    print("  DataLens AI — Visualization Assistant")
    print("  Open: http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=True, port=5000, use_reloader=False)
