"""Clear persisted demo events and face enrollments."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.database import SessionLocal, WATCHLIST_DB, init_db
from backend.models import Alert, Event


def main() -> int:
    init_db()
    db = SessionLocal()
    try:
        alerts = db.query(Alert).delete()
        events = db.query(Event).delete()
        db.commit()
    finally:
        db.close()

    watchlist_path = Path(WATCHLIST_DB)
    watchlist_rows = 0
    if watchlist_path.exists():
        with sqlite3.connect(watchlist_path) as watchlist:
            watchlist_rows = watchlist.execute("DELETE FROM watchlist").rowcount
            watchlist.commit()

    print(
        f"[reset] cleared {alerts} alerts, {events} analysis events, "
        f"and {watchlist_rows} watchlist entries"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
