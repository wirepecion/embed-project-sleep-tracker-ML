# utils_time.py
from dateutil import parser
from datetime import timezone
import zoneinfo

def parse_to_utc_iso(ts_str: str) -> str:
    """
    Accepts ISO8601 or human strings like:
      "December 5, 2025 at 2:52:08 PM UTC+7"
    Returns ISO8601 UTC string, e.g. '2025-12-05T07:52:08Z'
    """
    dt = parser.parse(ts_str)
    if dt.tzinfo is None:
        try:
            tz = zoneinfo.ZoneInfo("Asia/Bangkok")
            dt = dt.replace(tzinfo=tz)
        except Exception:
            dt = dt.replace(tzinfo=timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.replace(tzinfo=None).isoformat() + "Z"
