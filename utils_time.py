# utils_time.py
from dateutil import parser
from datetime import datetime, timezone
import zoneinfo

def parse_to_utc_iso(ts_str: str) -> str:
    """
    Accepts ISO8601 or human strings like:
      "December 5, 2025 at 2:52:08 PM UTC+7"
    Returns ISO8601 UTC string, e.g. '2025-12-05T07:52:08Z'
    """
    # parser.parse understands "UTC+7" and many human formats
    dt = parser.parse(ts_str)
    # if dt has no tzinfo, assume Asia/Bangkok (UTC+7) — optional policy
    if dt.tzinfo is None:
        try:
            tz = zoneinfo.ZoneInfo("Asia/Bangkok")
            dt = dt.replace(tzinfo=tz)
        except Exception:
            dt = dt.replace(tzinfo=timezone.utc)
    # convert to UTC
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.isoformat().replace("+00:00", "Z")

def iso_to_epoch_ms(iso_utc: str) -> int:
    dt = parser.isoparse(iso_utc)
    epoch_ms = int(dt.timestamp() * 1000)
    return epoch_ms

# example usage:
# iso = parse_to_utc_iso("December 5, 2025 at 2:52:08 PM UTC+7")
# ms = iso_to_epoch_ms(iso)
