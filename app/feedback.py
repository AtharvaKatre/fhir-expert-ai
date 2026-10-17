"""User feedback capture.

Feedback is appended to a JSONL file together with the orchestration trace it
refers to, so scripts/improve_from_feedback.py can turn it into router
examples and benchmark cases.
"""
import json
import os
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.experts import BASE_DIR

_lock = threading.Lock()
_traces: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
MAX_TRACES = 1000


def feedback_path() -> Path:
    return Path(os.getenv("FEEDBACK_PATH", BASE_DIR / "data" / "feedback.jsonl"))


def remember_trace(trace: Dict[str, Any]) -> None:
    with _lock:
        _traces[trace["trace_id"]] = trace
        while len(_traces) > MAX_TRACES:
            _traces.popitem(last=False)


def get_trace(trace_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        return _traces.get(trace_id)


def record_feedback(entry: Dict[str, Any]) -> Dict[str, Any]:
    record = {
        "received_at": datetime.now(timezone.utc).isoformat(),
        **entry,
        "trace": get_trace(entry["trace_id"]),
    }
    path = feedback_path()
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_feedback() -> List[Dict[str, Any]]:
    path = feedback_path()
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def feedback_summary() -> Dict[str, Any]:
    rows = load_feedback()
    up = sum(1 for r in rows if r.get("rating") == "up")
    return {
        "total": len(rows),
        "thumbs_up": up,
        "thumbs_down": len(rows) - up,
        "with_routing_correction": sum(1 for r in rows if r.get("expected_experts")),
        "with_key_points": sum(1 for r in rows if r.get("expected_key_points")),
    }
