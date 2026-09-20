"""One-shot access to the locked test partition.

The test partition is spent by looking at it, so access is gated and recorded in `config/final_selection.json` (tracked
in git, committed BEFORE the evaluation so the model choices are pre-registered):

  test_unlock_authorized   false until a human sets it (with `authorized_by` / `authorization_note`)
  test_accessed_at         written the moment the test frame is loaded -- any later attempt refuses, whether or not the
                           evaluation finished, so the test set cannot be re-examined after a first look

`load_test_frame()` is the ONLY code path to test households; `lib/identification/data.py` cannot load them.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from .data import _load_frame

ROOT = Path(__file__).resolve().parents[2]
SELECTION_PATH = ROOT / "config/final_selection.json"


class FinalTestLocked(RuntimeError):
    """Raised whenever the test partition may not be loaded."""


def read_selection(path: Path = SELECTION_PATH) -> dict:
    if not Path(path).exists():
        raise FinalTestLocked(f"{path} missing -- run scripts/freeze_identification_selection.py first")
    return json.loads(Path(path).read_text())


def assert_can_load(sel: dict) -> None:
    if not sel.get("test_unlock_authorized"):
        raise FinalTestLocked("test_unlock_authorized is false: a human must authorise the single test evaluation")
    if not sel.get("authorized_by"):
        raise FinalTestLocked("authorized_by is empty: record who authorised the test evaluation")
    if sel.get("test_accessed_at"):
        raise FinalTestLocked(f"the test partition was already accessed at {sel['test_accessed_at']}; it cannot be reused")


def load_test_frame(path: Path = SELECTION_PATH) -> pd.DataFrame:
    """Load test households once. Records the access before returning the data."""
    sel = read_selection(path)
    assert_can_load(sel)
    sel["test_accessed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    Path(path).write_text(json.dumps(sel, indent=2) + "\n")
    return _load_frame(("test",))


def record_evaluation(result_sha256: str, commit: str, path: Path = SELECTION_PATH) -> None:
    sel = read_selection(path)
    sel.update({"test_evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "result_sha256": result_sha256, "evaluated_at_commit": commit})
    Path(path).write_text(json.dumps(sel, indent=2) + "\n")
