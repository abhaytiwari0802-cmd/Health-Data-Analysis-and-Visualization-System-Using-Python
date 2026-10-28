"""
Module: history_manager.py
Description: Manages local analysis history using JSON files.
             Stores metadata, summary metrics, cleaning logs, and insights.
             Strictly avoids persisting raw patient personal data.
"""

import os
import json
import datetime
from typing import List, Dict, Any, Optional

HISTORY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "history")


def _ensure_history_dir():
    """Creates the history directory if it does not already exist."""
    os.makedirs(HISTORY_DIR, exist_ok=True)


def save_history_entry(
    dataset_name: str,
    record_count: int,
    column_count: int,
    audit_log: Optional[Dict[str, Any]] = None,
    key_metrics: Optional[Dict[str, Any]] = None,
    insights_summary: Optional[List[str]] = None
) -> str:
    """
    Saves an analysis snapshot metadata entry to local storage.
    Returns the generated entry ID.
    """
    _ensure_history_dir()
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate totals from audit log safely
    audit = audit_log or {}
    imputed_map = audit.get("imputed_details", {})
    imputed_count = sum(d.get("imputed_count", 0) for d in imputed_map.values() if isinstance(d, dict))
    
    entry_data = {
        "entry_id": entry_id,
        "timestamp": timestamp,
        "dataset_name": dataset_name,
        "record_count": record_count,
        "column_count": column_count,
        "duplicates_removed": audit.get("duplicates_removed", 0),
        "imputation_strategy": audit.get("imputation_strategy", "N/A"),
        "values_imputed": imputed_count,
        "bmi_recalculated": audit.get("bmi_recalculated_count", 0),
        "key_metrics": key_metrics or {},
        "top_insights": insights_summary or []
    }
    
    file_path = os.path.join(HISTORY_DIR, f"analysis_{entry_id}.json")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(entry_data, f, indent=2)
        return entry_id
    except Exception as e:
        print(f"Error saving analysis history: {e}")
        return ""


def load_history_entries() -> List[Dict[str, Any]]:
    """
    Loads all previous analysis history entries from disk, sorted descending by date.
    """
    _ensure_history_dir()
    entries = []
    
    if not os.path.exists(HISTORY_DIR):
        return entries
        
    for fname in os.listdir(HISTORY_DIR):
        if fname.startswith("analysis_") and fname.endswith(".json"):
            fpath = os.path.join(HISTORY_DIR, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    entries.append(data)
            except Exception as e:
                print(f"Skipping corrupt history file {fname}: {e}")
                
    entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return entries


def delete_history_entry(entry_id: str) -> bool:
    """Deletes an individual history file by entry ID."""
    _ensure_history_dir()
    file_path = os.path.join(HISTORY_DIR, f"analysis_{entry_id}.json")
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return True
        except Exception:
            return False
    return False


def clear_all_history() -> int:
    """Clears all saved history entries. Returns count of deleted entries."""
    _ensure_history_dir()
    deleted = 0
    for fname in os.listdir(HISTORY_DIR):
        if fname.startswith("analysis_") and fname.endswith(".json"):
            try:
                os.remove(os.path.join(HISTORY_DIR, fname))
                deleted += 1
            except Exception:
                pass
    return deleted


def export_history_json() -> str:
    """Exports all history entries as a single JSON formatted string."""
    entries = load_history_entries()
    return json.dumps(entries, indent=2)
