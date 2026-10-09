"""Resolve friendly IDs (e.g. ``competence_0``) back to their real titles.

Setup (call once at the top of your script):

    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "datas_management"))

    import name_from_id
    name_from_id.init()

Usage:

    name_from_id.get_course_title("course_0")      # -> real course title
    name_from_id.get_macro_title("macro_0")        # -> real macro-competence title
    name_from_id.get_competence_title("competence_0")  # -> real competence title
"""

import csv
import json
from pathlib import Path

_HERE = Path(__file__).parent

_data: list = []
_course_conv: dict = {}
_macro_conv: dict = {}
_competence_conv: dict = {}


def _load_csv(path, friendly_col, abstract_col):
    mapping = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row[friendly_col].strip()] = row[abstract_col].strip()
    return mapping


def init(
    data_json: str = "abstraction.json",
    course_csv: str = "conversion_table_courses.csv",
    macro_csv: str = "conversion_table_macro_competences.csv",
    competence_csv: str = "conversion_table_competences.csv",
):
    """Load all lookup files into memory. Call this once before any getter."""
    global _data, _course_conv, _macro_conv, _competence_conv

    with open(_HERE / data_json, encoding="utf-8") as f:
        _data = json.load(f)
    _course_conv = _load_csv(_HERE / course_csv, "new_id_cours", "id_cours")
    _macro_conv = _load_csv(_HERE / macro_csv, "new_id_macrocompetence", "id_macrocompetence")
    _competence_conv = _load_csv(_HERE / competence_csv, "new_id_competence", "id_competence")


def get_course_title(friendly_id: str) -> str:
    """Resolve a friendly course ID (e.g. ``course_0``) to its real title."""
    abstract_id = _course_conv.get(friendly_id)
    if abstract_id is None:
        return f"[ERROR] '{friendly_id}' not found in course conversion table."
    for record in _data:
        if record.get("id_cours") == abstract_id:
            return record["titre_cours"]
    return f"[ERROR] Abstract id_cours '{abstract_id}' not found in data."


def get_macro_title(friendly_id: str) -> str:
    """Resolve a friendly macro-competence ID to its real title."""
    abstract_id = _macro_conv.get(friendly_id)
    if abstract_id is None:
        return f"[ERROR] '{friendly_id}' not found in macro conversion table."
    for record in _data:
        if record.get("id_macrocompetence") == abstract_id:
            return record["titre_macrocompetence"]
    return f"[ERROR] Abstract id_macrocompetence '{abstract_id}' not found in data."


def get_competence_title(friendly_id: str) -> str:
    """Resolve a friendly competence ID to its real title."""
    abstract_id = _competence_conv.get(friendly_id)
    if abstract_id is None:
        return f"[ERROR] '{friendly_id}' not found in competence conversion table."
    for record in _data:
        if record.get("id_competence") == abstract_id:
            return record["titre_competence"]
    return f"[ERROR] Abstract id_competence '{abstract_id}' not found in data."