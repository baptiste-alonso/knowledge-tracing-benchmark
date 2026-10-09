"""Generate the ID conversion tables from the raw events export.

For each ID column, this writes a CSV mapping every distinct original ID to a
friendly alias (e.g. ``course_0``, ``course_1``, ...). These tables are then
consumed by :mod:`datas_management.data_cleaning`.

Run from the repository root:

    python -m datas_management.pandas_conversion_table
"""

from pathlib import Path

import pandas as pd
from pandas import DataFrame

DATA_DIR = Path(__file__).resolve().parent

# Maps each ID column to (output file, alias prefix).
ID_COLUMNS = {
    "id_cours": ("conversion_table_courses.csv", "course"),
    "id_test": ("conversion_table_tests.csv", "test"),
    "id_question": ("conversion_table_questions.csv", "question"),
    "id_competence": ("conversion_table_competences.csv", "competence"),
    "id_macrocompetence": ("conversion_table_macro_competences.csv", "macrocompetence"),
}


def id_to_csv(id_name: str, events: DataFrame, file_name, prefix: str) -> None:
    """Write a conversion table mapping each distinct ID to a friendly alias."""
    table = events[[id_name]].dropna().drop_duplicates()
    table["new_" + id_name] = [f"{prefix}_{i}" for i in range(len(table))]
    table.to_csv(file_name, index=False, sep=",")


def main() -> None:
    events = pd.read_csv(DATA_DIR / "events.csv")
    for id_name, (file_name, prefix) in ID_COLUMNS.items():
        id_to_csv(id_name, events, DATA_DIR / file_name, prefix)


if __name__ == "__main__":
    main()