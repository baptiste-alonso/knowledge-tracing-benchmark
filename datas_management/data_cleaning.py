"""Clean the raw events export and replace opaque IDs with friendly ones.

Run from the repository root:

    python -m datas_management.data_cleaning
"""

from pathlib import Path

import pandas as pd
from pandas import DataFrame

DATA_DIR = Path(__file__).resolve().parent

# Maps each ID column to the conversion table that holds its friendly aliases.
CONVERSION_TABLES = {
    "id_cours": "conversion_table_courses.csv",
    "id_test": "conversion_table_tests.csv",
    "id_question": "conversion_table_questions.csv",
    "id_competence": "conversion_table_competences.csv",
    "id_macrocompetence": "conversion_table_macro_competences.csv",
}


def clean_data(df: DataFrame) -> DataFrame:
    """Drop rows with missing values and exact duplicates."""
    return df.dropna().drop_duplicates()


def create_mapping(conversion_table: DataFrame) -> dict:
    """Build an {original_id: friendly_id} mapping from a two-column table."""
    original_col, friendly_col = conversion_table.columns[:2]
    return dict(zip(conversion_table[original_col], conversion_table[friendly_col]))


def rename_id(df: DataFrame, id_name: str, conversion_table: DataFrame) -> DataFrame:
    """Replace the values of a single ID column using its conversion table."""
    df[id_name] = df[id_name].map(create_mapping(conversion_table))
    return df


def rename_all_ids(df: DataFrame) -> DataFrame:
    """Replace every known ID column with its friendly alias."""
    for id_name, table_name in CONVERSION_TABLES.items():
        conversion_table = pd.read_csv(DATA_DIR / table_name)
        df = rename_id(df, id_name, conversion_table)
    return df


def save_cleaned_data(df: DataFrame, file_name: str) -> None:
    df.to_csv(file_name, index=False, sep=",")


def main() -> None:
    events = pd.read_csv(DATA_DIR / "events.csv")
    events = events[events["id_eleve"] != "cleverlearn"]
    cleaned_events = clean_data(events)
    cleaned_events = rename_all_ids(cleaned_events)
    save_cleaned_data(cleaned_events, DATA_DIR / "cleaned_events.csv")


if __name__ == "__main__":
    main()