import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def cleaning_df(df):
    """Rename and reformat raw events into the pyKT input schema."""
    df = df.drop(
        columns=[
            "id_event",
            "temps",
            "id_cours",
            "id_test",
            "id_macrocompetence",
        ]
    )
    df = df.rename(
        columns={
            "id_eleve": "uid",
            "id_question": "qid",
            "id_competence": "skill",
            "bonne_reponse": "response",
        }
    )
    df = str_id_to_int(df, "qid")
    df = str_id_to_int(df, "skill")
    df = df.sort_values(["uid", "timestamp"])
    return df


def str_id_to_int(df, column):
    """Extract the integer part of a string identifier column."""
    df[column] = df[column].str.extract(r"(\d+)", expand=False).astype(int)
    return df


def main():
    df = pd.read_csv(PROJECT_ROOT / "datas_management" / "cleaned_events.csv")
    df = cleaning_df(df)
    out_path = PROJECT_ROOT / "models" / "gkt" / "data" / "pyKT_formate.csv"
    df.to_csv(out_path, index=False, sep=",")
    print(f"Wrote {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()