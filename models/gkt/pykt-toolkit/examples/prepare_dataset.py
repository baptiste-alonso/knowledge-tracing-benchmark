import pandas as pd
import os

df = pd.read_csv("../data/my_dataset/my_dataset.csv")
df = df.sort_values(["uid", "timestamp"])

output_lines = []

for uid, group in df.groupby("uid"):
    seqlen = len(group)
    question_ids = ",".join(group["qid"].astype(str).tolist())
    concept_ids  = ",".join(group["skill"].astype(str).tolist())
    responses    = ",".join(group["response"].astype(str).tolist())
    timestamps   = ",".join(group["timestamp"].astype(str).tolist())

    output_lines.append(f"{uid},{seqlen}")
    output_lines.append(question_ids)
    output_lines.append(concept_ids)
    output_lines.append(responses)
    output_lines.append(timestamps)
    output_lines.append("NA")

os.makedirs("../data/my_dataset", exist_ok=True)
with open("../data/my_dataset/data.txt", "w") as f:
    f.write("\n".join(output_lines))

print(f"data.txt généré : {len(df['uid'].unique())} utilisateurs")