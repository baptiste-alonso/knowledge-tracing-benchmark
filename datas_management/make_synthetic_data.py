"""Generate a synthetic `cleaned_events.csv` with the same schema as the real data.

The Cleverlearn data is private and is not part of this repository. This script
simulates students who learn (ability grows with practice) and forget (gains
fade between sessions), so that every model of the benchmark can be run end to
end. Results on this data are only a sanity check.

Run from the repository root:

    python datas_management/make_synthetic_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "cleaned_events.csv"


def simulate(n_students=200, n_courses=3, macros_per_course=2, skills_per_macro=5,
             items_per_skill=8, sessions=12, answers_per_session=12, seed=0):
    rng = np.random.default_rng(seed)
    n_macro = n_courses * macros_per_course
    n_skills = n_macro * skills_per_macro
    n_items = n_skills * items_per_skill

    item_skill = np.repeat(np.arange(n_skills), items_per_skill)
    item_difficulty = rng.normal(0.0, 1.0, n_items)
    learn_rate = rng.uniform(0.05, 0.25, n_skills)
    forget_rate = 0.03  # per day

    t0 = 1_756_000_000
    rows, event = [], 0
    for s in range(n_students):
        ability = rng.normal(0.0, 0.8) + rng.normal(0.0, 0.4, n_skills)
        gain, last_seen = np.zeros(n_skills), np.zeros(n_skills)
        t = t0 + rng.uniform(0, 5) * 86400
        for session in range(sessions):
            t += rng.exponential(4.0) * 86400
            focus = rng.choice(n_skills, size=3, replace=False)
            for _ in range(answers_per_session):
                k = int(rng.choice(focus))
                j = int(rng.choice(np.flatnonzero(item_skill == k)))
                macro = k // skills_per_macro
                days = (t - last_seen[k]) / 86400 if last_seen[k] else 0.0
                gain[k] *= np.exp(-forget_rate * days)
                p = 1 / (1 + np.exp(-(ability[k] + gain[k] - item_difficulty[j] + 0.8)))
                correct = int(rng.random() < p)
                rows.append({
                    "id_event": f"ev-{event}",
                    "timestamp": int(t),
                    "id_eleve": s,
                    "id_cours": f"course_{macro // macros_per_course}",
                    "id_test": f"test_{s}_{session}",
                    "id_question": f"question_{j}",
                    "bonne_reponse": correct,
                    "temps": int(np.clip(rng.lognormal(3.0, 0.6), 2, 600)),
                    "id_competence": f"competence_{k}",
                    "id_macrocompetence": f"macrocompetence_{macro}",
                })
                event += 1
                gain[k] += learn_rate[k] * (1.0 if correct else 0.5)
                last_seen[k] = t
                t += rng.uniform(20, 120)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = simulate()
    df.to_csv(OUT, index=False)
    print(f"{len(df)} synthetic interactions written to {OUT}")
