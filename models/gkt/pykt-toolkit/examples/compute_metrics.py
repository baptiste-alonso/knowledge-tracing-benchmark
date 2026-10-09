import numpy as np
from sklearn.metrics import log_loss, mean_squared_error
import ast

filepath = "saved_model/my_dataset_gkt_qid_saved_model_42_0_0.01_0.5_64_transition_0_0_64/qid_test_predictions.txt"

all_ts, all_ps = [], []

with open(filepath, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data = ast.literal_eval(f"[{line}]")[0]
        all_ts.extend(data[1])
        all_ps.extend(data[4])

all_ts = np.array(all_ts)
all_ps = np.array(all_ps)

rmse    = np.sqrt(mean_squared_error(all_ts, all_ps))
logloss = log_loss(all_ts, all_ps)

print(f"RMSE     : {rmse:.4f}")
print(f"LogLoss  : {logloss:.4f}")
print(f"Samples  : {len(all_ts)}")