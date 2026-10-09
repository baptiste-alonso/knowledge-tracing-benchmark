import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss


class BKTVectorized:
    def __init__(
        self,
        df: pd.DataFrame,
        prediction_df: pd.DataFrame,
        mapping: dict,
        initial_parameters: dict,
    ):
        self.mapping_to_model(mapping)
        self.parameters_dict = initial_parameters  # order [p_0, p_T, p_G, p_S]
        self.df = df.sort_values([self.student_id, self.timestamp])
        self.prediction_df = prediction_df
        self._prepare_data()

    def mapping_to_model(self, mapping):
        self.student_id = mapping["student_id"]
        self.timestamp = mapping["timestamp"]
        self.skill_id = mapping["skill_id"]
        self.result = mapping["result"]

    def _prepare_data(self):
        """Pre-process: group the data once."""
        self.grouped_data = {}

        for skill in self.df[self.skill_id].unique():
            df_skill = self.df[self.df[self.skill_id] == skill]
            self.grouped_data[skill] = {}

            for student in df_skill[self.student_id].unique():
                results = df_skill[df_skill[self.student_id] == student][
                    self.result
                ].values
                self.grouped_data[skill][student] = results

    def em_algorithm(self, iterations: int, convergence_threshold=0) -> None:
        for iteration in range(iterations):
            l_dict = self.e_step()
            new_params = self.m_step(l_dict)

            converged = True
            for skill in new_params:
                if (
                    np.max(np.abs(self.parameters_dict[skill] - new_params[skill]))
                    > convergence_threshold
                ):
                    converged = False
                self.parameters_dict[skill] = new_params[skill]

            if converged and iteration > 5:
                print(f"Convergence reached at iteration {iteration}")
                break

    def e_step(self) -> dict:
        """Vectorized E-step."""
        l_dict = {}

        for skill in self.grouped_data:
            l_dict[skill] = {}

            for student, results_array in self.grouped_data[skill].items():
                l_forward_list, l_pred_list = self.forward_pass_vectorized(
                    results_array, skill
                )
                l_backward_list = self.backward_pass_vectorized(
                    l_forward_list, l_pred_list, results_array
                )
                l_dict[skill][student] = l_backward_list

        return l_dict

    def m_step(self, l_dict) -> dict:
        """Vectorized M-step."""
        new_params = {}

        for skill in l_dict:
            (
                enlt0,
                enlt1,
                enct1lt0,
                enct1lt1,
                enct0lt1,
                enct0lt0,
                learned_transitions,
            ) = self.expected_counts_vectorized(l_dict[skill], skill)

            new_prior, new_learn, new_guess, new_slip = self.estimation_vectorized(
                enlt0,
                enct1lt0,
                enct1lt1,
                enct0lt1,
                enct0lt0,
                learned_transitions,
                l_dict[skill],
            )

            new_params[skill] = np.array([new_prior, new_learn, new_guess, new_slip])

        return new_params

    def forward_pass_vectorized(self, results_array: np.ndarray, skill: str) -> tuple:
        """Forward pass: loop over time (state depends on the past)."""
        params = self.parameters_dict[skill]
        p_0, p_T, p_G, p_S = params

        T = len(results_array)
        l_forward_list = np.zeros(T + 1)
        l_pred_list = np.zeros(T)

        l_forward_list[0] = p_0
        l_forward = p_0

        for t in range(T):
            l_pred = l_forward + (1 - l_forward) * p_T
            l_pred_list[t] = l_pred

            if results_array[t] == 1:
                l_forward = ((1 - p_S) * l_pred) / (
                    l_pred * (1 - p_S) + (1 - l_pred) * p_G
                )
            else:
                l_forward = (p_S * l_pred) / (l_pred * p_S + (1 - l_pred) * (1 - p_G))

            l_forward_list[t + 1] = l_forward

        return l_forward_list, l_pred_list

    def backward_pass_vectorized(
        self,
        l_forward_list: np.ndarray,
        l_pred_list: np.ndarray,
        results_array: np.ndarray,
    ) -> list:
        """Backward pass with numerical guarding."""
        T = len(results_array)
        l_backward_list = [(l_forward_list[-1], results_array[-1])]
        l_backward = l_forward_list[-1]

        epsilon = 1e-10

        for t in range(T - 1, 0, -1):
            l_forward = l_forward_list[t]
            l_pred = l_pred_list[t - 1]

            if abs(l_pred) < epsilon or abs(1 - l_pred) < epsilon:
                if abs(l_pred) < epsilon:
                    l_backward = (
                        (1 - l_forward) * (1 - l_backward) / (1 - l_pred + epsilon)
                    )
                else:
                    l_backward = l_forward * l_backward / (l_pred + epsilon)
            else:
                l_backward = l_forward * l_backward / (l_pred + epsilon) + (
                    1 - l_forward
                ) * (1 - l_backward) / (1 - l_pred + epsilon)

            l_backward = np.clip(l_backward, 0, 1)

            l_backward_list.append((l_backward, results_array[t - 1]))

        return l_backward_list

    def expected_counts_vectorized(self, l_dict_skill: dict, skill: str) -> tuple:
        """Expected counts (vectorized)."""
        params = self.parameters_dict[skill]
        p_G, p_S, p_T = params[2], params[3], params[1]

        all_l_bwd = []
        all_results = []

        for student in l_dict_skill:
            for l_bwd, result in l_dict_skill[student]:
                all_l_bwd.append(l_bwd)
                all_results.append(result)

        l_bwd_array = np.array(all_l_bwd)
        results_array = np.array(all_results)

        enlt0 = np.sum(1 - l_bwd_array)
        enlt1 = np.sum(l_bwd_array)

        mask_correct = results_array == 1
        mask_incorrect = results_array == 0

        enct1lt0 = np.sum((1 - l_bwd_array[mask_correct]) * p_G)
        enct1lt1 = np.sum(l_bwd_array[mask_correct] * (1 - p_S))
        enct0lt1 = np.sum(l_bwd_array[mask_incorrect] * p_S)
        enct0lt0 = np.sum((1 - l_bwd_array[mask_incorrect]) * (1 - p_G))
        learned_transitions = np.sum((1 - l_bwd_array) * p_T)

        return (
            enlt0,
            enlt1,
            enct1lt0,
            enct1lt1,
            enct0lt1,
            enct0lt0,
            learned_transitions,
        )

    def estimation_vectorized(
        self,
        enlt0,
        enct1lt0,
        enct1lt1,
        enct0lt1,
        enct0lt0,
        learned_transitions,
        l_dict_skill,
    ) -> tuple:
        new_prior = np.mean(
            [l_dict_skill[s][-1][0] for s in l_dict_skill if len(l_dict_skill[s]) > 0]
        )
        new_prior = np.clip(new_prior, 0.01, 0.99)

        new_learn = (
            learned_transitions / enlt0
            if enlt0 > 0
            else self.parameters_dict.get("learn", 0.15)
        )
        new_learn = np.clip(new_learn, 0.01, 0.99)

        new_guess = (
            enct1lt0 / (enct1lt0 + enct0lt0) if (enct1lt0 + enct0lt0) > 0 else 0.1
        )
        new_guess = np.clip(new_guess, 0.01, 0.99)

        new_slip = (
            enct0lt1 / (enct0lt1 + enct1lt1) if (enct0lt1 + enct1lt1) > 0 else 0.1
        )
        new_slip = np.clip(new_slip, 0.01, 0.99)

        return new_prior, new_learn, new_guess, new_slip

    def predict_with_results(self) -> pd.DataFrame:

        predictions_list = []

        for skill in self.prediction_df[self.skill_id].unique():
            skill_df = self.prediction_df[self.prediction_df[self.skill_id] == skill]

            for student in skill_df[self.student_id].unique():

                student_df = skill_df[skill_df[self.student_id] == student]
                historical_results = student_df[self.result].values
                p_0, p_T, p_G, p_S = self.parameters_dict[skill]

                l_forward_list, _ = self.forward_pass_vectorized(
                    historical_results, skill
                )
                predictions_list.append(
                    {
                        self.student_id: student,
                        self.skill_id: skill,
                        "actual_result": historical_results[0],
                        "predicted_result": p_0,
                    }
                )
                for t in range(len(historical_results) - 1):
                    current_l = l_forward_list[t + 1]

                    l_next = current_l + (1 - current_l) * p_T
                    p_correct = l_next * (1 - p_S) + (1 - l_next) * p_G

                    predictions_list.append(
                        {
                            self.student_id: student,
                            self.skill_id: skill,
                            "actual_result": historical_results[t],
                            "predicted_result": p_correct,
                        }
                    )
        predictions_df = pd.DataFrame(predictions_list)

        print(f"{len(predictions_df)} predictions generated")
        print(f"Columns: {predictions_df.columns.tolist()}")

        return pd.DataFrame(predictions_list)

    def evaluate(self) -> dict:
        """Compute AUC, RMSE and log loss."""

        predictions_df = self.predict_with_results()

        y_true = np.asarray(predictions_df["actual_result"].values, dtype=np.float64)
        y_pred = np.asarray(predictions_df["predicted_result"].values, dtype=np.float64)

        auc = roc_auc_score(y_true, y_pred)

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        log_v = log_loss(y_true, y_pred)

        return {
            "auc": auc,
            "rmse": rmse,
            "log_v": log_v,
        }

    def print_metrics(self):
        metrics = self.evaluate()

        print("=" * 50)
        print("BKT EVALUATION METRICS")
        print("=" * 50)
        print(f"AUC:  {metrics['auc']:.4f}  (0.5=random, 1.0=perfect)")
        print(f"RMSE: {metrics['rmse']:.4f}  (0=perfect)")
        print(f"LOG_LOSS:  {metrics['log_v']:.4f}")
        print("=" * 50)