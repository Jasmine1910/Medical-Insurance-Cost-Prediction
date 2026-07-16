from sklearn.model_selection import KFold

import numpy as np
import time

from sklearn.metrics import (

    mean_absolute_error,
    mean_squared_error,
    r2_score

)


class BaseCV:

    def evaluate(

        self,
        y_true,
        y_pred

    ):

        return {

        "MAE":
        round(
            mean_absolute_error(
                y_true,
                y_pred
            ),
            2
        ),

        "RMSE":
        round(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred
                )
            ),
            2
        ),

        "R2":
        round(
            r2_score(
                y_true,
                y_pred
            ),
            4
        )

    }

    def run_cv(

        self,
        model,
        X,
        y

    ):

        kf = KFold(

            n_splits=5,
            shuffle=True,
            random_state=42

        )

        results = []

        fold = 1

        for train_idx, test_idx in kf.split(X):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            start_time = time.time()

            model.fit(
                X_train,
                y_train
            )

            training_time = (
                time.time()
                - start_time
            )

            pred = model.predict(
                X_test
            )

            metrics = self.evaluate(
                y_test,
                pred
            )

            metrics["Fold"] = fold

            metrics[
                "Training_Time_s"
            ] = round(
                training_time,
                4
            )

            results.append(
                metrics
            )

            fold += 1

        return results