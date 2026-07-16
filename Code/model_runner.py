import os
import time
import shap
import numpy as np
import pandas as pd

from sklearn.metrics import (
mean_absolute_error,
mean_squared_error,
r2_score
)

from model_factory import ModelFactory
from cv_factory import CVFactory

class ModelRunner:

    def __init__(

        self,

        X_train,
        X_test,

        y_train,
        y_test,

        result_folder

    ):

        self.X_train = X_train
        self.X_test = X_test

        self.y_train = y_train
        self.y_test = y_test

        self.result_folder = result_folder

        self.models_folder = os.path.join(
            result_folder,
            "Models"
        )

        self.shap_folder = os.path.join(
            result_folder,
            "SHAP"
        )

        os.makedirs(
            self.models_folder,
            exist_ok=True
        )

        os.makedirs(
            self.shap_folder,
            exist_ok=True
        )

    # ==========================================
    # EVALUATION
    # ==========================================

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

    # ==========================================
    # WITHOUT CV
    # ==========================================

    def run_without_cv(self):

        models = (
            ModelFactory
            .get_models()
        )

        results = []

        for model_name, model in models.items():

            print(
                f"\n{model_name} WITHOUT CV"
            )

            start_time = time.time()

            model.fit(
                self.X_train,
                self.y_train
            )

            training_time = (
                time.time()
                - start_time
            )

            pred = model.predict(
                self.X_test
            )

            metrics = self.evaluate(
                self.y_test,
                pred
            )

            results.append({

                "Model":
                model_name,

                "MAE":
                metrics["MAE"],

                "RMSE":
                metrics["RMSE"],

                "R2":
                metrics["R2"],

                "Training_Time_s":
                round(
                    training_time,
                    4
                )

            })

        return pd.DataFrame(
            results
        )

    # ==========================================
    # CV
    # ==========================================

    def run_cv(self):

        cv_models = (
            CVFactory
            .get_cv_models()
        )

        results = []

        for model_name, cv_model in cv_models.items():

            print(
                f"\n{model_name} 5-Fold CV"
            )

            fold_results = (
                cv_model.run(
                    self.X_train,
                    self.y_train
                )
            )

            fold_df = pd.DataFrame(
                fold_results
            )

            results.append({

                "Model":
                model_name,

                "MAE":
                round(
                    fold_df["MAE"].mean(),
                    2
                ),

                "RMSE":
                round(
                    fold_df["RMSE"].mean(),
                    2
                ),

                "R2":
                round(
                    fold_df["R2"].mean(),
                    4
                ),

                "Training_Time_s":
                round(
                    fold_df[
                        "Training_Time_s"
                    ].mean(),
                    4
                )

            })

        return pd.DataFrame(
            results
        )

    # ==========================================
    # SHAP
    # ==========================================

    def run_shap(self):

        models = (
            ModelFactory
            .get_models()
        )

        shap_models = {

            "Random_Forest":
            models["Random_Forest"],

            "XGBoost":
            models["XGBoost"]

        }

        sample_size = min(
            1000,
            len(self.X_train)
        )

        X_sample = (
            self.X_train
            .sample(
                sample_size,
                random_state=42
            )
        )

        for model_name, model in shap_models.items():

            print(
                f"\nGenerating SHAP: {model_name}"
            )

            model.fit(
                self.X_train,
                self.y_train
            )

            explainer = (
                shap.TreeExplainer(
                    model
                )
            )

            shap_values = (
                explainer.shap_values(
                    X_sample
                )
            )

            importance = np.abs(
                shap_values
            ).mean(axis=0)

            shap_df = pd.DataFrame({

                "Feature":
                X_sample.columns,

                "Importance":
                importance

            })

            shap_df = (

                shap_df

                .sort_values(

                    "Importance",

                    ascending=False

                )

                .reset_index(
                    drop=True
                )

            )

            output_file = os.path.join(

                self.shap_folder,

                f"{model_name}_SHAP_Importance.xlsx"

            )

            shap_df.to_excel(

                output_file,

                index=False

            )

            print(
                f"Saved:\n{output_file}"
            )

    # ==========================================
    # RUN
    # ==========================================

    def run(self):

        without_cv_df = (
            self.run_without_cv()
        )

        cv_df = (
            self.run_cv()
        )

        output_file = os.path.join(

            self.models_folder,

            "Model_Results.xlsx"

        )

        with pd.ExcelWriter(

            output_file,

            engine="openpyxl"

        ) as writer:

            without_cv_df.to_excel(

                writer,

                sheet_name="Without_CV",

                index=False

            )

            cv_df.to_excel(

                writer,

                sheet_name="CV",

                index=False

            )

        print(
            f"\nSaved:\n{output_file}"
        )

        self.run_shap()

        print(
            "\nModel Training Completed"
        )

