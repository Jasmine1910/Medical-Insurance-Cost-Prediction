import os
import time
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from shap_selector import SHAPSelector
from lasso_selector import LASSOSelector
from rfi_selector import RFISelector
from common_selector import CommonSelector

from cv_factory import CVFactory


class FeatureSelectionRunner:

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

        self.fs_folder = os.path.join(

            result_folder,

            "Feature_Selection"

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
    # FEATURE SET EVALUATION
    # ==========================================

    def evaluate_feature_set(

        self,

        feature_list,
        method_name,
        writer

    ):

        models = (
            CVFactory
            .get_cv_models()
        )

        results = []

        if len(feature_list) == 0:

            return

        X_subset = (
            self.X_train[
                feature_list
            ]
        )

        for model_name, model in models.items():

            fold_results = model.run(

                X_subset,

                self.y_train

            )

            fold_df = pd.DataFrame(
                fold_results
            )

            results.append({

                "Model":
                model_name,

                "Features":
                len(feature_list),

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

        pd.DataFrame(
            results
        ).to_excel(

            writer,

            sheet_name=method_name,

            index=False

        )

    # ==========================================
    # RUN
    # ==========================================

    def run(self):

        shap_features = (

            SHAPSelector(

                self.X_train,
                self.y_train,
                self.result_folder

            ).run()

        )

        lasso_features = (

            LASSOSelector(

                self.X_train,
                self.y_train,
                self.result_folder

            ).run()

        )

        rfi_features = (

            RFISelector(

                self.X_train,
                self.y_train,
                self.result_folder

            ).run()

        )

        common_features = (

            CommonSelector(

                self.result_folder

            ).run(

                shap_features,
                lasso_features,
                rfi_features

            )

        )

        output_file = os.path.join(

            self.fs_folder,
            "Feature_Selection_Model_Results.xlsx"

        )

        with pd.ExcelWriter(

            output_file,

            engine="openpyxl"

        ) as writer:

            self.evaluate_feature_set(

                shap_features,
                "SHAP",
                writer

            )

            self.evaluate_feature_set(

                lasso_features,
                "LASSO",
                writer

            )

            self.evaluate_feature_set(

                rfi_features,
                "RFI",
                writer

            )

            self.evaluate_feature_set(

                common_features,
                "COMMON",
                writer

            )

        print(
            "\nFeature Selection Completed"
        )

        print(
            f"Saved:\n{output_file}"
        )

        return {

            "SHAP":
            shap_features,

            "LASSO":
            lasso_features,

            "RFI":
            rfi_features,

            "COMMON":
            common_features

        }