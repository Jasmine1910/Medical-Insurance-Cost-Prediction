import os
import warnings
import pandas as pd

from cv_factory import CVFactory

warnings.filterwarnings("ignore")


class ForwardFeatureAddition:

    def __init__(

        self,

        X_train,
        y_train,

        result_folder

    ):

        self.X_train = X_train
        self.y_train = y_train

        self.result_folder = result_folder

        self.fs_folder = os.path.join(
            result_folder,
            "Feature_Selection"
        )

        self.ffa_folder = os.path.join(
            result_folder,
            "FFA"
        )

        os.makedirs(
            self.ffa_folder,
            exist_ok=True
        )

        self.models = (
            CVFactory.get_cv_models()
        )

    # =====================================================
    # FORWARD FEATURE ADDITION
    # =====================================================

    def run_ffa(

        self,

        feature_file,
        cv_model,
        model_name

    ):

        feature_df = pd.read_excel(
            feature_file
        )

        features = (

            feature_df["Feature"]

            .dropna()

            .tolist()

        )

        if len(features) == 0:

            return pd.DataFrame()

        results = []

        for i in range(

            1,

            len(features) + 1

        ):

            selected = (
                features[:i]
            )

            X_subset = (
                self.X_train[
                    selected
                ]
            )

            fold_results = (

                cv_model.run(

                    X_subset,

                    self.y_train

                )

            )

            fold_df = pd.DataFrame(
                fold_results
            )

            results.append({

                "Model":
                model_name,

                "Features":
                i,

                "Last_Feature":
                selected[-1],

                "MAE":
                round(
                    fold_df[
                        "MAE"
                    ].mean(),
                    2
                ),

                "RMSE":
                round(
                    fold_df[
                        "RMSE"
                    ].mean(),
                    2
                ),

                "R2":
                round(
                    fold_df[
                        "R2"
                    ].mean(),
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

    # =====================================================
    # RUN
    # =====================================================

    def run(self):

        files = {

            "SHAP":
            "SHAP_Top15_Features.xlsx",

            "LASSO":
            "LASSO_Features.xlsx",

            "RFI":
            "RFI_Features.xlsx",

            "COMMON":
            "Common_Features.xlsx"

        }

        output_file = os.path.join(

            self.ffa_folder,

            "FFA_Results.xlsx"

        )

        best_results = []

        with pd.ExcelWriter(

            output_file,

            engine="openpyxl"

        ) as writer:

            for method, filename in files.items():

                feature_file = os.path.join(

                    self.fs_folder,

                    filename

                )

                if not os.path.exists(
                    feature_file
                ):

                    print(
                        f"Missing: {filename}"
                    )

                    continue

                for model_name, model in self.models.items():

                    print(
                        f"{method} + {model_name}"
                    )

                    result_df = self.run_ffa(

                        feature_file,

                        model,

                        model_name

                    )

                    if result_df.empty:

                        continue

                    result_df.to_excel(

                        writer,

                        sheet_name=
                        f"{method}_{model_name}",

                        index=False

                    )

                    best_row = result_df.loc[
                        result_df[
                            "R2"
                        ].idxmax()
                    ]

                    best_results.append({

                        "Method":
                        method,

                        "Model":
                        model_name,

                        "Features":
                        int(
                            best_row[
                                "Features"
                            ]
                        ),

                        "Last_Feature":
                        best_row[
                            "Last_Feature"
                        ],

                        "MAE":
                        best_row[
                            "MAE"
                        ],

                        "RMSE":
                        best_row[
                            "RMSE"
                        ],

                        "R2":
                        best_row[
                            "R2"
                        ],

                        "Training_Time_s":
                        best_row[
                            "Training_Time_s"
                        ]

                    })

                    print(

                        f"Best -> "

                        f"{method} + "

                        f"{model_name} | "

                        f"{best_row['Features']} "

                        f"features | "

                        f"R2={best_row['R2']}"

                    )

            if len(best_results) > 0:

                best_df = pd.DataFrame(
                    best_results
                )

                best_df = (

                    best_df

                    .sort_values(

                        "R2",

                        ascending=False

                    )

                )

                best_df.to_excel(

                    writer,

                    sheet_name=
                    "Best_FFA",

                    index=False

                )

        print(
            "\nFFA Completed"
        )

        print(
            f"Saved:\n{output_file}"
        )