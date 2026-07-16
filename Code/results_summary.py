import os
import pandas as pd
import matplotlib.pyplot as plt

class ResultsSummary:


    def __init__(

        self,

        result_folder

    ):

        self.result_folder = result_folder

        self.models_file = os.path.join(

            result_folder,

            "Models",

            "Model_Results.xlsx"

        )

        self.fs_file = os.path.join(

            result_folder,

            "Feature_Selection",

            "Feature_Selection_Model_Results.xlsx"

        )

        self.ffa_file = os.path.join(

            result_folder,

            "FFA",

            "FFA_Results.xlsx"

        )

        self.output_file = os.path.join(

            result_folder,

            "Final_Results_Summary.xlsx"

        )

        self.visualization_folder = os.path.join(

            result_folder,

            "Visualization"

        )

        os.makedirs(

            self.visualization_folder,

            exist_ok=True

        )

    # =====================================================
    # MODEL SUMMARY TABLE
    # =====================================================

    def create_model_summary(

        self,

        model_name,

        short_name,

        model_results,
        fs_results

    ):

        summary = {

            "Metric": [

                "RMSE",
                "MAE",
                "R2",
                "Training_Time_s"

            ]

        }

        base_row = (

            model_results[
                model_results["Model"]
                == model_name
            ]

            .iloc[0]

        )

        summary[short_name] = [

            base_row["RMSE"],
            base_row["MAE"],
            base_row["R2"],
            base_row["Training_Time_s"]

        ]

        cv_row = (

            model_results[
                model_results["Model"]
                == model_name
            ]

            .iloc[1]

        )

        summary[f"{short_name}cv"] = [

            cv_row["RMSE"],
            cv_row["MAE"],
            cv_row["R2"],
            cv_row["Training_Time_s"]

        ]

        methods = [

            "SHAP",
            "LASSO",
            "RFI",
            "COMMON"

        ]

        for method in methods:

            row = (

                fs_results[method]

                [

                    fs_results[method]["Model"]

                    == model_name

                ]

                .iloc[0]

            )

            summary[

                f"{short_name}cv+{method}"

            ] = [

                row["RMSE"],
                row["MAE"],
                row["R2"],
                row["Training_Time_s"]

            ]

        return pd.DataFrame(
            summary
        )


    # =====================================================
    # RUN
    # =====================================================

    def run(self):

        model_book = pd.ExcelFile(
            self.models_file
        )

        without_cv = pd.read_excel(

            model_book,

            sheet_name="Without_CV"

        )

        cv = pd.read_excel(

            model_book,

            sheet_name="CV"

        )

        model_results = pd.concat(

            [

                without_cv,

                cv

            ],

            ignore_index=True

        )

        fs_book = pd.ExcelFile(
            self.fs_file
        )

        fs_results = {

            "SHAP":
            pd.read_excel(
                fs_book,
                sheet_name="SHAP"
            ),

            "LASSO":
            pd.read_excel(
                fs_book,
                sheet_name="LASSO"
            ),

            "RFI":
            pd.read_excel(
                fs_book,
                sheet_name="RFI"
            ),

            "COMMON":
            pd.read_excel(
                fs_book,
                sheet_name="COMMON"
            )

        }

        best_ffa = pd.read_excel(

            self.ffa_file,

            sheet_name="Best_FFA"

        )

        overall_ranking = (

            best_ffa

            .sort_values(

                [

                    "R2",
                    "RMSE"

                ],

                ascending=[

                    False,
                    True

                ]

            )

            .reset_index(
                drop=True
            )

        )

        overall_ranking.insert(

            0,

            "Rank",

            range(

                1,

                len(
                    overall_ranking
                ) + 1

            )

        )

        overall_ranking = overall_ranking[

            [

                "Rank",

                "Model",

                "Method",

                "Features",

                "RMSE",

                "MAE",

                "R2",

                "Training_Time_s"

            ]

        ]

        with pd.ExcelWriter(

            self.output_file,

            engine="openpyxl"

        ) as writer:

            lr_summary = (

                self.create_model_summary(

                    "Linear_Regression",

                    "LR",

                    model_results,

                    fs_results

                )

            )

            rf_summary = (

                self.create_model_summary(

                    "Random_Forest",

                    "RF",

                    model_results,

                    fs_results

                )

            )

            xgb_summary = (

                self.create_model_summary(

                    "XGBoost",

                    "XGB",

                    model_results,

                    fs_results

                )

            )

            comparison_sheet = "Model_Comparison"

            current_row = 0

            lr_summary.to_excel(

                writer,

                sheet_name=comparison_sheet,

                startrow=current_row,

                index=False

            )

            current_row += len(lr_summary) + 3

            rf_summary.to_excel(

                writer,

                sheet_name=comparison_sheet,

                startrow=current_row,

                index=False

            )

            current_row += len(rf_summary) + 3

            xgb_summary.to_excel(

                writer,

                sheet_name=comparison_sheet,

                startrow=current_row,

                index=False

            )

            overall_ranking.to_excel(

                writer,

                sheet_name="Best_Model_Ranking",

                index=False

            )

        print(

            "\nResults Summary Completed"

        )

        print(

            f"Saved:\n"

            f"{self.output_file}"

        )

        #self.create_rf_line_charts()

