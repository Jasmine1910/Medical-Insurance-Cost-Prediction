import os
import pandas as pd
import matplotlib.pyplot as plt


class VisualizationRunner:

    def __init__(

        self,

        result_folder

    ):

        self.result_folder = result_folder

        self.ffa_file = os.path.join(

            result_folder,

            "FFA",

            "FFA_Results.xlsx"

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
    # RF COMPARISON CHARTS
    # =====================================================

    def create_rf_comparison_charts(self):

        methods = {

            "SHAP":
            "SHAP_Random_Forest",

            "LASSO":
            "LASSO_Random_Forest",

            "RFI":
            "RFI_Random_Forest",

            "COMMON":
            "COMMON_Random_Forest"

        }

        metrics = [

            "MAE",
            "RMSE",
            "R2",
            "Training_Time_s"

        ]

        data = {}

        for method, sheet in methods.items():

            data[method] = pd.read_excel(

                self.ffa_file,

                sheet_name=sheet

            )

        for metric in metrics:

            plt.figure(

                figsize=(12, 6)

            )

            for method, df in data.items():

                plt.plot(

                    df["Features"],

                    df[metric],

                    marker="o",

                    linewidth=2,

                    label=method

                )

            plt.title(

                f"Random Forest Comparison - {metric}"

            )

            plt.xlabel(

                "Number of Features"

            )

            plt.ylabel(

                metric

            )

            plt.legend()

            plt.grid(True)

            plt.tight_layout()

            output_file = os.path.join(

                self.visualization_folder,

                f"RF_Comparison_{metric}.png"

            )

            plt.savefig(

                output_file,

                dpi=1200

            )

            plt.close()

            print(

                f"Saved: {output_file}"

            )

    # =====================================================
    # RF INDIVIDUAL CHARTS
    # =====================================================

    def create_rf_individual_charts(self):

        methods = {

            "SHAP":
            "SHAP_Random_Forest",

            "LASSO":
            "LASSO_Random_Forest",

            "RFI":
            "RFI_Random_Forest",

            "COMMON":
            "COMMON_Random_Forest"

        }

        metrics = [

            "MAE",
            "RMSE",
            "R2",
            "Training_Time_s"

        ]

        for method, sheet in methods.items():

            df = pd.read_excel(

                self.ffa_file,

                sheet_name=sheet

            )

            for metric in metrics:

                plt.figure(

                    figsize=(12, 6)

                )

                plt.plot(

                    df["Features"],

                    df[metric],

                    marker="o",

                    linewidth=2

                )

                # =====================================
                # Point Labels
                # =====================================

                for x, y in zip(

                    df["Features"],

                    df[metric]

                ):

                    if metric == "R2":

                        label = f"{y:.4f}"

                    elif metric == "Training_Time_s":

                        label = f"{y:.3f}"

                    else:

                        label = f"{y:.2f}"

                    plt.annotate(

                        label,

                        (x, y),

                        textcoords="offset points",

                        xytext=(0, 10),

                        ha="center",

                        fontsize=8

                    )

                plt.title(

                    f"Random Forest + {method} - {metric}"

                )

                plt.xlabel(

                    "Number of Features"

                )

                plt.ylabel(

                    metric

                )

                plt.grid(True)

                plt.tight_layout()

                output_file = os.path.join(

                    self.visualization_folder,

                    f"RF_{method}_{metric}.png"

                )

                plt.savefig(

                    output_file,

                    dpi=1200

                )

                plt.close()

                print(

                    f"Saved: {output_file}"

                )

    # =====================================================
    # RF SHAP vs LASSO vs RFI Dashboard
    # =====================================================

    def create_rf_fs_dashboard(self):

        plt.rcParams["font.family"] = "Times New Roman"

        methods = {

            "SHAP":
            "SHAP_Random_Forest",

            "LASSO":
            "LASSO_Random_Forest",

            "RFI":
            "RFI_Random_Forest"

        }

        metrics = [

            "RMSE",
            "MAE",
            "R2",
            "Training_Time_s"

        ]

        data = {}

        for method, sheet in methods.items():

            data[method] = pd.read_excel(

                self.ffa_file,

                sheet_name=sheet

            )

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(10, 6)
        )

        axes = axes.flatten()
        display_names = {
            "MAE": "MAE",
            "RMSE": "RMSE",
            "R2": "R²",
            "Training_Time_s": "Training Time"
        }

        y_labels = {
            "MAE": "MAE",
            "RMSE": "RMSE",
            "R2": "R²",
            "Training_Time_s": "Training Time (s)"
        }

        for idx, metric in enumerate(metrics):

            ax = axes[idx]

            for method, df in data.items():

                ax.plot(
                df["Features"],
                df[metric],
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=method
            )

            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax * 1.15)

            ax.set_title(
                display_names[metric],
                fontsize=11,
                fontweight="bold"
            )

            ax.set_xlabel(
                "Number of Features",
                fontsize=9
            )

            ax.set_ylabel(
                y_labels[metric],
                fontsize=9
            )

            # =====================================
            # Remove Grid
            # =====================================

            ax.grid(False)

            # =====================================
            # Keep Only Left & Bottom Border
            # =====================================

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.spines["left"].set_linewidth(1.2)
            ax.spines["bottom"].set_linewidth(1.2)

            # =====================================
            # Tick Style
            # =====================================

            ax.tick_params(
                axis="both",
                labelsize=9,
                direction="out",
                length=3,
                width=0.8
            )

            ax.legend(
                fontsize=9,
                frameon=False,
                loc="best"
            )

        plt.suptitle(
            "Random Forest Performance Using Forward Feature Addition",
            fontsize=12,
            fontweight="bold"
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        output_file = os.path.join(

            self.visualization_folder,

            "RF_FS_Dashboard.png"

        )

        plt.savefig(
            output_file.replace(".png", ".svg"),
            dpi=1200,
            bbox_inches="tight"
        )

        plt.close()

        print(

            f"Saved: {output_file}"

        )

    # =====================================================
    # RUN
    # =====================================================

    def run(self):

        print("\nGenerating Visualizations...")

        self.create_rf_comparison_charts()
        self.create_rf_individual_charts()
        self.create_rf_fs_dashboard()

        print("\nVisualization Completed")