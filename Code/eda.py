import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


class EDAAnalyzer:

    plt.rcParams["font.family"] = "Times New Roman"
    def __init__(
        self,
        df,
        target,
        result_folder
    ):

        self.df = df
        self.target = target

        self.eda_folder = os.path.join(
            result_folder,
            "EDA"
        )

        self.visual_folder = os.path.join(
            result_folder,
            "Visualization"
        )

        os.makedirs(
            self.eda_folder,
            exist_ok=True
        )

        os.makedirs(
            self.visual_folder,
            exist_ok=True
        )

    # =====================================================
    # EDA REPORT
    # =====================================================

    def create_eda_report(self):

        output_file = os.path.join(
            self.eda_folder,
            "EDA_Report.xlsx"
        )

        overview_df = pd.DataFrame({

            "Metric": [

                "Rows",
                "Columns",
                "Missing Values",
                "Duplicate Rows"

            ],

            "Value": [

                self.df.shape[0],
                self.df.shape[1],
                self.df.isnull().sum().sum(),
                self.df.duplicated().sum()

            ]

        })

        dtype_df = pd.DataFrame({

            "Feature":
            self.df.columns,

            "Data_Type":
            self.df.dtypes.astype(str)

        })

        missing_df = pd.DataFrame({

            "Feature":
            self.df.columns,

            "Missing_Count":
            self.df.isnull().sum(),

            "Missing_Percentage":
            round(
                self.df.isnull().mean() * 100,
                2
            )

        })

        numerical_df = (

            self.df
            .select_dtypes(
                include=np.number
            )
            .describe()
            .T
            .round(2)

        )

        categorical_cols = (

            self.df
            .select_dtypes(
                exclude=np.number
            )
            .columns

        )

        categorical_summary = []

        for col in categorical_cols:

            temp = (

                self.df[col]
                .value_counts()
                .reset_index()

            )

            temp.columns = [

                "Category",
                "Count"

            ]

            temp.insert(
                0,
                "Feature",
                col
            )

            categorical_summary.append(
                temp
            )

        if len(categorical_summary) > 0:

            categorical_df = pd.concat(
                categorical_summary,
                ignore_index=True
            )

        else:

            categorical_df = pd.DataFrame(
                columns=[
                    "Feature",
                    "Category",
                    "Count"
                ]
            )

        corr_matrix = (

            self.df
            .select_dtypes(
                include=np.number
            )
            .corr()
            .round(4)

        )

        target_corr = (

            corr_matrix[
                self.target
            ]
            .sort_values(
                ascending=False
            )
            .to_frame(
                "Correlation"
            )

        )

        with pd.ExcelWriter(
            output_file,
            engine="openpyxl"
        ) as writer:

            overview_df.to_excel(
                writer,
                sheet_name="Dataset_Overview",
                index=False
            )

            dtype_df.to_excel(
                writer,
                sheet_name="Data_Types",
                index=False
            )

            missing_df.to_excel(
                writer,
                sheet_name="Missing_Values",
                index=False
            )

            numerical_df.to_excel(
                writer,
                sheet_name="Numerical_Summary"
            )

            categorical_df.to_excel(
                writer,
                sheet_name="Categorical_Summary",
                index=False
            )

            corr_matrix.to_excel(
                writer,
                sheet_name="Correlation_Matrix"
            )

            target_corr.to_excel(
                writer,
                sheet_name="Target_Correlation"
            )

        print(
            f"EDA Report Saved:\n{output_file}"
        )

    # =====================================================
    # HELPER
    # =====================================================

    def save_histogram(
        self,
        column,
        title,
        filename,
        color="skyblue",
        bins=30,
        kde=False,
        edgecolor="black",
        linewidth=1.5
    ):

        plt.figure(figsize=(10, 6))

        sns.histplot(
            self.df[column],
            bins=bins,
            kde=kde,
            color=color,
            edgecolor=edgecolor,
            linewidth=linewidth
        )

        plt.title(title, fontsize=24, fontweight="bold")

        if column == "annual_medical_cost":
            plt.xlabel("Annual Medical Cost", fontsize=20)
            plt.ylabel("Number of Policy Holders", fontsize=20)
        else:
            plt.xlabel(column, fontsize=20)
            plt.ylabel("Number of Policy Holders", fontsize=20)

        ax = plt.gca()
        ax.tick_params(labelsize=18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

        plt.tight_layout()

        # Original figure
        plt.savefig(
            os.path.join(self.visual_folder, filename),
            dpi=300
        )

        # Extra zoomed figure for Annual Medical Cost
        if column == "annual_medical_cost":

            plt.xlim(0, 15000)

            plt.savefig(
                os.path.join(
                    self.visual_folder,
                    "annual_medical_cost_lim15000.svg"
                ),
                dpi=1200
            )

        plt.close()

    def save_barplot(
        self,
        column,
        title,
        filename,
        color="mediumpurple"
    ):

        plt.figure(figsize=(10, 6))

        sns.countplot(
            data=self.df,
            x=column,
            color=color,
            edgecolor="black"
        )

        plt.title(title, fontsize=24)
        plt.xlabel(column, fontsize=18)
        plt.ylabel("count", fontsize=18)

        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.savefig(
            os.path.join(self.visual_folder, filename),
            dpi=300
        )

        plt.close()

    # =====================================================
    # CORRELATION HEATMAP
    # =====================================================

    def correlation_heatmap(self):

        corr = (
            self.df
            .select_dtypes(include=np.number)
            .corr()
            .abs()
        )

        plt.figure(figsize=(18, 12))

        sns.heatmap(
            corr,
            cmap="RdPu_r",
            vmin=0,
            vmax=1,
            annot=False
        )

        plt.title(
            "Correlation Heatmap",
            fontsize=16
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.visual_folder,
                "Correlation_Heatmap.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # =====================================================
    # TOP 15 CORRELATION
    # =====================================================

    def top_target_correlation(self):

        corr = (
            self.df
            .select_dtypes(include=np.number)
            .corr()[self.target]
            .abs()
            .sort_values(
                ascending=False
            )
            .head(15)
            .to_frame()
        )

        plt.figure(figsize=(8, 10))

        sns.heatmap(
            corr,
            annot=True,
            cmap="RdPu_r",
            vmin=0,
            vmax=1
        )

        plt.title(
            "Top 15 Correlation with Annual Medical Costs",
            fontsize=18
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.visual_folder,
                "Top_15_Target_Correlation_Heatmap.png"
            ),
            dpi=300
        )

        plt.close()

    # =====================================================
    # CHRONIC ILLNESS
    # =====================================================

    def chronic_illness_distribution(self):

        plt.figure(figsize=(10, 6))

        sns.countplot(
            x="chronic_count",
            data=self.df,
            color="yellowgreen",
            edgecolor="black"
        )

        plt.title(
            "Chronic Illness Count",
            fontsize=24
        )

        plt.xlabel(
            "illness_count",
            fontsize=18
        )

        plt.ylabel(
            "count",
            fontsize=18
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.visual_folder,
                "Chronic_Illness_Count.png"
            ),
            dpi=300
        )

        plt.close()

    # =====================================================
    # MEDICAL COST VS PREMIUM DISTRIBUTION
    # =====================================================

    def cost_premium_distribution(self):

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(10, 12)
        )

        # Annual Medical Cost
        sns.histplot(
            self.df["annual_medical_cost"],
            bins=30,
            kde=True,
            color="khaki",
            edgecolor="black",
            linewidth=1.2,
            ax=axes[0]
        )

        axes[0].set_title(
            "Annual Medical Cost Distribution",
            fontsize=14
        )

        axes[0].set_xlabel(
            "Annual Medical Insurance Cost",
            fontsize=8
        )

        axes[0].set_ylabel(
            "Count",
            fontsize=8
        )

        # Annual Premium
        sns.histplot(
            self.df["annual_premium"],
            bins=30,
            kde=True,
            color="#a8d672",
            edgecolor="black",
            linewidth=1.2,
            ax=axes[1]
        )

        axes[1].set_title(
            "Annual Insurance Premium Distribution",
            fontsize=20
        )

        axes[1].set_xlabel(
            "Annual Insurance Premium",
            fontsize=14
        )

        axes[1].set_ylabel(
            "Count",
            fontsize=14
        )

        # Remove top/right border only
        for ax in axes:

            ax.spines["top"].set_visible(False)

            ax.spines["right"].set_visible(False)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.visual_folder,
                "Cost_Premium_Distribution.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # =====================================================
    # EDA ONLY
    # =====================================================

    def run_eda(self):

        self.create_eda_report()

        print("\nEDA Completed")


    # =====================================================
    # VISUALIZATION ONLY
    # =====================================================

    def run_visualization(self):

        self.correlation_heatmap()

        self.top_target_correlation()

        self.save_histogram(
            "annual_premium",
            "Annual Insurance Premium Distribution",
            "Insurance_Premium_Distribution.png",
            color="#a8d672"
        )

        self.save_histogram(
            "annual_medical_cost",
            "Annual Medical Cost Distribution",
            "Medical_Cost_Distribution.png",
            color="khaki"
        )

        self.save_histogram(
            "age",
            "Age Distribution",
            "Age_Distribution.png",
            color="thistle"
        )

        self.save_histogram(
            "income",
            "Annual Income Distribution",
            "Annual_Income_Distribution.png",
            color="darkkhaki"
        )

        self.save_histogram(
            "bmi",
            "BMI Distribution",
            "BMI_Distribution.png",
            color="#e8c68f"
        )

        self.save_histogram(
            "systolic_bp",
            "Systolic Blood Pressure Distribution",
            "Systolic_BP_Distribution.png",
            color="#e88d98"
        )

        self.save_histogram(
            "diastolic_bp",
            "Diastolic Blood Pressure Distribution",
            "Diastolic_BP_Distribution.png",
            color="#73e0c4"
        )

        self.save_histogram(
            "days_hospitalized_last_3yrs",
            "Days Hospitalized",
            "Days_Hospitalized_Distribution.png",
            color="#3d8b7d",
            kde=False
        )

        self.save_histogram(
            "visits_last_year",
            "Doctor Visits Last Year",
            "Doctor_Visits_Distribution.png",
            color="royalblue",
            kde=False
        )

        self.cost_premium_distribution()

        self.save_barplot(
            "region",
            "Region Distribution",
            "Region_Distribution.png",
            color="lightblue"
        )

        self.save_barplot(
            "education",
            "Education Level Distribution",
            "Education_Level_Distribution.png",
            color="mediumpurple"
        )

        self.save_barplot(
            "employment_status",
            "Employment Status Distribution",
            "Employment_Status_Distribution.png",
            color="plum"
        )

        self.save_barplot(
            "sex",
            "Gender Distribution",
            "Gender_Distribution.png",
            color="darkseagreen"
        )

        self.chronic_illness_distribution()

        print("\nVisualization Completed")


    # =====================================================
    # RUN ALL
    # =====================================================

    def run(self):

        self.run_eda()

        self.run_visualization()