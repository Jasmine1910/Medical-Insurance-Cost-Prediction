import os
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
StandardScaler,
OneHotEncoder
)

class InsurancePreprocessor:

    def __init__(
        self,
        dataset_path,
        target="annual_medical_cost",
        test_size=0.30,
        random_state=42,
        result_folder="RESULTS"
    ):

        self.dataset_path = dataset_path
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        
        today = datetime.now().strftime("%d%m%y")

        self.result_folder = os.path.join(
            result_folder,
            f"Version_{today}"
        )

        os.makedirs(
            self.result_folder,
            exist_ok=True
        )

        print(
            f"\nResults Folder:\n"
            f"{self.result_folder}"
)

    # ==========================================
    # LOAD DATA
    # ==========================================

    def load_data(self):

        print("\nLoading Dataset...")

        self.df = pd.read_csv(
            self.dataset_path
        )

        print(
            f"Dataset Shape: "
            f"{self.df.shape}"
        )

        return self.df

    # ==========================================
    # REMOVE LEAKAGE FEATURES
    # ==========================================

    def remove_leakage_features(
        self,
        leakage_features=None
    ):

        if leakage_features is None:

            leakage_features = []

        existing_features = [

            col for col in leakage_features
            if col in self.df.columns

        ]

        if existing_features:

            print(
                "\nRemoving Leakage Features:"
            )

            print(existing_features)

            self.df.drop(
                columns=existing_features,
                inplace=True
            )

        return self.df
    
    # ==========================================
    # DATA CLEANING
    # ==========================================

    def clean_data(self):

        print("\nCleaning Dataset...")

        # Replace alcohol='None' with 'Never'

        if "alcohol_freq" in self.df.columns:

            self.df["alcohol_freq"] = self.df["alcohol_freq"].replace(
                "None",
                "Never"
            )

            print(
                "Replaced alcohol_freq: "
                "'None' -> 'Never'"
            )

        # ======================================
        # REMOVE person_id
        # ======================================

        if "person_id" in self.df.columns:

            self.df.drop(
                columns=["person_id"],
                inplace=True
            )

            print(
                "Removed column: person_id"
            )

        # ======================================
        # SAVE CLEANED DATASET
        # ======================================

        cleaned_file = os.path.join(
            self.result_folder,
            "medical_insurance_cleaned.csv"
        )

        self.df.to_csv(
            cleaned_file,
            index=False
        )

        print(
            f"Cleaned dataset saved:\n"
            f"{cleaned_file}"
        )

        return self.df
    # ==========================================
    # SPLIT DATA
    # ==========================================

    def split_data(self):

        X = self.df.drop(
            columns=[self.target]
        )

        y = self.df[self.target]

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test

        ) = train_test_split(

            X,
            y,

            test_size=self.test_size,
            random_state=self.random_state

        )

        print(
            "\nTrain Shape:",
            self.X_train.shape
        )

        print(
            "Test Shape:",
            self.X_test.shape
        )

        return (

            self.X_train,
            self.X_test,

            self.y_train,
            self.y_test

        )

    # ==========================================
    # PREPROCESS DATA
    # ==========================================

    def preprocess(self):

        numeric_features = (

            self.X_train
            .select_dtypes(
                include=np.number
            )
            .columns

        )

        categorical_features = (

            self.X_train
            .select_dtypes(
                exclude=np.number
            )
            .columns

        )

        self.preprocessor = (

            ColumnTransformer(

                transformers=[

                    (

                        "num",

                        StandardScaler(),

                        numeric_features

                    ),

                    (

                        "cat",

                        OneHotEncoder(
                            handle_unknown="ignore"
                        ),

                        categorical_features

                    )

                ]

            )

        )

        X_train_processed = (

            self.preprocessor
            .fit_transform(
                self.X_train
            )

        )

        X_test_processed = (

            self.preprocessor
            .transform(
                self.X_test
            )

        )

        feature_names = (

            self.preprocessor
            .get_feature_names_out()

        )

        self.X_train_processed_df = (

            pd.DataFrame(

                X_train_processed,

                columns=feature_names

            )

        )

        self.X_test_processed_df = (

            pd.DataFrame(

                X_test_processed,

                columns=feature_names

            )

        )

        print(
            "\nProcessed Train Shape:",
            self.X_train_processed_df.shape
        )

        print(
            "Processed Test Shape:",
            self.X_test_processed_df.shape
        )

        return (

            self.X_train_processed_df,
            self.X_test_processed_df,

            self.y_train,
            self.y_test,

            feature_names,
            self.preprocessor

        )

    # ==========================================
    # SAVE PROCESSED DATA
    # ==========================================

    def save_processed_data(self):

        output_file = os.path.join(

            self.result_folder,

            "Processed_Data.xlsx"

        )

        with pd.ExcelWriter(
            output_file,
            engine="openpyxl"
        ) as writer:

            self.X_train_processed_df.to_excel(

                writer,

                sheet_name="X_Train",

                index=False

            )

            self.X_test_processed_df.to_excel(

                writer,

                sheet_name="X_Test",

                index=False

            )

            self.y_train.to_frame(
                name=self.target
            ).to_excel(

                writer,

                sheet_name="y_Train",

                index=False

            )

            self.y_test.to_frame(
                name=self.target
            ).to_excel(

                writer,

                sheet_name="y_Test",

                index=False

            )

        print(
            f"\nProcessed Data Saved:\n"
            f"{output_file}"
        )

    # ==========================================
    # RUN ALL
    # ==========================================

    def run(self):

        self.load_data()

        self.clean_data()

        self.split_data()

        self.preprocess()

        self.save_processed_data()

        return (

            self.X_train_processed_df,
            self.X_test_processed_df,

            self.y_train,
            self.y_test,

            self.preprocessor

        )