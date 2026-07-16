import os
import pandas as pd


class CommonSelector:

    def __init__(

        self,
        result_folder

    ):

        self.result_folder = result_folder
        self.fs_folder = os.path.join(

            result_folder,
            "Feature_Selection"

        )

    def run(

        self,

        shap_features,
        lasso_features,
        rfi_features

    ):

        common = list(

            set(shap_features)
            &
            set(lasso_features)

            &
            set(rfi_features)

        )

        df = pd.DataFrame({

            "Feature":
            common

        })

        output_file = os.path.join(

            self.fs_folder,
            "Common_Features.xlsx"

        )

        df.to_excel(

            output_file,
            index=False
        )

        return common