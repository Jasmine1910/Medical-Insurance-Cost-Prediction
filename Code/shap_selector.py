import os
import pandas as pd

from base_selector import BaseSelector


class SHAPSelector(
    BaseSelector
):

    def run(self):

        shap_file = os.path.join(

            self.result_folder,

            "SHAP",

            "Random_Forest_SHAP_Importance.xlsx"

        )

        shap_df = pd.read_excel(
            shap_file
        )

        shap_df = (
            shap_df
            .head(15)
            .copy()
        )

        shap_df.insert(

            0,

            "Rank",

            range(
                1,
                len(shap_df)+1
            )

        )

        output_file = os.path.join(

            self.fs_folder,

            "SHAP_Top15_Features.xlsx"

        )

        shap_df.to_excel(

            output_file,

            index=False

        )

        return (
            shap_df["Feature"]
            .tolist()
        )