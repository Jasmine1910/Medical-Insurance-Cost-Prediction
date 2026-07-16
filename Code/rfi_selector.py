import os
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from base_selector import BaseSelector



class RFISelector(
    BaseSelector
):

    def run(self):

        model = RandomForestRegressor(

            n_estimators=100,
            random_state=42,
            n_jobs=-1

        )

        model.fit(

            self.X_train,
            self.y_train

        )

        df = pd.DataFrame({

            "Feature":
            self.X_train.columns,

            "Importance":
            model.feature_importances_

        })

        df = (

            df

            .sort_values(
                "Importance",
                ascending=False
            )

            .head(15)
            .reset_index(
                drop=True
            )

        )

        df.insert(

            0,
            "Rank",

            range(
                1,
                len(df)+1
            )

        )

        output_file = os.path.join(

            self.fs_folder,
            "RFI_Features.xlsx"

        )

        df.to_excel(

            output_file,
            index=False

        )

        return (
            df["Feature"]
            .tolist()
        )