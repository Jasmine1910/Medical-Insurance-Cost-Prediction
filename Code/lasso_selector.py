import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LassoCV
from base_selector import BaseSelector

class LASSOSelector(
    BaseSelector
):

    def run(self):

        model = LassoCV(

            cv=5,
            random_state=42,
            n_jobs=-1

        )

        model.fit(

            self.X_train,
            self.y_train

        )

        importance = np.abs(
            model.coef_
        )

        df = pd.DataFrame({

            "Feature":
            self.X_train.columns,

            "Importance":
            importance

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
            "LASSO_Features.xlsx"

        )

        df.to_excel(

            output_file,

            index=False

        )

        return (
            df["Feature"]
            .tolist()
        )