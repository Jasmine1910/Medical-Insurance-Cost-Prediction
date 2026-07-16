from base_cv import BaseCV

from xgboost import XGBRegressor


class XGBoostCV(
    BaseCV
):

    def run(

        self,
        X,
        y

    ):

        model = XGBRegressor(

            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            verbosity=0

        )

        return self.run_cv(

            model,
            X,
            y

        )