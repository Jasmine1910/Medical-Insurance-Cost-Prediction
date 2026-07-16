from base_cv import BaseCV

from sklearn.ensemble import  RandomForestRegressor


class RandomForestCV(
    BaseCV
):

    def run(

        self,
        X,
        y

    ):

        model = RandomForestRegressor(

            n_estimators=100,
            random_state=42,
            n_jobs=-1

        )

        return self.run_cv(

            model,
            X,
            y

        )