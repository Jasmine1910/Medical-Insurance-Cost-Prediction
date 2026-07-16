from base_cv import BaseCV

from sklearn.linear_model import (
    LinearRegression
)


class LinearRegressionCV(
    BaseCV
):

    def run(

        self,
        X,
        y

    ):

        model = LinearRegression()

        return self.run_cv(

            model,
            X,
            y

        )