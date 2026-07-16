from linear_regression_cv import LinearRegressionCV

from random_forest_cv import RandomForestCV

from xgboost_cv import XGBoostCV


class CVFactory:

    @staticmethod
    def get_cv_models():

        return {

            "Linear_Regression":
            LinearRegressionCV(),

            "Random_Forest":
            RandomForestCV(),

            "XGBoost":
            XGBoostCV()

        }