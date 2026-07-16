from linear_regression_model import LinearRegressionModel
from random_forest_model import RandomForestModel
from xgboost_model import XGBoostModel


class ModelFactory:

    @staticmethod
    def get_models():

        return {

            "Linear_Regression":

            LinearRegressionModel()
            .get_model(),

            "Random_Forest":

            RandomForestModel()
            .get_model(),

            "XGBoost":

            XGBoostModel()
            .get_model()

        }