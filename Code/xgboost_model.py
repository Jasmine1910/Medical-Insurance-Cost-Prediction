from xgboost import XGBRegressor

class XGBoostModel:

    def get_model(self):

        return XGBRegressor(

            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )