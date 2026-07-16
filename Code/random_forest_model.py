from sklearn.ensemble import RandomForestRegressor

class RandomForestModel:

    def get_model(self):

        return RandomForestRegressor(

            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )