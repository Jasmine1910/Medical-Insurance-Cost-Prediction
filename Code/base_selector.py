import os
import pandas as pd


class BaseSelector:

    def __init__(

        self,

        X_train,
        y_train,

        result_folder

    ):

        self.X_train = X_train
        self.y_train = y_train

        self.result_folder = result_folder

        self.fs_folder = os.path.join(
            result_folder,
            "Feature_Selection"
        )

        os.makedirs(
            self.fs_folder,
            exist_ok=True
        )