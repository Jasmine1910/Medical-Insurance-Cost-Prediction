from preprocessing import InsurancePreprocessor
from eda import EDAAnalyzer

from model_runner import ModelRunner
from feature_selection_runner import FeatureSelectionRunner
from forward_feature_addition import ForwardFeatureAddition

from results_summary import ResultsSummary
from visualization_runner import VisualizationRunner

import os

# ==========================================
# DATASET
# ==========================================

dataset_path = os.path.join(
    "Dataset",
    "medical_insurance.csv"
)

# ==========================================
# PREPROCESSING
# ==========================================

processor = InsurancePreprocessor(

    dataset_path=dataset_path,

    target="annual_medical_cost",

    result_folder="Results"

)

X_train_processed_df, \
X_test_processed_df, \
y_train, \
y_test, \
preprocessor = processor.run()

VERSION_FOLDER = processor.result_folder

# ==========================================
# EDA
# ==========================================

eda = EDAAnalyzer(
    df=processor.df,
    target="annual_medical_cost",
    result_folder=VERSION_FOLDER
)

eda.run()

# ==========================================
# MODELS + CV + SHAP
# ==========================================

model_runner = ModelRunner(

    X_train=X_train_processed_df,
    X_test=X_test_processed_df,

    y_train=y_train,
    y_test=y_test,

    result_folder=VERSION_FOLDER

)

model_runner.run()

# ==========================================
# FEATURE SELECTION
# ==========================================

fs = FeatureSelectionRunner(

    X_train=X_train_processed_df,
    X_test=X_test_processed_df,

    y_train=y_train,
    y_test=y_test,

    result_folder=VERSION_FOLDER

)

selected_features = fs.run()

# ==========================================
# FORWARD FEATURE ADDITION
# ==========================================

ffa = ForwardFeatureAddition(

    X_train=X_train_processed_df,

    y_train=y_train,

    result_folder=VERSION_FOLDER

)

ffa.run()

print(
    "\nALL EXPERIMENTS COMPLETED"
)

summary = ResultsSummary(

    result_folder=VERSION_FOLDER

)

summary.run()

viz = VisualizationRunner(

    VERSION_FOLDER

)

viz.run()