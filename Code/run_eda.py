import pandas as pd

from eda import EDAAnalyzer

VERSION_FOLDER = "Results/Version_210626"

df = pd.read_csv(
    f"{VERSION_FOLDER}/medical_insurance_cleaned.csv"
)

eda = EDAAnalyzer(
    df=df,
    target="annual_medical_cost",
    result_folder=VERSION_FOLDER
)

eda.run()