import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from config import BOSTON_PATH, FRENCH_PATH, DATASETS_NAMES_LABELS



def _load_boston() -> pd.DataFrame:
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    return pd.read_csv(BOSTON_PATH, header=None, delimiter=r"\s+", names=column_names)


def _load_french() -> pd.DataFrame:
    return pd.read_csv(FRENCH_PATH)


def load_data(dataset_name: str) -> pd.DataFrame:
    if dataset_name not in DATASETS_NAMES_LABELS:
        print('no such data set')

    if dataset_name == 'boston_housing':
        return _load_boston()
    elif dataset_name == 'french_motor_claims':
        return _load_french()




