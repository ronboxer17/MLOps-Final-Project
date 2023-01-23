from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split
from config import SEED, DATASETS_NAMES_LABELS, RESULTS_PATH

from xgboost import XGBRegressor

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectFromModel
import numpy as np

REMOVE_STEP = 'need_to_remove_this_step'

class AbstractModel(ABC):
    def __init__(
            self, x_train, x_test, y_train, y_test,
            max_features=None, degree=1, interaction_only=False
        ):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.categorical_cols = [
            i for i, c in enumerate(self.x_train.columns)
            if self.x_train[c].nunique() < 10 and self.x_train[c].dtype == "object"
        ]
        self.numerical_cols = [
            i for i, c in enumerate(self.x_train.columns)
            if self.x_train[c].dtype in ['int64', 'float64']
        ]

        # params for PolynomialFeatures
        self.degree = degree
        self.interaction_only = interaction_only

        # params for SelectFromModel
        self.max_features = max_features

        self.preprocessor = None
        self.pipeline = None
        self.features_pipeline = None

    @abstractmethod
    def fit(self, **kargs):
        raise NotImplemented

    @abstractmethod
    def predict(self, **kargs):
        raise NotImplemented


class xgboost(AbstractModel):
    def __init__(self,
                 x: pd.DataFrame,
                 y: pd.Series,
                 test_size: float = 0.2,
                 is_benchmark: bool = False,
                 max_features: int = None,
                 degree=1,
                 interaction_only=False
                 ):

        self.is_benchmark = is_benchmark
        n_jobs = 5
        self.model = XGBRegressor(n_jobs=n_jobs)
        self.select_model = XGBRegressor(n_jobs=n_jobs)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=SEED)
        super().__init__(x_train, x_test, y_train, y_test, max_features, degree, interaction_only)

    def fit(self):
        numerical_transformer = SimpleImputer(strategy='constant')

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols),

            ])

        steps = [
            ('preprocessor', self.preprocessor),
            REMOVE_STEP if self.is_benchmark\
                else ('poly', PolynomialFeatures(degree=self.degree, interaction_only=self.interaction_only)),
            REMOVE_STEP if self.is_benchmark\
                else ('selector', SelectFromModel(estimator=self.select_model, max_features=self.max_features,
                                         threshold=-np.inf if self.max_features else None
                                         )),
        ]
        steps = [s for s in steps if s != REMOVE_STEP]

        self.features_pipeline = Pipeline(steps=steps)

        x_train_transformed = self.features_pipeline.fit_transform(self.x_train, self.y_train)
        self.model.fit(x_train_transformed, self.y_train)

    def predict(self, **kargs):
        x_test_transformed = self.features_pipeline.transform(self.x_test)
        return self.model.predict(x_test_transformed)



if __name__ == '__main__':
    from utils.data_processing import load_data
    from config import RESULTS_PATH_DICT

    degree = 3
    dataset_name = 'boston_housing'
    label = DATASETS_NAMES_LABELS.get(dataset_name)
    for n_featuers in range(10, 30):
        for interaction_only in [False]:
            print(f'\n\nstaring training:\n{dataset_name=}\n{n_featuers=}\n{degree=}\n{interaction_only=}')
            df = load_data(dataset_name)
            x = df.drop(label, axis=1)
            y = df[label]
            print(f'{x.shape=}')
            try:
                xgb_pipe = xgboost(
                    x,
                    y,
                    is_benchmark=True,
                    degree=degree,
                    interaction_only=interaction_only,
                    max_features=n_featuers
                )
                xgb_pipe.fit()
                y_hat = xgb_pipe.predict()
                y = xgb_pipe.y_test.values
                mse = mean_absolute_error(y_hat, y)
                print(mse)
                break
            except:
                print('asdsa')
            #     results_data = pd.read_csv(RESULTS_PATH_DICT.get(dataset_name))
            #     new_result = {
            #         'dataset_name': dataset_name,
            #         'degree': degree,
            #         'interaction_only': interaction_only,
            #         'max_features': n_featuers,
            #         'is_benchmark': False,
            #         'mse': mse
            #     }
            #     results_data = results_data.append(new_result, ignore_index=True)
            #     results_data.to_csv(RESULTS_PATH_DICT.get(dataset_name), index=False)
            #
            # except:
            #     print(n_featuers, 'didnt worked')
