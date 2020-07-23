import uuid
from typing import Union, Tuple, List
from dataclasses import dataclass

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from pandas import DataFrame, Series


@dataclass()
class ModelDefinition:
    regressor: Union[GradientBoostingRegressor, RandomForestRegressor]
    n_estimators: int
    max_depth: int
    max_features_method: str
    n_jobs: int = 1


@dataclass()
class ExperimentDefinition:
    dataset_file_path: str
    target_column_name: str
    buurt_divider_column_name: str
    feature_column_nickname: str
    feature_column_names: List[str]
    cols_to_one_hot_encode: List[str]
    number_of_folds: int
    get_feature_importance: bool
    random_seed: int
    get_rmse: bool
    get_r2: bool

    model_definition: ModelDefinition

    name: str = None
    model_per_buurt: bool = True
    id: str = str(uuid.uuid4())


@dataclass()
class Config:
    experiment_definitions: Tuple[ExperimentDefinition]
    results_path: str
    cores_to_use: Union[int, None]


@dataclass()
class Fold:
    x_train: DataFrame
    x_test: Series
    y_train: DataFrame
    y_test: Series
    is_buurt_fold: bool = False
    buurt_code: str = None
    train_size: int = None
    test_size: int = None

    def __post_init__(self):
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]


@dataclass()
class ErrorMetricsResults:
    r2: float = None
    rmse: float = None


@dataclass()
class FeatureImportanceResults:
    pass


@dataclass()
class ExperimentResults:

    experiment_definition: ExperimentDefinition
    error_metrics: ErrorMetricsResults
    train_size: int
    test_size: int
    buurtcode: str = None
    stadsdeel_code: str = None
    # feature_importance: FeatureImportanceResults

    def __post_init__(self):
        if self.buurtcode:
            self.stadsdeel_code = self.buurtcode[0]

    def return_tabular(self):
        performance_results = {
            "id": self.experiment_definition.id,
            "name": self.experiment_definition.name,
            "buurtcode": self.buurtcode,
            "feature_column_nickname": self.experiment_definition.feature_column_nickname,
            "regressor": self.experiment_definition.model_definition.regressor.__name__,
            "n_estimators": self.experiment_definition.model_definition.n_estimators,
            "max_depth": self.experiment_definition.model_definition.max_depth,
            "max_features_method": self.experiment_definition.model_definition.max_features_method,
            "r2": self.error_metrics.r2,
            "rmse": self.error_metrics.rmse,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "stadsdeel_code": self.stadsdeel_code,
        }

        performance_results_df = DataFrame(data=[performance_results])

        return performance_results_df
