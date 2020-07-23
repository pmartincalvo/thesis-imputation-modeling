from typing import Tuple, List
from math import sqrt

from pandas import DataFrame, Series
import pandas
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import r2_score, mean_squared_error

from imputation_modeling.data_classes import (
    Fold,
    ExperimentDefinition,
    ErrorMetricsResults,
    ExperimentResults,
)


def load_dataset(dataset_path: str) -> DataFrame:
    dataset = pandas.read_csv(dataset_path)
    return dataset


def preprocess_dataset(
    dataset: DataFrame, experiment_definition: ExperimentDefinition
) -> DataFrame:
    """
    Method for preprocessing the dataset
    :param dataset: the raw dataset
    :param experiment_definition: details on data preprocessing and model parameters
    :return: preprocessed dataset
    """

    for column_name in experiment_definition.cols_to_one_hot_encode:
        hot_encoded = pandas.get_dummies(dataset[column_name])
        dataset = dataset.join(hot_encoded)
        experiment_definition.feature_column_names.extend(
            list(hot_encoded.columns.values)
        )
        experiment_definition.feature_column_names.remove(column_name)

    return dataset


def obtain_training_folds(
    dataset: DataFrame,
    number_of_folds: int,
    random_seed: int,
    feature_column_names: List,
    target_column_name: str,
    buurt_column_name: str,
    model_per_buurt: bool,
) -> Tuple[Fold]:
    """
    Method for splitting the dataset in folds
    :param dataset: the dataset
    :param number_of_folds: number of desired folds
    :param random_seed: a random seed to maintain results constant if desired
    :param feature_column_names: columns which should be used as predictive features
    :param target_column_name: column which contains the target data to be predicted
    :param buurt_column_name: column which contains the code of the buurt
    :param model_per_buurt: whether folds need to include a single buurt or not
    :return: all the generated folds
    """

    if model_per_buurt:
        x = dataset[[buurt_column_name, *feature_column_names]]
    else:
        x = dataset[feature_column_names]
    y = dataset[target_column_name]

    folds = []

    if model_per_buurt:
        stratifier = StratifiedKFold(n_splits=number_of_folds, random_state=random_seed)
        fold_indices = stratifier.split(
            x, x[buurt_column_name]
        )
    else:
        stratifier = KFold(n_splits=number_of_folds, random_state=random_seed)
        fold_indices = stratifier.split(x)

    for train_indices, test_indices in fold_indices:
        x_train, x_test = (x.iloc[train_indices], x.iloc[test_indices])
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        fold = Fold(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
        folds.append(fold)

    return tuple(folds)


def prepare_buurt_folds(
    input_fold: Fold, experiment_definition: ExperimentDefinition
) -> Tuple[Fold]:
    observations = (
        input_fold.x_test.groupby(experiment_definition.buurt_divider_column_name)
        .size()
        .reset_index()
    )

    f = observations[0] > 25
    buurtcodes = observations[f].buurtcode.values

    buurt_folds = []

    for buurtcode in buurtcodes:
        train_indices = input_fold.x_train.buurtcode == buurtcode
        test_indices = input_fold.x_test.buurtcode == buurtcode

        buurt_x_train = input_fold.x_train[train_indices][experiment_definition.feature_column_names]
        buurt_x_test = input_fold.x_test[test_indices][experiment_definition.feature_column_names]
        buurt_y_train = input_fold.y_train[train_indices]
        buurt_y_test = input_fold.y_test[test_indices]

        buurt_folds.append(
            Fold(
                x_train=buurt_x_train,
                x_test=buurt_x_test,
                y_train=buurt_y_train,
                y_test=buurt_y_test,
                is_buurt_fold=True,
                buurt_code=buurtcode,
            )
        )

    return tuple(buurt_folds)


def get_error_metrics(
    y_test: Series, y_predicted: Series, experiment_definition: ExperimentDefinition
):
    r2 = None
    if experiment_definition.get_r2:
        r2 = r2_score(y_test, y_predicted)

    rmse = None
    if experiment_definition.get_rmse:
        mse = mean_squared_error(y_test, y_predicted)
        rmse = sqrt(mse)

    return ErrorMetricsResults(r2=r2, rmse=rmse)


def get_feature_importance():
    pass


def format_many_fold_results(folds_results: Tuple[ExperimentResults]) -> DataFrame:
    all_folds_results = pandas.concat(
        [result.return_tabular() for result in folds_results]
    )

    return all_folds_results
