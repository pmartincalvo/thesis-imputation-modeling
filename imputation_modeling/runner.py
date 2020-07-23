import os
from typing import Union
from concurrent.futures import ProcessPoolExecutor

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from imputation_modeling.data_classes import (
    Config,
    ExperimentDefinition,
    Fold,
    ExperimentResults,
)
from imputation_modeling.models import prepare_model
from imputation_modeling.utils import (
    load_dataset,
    preprocess_dataset,
    obtain_training_folds,
    prepare_buurt_folds,
    get_error_metrics,
    format_many_fold_results,
)


def run_all_experiments(config: Config) -> None:
    """
    Wrapper method to individually run several experiments
    :param config: config object
    :return: None
    """

    if not config.cores_to_use:
        print("Running sequentially")
        for experiment_definition in config.experiment_definitions:
            run_experiment(config, experiment_definition)
        return

    with ProcessPoolExecutor(max_workers=config.cores_to_use) as executor:
        print(f"Running in parallel with {config.cores_to_use} cores")
        for experiment_definition in config.experiment_definitions:
            executor.submit(run_experiment, config, experiment_definition)


def run_experiment(config: Config, experiment_definition: ExperimentDefinition) -> None:
    """
    Entire process of loading and preparing data, training the models, evaluating them and storing the results.
    :param config: config object
    :param experiment_definition: details on data preprocessing and model parameters
    :return: None
    """
    print(f"Starting experiment {experiment_definition.name}")

    dataset = load_dataset(experiment_definition.dataset_file_path)

    dataset = preprocess_dataset(dataset, experiment_definition)

    training_folds = obtain_training_folds(
        dataset,
        experiment_definition.number_of_folds,
        experiment_definition.random_seed,
        experiment_definition.feature_column_names,
        experiment_definition.target_column_name,
        experiment_definition.buurt_divider_column_name,
        experiment_definition.model_per_buurt,
    )

    model = prepare_model(
        experiment_definition.model_definition, experiment_definition.random_seed
    )

    folds_results = []
    for fold in training_folds:
        if experiment_definition.model_per_buurt:
            buurt_folds = prepare_buurt_folds(fold, experiment_definition)

            for buurt_fold in buurt_folds:
                results = run_fold(
                    per_buurt=True,
                    fold=buurt_fold,
                    model=model,
                    experiment_definition=experiment_definition,
                )
                folds_results.append(results)
        else:
            results = run_fold(
                per_buurt=False,
                fold=fold,
                model=model,
                experiment_definition=experiment_definition,
            )
            folds_results.append(results)

    folds_results = tuple(folds_results)

    formated_results = format_many_fold_results(folds_results)

    formated_results.to_csv(
        os.path.join(config.results_path, f"{experiment_definition.name}.csv"),
        index=False,
    )

    print(f"Finished experiment {experiment_definition.name}")


def run_fold(
    per_buurt: bool,
    fold: Fold,
    model: Union[GradientBoostingRegressor, RandomForestRegressor],
    experiment_definition: ExperimentDefinition,
):
    """
    Runs training and evaluation for an specific fold.
    :param per_buurt: Whether the fold only contains data for one buurt.
    :param fold: the fold with the training and testing data
    :param model: model object ready for training
    :param experiment_definition: details on data preprocessing and model parameters
    :return: evaluation results
    """
    model.fit(fold.x_train, fold.y_train)

    y_predicted = model.predict(fold.x_test)

    error_metrics = get_error_metrics(fold.y_test, y_predicted, experiment_definition)

    # if experiment_definition.get_feature_importance:
    #    feature_importance_results = get_feature_importance()

    if per_buurt:
        results = ExperimentResults(
            experiment_definition,
            error_metrics,
            fold.train_size,
            fold.test_size,
            fold.buurt_code,
        )
    else:
        results = ExperimentResults(
            experiment_definition, error_metrics, fold.train_size, fold.test_size
        )

    return results
