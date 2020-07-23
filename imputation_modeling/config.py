import json

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from imputation_modeling.data_classes import (
    Config,
    ExperimentDefinition,
    ModelDefinition,
)


def load_config(config_file_path: str) -> Config:
    """
    Loads the configuration from a json file.
    :param config_file_path: path to json file
    :return: config data class
    """
    with open(config_file_path, "r") as config_file:
        config_data = json.load(config_file)

    all_experiments = []
    for experiment_definition_path in config_data["experiments_paths"]:
        all_experiments.append(load_experiment(experiment_definition_path))

    all_experiments = tuple(all_experiments)

    return Config(
        experiment_definitions=all_experiments,
        results_path=config_data["results_path"],
        cores_to_use=config_data["cores_to_use"],
    )


def load_experiment(experiment_definition_path: str) -> ExperimentDefinition:
    """
    Loads an experiment definition from a json file.
    :param experiment_definition_path: path to json file
    :return: experiment definition data class
    """
    with open(experiment_definition_path, "r") as experiment_file:
        experiment_data = json.load(experiment_file)

    definition = ExperimentDefinition(
        dataset_file_path=experiment_data["dataset_file_path"],
        model_per_buurt=experiment_data["model_per_buurt"],
        target_column_name=experiment_data["target_column_name"],
        buurt_divider_column_name=experiment_data["buurt_dividider_column_name"],
        feature_column_nickname=experiment_data["feature_column_nickname"],
        feature_column_names=experiment_data["feature_column_names"],
        cols_to_one_hot_encode=experiment_data["cols_to_one_hot_encode"],
        number_of_folds=experiment_data["number_of_folds"],
        get_feature_importance=experiment_data["get_feature_importance"],
        random_seed=experiment_data["random_seed"],
        get_rmse=experiment_data["get_rmse"],
        get_r2=experiment_data["get_r2"],
        model_definition=load_model_definition(experiment_data["model_definition"]),
    )

    if "name" in experiment_data:
        definition.name = experiment_data["name"]
    if "model_per_buurt" in experiment_data:
        definition.model_per_buurt = experiment_data["model_per_buurt"]

    return definition


def load_model_definition(model_definition_data: dict) -> ModelDefinition:
    """
    Loads a model definition from a dictionary.
    :param model_definition_data: dictionary with the data
    :return: model definition data class
    """
    regressor_string_to_object = {
        "gbm": GradientBoostingRegressor,
        "rf": RandomForestRegressor,
    }

    definition = ModelDefinition(
        regressor=regressor_string_to_object[model_definition_data["regressor"]],
        n_estimators=model_definition_data["n_estimators"],
        max_depth=model_definition_data["max_depth"],
        max_features_method=model_definition_data["max_features_method"],
    )

    if "n_jobs" in model_definition_data:
        definition.n_jobs = model_definition_data["n_jobs"]

    return definition
