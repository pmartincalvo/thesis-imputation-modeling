from typing import Union

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from imputation_modeling.data_classes import ModelDefinition


def prepare_model(
    model_definition: ModelDefinition, random_seed: int
) -> Union[GradientBoostingRegressor, RandomForestRegressor]:
    """
    Prepares a model ready to be trained according to the passed definition.
    :param model_definition: a data class with the model parameters
    :param random_seed: a random seed to maintain results constant if desired
    :return: the model object ready to be trained
    """

    model_parameters = {
        "n_estimators": model_definition.n_estimators,
        "max_depth": model_definition.max_depth,
        "max_features": model_definition.max_features_method,
        "random_state": random_seed,
    }

    if model_definition.regressor == RandomForestRegressor:
        model_parameters["n_jobs"] = model_definition.n_jobs

    model = model_definition.regressor(**model_parameters)

    return model
