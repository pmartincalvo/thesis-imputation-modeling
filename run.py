import argparse

from imputation_modeling.config import load_config
from imputation_modeling.runner import run_all_experiments


def main(config_file_path: str) -> None:
    config = load_config(config_file_path)

    run_all_experiments(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path", default=None, type=str)
    config_file_path = parser.parse_args().config_file_path
    main(config_file_path)
