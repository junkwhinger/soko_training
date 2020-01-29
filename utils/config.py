from pathlib import Path

import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler

import json
from easydict import EasyDict
from pprint import pprint

def setup_logging(log_dir):
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler('{}exp_debug.log'.format(log_dir), maxBytes=10 ** 6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler('{}exp_error.log'.format(log_dir), maxBytes=10 ** 6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)


def get_config_from_json(json_file):

    with open(json_file, 'r') as config_file:
        try:
            config_dict = json.load(config_file)
            config = EasyDict(config_dict)
            return config, config_dict
        except ValueError:
            print("Invalid json file.")
            exit(-1)


def process_config(json_file):

    config, _ = get_config_from_json(json_file)
    print("*-*-* CONFIGURATION *-*-*")
    pprint(config)

    # summary path
    summary_dir = Path("experiments")
    config.summary_dir = summary_dir / config.exp_name / "summaries/"
    config.checkpoint_dir = summary_dir / config.exp_name / "checkpoints/"
    config.out_dir = summary_dir / config.exp_name / "out/"
    config.log_dir = summary_dir / config.exp_name / "logs/"

    # create summary path directories
    config.summary_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(exist_ok=True)
    config.out_dir.mkdir(exist_ok=True)
    config.log_dir.mkdir(exist_ok=True)

    setup_logging(config.log_dir)

    logging.getLogger().info("Root-!")
    logging.getLogger().info("------")

    return config