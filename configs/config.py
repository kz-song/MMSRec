import os
import yaml
import argparse
from easydict import EasyDict


class BasicConfigs(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def _parse_args_with_config(self):
        parsed_args = self.parser.parse_args()
        args = EasyDict(vars(parsed_args))

        assert os.path.exists(args.config)
        with open(args.config, "r", encoding="utf-8") as cfg_file:
            configs = yaml.safe_load(cfg_file)
            for key, value in configs.items():
                args[key] = value
        return args

    def get_training_args(self):
        self.parser.add_argument("--config",
                                 type=str,
                                 help="Config file")

        self.args = self._parse_args_with_config()
        return self.args

