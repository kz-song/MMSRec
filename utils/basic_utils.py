import os
import sys
import wandb
import torch
import random
import pynvml
import logging
import numpy as np
import torch.backends.cudnn as cudnn


if torch.cuda.is_available():
    pynvml.nvmlInit()


def wandb_init(args):
    project_name = args.project_name
    display_name = args.display_name

    wandb.init(project=project_name, name=display_name, config=args)


def check_dirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs, exist_ok=True)


def seeds_init(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def log_gpu_memory(prefix):
    torch.cuda.empty_cache()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    info.free = round(info.free / 1024**2)
    info.used = round(info.used / 1024**2)
    print(f"[{prefix}]GPU memory free: {info.free} MB, memory used: {info.used} MB")


class Logger(object):
    def __init__(self, name):
        self.name = name
        self.logger = logging.getLogger(self.name)

    def get_logger(self, log_file, console_level=logging.INFO, file_level=logging.DEBUG):
        self.log_file = log_file
        check_dirs(os.path.split(self.log_file)[0])
        self.console_level = console_level
        self.file_level = file_level

        # add a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.console_level)
        console_format = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                                           datefmt="%m/%d/%Y %H:%M:%S")
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # add a file handler
        file_handler = logging.FileHandler(self.log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(self.file_level)
        file_format = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                                        datefmt="%m/%d/%Y %H:%M:%S")
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(min(self.console_level, self.file_level))

        return self.logger


