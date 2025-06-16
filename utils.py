# # Rewritten utils.py based on latest torch 2.4 and python 3.12 best practices
# Compatible with PyTorch-Lightning 2.5+

import os
import random
import logging
from argparse import ArgumentParser
from logging.handlers import TimedRotatingFileHandler

import cv2
import numpy as np
import pandas as pd
import torch

IMG_SHAPE = (1365, 2048, 3)
IMAGE_FOLDER = "data/plant_pathodolgy_data/images"
LOG_FOLDER = "logs"


def mkdir(path: str):
    """Create directory if it does not exist."""
    if path and not os.path.exists(path):
        os.makedirs(path)


def seed_reproducer(seed=2020):
    """Fix random seeds to ensure reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



from argparse import ArgumentParser

def init_hparams():
    parser = ArgumentParser()
    
    # Basic Training Control
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=2020, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='L2 regularization coefficient')

    # LR Scheduler 学习率调度器
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str , default='cosine')
    parser.add_argument('--lr_decay_steps', default=20, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    # Training Info
    parser.add_argument('--dataset', default='standard_data', type=str)
    parser.add_argument('--data_dir', default='data/plant_pathodolgy_data', type=str)
    parser.add_argument('--model_name', default='general_backbone_classifier', type=str) #默认 backbone 模型
    parser.add_argument('--loss', default='soft_cross_entropy', type=str)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--log_dir', default='lightning_logs', type=str)
    
    # Model Hyperparameters
    parser.add_argument('--hid', default=64, type=int)
    parser.add_argument('--block_num', default=8, type=int)
    parser.add_argument('--in_channel', default=3, type=int)
    parser.add_argument('--layer_num', default=5, type=int)

    # KFold Support
    parser.add_argument('--kfold', default=0, type=int)
    parser.add_argument('--fold_num', default=0, type=int)

    # Other
    parser.add_argument('--aug_prob', default=0.5, type=float)
    parser.add_argument("--backbone", type=str, default="seresnext50_32x4d")
    parser.add_argument("--image_size", nargs='+', type=int, default=[256, 256])
    parser.add_argument("--soft_labels_filename", type=str, default="data/plant_pathodolgy_data/soft_labels.csv")

    # Add trainer arguments
    parser.add_argument('--max_epochs', default=200, type=int)
    parser.add_argument('--accelerator', default='auto', type=str)
    parser.add_argument('--devices', default='auto', type=str)
    parser.add_argument('--strategy', default='auto', type=str)
    parser.add_argument('--precision', default='32', type=str)
    parser.add_argument('--gradient_clip_val', default=None, type=float)
    parser.add_argument('--accumulate_grad_batches', default=1, type=int)
    parser.add_argument('--num_nodes', default=1, type=int)
    parser.add_argument('--log_every_n_steps', default=50, type=int)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--sync_batchnorm', action='store_true')
    parser.add_argument('--reload_dataloaders_every_n_epochs', default=0, type=int)
    parser.add_argument('--auto_scale_batch_size', action='store_true')
    parser.add_argument('--auto_lr_find', action='store_true')
    parser.add_argument('--replace_sampler_ddp', action='store_true')
    parser.add_argument('--detect_anomaly', action='store_true')
    parser.add_argument('--profiler', default=None, type=str)
    parser.add_argument('--enable_progress_bar', action='store_true')
    parser.add_argument('--enable_model_summary', action='store_true')
    parser.add_argument('--enable_checkpointing', action='store_true')
    parser.add_argument('--inference_mode', action='store_true')
    parser.add_argument('--use_distributed_sampler', action='store_true')

    args = parser.parse_args()

    # List Arguments
    args.mean_sen = [0.485, 0.456, 0.406]
    args.std_sen = [0.229, 0.224, 0.225]

    return args

def load_data(logger, frac=1.0):
    """Load training and test data."""
    train_csv = "data/plant_pathodolgy_data/train.csv"
    test_csv = "data/plant_pathodolgy_data/test.csv"

    data = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)

    if frac < 1.0:
        logger.info(f"Sampling fraction: {frac}")
        data = data.sample(frac=frac, random_state=42).reset_index(drop=True)
        test_data = test_data.sample(frac=frac, random_state=42).reset_index(drop=True)

    return data, test_data


def init_logger(log_name: str, log_dir: str = LOG_FOLDER):
    """Initialize a logger that outputs to both console and rotating file."""
    mkdir(log_dir)

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(asctime)s] %(filename)s[%(lineno)d]: %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = TimedRotatingFileHandler(os.path.join(log_dir, f"{log_name}.log"), when="D", backupCount=7)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def read_image(image_path: str):
    """Read an image and convert to RGB."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#打印超参数
def print_hparams(hparams, mode="table"):
    d = vars(hparams)
    print("\n" + "="*20 + " HYPERPARAMETERS " + "="*20)
    if mode == "table":
        for k, v in sorted(d.items()):
            print(f"{k:<25}: {v}")
    elif mode == "json":
        import json
        print(json.dumps(d, indent=4))
    elif mode == "pretty":
        import pprint
        pprint.pprint(d)
    print("="*60 + "\n")

