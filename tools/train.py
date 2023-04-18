import os
import time

import argparse

from mmseg.models import build_model
from mmseg.apis import set_random_seed, train_model
from mmseg import __version__ as mmseg_version

import utils

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument( '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher'
    )
   
def main():
    meta = {'timestamp': time.strftime('%Y%m%d_%H%M%S', time.localtime())}
    
    args = parse_args()
    cfg = utils.initConfig(args)
    logger = utils.initLogger(cfg, meta)
    
    # log some basic info
    distributed = (args.launcher != 'none')
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')
    
    # set random seeds
    logger.info(f'Set random seed to {cfg.seed}, '
                f'deterministic: {cfg.deterministic}')
    set_random_seed(cfg.seed, deterministic=cfg.deterministic)
    
    meta['seed'] = cfg.seed
    meta['exp_name'] = os.path.basename(args.config)
    
    # init dataset
    datasets = utils.initDataset(cfg)
    
    # init model
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')
    )
    model.init_weights()
    model.CLASSES = datasets[0].CLASSES
    model.PALETTE = datasets[0].PALETTE  if hasattr(datasets[0], 'PALETTE') else None

    logger.info(f'Model:\n{model}')
    
    if (cfg.checkpoint_config is not None):
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=mmseg_version,
            config=cfg.pretty_text,
            CLASSES=model.CLASSES,
            PALETTE=model.PALETTE
        )
        
    train_model(
        model, datasets, cfg,
        distributed=True, validate=True,
        timestamp=meta["timestamp"]
    )
