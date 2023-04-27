import os
import time

import argparse

from mmcv.runner import load_checkpoint
from mmcv.cnn import fuse_conv_bn

from mmsegBEV.models import build_model
from mmsegBEV.apis import test_model
from mmsegBEV.datasets import build_dataset

from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

import utils


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument( '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher'
    )
    parser.add_argument('--local_rank', type=int, default=0)

    # parser.add_argument('--seed', type=int, default=0, help='random seed')
    # parser.add_argument(
    #     '--deterministic',
    #     action='store_true',
    #     help='whether to set deterministic options for CUDNN backend.')
        
    return parser.parse_args()
   
def main():
    meta = {'timestamp': time.strftime('%Y%m%d_%H%M%S', time.localtime())}
    
    args = parse_args()
    cfg = utils.initConfig(args)
    
    # log some basic info
    distributed = (args.launcher != 'none')
    
    # set random seeds
    set_random_seed(cfg.seed, deterministic=cfg.deterministic)
    
    meta['seed'] = cfg.seed
    meta['exp_name'] = os.path.basename(args.config)
    
    # init dataset
    dataset = build_dataset(cfg.data.test)
    
    # init model
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is not None:
    #     wrap_fp16_model(model)
    
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
        
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE
    
    if (cfg.checkpoint_config is not None):
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=mmseg_version,
            config=cfg.pretty_text,
            CLASSES=model.CLASSES,
            PALETTE=model.PALETTE
        )

    test_model(
        model, dataset, cfg,
        distributed=True, validate=True,
        timestamp=meta["timestamp"]
    )


if __name__ == '__main__':
    main()