import copy
import os

import torch

import mmcv
from mmcv import Config
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import TORCH_VERSION, digit_version

import mmseg
from mmsegBEV.datasets import build_dataset
from mmsegBEV.utils import get_root_logger
   
def initConfig(args):        
    cfg = Config.fromfile(args.config)
        
    ### import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
        
    ### set work_dir (priority: CLI > segment in file > filename)
    if (args.work_dir is not None):
        cfg.work_dir = args.work_dir
        
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = os.path.join(
            './work_dirs',
            os.path.splitext(os.path.basename(args.config))[0]
        )    
        
    ### set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    ### set dist
    if args.launcher != 'none':
        init_dist(args.launcher, **cfg.dist_params)
        
        ### re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
                
    ### set resume 
    if (args.resume_from is not None) and (os.path.isfile(args.resume_from)):
        cfg.resume_from = args.resume_from
        
    ### set optimizer
    if (digit_version(TORCH_VERSION) == digit_version('1.8.1')) and (cfg.optimizer['type'] == 'AdamW'):
        cfg.optimizer['type'] = 'AdamW2' # fix bug in Adamw
    
    # create work_dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    # dump config
    cfg.dump(os.path.join(cfg.work_dir, os.path.basename(args.config)))
    
    return cfg    
    
def initLogger(cfg, meta):    
    log_file = os.path.join(cfg.work_dir, f'{meta["timestamp"]}.log')
    
    logger =  get_root_logger(log_file=log_file, log_level=cfg.log_level)
       
    # log env info 
    env_info_dict = mmseg.utils.collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    return logger

def initDataset(cfg):
    datasets = [build_dataset(cfg.data.train)]
    # if len(cfg.workflow) == 2:
    #     val_dataset = copy.deepcopy(cfg.data.val)
        
    #     # in case we use a dataset wrapper
    #     if 'dataset' in cfg.data.train:
    #         val_dataset.pipeline = cfg.data.train.dataset.pipeline
    #     else:
    #         val_dataset.pipeline = cfg.data.train.pipeline
            
    #     # set test_mode=False here in deep copied config, which do not affect AP/AR calculation later
    #     # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
    #     val_dataset.test_mode = False
    #     datasets.append(build_dataset(val_dataset)) 
    return datasets
