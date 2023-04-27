# from .mmdet_train import custom_train_detector
from mmseg.apis import train_segmentor
from mmdet.apis import train_detector


import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, get_dist_info)
from mmcv.utils import build_from_cfg


import time
from os import path

from mmsegBEV.utils import get_root_logger
from mmsegBEV.evaluation import DistEvalHook, EvalHook

from .test import multi_gpu_test, single_gpu_test
from .utils import getDataLoader, loadModel2GPU

def train_model(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=False,
        timestamp=None,
        eval_model=None,
        meta=None
    ):
    logger = get_root_logger()     # get the initialized looger

    ## prepare data loader
    datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]
    dataloader_train = getDataLoader(datasets[0], cfg, distributed)
    dataloader_val = None
    
    if validate:
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        dataloader_val = getDataLoader(
            datasets[1], cfg, distributed, samples_per_gpu=val_samples_per_gpu, shuffle=False
        )
        
    # put model on gpus
    model = loadModel2GPU(model, cfg, distributed)
    if eval_model is not None: 
        eval_model = loadModel2GPU(eval_model, cfg, distributed)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    
    runner_default_args = dict(
        model=model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta
    )
    if eval_model is not None:
        runner_default_args.setdefault("eval_model", eval_model)
    
    runner = build_runner(
        cfg.runner,
        default_args=runner_default_args
    )

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp
    
    registerTrainHooks(runner, cfg, distributed)
    
    if distributed and isinstance(runner, EpochBasedRunner):
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        registerValHooks(runner, dataloader_val, cfg, distributed)

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)       
        
    runner.run(dataloader_train, cfg.workflow)

# def registerHooks(runner, cfg, distributed=False, val=False):
#     registerTrainHooks(runner, cfg, distributed)
    
#     if val:
#         registerValHooks(runner, cfg, distributed)
    

def registerTrainHooks(runner, cfg, distributed=False):
    optimizer_config = OptimizerHook(**cfg.optimizer_config)
    
    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed
        )
        
   # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    
def registerValHooks(runner, dataloader_val, cfg, distributed=False):
    if dataloader_val is None: 
        return
    
    eval_cfg = cfg.get('evaluation', {})
    eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
    eval_cfg['jsonfile_prefix'] = path.join('val', cfg.work_dir, time.ctime().replace(' ','_').replace(':','_'))
    
    if distributed:
        eval_hook = DistEvalHook(dataloader_val, test_fn=multi_gpu_test, **eval_cfg)
    else:
        eval_hook = EvalHook(dataloader_val, test_fn=single_gpu_test, **eval_cfg)
    
    runner.register_hook(eval_hook)

# def train_model(model,
#                 dataset,
#                 cfg,
#                 distributed=False,
#                 validate=False,
#                 timestamp=None,
#                 meta=None):
#     """A function wrapper for launching model training according to cfg.

#     Because we need different eval_hook in runner. Should be deprecated in the
#     future.
#     """
#     train_segmentor(
#         model,
#         dataset,
#         cfg,
#         distributed=distributed,
#         validate=validate,
#         timestamp=timestamp,
#         meta=meta)
