import torch.cuda
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmsegBEV.datasets import build_dataloader

def getDataLoader(dataset, cfg, distributed=False, samples_per_gpu=None, shuffle=None):
    if samples_per_gpu is None: samples_per_gpu = cfg.data.samples_per_gpu
    if shuffle is None: shuffle = cfg.data.shuffle
    return build_dataloader(
        dataset,
        samples_per_gpu,
        cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=shuffle,
        seed=cfg.seed,
        shuffler_sampler=cfg.data.shuffler_sampler,  # dict(type='DistributedGroupSampler'),
        nonshuffler_sampler=cfg.data.nonshuffler_sampler
    )
    
def loadModel2GPU(model, cfg, distributed=False):
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids
        )
        
    return model