# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------

import os.path
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

# from mmdet.core import encode_mask_results
import mmcv
import numpy as np
import pycocotools.mask as mask_util


from .utils import getDataLoader, loadModel2GPU

def test_model( model, dataset, cfg, distributed=True, format_only=False,timestamp=None):
    ## prepare data loader
    dataloader = getDataLoader(dataset, cfg, distributed, shuffle=False)
    model = loadModel2GPU(model, cfg, distributed)
    
    outputs = multi_gpu_test(model, dataloader)
    
    rank, _ = get_dist_info()
    if rank == 0:
        if format_only:            
            kwargs = {} 
            kwargs['jsonfile_prefix'] = os.path.join(
                'test', 
                cfg.filename.split('/')[-1].split('.')[-2], 
                time.ctime().replace(' ', '_').replace(':', '_')
            )
        
            dataset.format_results(outputs, **kwargs)
        else:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)

            print(dataset.evaluate(outputs, **eval_kwargs))
    

def single_gpu_test(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results

def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, encode_results=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    
    with torch.no_grad():
        for data in data_loader:  
            output = model(return_loss=False, rescale=True, **data)
            
            if isinstance(output, dict):
                # encode mask results
                masks_bev = custom_encode_mask_results(output.get("masks_bev")) if encode_results else output.get("masks_bev")
                
                mask_results.extend(masks_bev)
            else:
                # normally take this branch
                batch_size = len(output)
                results.extend(output)
            
    if rank == 0:
        for _ in range(batch_size * world_size):
            prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        mask_results = collect_results_gpu(mask_results, len(dataset))
    else:
        mask_results = collect_results_cpu(mask_results, len(dataset), tmpdir)
        # tmpdir = tmpdir+'_mask' if tmpdir is not None else None

    return {'mask_results': mask_results}


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
        
    # dump the part result to the dir
    mmcv.dump(result_part, os.path.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    
    # collect all parts
    if rank != 0:
        return None

    # load results of all parts from tmp dir
    part_list = []
    for i in range(world_size):
        part_file = os.path.join(tmpdir, f'part_{i}.pkl')
        part_list.append(mmcv.load(part_file))
    # sort the results
    ordered_results = []
    '''
    bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample,
    '''
    #for res in zip(*part_list):
    for res in part_list:  
        ordered_results.extend(list(res))
        
    # the dataloader may pad some samples
    ordered_results = ordered_results[:size]
    
    # remove tmp dir
    shutil.rmtree(tmpdir)
    return ordered_results


def collect_results_gpu(result_part, size):
    collect_results_cpu(result_part, size)
    
def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])  # encoded with RLE
    return [encoded_mask_results]