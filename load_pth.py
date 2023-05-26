import os 
import torch
import numpy as np
from mmsegBEV.core.utils import visualize_map

map_classes = ['drivable_area', 'ped_crossing', 
               'walkway', 'stop_line',
               'carpark_area', 'divider']


def main():
    output = torch.load("ckpts/bevformer_r101_dcn_24ep.pth")

    new_state_dict = {}
    pred = {}
    for name, weight in output['state_dict'].items():
        if 'img_backbone' in name or 'img_neck' in name:
            print(f"name: {name}")
            new_state_dict[name] = weight

    new_model = {
        'state_dict': new_state_dict
    }

    torch.save(new_model, 'ckpts/r101_fpn_pretrained.pth')

        # print(f"out: {out}")
        # if type(out) is not torch.Size:
        #     pred[name] = out
            # print(f"mean of out: {out.mean()}")
    
    # gt_map = pred['targets']
    # print(gt_map)
    # gt_map = gt_map.cpu().detach().numpy()
    # gt_map = gt_map.astype(np.bool)

    # visualize_map(
    #     os.path.join('test_gt.png'),
    #     gt_map,
    #     classes=map_classes
    # )

    # pred_map = pred['inputs']
    # print(pred_map)
    # pred_map = pred_map.cpu().detach().numpy()
    # # pred_map

    # visualize_map(
    #     os.path.join('test_pred.png'),
    #     gt_map,
    #     classes=map_classes
    # )

if __name__ == "__main__":
    main()