import os 
import torch
import numpy as np
from mmsegBEV.core.utils import visualize_map

map_classes = ['drivable_area', 'ped_crossing', 
               'walkway', 'stop_line',
               'carpark_area', 'divider']


def main():
    output = torch.load('/data/ptpyip_share/BEVSegmentation/work_dirs/bevformer_1/temp_out_epoach_1.pth')

    pred = {}
    for name, out in output.items():
        if type(out) is not torch.Size:
            pred[name] = out
            # print(f"mean of out: {out.mean()}")

    gt_map = pred['targets'][0]
    # print(gt_map)
    gt_map = gt_map.cpu().detach().numpy()
    gt_map = gt_map.astype(np.bool)

    visualize_map(
        os.path.join('vis/test_gt.png'),
        gt_map,
        classes=map_classes
    )

    pred_map = pred['inputs'][0]
    pred_map = torch.sigmoid(pred_map) #1 / (1 + torch.exp(pred_map))
    pred_map = pred_map.permute((1, 2, 0)).cpu().detach().numpy()
    # print(np.max(pred_map, axis=2, keepdims=2))
    # pred_map_max = np.max(pred_map, axis=2, keepdims=2) == pred_map
    # print(pred_map[pred_map_max])
    # # pred_map[pred_map_max]
    # print(pred_map.shape)
    # print(f"Predicted value shape : {pred_map.shape}")
    
    H, W, C = pred_map.shape
    zeros = np.zeros_like((C,H,W))
    # pred_map = pred_map.transpose(1, 2, 0)
    for h in range(H):
        for w in range(W):
            c = pred_map[h, w]
            c[c != c.max()] = 0
            pred_map[h, w] = c
    
    pred_map = pred_map >= 0.7

    # print(pred_map)

    visualize_map(
        os.path.join('vis/test_pred.png'),
        pred_map.transpose((2, 0, 1)),
        classes=map_classes
    )

    

if __name__ == "__main__":
    main()