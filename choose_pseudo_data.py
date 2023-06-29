import json
from random import sample
import torch
from mmdet.core import BitmapMasks
import pycocotools.mask as maskUtils
import copy
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='choose pseudo data')

parser.add_argument('--ratio', type=int, choices=[1,5,10], default=10, help='sample ratio')
parser.add_argument('--dataset', type=str, choices=['refcoco-unc', 'refcocoplus-unc', 'refcocog-umd'], default='refcoco-unc', help='input dataset name')

args = parser.parse_args()


def polygonFromMask(maskedArr):
    # adapted from https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
    contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    valid_poly = 0
    for contour in contours:
    # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.astype(float).flatten().tolist())
            valid_poly += 1
    if valid_poly == 0:
        raise ValueError
    return segmentation

def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss
    
def compute_project_term(mask_scores, gt_bitmasks):
    mask_losses_y = dice_coefficient(
        mask_scores.max(dim=2, keepdim=True)[0],
        gt_bitmasks.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = dice_coefficient(
        mask_scores.max(dim=3, keepdim=True)[0],
        gt_bitmasks.max(dim=3, keepdim=True)[0]
    )
    return (mask_losses_x + mask_losses_y).mean()

annsfile = "./data/annotations/{}/instances.json".format(args.dataset)
sample_annsfile = "./data/annotations/{}/instances_sample_{}.json".format(args.dataset, args.ratio)
outfile = "./data/annotations/{}/instances_sample_{}_bbox_pick.json".format(args.dataset, args.ratio)
predict = torch.load("./refine_data/{}_instances_refine.pth".format(args.dataset))

anns_all = json.load(open(annsfile, 'r'))
anns_sample = json.load(open(sample_annsfile, 'r'))
train_all = anns_all["train"]
train_sample = anns_sample["train"]
refine_train_sample = copy.deepcopy(train_sample)

count = 0
all_zero_mask = 0
project_term_sum = []
for i, train_inst in enumerate(train_all):
    if train_inst not in train_sample:
        # Generated data has a lot of noise, use "try...except..." to filter out the noise.
        try:
            mask_origin = maskUtils.decode(maskUtils.frPyObjects(train_inst['mask'], train_inst['height'], train_inst['width'])) 
            train_inst['mask'] = polygonFromMask(maskUtils.decode(predict[i][-1]))
            mask_dataloader = maskUtils.decode(maskUtils.frPyObjects(train_inst['mask'], train_inst['height'], train_inst['width'])) 
            box_mask = np.zeros_like(mask_dataloader)
            # box_mask = torch.zeros(bs, 1000, 1000).to(targets.device)
            x1 = int(train_inst['bbox'][0])
            y1 = int(train_inst['bbox'][1])
            x2 = int(train_inst['bbox'][2]) + x1
            y2 = int(train_inst['bbox'][3]) + y1
            box_mask[y1: y2, x1:x2] = 1
            project_term = compute_project_term(torch.from_numpy(mask_dataloader).permute(2,0,1).unsqueeze(0), torch.from_numpy(box_mask).permute(2,0,1).unsqueeze(0))
        except:
            pass
    
        try:
            contour, _ = cv2.findContours(mask_dataloader, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        except:
            contour = []
        try:
            contour = sorted(
            contour, key=lambda x: cv2.contourArea(x), reverse=True)
        except:
            contour = []
        try:
            contour = contour[0][:, 0, :]  # x, y coordinate of contour
        except:
            contour = []
        
        if project_term < 0.1:
            project_term_sum.append(project_term)
        else:
            contour=[]
            
        # if mask_dataloader.max()==0 or contour==[]:
        if contour==[]:
            all_zero_mask += 1
        else:
            refine_train_sample.append(train_inst)
            print(len(refine_train_sample), i)
        
anns_sample = {'train':refine_train_sample, 'val':anns_all["val"], 'test':anns_all["test"]}
print(len(refine_train_sample))
with open(outfile,'w') as f:
    json.dump(anns_sample, f)