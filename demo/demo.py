# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
from glob import glob
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_model
from utils.preprocessing import load_img, process_bbox, generate_patch_image, get_iou
from utils.vis import vis_keypoints_with_skeleton
from utils.mano import mano

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    args = parser.parse_args()

    assert args.gpu_ids, "Please set proper gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    return args

args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True

# snapshot load
model_path = './snapshot_6.pth'
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model('test')
model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

# prepare save paths
input_img_path = './images'
output_save_path = './outputs'
os.makedirs(output_save_path, exist_ok=True)

# load paths of input images
img_path_list = glob(osp.join(input_img_path, '*.jpg')) + glob(osp.join(input_img_path, '*.png')) + glob(osp.join(input_img_path, '*.jpeg'))

# for each input image
for img_path in tqdm(img_path_list):
    file_name = img_path.split('/')[-1][:-4]
    
    # load image and make its aspect ratio follow cfg.input_img_shape
    original_img = load_img(img_path) 
    img_height, img_width = original_img.shape[:2]
    bbox = [0, 0, img_width, img_height]
    bbox = process_bbox(bbox, img_width, img_height)
    img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
    transform = transforms.ToTensor()
    img = transform(img.astype(np.float32))/255
    img = img.cuda()[None,:,:,:]

    # forward to InterWild
    inputs = {'img': img}
    targets = {}
    meta_info = {}
    with torch.no_grad():
        out = model(inputs, targets, meta_info, 'test')
    
    # check IoU between boxes of two hands
    rhand_bbox = out['rhand_bbox'].cpu().numpy()[0]
    lhand_bbox = out['lhand_bbox'].cpu().numpy()[0]
    iou = get_iou(rhand_bbox, lhand_bbox, 'xyxy')
    if iou > 0:
        is_th = True
    else:
        is_th = False

    # for each right and left hand
    vis_box = original_img.copy()[:,:,::-1]
    vis_skeleton = original_img.copy()[:,:,::-1]
    for h in ('right', 'left'):
        # get outputs
        hand_bbox = out[h[0] + 'hand_bbox'].cpu().numpy()[0].reshape(2,2) # xyxy
        hand_bbox_conf = float(out[h[0] + 'hand_bbox_conf'].cpu().numpy()[0]) # bbox confidence
        joint_img = out[h[0] + 'joint_img'].cpu().numpy()[0] # 2.5D pose

        # bbox save
        hand_bbox[:,0] = hand_bbox[:,0] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        hand_bbox[:,1] = hand_bbox[:,1] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
        hand_bbox_xy1 = np.concatenate((hand_bbox, np.ones_like(hand_bbox[:,:1])),1)
        hand_bbox = np.dot(bb2img_trans, hand_bbox_xy1.transpose(1,0)).transpose(1,0)
        if h == 'right':
            color = (255,0,255) # purple
        else:
            color = (102,255,102) # green
        vis_box = cv2.rectangle(vis_box.copy(), (int(hand_bbox[0,0]), int(hand_bbox[0,1])), (int(hand_bbox[1,0]), int(hand_bbox[1,1])), color, 3)

        # 2D skeleton
        joint_img_xy1 = np.concatenate((joint_img[:,:2], np.ones_like(joint_img[:,:1])),1)
        joint_img = np.dot(bb2img_trans, joint_img_xy1.transpose(1,0)).transpose(1,0)
        if h == 'right':
            color = (255,0,255) # purple
        else:
            color = (102,255,102) # green
        vis_skeleton = vis_keypoints_with_skeleton(vis_skeleton, joint_img, mano.sh_skeleton, color)
 
    # save box
    cv2.imwrite(osp.join(output_save_path, file_name + '_box.jpg'), vis_box)

    # save 2D skeleton
    cv2.imwrite(osp.join(output_save_path, file_name + '_skeleton.jpg'), vis_skeleton)

