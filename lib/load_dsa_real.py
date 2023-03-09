import os
import torch
import numpy as np
import imageio
import json
import re
import random
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_dsa_real_data(basedir, half_res=False, testskip=1,train_num=70,radius_=0.5985,focal_=4086,size = 1024, tivox=False, time_mlp=False):
    # len_allframe = 133
    len_allframe = 133
    # len_allframe = 397
    fh = open(os.path.join(basedir, 'angle.txt'), 'r')
    angle_list = np.zeros(len_allframe)
    angle2_list = np.zeros(len_allframe)
    for line_index, line in enumerate(fh):
        line = line.rstrip()
        words = line.split()
        angle_list[line_index] = words[1]
        angle2_list[line_index] = words[2]

    val_index = np.array([  2,  10,  17,  23,  24,  25,  31,  33,  45,  46,  51,  52,  55,\
        59,  62,  65,  66,  71,  77,  81,  84,  89,  91,  93,  97,  98,\
        103, 107, 109, 113, 124, 130, 132])    # 33
    # val_index = np.array([  2,  10,  17,  23,  24,  25,  31,  33,  45,  46,  51,  52,  55,\
    #     59,  62,  65,  66,  71,  77,  81,  84,  89,  91,  93,  97,  98,\
    #     103, 107, 109, 113, 124, 130])    # 33
    if train_num == 10:
        train_index = np.array([11, 29, 30, 58, 69, 73, 86, 94, 99, 114])   # 10
        test_index = np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 18, 19, 20, \
            21, 22, 26, 27, 28, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 47, 48, 49, 50, \
                53, 54, 56, 57, 60, 61, 63, 64, 67, 68, 70, 72, 74, 75, 76, 78, 79, 80, 82, 83, \
                    85, 87, 88, 90, 92, 95, 96, 100, 101, 102, 104, 105, 106, 108, 110, 111, 112, \
                        115, 116, 117, 118, 119, 120, 121, 122, 123, 125, 126, 127, 128, 129, 131]) # 90
        test_index = np.union1d(test_index, val_index)
    elif train_num == 20:
        train_index = np.array([6, 11, 16, 18, 37, 42, 44, 54, 56, 61, 64, 75, 90, 94, 99, 102, 106, 115, 118, 127])   # 20
        test_index = np.array([0, 1, 3, 4, 5, 7, 8, 9, 12, 13, 14, 15, 19, 20, 21, 22, 26, 27, 28, 29, 30, 32, 34, \
            35, 36, 38, 39, 40, 41, 43, 47, 48, 49, 50, 53, 57, 58, 60, 63, 67, 68, 69, 70, 72, 73, 74, 76, 78, 79,\
                80, 82, 83, 85, 86, 87, 88, 92, 95, 96, 100, 101, 104, 105, 108, 110, 111, 112, 114, 116, 117, 119,\
                120, 121, 122, 123, 125, 126, 128, 129, 131]) # 80
        test_index = np.union1d(test_index, val_index)
    elif train_num == 30:
        test_index = np.array([  0,   1,   4,   5,   8,   9,  11,  12,  13,  14,  15,  16,  18,\
            21,  22,  26,  27,  32,  34,  35,  36,  39,  40,  42,  43,  44,\
            48,  49,  50,  53,  56,  57,  58,  61,  63,  64,  67,  68,  69,\
            70,  73,  74,  75,  76,  78,  79,  80,  82,  83,  85,  86,  88,\
            94,  95,  96, 104, 108, 110, 112, 114, 115, 116, 117, 118, 120,\
            122, 123, 128, 129, 131])  # 70
        # test_index = range(0,133)
        train_index = np.array([3, 6, 7, 19, 20, 28, 29, 30, 37, 38, 41, 47, 54, 60, 72, 87, 90, \
            92, 99, 100, 101, 102, 105, 106, 111, 119, 121, 125, 126, 127]) # 30
        test_index = np.union1d(test_index, val_index)
    elif train_num == 40:
        train_index = np.array([1, 6, 7, 13, 14, 16, 18, 26, 28, 32, 37, 38, 39, 41, 42, 44, 48, \
            50, 53, 58, 60, 61, 68, 70, 75, 78, 79, 80, 82, 90, 100, 110, 112, 114, 115, 117, 118, 120, 125, 128])
        test_index = np.array(list(set(range(0, len_allframe))-set(train_index)))
    elif train_num == 50:
        train_index = np.array([1, 4, 6, 8, 12, 13, 14, 19, 20, 21, 32, 34, 36, 37, 38, 39, 41, 44,\
             48, 49, 56, 58, 67, 68, 69, 73, 74, 76, 78, 83, 86, 87, 88, 94, 95, 96, 99, 100, 102, \
                 106, 110, 112, 114, 115, 116, 117, 122, 123, 127, 128])
        # test_index = np.array([])
        # test_index.sort()
        # test_index = np.union1d(test_index, val_index)
        test_index = np.array(list(set(range(0, len_allframe))-set(train_index)))
    elif train_num == 60:
        train_index = np.array([3, 5, 6, 7, 8, 11, 12, 13, 15, 18, 20, 28, 29, 32, 34, 35, 37,\
             38, 39, 40, 42, 44, 47, 50, 53, 54, 56, 58, 63, 64, 67, 68, 69, 70, 73, 75, 78, 82,\
                  83, 85, 87, 90, 92, 94, 95, 96, 100, 101, 105, 106, 110, 114, 118, 120, 121, 122,\
                       126, 127, 128, 129])
        test_index = np.array([0, 1, 4, 9, 14, 16, 19, 21, 22, 26, 27, 30, 36, 41, 43, 48, 49, 57, \
            60, 61, 72, 74, 76, 79, 80, 86, 88, 99, 102, 104, 108, 111, 112, 115, 116, 117, 119, 123, 125, 131])
        test_index = np.union1d(test_index, val_index)
    elif train_num == 70:
        train_index = np.array([  0,   1,   4,   5,   8,   9,  11,  12,  13,  14,  15,  16,  18,\
            21,  22,  26,  27,  32,  34,  35,  36,  39,  40,  42,  43,  44,\
            48,  49,  50,  53,  56,  57,  58,  61,  63,  64,  67,  68,  69,\
            70,  73,  74,  75,  76,  78,  79,  80,  82,  83,  85,  86,  88,\
            94,  95,  96, 104, 108, 110, 112, 114, 115, 116, 117, 118, 120,\
            122, 123, 128, 129, 131])  # 70
        test_index = np.array([3, 6, 7, 19, 20, 28, 29, 30, 37, 38, 41, 47, 54, 60, 72, 87, 90, \
            92, 99, 100, 101, 102, 105, 106, 111, 119, 121, 125, 126, 127]) # 30
        test_index = np.union1d(test_index, val_index)
    elif train_num == 80:
        train_index = np.array([0, 1, 3, 4, 6, 7, 8, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, \
            26, 27, 29, 30, 34, 35, 37, 40, 41, 43, 44, 47, 48, 49, 50, 56, 57, 58, 60, 63, 67, 68, \
                69, 70, 72, 73, 75, 76, 79, 80, 83, 85, 87, 90, 92, 95, 96, 99, 100, 101, 102, 105, \
                    106, 108, 110, 111, 112, 115, 116, 117, 118, 119, 120, 121, 122, 123, 125, 126, \
                        127, 128, 129, 131])
        test_index = np.array(list(set(range(0, len_allframe))-set(train_index)))
    elif train_num == 90:
        train_index = np.array([0, 1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19,\
             20, 21, 22, 26, 28, 30, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 47,\
                  49, 50, 53, 54, 56, 57, 58, 60, 61, 64, 67, 68, 69, 70, 72, 73, 74, 75,\
                       78, 79, 80, 82, 85, 86, 87, 88, 90, 92, 94, 95, 96, 99, 100, 101,\
                            102, 104, 105, 106, 108, 110, 112, 114, 115, 116, 118, 119, 120,\
                                 121, 122, 123, 125, 126, 127, 128, 129])
        test_index = np.array(list(set(range(0, len_allframe))-set(train_index)))
    elif train_num == 100:
        train_index = np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19,\
             20, 21, 22, 26, 27, 28, 29, 30, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,\
                  47, 48, 49, 50, 53, 54, 56, 57, 58, 60, 61, 63, 64, 67, 68, 69, 70, 72, 73,\
                       74, 75, 76, 78, 79, 80, 82, 83, 85, 86, 87, 88, 90, 92, 94, 95, 96, 99,\
                            100, 101, 102, 104, 105, 106, 108, 110, 111, 112, 114, 115, 116,\
                                 117, 118, 119, 120, 121, 122, 123, 125, 126, 127, 128, 129, 131])
        test_index = np.array(list(set(range(0, len_allframe))-set(train_index)))
    elif train_num == 2:
        train_index = np.array([65 , 130])
        test_index = np.array(list(set(range(0, len_allframe))-set(train_index)))
    elif train_num == 4:
        train_index = np.array([30, 60, 90, 120])
        test_index = np.array(list(set(range(0, len_allframe))-set(train_index)))
    elif train_num == 6:
        train_index = np.array([20,40,60,80,100,120])
        test_index = np.array(list(set(range(0, len_allframe))-set(train_index)))
    elif train_num == 8:
        train_index = np.array([15,30,45,60,75,90,105,120])
        test_index = np.array(list(set(range(0, len_allframe))-set(train_index)))
    elif train_num == 200:
        train_index = np.array([4, 7, 8, 10, 11, 14, 15, 16, 19, 20, 21, 24, 26, 28, 30, 31, 34,\
         35, 37, 38, 40, 42, 45, 50, 51, 52, 53, 57, 58, 59, 60, 63, 64, 65, 67, 70, 73, 75, 76, 77,\
         78, 79, 82, 83, 85, 87, 90, 91, 92, 93, 94, 96, 99, 103, 107, 108, 109, 110, 112, 114, 118, 120,\
         121, 125, 126, 133, 135, 139, 144, 145, 147, 148, 149, 153, 156, 158, 161, 163, 164, 167, 174, 175,\
         176, 177, 183, 185, 186, 187, 189, 190, 191, 193, 194, 195, 196, 200, 202, 204, 205, 208, 209, 210, 211,\
         212, 215, 218, 219, 220, 227, 228, 231, 234, 237, 239, 241, 242, 250, 251, 255, 256, 257, 258, 259, 260, \
        262, 263, 264, 265, 266, 267, 269, 270, 271, 273, 274, 276, 281, 282, 283, 287, 292, 294, 296, 297, 299, 302,\
         303, 304, 305, 307, 309, 311, 312, 313, 314, 315, 317, 323, 325, 327, 328, 330, 332, 333, 336, 338, 339, 340, \
        342, 344, 347, 348, 349, 350, 353, 354, 355, 356, 357, 359, 360, 363, 365, 368, 369, 370, 371, 373, 376, 377, 379,\
         382, 385, 387, 388, 389, 391, 393, 394, 395])
        test_index = np.array(list(set(range(0, len_allframe))-set(train_index)))
    elif train_num == 1010:
        train_index = np.array(list(set(range(0, len_allframe,2))))
        train_index.sort()
        test_index = np.array(list(set(range(0, len_allframe))-set(train_index)))
        test_index.sort()

    all_imgs = []
    all_poses = []

    radius = radius_# radius = 0.5985
    for i in range(len_allframe):
        imgs = cv2.imread(os.path.join(basedir, 'traindata_{}.jpg'.format(i)),0)
        # real 
        img_tmp = 255 - imgs
        img_tmp[0:116,:] = 0
        img_tmp[908:,:] = 0

        if True: # mask
            imgs_mask = cv2.imread(os.path.join(basedir[:-4]+'mask', 'traindata_{}.jpg'.format(i)),0)
            img_tmp = img_tmp.astype(np.float32) * imgs_mask.astype(np.float32) / imgs_mask.max()

        if img_tmp.shape[0] != size:
            img_tmp = cv2.resize(img_tmp, (size, size))
        img_tmp = img_tmp / 255  
        img_tmp = 1 - img_tmp   # keep vessel is dark!!!!

        angle = angle_list[i]
        angle2 = angle2_list[i]
        poses = pose_spherical(angle, angle2, radius)  # why 4.0?  voxel 大约是4*4*4??
        all_imgs.append(np.expand_dims(img_tmp, axis=0))
        all_poses.append(poses.unsqueeze(0).cpu().numpy())

    i_split = []
    i_split.append(train_index)
    i_split.append(val_index)
    i_split.append(test_index)

    imgs = np.concatenate(all_imgs, 0)
    imgs = np.expand_dims(imgs, axis=3)
    imgs = np.repeat(imgs, 3, 3)
    poses = np.concatenate(all_poses, 0)

    render_poses = torch.stack([pose_spherical(angle, -0.7, radius) for angle in np.linspace(-180,180,10+1)[:-1]], 0)
    if tivox or time_mlp:
        # times = torch.linspace(0., 1., 133)
        times = torch.from_numpy(np.array(np.array(range(133))/len_allframe)).cuda()
        render_times = torch.linspace(0., 1., render_poses.shape[0])

    H, W = imgs[0].shape[:2]
    focal = focal_ # 2430 for idea, 4086 for real

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    if tivox or time_mlp:
        return imgs, poses, times, render_poses, render_times, [H, W, focal], i_split
    else:
        return imgs, poses, render_poses, [H, W, focal], i_split