# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import os.path
from glob import glob
import time
import re
import bisect
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import scipy.ndimage
from scipy.ndimage.morphology import binary_dilation
import scipy.misc
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import math
import PIL.Image
import skimage
import skimage.io
import skimage.transform
import cv2

import config
import misc
import tfutil
from run import setup_snapshot_image_grid, my_swap_h, my_swap_w, block_permutation
from loss import fp32
import dataset

from custom_vgg19 import *

from logging import getLogger
from logging import config as logconfig
import json

with open('./log_config.json', 'r') as f:
    log_conf = json.load(f)
logconfig.dictConfig(log_conf)
logger = getLogger("util_scripts_log")

def gkern_corners_for_scale(out_length):
    sig = float(out_length) / 6.0
    ax = np.arange(0.0, float(out_length))
    xx, yy = np.meshgrid(ax, ax)
    kernel_ul = np.exp(-(xx**2 + yy**2) / (2. * sig**2))
    kernel_ul = (kernel_ul - np.amin(kernel_ul)) / (np.amax(kernel_ul) - np.amin(kernel_ul))
    kernel_ul = 1.0 - kernel_ul
    kernel_bl = np.rot90(kernel_ul)
    kernel_br = np.rot90(kernel_bl)
    kernel_ur = np.rot90(kernel_br)
    return kernel_ul, kernel_ur, kernel_bl, kernel_br

def linkern_for_weight_square(out_length, latent_res):
    step = 1.0 / (out_length - 2.0*latent_res - 1.0)
    ax = np.arange(start=0.0, stop=1.0+step, step=step)
    ax = np.concatenate((np.zeros(latent_res), ax, np.ones(latent_res)))
    X, Y = np.meshgrid(ax, ax)
    weight_br = X * Y
    weight_ur = np.rot90(weight_br)
    weight_ul = np.rot90(weight_ur)
    weight_bl = np.rot90(weight_ul)
    return weight_ul, weight_ur, weight_bl, weight_br

def gkern_for_weight(in_length, out_length, x, y, sig_div=6.0):
    cx = float(x + in_length//2)
    cy = float(y + in_length//2)
    sig = float(out_length) / sig_div
    ax = np.arange(0.0, float(out_length))
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-((xx-cx)**2 + (yy-cy)**2) / (2. * sig**2))
    kernel = (kernel - np.amin(kernel)) / (np.amax(kernel) - np.amin(kernel))
    return kernel

def gkern_for_scale(in_length, out_length, x, y):
    cx = float(x + in_length//2)
    cy = float(y + in_length//2)
    sig = 32.0 / 12.0
    ax = np.arange(0.0, float(out_length))
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-((xx-cx)**2 + (yy-cy)**2) / (2. * sig**2))
    kernel = (kernel - np.amin(kernel)) / (np.amax(kernel) - np.amin(kernel))
    kernel = 1.0 - kernel
    return kernel

def linkern_for_weight_horizontal(out_shape, latent_res):
    kernel = np.reshape(np.linspace(start=0.0, stop=1.0, num=out_shape[3]-2*latent_res)[::-1], [1,1,1,out_shape[3]-2*latent_res])
    kernel = np.concatenate((np.ones((1,1,1,latent_res)), kernel, np.zeros((1,1,1,latent_res))), axis=3)
    kernel = np.tile(kernel, list(out_shape[:3])+[1])
    return kernel.astype(np.float32)

def linkern_for_weight_arbitrary_shape(out_h, out_w, latent_res):
    kernel_h = np.reshape(np.linspace(start=0.0, stop=1.0, num=out_h-2*latent_res)[::-1], [out_h-2*latent_res,1])
    kernel_h = np.concatenate((np.ones((latent_res,1)), kernel_h, np.zeros((latent_res,1))), axis=0)
    kernel_h = np.tile(kernel_h, [1,out_w])
    kernel_w = np.reshape(np.linspace(start=0.0, stop=1.0, num=out_w-2*latent_res)[::-1], [1,out_w-2*latent_res])
    kernel_w = np.concatenate((np.ones((1,latent_res)), kernel_w, np.zeros((1,latent_res))), axis=1)
    kernel_w = np.tile(kernel_w, [out_h,1])
    weight_ul = kernel_h * kernel_w
    weight_ur = kernel_h * (1.0 - kernel_w)
    weight_bl = (1.0 - kernel_h) * kernel_w
    weight_br = (1.0 - kernel_h) * (1.0 - kernel_w)
    return weight_ul, weight_ur, weight_bl, weight_br

def gkern_for_weight_arbitrary_shape(out_h, out_w , x, y, sig_div):
    cx = float(x)
    cy = float(y)
    sig = float(out_w) / sig_div
    ax = np.arange(0.0, float(out_w))
    ay = np.arange(0.0, float(out_h))
    xx, yy = np.meshgrid(ax, ay)
    #kernel = np.exp(-((xx-cx)**2 + (yy-cy)**2) / (2. * sig**2))
    kernel = np.exp(-(np.maximum(np.absolute(xx-cx)-sig, 0.0)**2 + (yy-cy)**2) / (2. * sig**2))
    kernel = (kernel - np.amin(kernel)) / (np.amax(kernel) - np.amin(kernel))
    return kernel

def gkern_for_weight_arbitrary_shape_hybridization(out_h, out_w , x, y, sig_div):
    cx = float(x)
    cy = float(y)
    sig = float(out_w) / sig_div
    ax = np.arange(0.0, float(out_w))
    ay = np.arange(0.0, float(out_h))
    xx, yy = np.meshgrid(ax, ay)
    kernel = np.exp(-((xx-cx)**2 + (yy-cy)**2) / (2. * sig**2))
    #kernel = np.exp(-(np.maximum(np.absolute(xx-cx)-sig, 0.0)**2 + (yy-cy)**2) / (2. * sig**2))
    return kernel

def l2(x1, y1, x2, y2):
    return (x1 - x2)**2 + (y1 - y2)**2

def dist2square(x, y, ul_x, ul_y, size):
    x_min = ul_x
    x_max = x_min + size
    y_min = ul_y
    y_max = y_min + size
    if x < x_min:
        if y < y_min:
            dist2 = l2(x, y, x_min, y_min)
        elif y >= y_min and y <= y_max:
            dist2 = (x - x_min)**2
        else:
            dist2 = l2(x, y, x_min, y_max)
    elif x >= x_min and x <= x_max:
        if y < y_min:
            dist2 = (y - y_min)**2
        elif y >= y_min and y <= y_max:
            dist2 = 0.0
        else:
            dist2 = (y - y_max)**2
    else:
        if y < y_min:
            dist2 = l2(x, y, x_max, y_min)
        elif y >= y_min and y <= y_max:
            dist2 = (x - x_max)**2
        else:
            dist2 = l2(x, y, x_max, y_max)
    return dist2

def gkern_for_weight_grid_shape_hybridization(out_h, out_w , cx, cy, size, sig_div):
    cx = float(cx)
    cy = float(cy)
    sig = min([float(out_h), float(out_w)]) / sig_div
    kernel = np.zeros([out_h, out_w])
    for i in range(out_h):
        y = float(i)
        for j in range(out_w):
            x = float(j)
            kernel[i, j] = np.exp(-dist2square(x, y, cx, cy, size) / (2. * sig**2))
    return kernel

def gkern_for_scale_horizontal(out_shape, latent_res):
    '''
    sig = 6.0
    ax = np.arange(0.0, float(out_shape[3]-latent_res))
    kernel = np.exp(-ax**2 / (2. * sig**2))
    kernel = (kernel - np.amin(kernel)) / (np.amax(kernel) - np.amin(kernel))
    kernel = 1.0 - kernel
    kernel = np.reshape(kernel, [1,1,1,out_shape[3]-latent_res])
    kernel = np.concatenate((np.zeros((1,1,1,latent_res)), kernel), axis=3)
    kernel = np.tile(kernel, list(out_shape[:3])+[1])
    '''
    kernel = np.concatenate((np.zeros((1,1,1,latent_res)), np.ones((1,1,1,out_shape[3]-latent_res))), axis=3)
    return kernel.astype(np.float32)

def move(x, y, vx, vy, in_length, out_length, still=False):
    if still:
        x_new = x; y_new = y; vx_new = 0; vy_new = 0
        return x_new, y_new, vx_new, vy_new
    assert x>=0 and x<out_length-in_length and y>=0 and y<out_length-in_length and vx in [-1,0,1] and vy in [-1,0,1] and (vx!=0 or vy!=0)
    # bounce at boundary
    if x==0 and vx==-1 or x==out_length-in_length-1 and vx==1:
        vx *= -1
    if y==0 and vy==-1 or y==out_length-in_length-1 and vy==1:
        vy *= -1
    # move
    x_new = x + vx
    y_new = y + vy
    # random velocity biase
    rand = np.random.uniform()
    if rand<0.5:
        vx_new = vx
        vy_new = vy
    else:
        if abs(vx)==1 and abs(vy)==1:
            if rand<0.75:
                vx_new = vx
                vy_new = 0
            else:
                vx_new = 0
                vy_new = vy
        elif vx==0:
            if rand<0.75:
                vx_new = -1
            else:
                vx_new = 1
            vy_new = vy
        elif vy==0:
            vx_new = vx
            if rand<0.75:
                vy_new = -1
            else:
                vy_new = 1
    return x_new, y_new, vx_new, vy_new

def transform(im0, height, width, hight_crop=None, width_crop=None, transpose=False):
    h = im0.shape[0]
    w = im0.shape[1]
    if h == height:
        i = 0
    else:
        if hight_crop is None:
            i = np.random.randint(0, h-height)
        else:
            i = hight_crop
    if w == width:
        j = 0
    else:
        if width_crop is None:
            j = np.random.randint(0, w-width)
        else:
            j = width_crop
    im1 = im0[i:i+height, j:j+width, :]
    if transpose:
        im1 = np.transpose(im1, [2, 0, 1])
    return im1

def render(im0, height, width, reflectence=None, rotation=None, scale=None):
    input_height = im0.shape[0]
    input_width = im0.shape[1]
    flag = True
    while flag:
        flag = False
        if reflectence is None:
            reflectence = np.random.randint(0, 4)
        if reflectence == 0:
            im1 = np.array(im0).astype(np.uint8)
        elif reflectence == 1:
            im1 = np.fliplr(im0).astype(np.uint8)
        elif reflectence == 2:
            im1 = np.flipud(im0).astype(np.uint8)
        elif reflectence == 3:
            im1 = np.flipud(np.fliplr(im0)).astype(np.uint8)
        if rotation is None:
            rotation = np.random.uniform(-90, 90)
        IM1 = PIL.Image.fromarray(im1, 'RGB')
        IM2 = IM1.rotate(rotation, expand=True)
        im2 = np.asarray(IM2).astype(np.uint8)
        w_new, h_new = largest_rotated_rect(im1.shape[1], im1.shape[0], math.radians(rotation))
        im2 = crop_around_center(im2, w_new, h_new)
        if scale is None:
            log_scale = np.random.uniform(-np.log(4.0), 0.0)
            scale = np.exp(log_scale)
        IM2 = PIL.Image.fromarray(im2, 'RGB')
        IM3 = IM2.resize((int(im2.shape[1]*scale), int(im2.shape[0]*scale)))
        im3 = np.asarray(IM3).astype(np.uint8)
        if im3.shape[0] < height or im3.shape[1] < width:
            flag = True
    im3 = im3 * 255.0
    return im3.astype(np.uint8)

def largest_rotated_rect(w, h, angle):
    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi
    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)
    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)
    delta = math.pi - alpha - gamma
    length = h if (w < h) else w
    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)
    y = a * math.cos(gamma)
    x = y * math.tan(gamma)
    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

def crop_around_center(image, width, height):
    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))
    if(width > image_size[0]):
        width = image_size[0]
    if(height > image_size[1]):
        height = image_size[1]
    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)
    return image[y1:y2, x1:x2]

def multi_layer_diff(feature, feature_, src1=None, src2=None, return_per_sample=False):
    l = 0.0
    if return_per_sample:
        l_per_sample = 0.0
    if src1 is None or src2 is None:
        for f, f_ in zip(feature, feature_):
            l += np.mean(np.absolute(f - f_))
            if return_per_sample:
                l_per_sample += np.absolute(f - f_)
    else:
        for f, f_, s1, s2 in zip(feature, feature_, src1, src2):
            temp = np.mean(np.absolute(f - f_), axis=(1,2)) / np.mean(np.absolute(s1 - s2), axis=(1,2))
            l += np.mean(np.nan_to_num(temp))
            if return_per_sample:
                l_per_sample += np.nan_to_num(temp)
    if return_per_sample:
        return l / np.float32(len(feature)), l_per_sample / np.float32(len(feature))
    else:
        return l / np.float32(len(feature))

def multi_layer_cos(src1, center, src2, return_per_sample=False):
    l = 0.0
    if return_per_sample:
        l_per_sample = 0.0
    for s1, f, s2 in zip(src1, center, src2):
        temp = np.sum((f - s1) * (s2 - f), axis=(1,2)) / np.sqrt(np.sum((f - s1) * (f - s1), axis=(1,2))) / np.sqrt(np.sum((s2 - f) * (s2 - f), axis=(1,2)))
        l += np.mean(np.nan_to_num(temp))
        if return_per_sample:
            l_per_sample += np.nan_to_num(temp)
    if return_per_sample:
        return l / np.float32(len(center)), l_per_sample / np.float32(len(center))
    else:
        return l / np.float32(len(center))

#----------------------------------------------------------------------------
# Generate MP4 video of random interpolations using a previously trained network.
# To run, uncomment the appropriate line in config.py and launch train.py.

def texture_dissolve_video(model_path, imageStartUL_path, imageStartUR_path, imageStartBL_path, imageStartBR_path, imageEndUL_path, imageEndUR_path, imageEndBL_path, imageEndBR_path, out_dir, duration_sec=4.0, scale_h=8, scale_w=8, image_shrink=1, image_zoom=1, mp4=None, mp4_fps=30, mp4_codec='libx265', mp4_bitrate='16M', minibatch_size=1):
    changes = 1
    if not os.path.isdir(out_dir): os.makedirs(out_dir)

    # Load model
    E_zg, E_zl, G, D_rec, D_interp, D_blend, Es_zg, Es_zl, Gs = misc.load_pkl(model_path)
    Gs_fcn = tfutil.Network('Gs', reuse=True, num_channels=Gs.output_shape[1], resolution=Gs.output_shape[2], scale_h=scale_h, scale_w=scale_w, **config.G)

    # Load dataset
    grid_reals_ul = np.zeros([2]+Es_zl.input_shape[1:]).astype(np.float32)
    imageStartUL = np.transpose(misc.adjust_dynamic_range(np.array(PIL.Image.open(imageStartUL_path)).astype(np.float32), [0,255], [-1,1]), axes=[2,0,1])
    grid_reals_ul[0,:,:,:] = imageStartUL
    imageEndUL = np.transpose(misc.adjust_dynamic_range(np.array(PIL.Image.open(imageEndUL_path)).astype(np.float32), [0,255], [-1,1]), axes=[2,0,1])
    grid_reals_ul[1,:,:,:] = imageEndUL
    grid_reals_ur = np.zeros([2]+Es_zl.input_shape[1:]).astype(np.float32)
    imageStartUR = np.transpose(misc.adjust_dynamic_range(np.array(PIL.Image.open(imageStartUR_path)).astype(np.float32), [0,255], [-1,1]), axes=[2,0,1])
    grid_reals_ur[0,:,:,:] = imageStartUR
    imageEndUR = np.transpose(misc.adjust_dynamic_range(np.array(PIL.Image.open(imageEndUR_path)).astype(np.float32), [0,255], [-1,1]), axes=[2,0,1])
    grid_reals_ur[1,:,:,:] = imageEndUR
    grid_reals_bl = np.zeros([2]+Es_zl.input_shape[1:]).astype(np.float32)
    imageStartBL = np.transpose(misc.adjust_dynamic_range(np.array(PIL.Image.open(imageStartBL_path)).astype(np.float32), [0,255], [-1,1]), axes=[2,0,1])
    grid_reals_bl[0,:,:,:] = imageStartBL
    imageEndBL = np.transpose(misc.adjust_dynamic_range(np.array(PIL.Image.open(imageEndBL_path)).astype(np.float32), [0,255], [-1,1]), axes=[2,0,1])
    grid_reals_bl[1,:,:,:] = imageEndBL
    grid_reals_br = np.zeros([2]+Es_zl.input_shape[1:]).astype(np.float32)
    imageStartBR = np.transpose(misc.adjust_dynamic_range(np.array(PIL.Image.open(imageStartBR_path)).astype(np.float32), [0,255], [-1,1]), axes=[2,0,1])
    grid_reals_br[0,:,:,:] = imageStartBR
    imageEndBR = np.transpose(misc.adjust_dynamic_range(np.array(PIL.Image.open(imageEndBR_path)).astype(np.float32), [0,255], [-1,1]), axes=[2,0,1])
    grid_reals_br[1,:,:,:] = imageEndBR

    # encode zg latent tensors
    print('zg encoding...')
    zg_mu_ul, zg_log_sigma_ul = Es_zg.run(grid_reals_ul, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    zg_mu_ur, zg_log_sigma_ur = Es_zg.run(grid_reals_ur, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    zg_mu_bl, zg_log_sigma_bl = Es_zg.run(grid_reals_bl, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    zg_mu_br, zg_log_sigma_br = Es_zg.run(grid_reals_br, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    zg_latents_ul = np.array(zg_mu_ul)
    zg_latents_ur = np.array(zg_mu_ur)
    zg_latents_bl = np.array(zg_mu_bl)
    zg_latents_br = np.array(zg_mu_br)

    # encode zl latent tensors
    print('zl encoding...')
    zl_mu_ul, zl_log_sigma_ul = Es_zl.run(grid_reals_ul, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    zl_mu_ur, zl_log_sigma_ur = Es_zl.run(grid_reals_ur, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    zl_mu_bl, zl_log_sigma_bl = Es_zl.run(grid_reals_bl, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    zl_mu_br, zl_log_sigma_br = Es_zl.run(grid_reals_br, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    zl_latents_ul = np.array(zl_mu_ul)
    zl_latents_ur = np.array(zl_mu_ur)
    zl_latents_bl = np.array(zl_mu_bl)
    zl_latents_br = np.array(zl_mu_br)

    # compute permutation matrices
    def permutation_matrix_w_sampler(scale_w):
        if config.block_size == 0:
            temp_perm = np.eye(Es_zl.output_shapes[0][3]*scale_w)
            for idx in range(int(np.log(Es_zl.output_shapes[0][3]))):
                block_size_temp = int(2**idx)
                if config.perm:
                    perm = np.random.permutation(np.eye(Es_zl.output_shapes[0][3]*scale_w//block_size_temp))
                else:
                    perm = my_swap_w(np.eye(Es_zl.output_shapes[0][3]*scale_w//block_size_temp))
                cur_perm = block_permutation(perm, block_size_temp)
                temp_perm = np.matmul(cur_perm, temp_perm)
            permutation_matrix_w = np.tile(temp_perm, [1,Es_zl.output_shapes[0][1],1,1])
        else:
            if config.perm:
                perm = np.random.permutation(np.eye(Es_zl.output_shapes[0][3]*scale_w//config.block_size))
            else:
                perm = my_swap_w(np.eye(Es_zl.output_shapes[0][3]*scale_w//config.block_size))
            permutation_matrix_w = block_permutation(perm, config.block_size)
            permutation_matrix_w = np.tile(permutation_matrix_w, [1,Es_zl.output_shapes[0][1],1,1])
        return permutation_matrix_w

    def permutation_matrix_h_sampler(scale_h):
        if config.block_size == 0:
            temp_perm = np.eye(Es_zl.output_shapes[0][2]*scale_h)
            for idx in range(int(np.log(Es_zl.output_shapes[0][2]))):
                block_size_temp = int(2**idx)
                if config.perm:
                    perm = np.random.permutation(np.eye(Es_zl.output_shapes[0][2]*scale_h//block_size_temp))
                else:
                    perm = my_swap_h(np.eye(Es_zl.output_shapes[0][2]*scale_h//block_size_temp))
                cur_perm = block_permutation(perm, block_size_temp)
                temp_perm = np.matmul(temp_perm, cur_perm)
            permutation_matrix_h = np.tile(temp_perm, [1,Es_zl.output_shapes[0][1],1,1])
        else:
            if config.perm:
                perm = np.random.permutation(np.eye(Es_zl.output_shapes[0][2]*scale_h//config.block_size))
            else:
                perm = my_swap_h(np.eye(Es_zl.output_shapes[0][2]*scale_h//config.block_size))
            permutation_matrix_h = block_permutation(perm, config.block_size)
            permutation_matrix_h = np.tile(permutation_matrix_h, [1,Es_zl.output_shapes[0][1],1,1])
        return permutation_matrix_h

    permutation_matrix_h_0_ul_list = []
    permutation_matrix_w_0_ul_list = []
    permutation_matrix_h_0_ur_list = []
    permutation_matrix_w_0_ur_list = []
    permutation_matrix_h_0_bl_list = []
    permutation_matrix_w_0_bl_list = []
    permutation_matrix_h_0_br_list = []
    permutation_matrix_w_0_br_list = []
    for count in range(changes+1):
        permutation_matrix_h_0_ul_list.append(permutation_matrix_h_sampler(scale_h))
        permutation_matrix_w_0_ul_list.append(permutation_matrix_w_sampler(scale_w))
        permutation_matrix_h_0_ur_list.append(permutation_matrix_h_sampler(scale_h))
        permutation_matrix_w_0_ur_list.append(permutation_matrix_w_sampler(scale_w))
        permutation_matrix_h_0_bl_list.append(permutation_matrix_h_sampler(scale_h))
        permutation_matrix_w_0_bl_list.append(permutation_matrix_w_sampler(scale_w))
        permutation_matrix_h_0_br_list.append(permutation_matrix_h_sampler(scale_h))
        permutation_matrix_w_0_br_list.append(permutation_matrix_w_sampler(scale_w))

    # compute alpha blending matt
    #kernel_weight_ul, kernel_weight_ur, kernel_weight_bl, kernel_weight_br = linkern_for_weight_square(out_length=Gs_fcn.input_shapes[1][2], latent_res=Gs.input_shapes[1][2])
    kernel_weight_ul, kernel_weight_ur, kernel_weight_bl, kernel_weight_br = linkern_for_weight_arbitrary_shape(out_h=Gs_fcn.input_shapes[1][2], out_w=Gs_fcn.input_shapes[1][2], latent_res=Gs.input_shapes[1][2])
    kernel_weight_ul = np.tile(kernel_weight_ul[np.newaxis, np.newaxis, :, :], [1, Es_zl.output_shapes[0][1], 1, 1])
    kernel_weight_ur = np.tile(kernel_weight_ur[np.newaxis, np.newaxis, :, :], [1, Es_zl.output_shapes[0][1], 1, 1])
    kernel_weight_bl = np.tile(kernel_weight_bl[np.newaxis, np.newaxis, :, :], [1, Es_zl.output_shapes[0][1], 1, 1])
    kernel_weight_br = np.tile(kernel_weight_br[np.newaxis, np.newaxis, :, :], [1, Es_zl.output_shapes[0][1], 1, 1])

    # Frame generation func for moviepy.
    def make_frame(t):
        for c in range(1, changes+1): 
            r = float(c)/float(changes)
            if r >= t/duration_sec:
                break
        
        #zg temporal interpolation
        zg_latents_ul_head = np.array(zg_latents_ul[c-1:c,:,:,:])
        zg_latents_ur_head = np.array(zg_latents_ur[c-1:c,:,:,:])
        zg_latents_bl_head = np.array(zg_latents_bl[c-1:c,:,:,:])
        zg_latents_br_head = np.array(zg_latents_br[c-1:c,:,:,:])
        zg_latents_ul_tail = np.array(zg_latents_ul[c:c+1,:,:,:])
        zg_latents_ur_tail = np.array(zg_latents_ur[c:c+1,:,:,:])
        zg_latents_bl_tail = np.array(zg_latents_bl[c:c+1,:,:,:])
        zg_latents_br_tail = np.array(zg_latents_br[c:c+1,:,:,:])
        cur_zg_latents_ul = zg_latents_ul_head * (r - t/duration_sec) * float(changes) + zg_latents_ul_tail * (1.0 - (r - t/duration_sec) * float(changes))
        cur_zg_latents_ur = zg_latents_ur_head * (r - t/duration_sec) * float(changes) + zg_latents_ur_tail * (1.0 - (r - t/duration_sec) * float(changes))
        cur_zg_latents_bl = zg_latents_bl_head * (r - t/duration_sec) * float(changes) + zg_latents_bl_tail * (1.0 - (r - t/duration_sec) * float(changes))
        cur_zg_latents_br = zg_latents_br_head * (r - t/duration_sec) * float(changes) + zg_latents_br_tail * (1.0 - (r - t/duration_sec) * float(changes))

        # zg spatial interpolation
        interp_zg_latents_ul = np.tile(cur_zg_latents_ul, [1, 1, Gs_fcn.input_shapes[1][2], Gs_fcn.input_shapes[1][3]])
        interp_zg_latents_ur = np.tile(cur_zg_latents_ur, [1, 1, Gs_fcn.input_shapes[1][2], Gs_fcn.input_shapes[1][3]])
        interp_zg_latents_bl = np.tile(cur_zg_latents_bl, [1, 1, Gs_fcn.input_shapes[1][2], Gs_fcn.input_shapes[1][3]])
        interp_zg_latents_br = np.tile(cur_zg_latents_br, [1, 1, Gs_fcn.input_shapes[1][2], Gs_fcn.input_shapes[1][3]])
        interp_zg_latents = interp_zg_latents_ul * kernel_weight_ul + interp_zg_latents_ur * kernel_weight_ur + interp_zg_latents_bl * kernel_weight_bl + interp_zg_latents_br * kernel_weight_br

        # zl temporal interpolation
        zl_latents_ul_head = np.array(zl_latents_ul[c-1:c,:,:,:])
        zl_latents_ur_head = np.array(zl_latents_ur[c-1:c,:,:,:])
        zl_latents_bl_head = np.array(zl_latents_bl[c-1:c,:,:,:])
        zl_latents_br_head = np.array(zl_latents_br[c-1:c,:,:,:])
        zl_latents_ul_tail = np.array(zl_latents_ul[c:c+1,:,:,:])
        zl_latents_ur_tail = np.array(zl_latents_ur[c:c+1,:,:,:])
        zl_latents_bl_tail = np.array(zl_latents_bl[c:c+1,:,:,:])
        zl_latents_br_tail = np.array(zl_latents_br[c:c+1,:,:,:])
        cur_zl_latents_ul = zl_latents_ul_head * (r - t/duration_sec) * float(changes) + zl_latents_ul_tail * (1.0 - (r - t/duration_sec) * float(changes))
        cur_zl_latents_ur = zl_latents_ur_head * (r - t/duration_sec) * float(changes) + zl_latents_ur_tail * (1.0 - (r - t/duration_sec) * float(changes))
        cur_zl_latents_bl = zl_latents_bl_head * (r - t/duration_sec) * float(changes) + zl_latents_bl_tail * (1.0 - (r - t/duration_sec) * float(changes))
        cur_zl_latents_br = zl_latents_br_head * (r - t/duration_sec) * float(changes) + zl_latents_br_tail * (1.0 - (r - t/duration_sec) * float(changes))

        # zl spatial interpolation
        permutation_matrix_h_0_ul = permutation_matrix_h_0_ul_list[0]
        permutation_matrix_h_0_ur = permutation_matrix_h_0_ur_list[0]
        permutation_matrix_h_0_bl = permutation_matrix_h_0_bl_list[0]
        permutation_matrix_h_0_br = permutation_matrix_h_0_br_list[0]
        permutation_matrix_w_0_ul = permutation_matrix_w_0_ul_list[0]
        permutation_matrix_w_0_ur = permutation_matrix_w_0_ur_list[0]
        permutation_matrix_w_0_bl = permutation_matrix_w_0_bl_list[0]
        permutation_matrix_w_0_br = permutation_matrix_w_0_br_list[0]
        interp_zl_latents_ul = np.matmul(np.matmul(permutation_matrix_h_0_ul, np.tile(cur_zl_latents_ul, [1,1,scale_h,scale_w])), permutation_matrix_w_0_ul)
        interp_zl_latents_ur = np.matmul(np.matmul(permutation_matrix_h_0_ur, np.tile(cur_zl_latents_ur, [1,1,scale_h,scale_w])), permutation_matrix_w_0_ur)
        interp_zl_latents_bl = np.matmul(np.matmul(permutation_matrix_h_0_bl, np.tile(cur_zl_latents_bl, [1,1,scale_h,scale_w])), permutation_matrix_w_0_bl)
        interp_zl_latents_br = np.matmul(np.matmul(permutation_matrix_h_0_br, np.tile(cur_zl_latents_br, [1,1,scale_h,scale_w])), permutation_matrix_w_0_br)
        interp_zl_latents = interp_zl_latents_ul * kernel_weight_ul + interp_zl_latents_ur * kernel_weight_ur + interp_zl_latents_bl * kernel_weight_bl + interp_zl_latents_br * kernel_weight_br

        # generation
        images = Gs_fcn.run(interp_zg_latents, interp_zl_latents, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_mul=127.5, out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8)
        grid = misc.create_image_grid(images, [1, 1]).transpose(1, 2, 0) # HWC

        images_ul = grid_reals_ul[c-1:c,:,:,:] * (r - t/duration_sec) * float(changes) + grid_reals_ul[c:c+1,:,:,:] * (1.0 - (r - t/duration_sec) * float(changes))
        images_ul = images_ul * 127.5 + 127.5; images_ul = images_ul.astype(np.uint8)
        images_ur = grid_reals_ur[c-1:c,:,:,:] * (r - t/duration_sec) * float(changes) + grid_reals_ur[c:c+1,:,:,:] * (1.0 - (r - t/duration_sec) * float(changes))
        images_ur = images_ur * 127.5 + 127.5; images_ur = images_ur.astype(np.uint8)
        images_bl = grid_reals_bl[c-1:c,:,:,:] * (r - t/duration_sec) * float(changes) + grid_reals_bl[c:c+1,:,:,:] * (1.0 - (r - t/duration_sec) * float(changes))
        images_bl = images_bl * 127.5 + 127.5; images_bl = images_bl.astype(np.uint8)
        images_br = grid_reals_br[c-1:c,:,:,:] * (r - t/duration_sec) * float(changes) + grid_reals_br[c:c+1,:,:,:] * (1.0 - (r - t/duration_sec) * float(changes))
        images_br = images_br * 127.5 + 127.5; images_br = images_br.astype(np.uint8)
        return grid

    # Generate video.
    import moviepy.editor # pip install moviepy
    moviepy.editor.VideoClip(make_frame, duration=duration_sec).write_videofile(os.path.join(out_dir, 'texture_dissolve.mp4'), fps=mp4_fps, codec='libx264', bitrate=mp4_bitrate)

def im2graph(im):
    h = im.shape[0]; w = im.shape[1]
    graph = np.zeros((h*w, h*w)).astype(np.uint8)
    y, x = np.where(im>0)
    for i in range(len(y)):
        y0 = y[i]; x0 = x[i]
        idx0 = y0 + x0 * h
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if abs(dy) + abs(dx) > 0:
                    if im[y0+dy, x0+dx] > 0:
                        idx1 = (y0+dy) + (x0+dx) * h
                        graph[idx0, idx1] = 1
                        graph[idx1, idx0] = 1
    return graph

def im2trajectory(im):
    h = im.shape[0]; w = im.shape[1]
    graph = im2graph(im)
    cur_idx = None
    trajectory = []
    while np.sum(graph) > 0:
        if cur_idx is None:
            degree = np.sum(graph, axis=1)
            idx_valid = np.where(degree>0)[0]
            degree_valid = degree[idx_valid]
            idx_min = np.argmin(degree_valid)
            cur_idx = idx_valid[idx_min]
        else:
            cur_y = cur_idx % h
            cur_x = cur_idx // h
            trajectory.append([cur_y, cur_x])
            if np.sum(graph[cur_idx,:]) == 0:
                cur_idx = None
            else:
                next_idx = np.where(graph[cur_idx,:]>0)[0][0]
                graph[cur_idx, next_idx] = 0
                graph[next_idx, cur_idx] = 0
                cur_idx = next_idx
    if cur_idx is not None:
        cur_y = cur_idx % h
        cur_x = cur_idx // h
        trajectory.append([cur_y, cur_x])
    return trajectory

def texture_brush_video(model_path, imageBgUL_path, imageBgUR_path, imageBgBL_path, imageBgBR_path, imageFgUL_path, imageFgUR_path, imageFgBL_path, imageFgBR_path, stroke1_path, stroke2_path, stroke3_path, stroke4_path, out_dir, scale_h_bg=4, scale_w_bg=16, scale_h_fg=8, scale_w_fg=8, stroke_radius_div=128.0, minibatch_size=1, mp4_fps=60, mp4_codec='libx265', mp4_bitrate='16M'):
    if not os.path.isdir(out_dir): os.makedirs(out_dir)

    # Load model
    E_zg, E_zl, G, D_rec, D_interp, D_blend, Es_zg, Es_zl, Gs = misc.load_pkl(model_path)
    Gs_fcn_bg = tfutil.Network('Gs', reuse=True, num_channels=Gs.output_shape[1], resolution=Gs.output_shape[2], scale_h=scale_h_bg, scale_w=scale_w_bg, **config.G)
    Gs_fcn_fg = tfutil.Network('Gs', reuse=True, num_channels=Gs.output_shape[1], resolution=Gs.output_shape[2], scale_h=scale_h_fg, scale_w=scale_w_fg, **config.G)  
    
    sample_fg_1 = np.array([0, 0, 2*Gs.input_shapes[0][2], 2*Gs.input_shapes[0][3]]) # brush_1 sampling position in the fg palatte
    sample_fg_2 = np.array([0, 0, 2*Gs.input_shapes[0][2], 6*Gs.input_shapes[0][3]]) # brush_2 sampling position in the fg palatte
    sample_fg_3 = np.array([0, 0, 6*Gs.input_shapes[0][2], 2*Gs.input_shapes[0][3]]) # brush_3 sampling position in the fg palatte
    sample_fg_4 = np.array([0, 0, 6*Gs.input_shapes[0][2], 6*Gs.input_shapes[0][3]]) # brush_4 sampling position in the fg palatte
    bg_ul = np.transpose(misc.adjust_dynamic_range(np.array(PIL.Image.open(imageBgUL_path)).astype(np.float32), [0,255], [-1,1]), axes=[2,0,1])[np.newaxis,:,:,:]
    bg_ur = np.transpose(misc.adjust_dynamic_range(np.array(PIL.Image.open(imageBgUR_path)).astype(np.float32), [0,255], [-1,1]), axes=[2,0,1])[np.newaxis,:,:,:]
    bg_bl = np.transpose(misc.adjust_dynamic_range(np.array(PIL.Image.open(imageBgBL_path)).astype(np.float32), [0,255], [-1,1]), axes=[2,0,1])[np.newaxis,:,:,:]
    bg_br = np.transpose(misc.adjust_dynamic_range(np.array(PIL.Image.open(imageBgBR_path)).astype(np.float32), [0,255], [-1,1]), axes=[2,0,1])[np.newaxis,:,:,:]
    fg_ul = np.transpose(misc.adjust_dynamic_range(np.array(PIL.Image.open(imageFgUL_path)).astype(np.float32), [0,255], [-1,1]), axes=[2,0,1])[np.newaxis,:,:,:]
    fg_ur = np.transpose(misc.adjust_dynamic_range(np.array(PIL.Image.open(imageFgUR_path)).astype(np.float32), [0,255], [-1,1]), axes=[2,0,1])[np.newaxis,:,:,:]
    fg_bl = np.transpose(misc.adjust_dynamic_range(np.array(PIL.Image.open(imageFgBL_path)).astype(np.float32), [0,255], [-1,1]), axes=[2,0,1])[np.newaxis,:,:,:]
    fg_br = np.transpose(misc.adjust_dynamic_range(np.array(PIL.Image.open(imageFgBR_path)).astype(np.float32), [0,255], [-1,1]), axes=[2,0,1])[np.newaxis,:,:,:]

    # encode zg
    print('zg encoding...')
    zg_mu_bg_ul, zg_log_sigma_bg_ul = Es_zg.run(bg_ul, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    zg_mu_bg_ur, zg_log_sigma_bg_ur = Es_zg.run(bg_ur, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    zg_mu_bg_bl, zg_log_sigma_bg_bl = Es_zg.run(bg_bl, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    zg_mu_bg_br, zg_log_sigma_bg_br = Es_zg.run(bg_br, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    zg_latents_bg_ul = np.array(zg_mu_bg_ul)
    zg_latents_bg_ur = np.array(zg_mu_bg_ur)
    zg_latents_bg_bl = np.array(zg_mu_bg_bl)
    zg_latents_bg_br = np.array(zg_mu_bg_br)
    zg_mu_fg_ul, zg_log_sigma_fg_ul = Es_zg.run(fg_ul, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    zg_mu_fg_ur, zg_log_sigma_fg_ur = Es_zg.run(fg_ur, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    zg_mu_fg_bl, zg_log_sigma_fg_bl = Es_zg.run(fg_bl, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    zg_mu_fg_br, zg_log_sigma_fg_br = Es_zg.run(fg_br, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    zg_latents_fg_ul = np.array(zg_mu_fg_ul)
    zg_latents_fg_ur = np.array(zg_mu_fg_ur)
    zg_latents_fg_bl = np.array(zg_mu_fg_bl)
    zg_latents_fg_br = np.array(zg_mu_fg_br)

    # encode zl
    print('zl encoding...')
    zl_mu_bg_ul, zl_log_sigma_bg_ul = Es_zl.run(bg_ul, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    zl_mu_bg_ur, zl_log_sigma_bg_ur = Es_zl.run(bg_ur, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    zl_mu_bg_bl, zl_log_sigma_bg_bl = Es_zl.run(bg_bl, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    zl_mu_bg_br, zl_log_sigma_bg_br = Es_zl.run(bg_br, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    zl_latents_bg_ul = np.array(zl_mu_bg_ul)
    zl_latents_bg_ur = np.array(zl_mu_bg_ur)
    zl_latents_bg_bl = np.array(zl_mu_bg_bl)
    zl_latents_bg_br = np.array(zl_mu_bg_br)
    zl_mu_fg_ul, zl_log_sigma_fg_ul = Es_zl.run(fg_ul, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    zl_mu_fg_ur, zl_log_sigma_fg_ur = Es_zl.run(fg_ur, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    zl_mu_fg_bl, zl_log_sigma_fg_bl = Es_zl.run(fg_bl, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    zl_mu_fg_br, zl_log_sigma_fg_br = Es_zl.run(fg_br, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    zl_latents_fg_ul = np.array(zl_mu_fg_ul)
    zl_latents_fg_ur = np.array(zl_mu_fg_ur)
    zl_latents_fg_bl = np.array(zl_mu_fg_bl)
    zl_latents_fg_br = np.array(zl_mu_fg_br)

    def permutation_matrix_w_sampler(scale_w):
        if config.block_size == 0:
            temp_perm = np.eye(Es_zl.output_shapes[0][3]*scale_w)
            for idx in range(int(np.log(Es_zl.output_shapes[0][3]))):
                block_size_temp = int(2**idx)
                if config.perm:
                    perm = np.random.permutation(np.eye(Es_zl.output_shapes[0][3]*scale_w//block_size_temp))
                else:
                    perm = my_swap_w(np.eye(Es_zl.output_shapes[0][3]*scale_w//block_size_temp))
                cur_perm = block_permutation(perm, block_size_temp)
                if scale_w == 8:
                    im = cur_perm.astype(np.uint8) * 255
                    im = np.dstack((im, im, im))
                temp_perm = np.matmul(cur_perm, temp_perm)
            if scale_w == 8:
                im = temp_perm.astype(np.uint8) * 255
                im = np.dstack((im, im, im))
            permutation_matrix_w = np.tile(temp_perm, [1,Es_zl.output_shapes[0][1],1,1])
        else:
            if config.perm:
                perm = np.random.permutation(np.eye(Es_zl.output_shapes[0][3]*scale_w//config.block_size))
            else:
                perm = my_swap_w(np.eye(Es_zl.output_shapes[0][3]*scale_w//config.block_size))
            permutation_matrix_w = block_permutation(perm, config.block_size)
            permutation_matrix_w = np.tile(permutation_matrix_w, [1,Es_zl.output_shapes[0][1],1,1])
        return permutation_matrix_w

    def permutation_matrix_h_sampler(scale_h):
        if config.block_size == 0:
            temp_perm = np.eye(Es_zl.output_shapes[0][2]*scale_h)
            for idx in range(int(np.log(Es_zl.output_shapes[0][2]))):
                block_size_temp = int(2**idx)
                if config.perm:
                    perm = np.random.permutation(np.eye(Es_zl.output_shapes[0][2]*scale_h//block_size_temp))
                else:
                    perm = my_swap_h(np.eye(Es_zl.output_shapes[0][2]*scale_h//block_size_temp))
                cur_perm = block_permutation(perm, block_size_temp)
                if scale_h == 8:
                    im = cur_perm.astype(np.uint8) * 255
                    im = np.dstack((im, im, im))
                temp_perm = np.matmul(temp_perm, cur_perm)
            if scale_h == 8:
                im = temp_perm.astype(np.uint8) * 255
                im = np.dstack((im, im, im))
            permutation_matrix_h = np.tile(temp_perm, [1,Es_zl.output_shapes[0][1],1,1])
        else:
            if config.perm:
                perm = np.random.permutation(np.eye(Es_zl.output_shapes[0][2]*scale_h//config.block_size))
            else:
                perm = my_swap_h(np.eye(Es_zl.output_shapes[0][2]*scale_h//config.block_size))
            permutation_matrix_h = block_permutation(perm, config.block_size)
            permutation_matrix_h = np.tile(permutation_matrix_h, [1,Es_zl.output_shapes[0][1],1,1])
        return permutation_matrix_h

    # compute permutation matrices
    permutation_matrix_w_0_bg_ul = permutation_matrix_w_sampler(scale_w_bg)
    permutation_matrix_h_0_bg_ul = permutation_matrix_h_sampler(scale_h_bg)
    permutation_matrix_w_0_bg_ur = permutation_matrix_w_sampler(scale_w_bg)
    permutation_matrix_h_0_bg_ur = permutation_matrix_h_sampler(scale_h_bg)
    permutation_matrix_w_0_bg_bl = permutation_matrix_w_sampler(scale_w_bg)
    permutation_matrix_h_0_bg_bl = permutation_matrix_h_sampler(scale_h_bg)
    permutation_matrix_w_0_bg_br = permutation_matrix_w_sampler(scale_w_bg)
    permutation_matrix_h_0_bg_br = permutation_matrix_h_sampler(scale_h_bg)
    permutation_matrix_w_0_fg_ul = permutation_matrix_w_sampler(scale_w_fg)
    permutation_matrix_h_0_fg_ul = permutation_matrix_h_sampler(scale_h_fg)
    permutation_matrix_w_0_fg_ur = permutation_matrix_w_sampler(scale_w_fg)
    permutation_matrix_h_0_fg_ur = permutation_matrix_h_sampler(scale_h_fg)
    permutation_matrix_w_0_fg_bl = permutation_matrix_w_sampler(scale_w_fg)
    permutation_matrix_h_0_fg_bl = permutation_matrix_h_sampler(scale_h_fg)
    permutation_matrix_w_0_fg_br = permutation_matrix_w_sampler(scale_w_fg)
    permutation_matrix_h_0_fg_br = permutation_matrix_h_sampler(scale_h_fg)

    # compute alpha blending matt
    kernel_weight_bg_ul, kernel_weight_bg_ur, kernel_weight_bg_bl, kernel_weight_bg_br = linkern_for_weight_arbitrary_shape(out_h=Gs_fcn_bg.input_shape[2], out_w=Gs_fcn_bg.input_shape[3], latent_res=Gs.input_shapes[0][2])
    kernel_weight_bg_ul = np.tile(kernel_weight_bg_ul[np.newaxis, np.newaxis, :, :], [1, Es_zl.output_shapes[0][1], 1, 1])
    kernel_weight_bg_ur = np.tile(kernel_weight_bg_ur[np.newaxis, np.newaxis, :, :], [1, Es_zl.output_shapes[0][1], 1, 1])
    kernel_weight_bg_bl = np.tile(kernel_weight_bg_bl[np.newaxis, np.newaxis, :, :], [1, Es_zl.output_shapes[0][1], 1, 1])
    kernel_weight_bg_br = np.tile(kernel_weight_bg_br[np.newaxis, np.newaxis, :, :], [1, Es_zl.output_shapes[0][1], 1, 1])
    kernel_weight_fg_ul, kernel_weight_fg_ur, kernel_weight_fg_bl, kernel_weight_fg_br = linkern_for_weight_arbitrary_shape(out_h=Gs_fcn_fg.input_shape[2], out_w=Gs_fcn_fg.input_shape[3], latent_res=Gs.input_shapes[0][2])
    kernel_weight_fg_ul = np.tile(kernel_weight_fg_ul[np.newaxis, np.newaxis, :, :], [1, Es_zl.output_shapes[0][1], 1, 1])
    kernel_weight_fg_ur = np.tile(kernel_weight_fg_ur[np.newaxis, np.newaxis, :, :], [1, Es_zl.output_shapes[0][1], 1, 1])
    kernel_weight_fg_bl = np.tile(kernel_weight_fg_bl[np.newaxis, np.newaxis, :, :], [1, Es_zl.output_shapes[0][1], 1, 1])
    kernel_weight_fg_br = np.tile(kernel_weight_fg_br[np.newaxis, np.newaxis, :, :], [1, Es_zl.output_shapes[0][1], 1, 1])

    # interpolate zg
    interp_zg_latents_bg_ul = np.tile(zg_latents_bg_ul, [1, 1, Gs_fcn_bg.input_shapes[0][2], Gs_fcn_bg.input_shapes[0][3]])
    interp_zg_latents_bg_ur = np.tile(zg_latents_bg_ur, [1, 1, Gs_fcn_bg.input_shapes[0][2], Gs_fcn_bg.input_shapes[0][3]])
    interp_zg_latents_bg_bl = np.tile(zg_latents_bg_bl, [1, 1, Gs_fcn_bg.input_shapes[0][2], Gs_fcn_bg.input_shapes[0][3]])
    interp_zg_latents_bg_br = np.tile(zg_latents_bg_br, [1, 1, Gs_fcn_bg.input_shapes[0][2], Gs_fcn_bg.input_shapes[0][3]])
    interp_zg_latents_bg = interp_zg_latents_bg_ul * kernel_weight_bg_ul + interp_zg_latents_bg_ur * kernel_weight_bg_ur + interp_zg_latents_bg_bl * kernel_weight_bg_bl + interp_zg_latents_bg_br * kernel_weight_bg_br
    interp_zg_latents_fg_ul = np.tile(zg_latents_fg_ul, [1, 1, Gs_fcn_fg.input_shapes[0][2], Gs_fcn_fg.input_shapes[0][3]])
    interp_zg_latents_fg_ur = np.tile(zg_latents_fg_ur, [1, 1, Gs_fcn_fg.input_shapes[0][2], Gs_fcn_fg.input_shapes[0][3]])
    interp_zg_latents_fg_bl = np.tile(zg_latents_fg_bl, [1, 1, Gs_fcn_fg.input_shapes[0][2], Gs_fcn_fg.input_shapes[0][3]])
    interp_zg_latents_fg_br = np.tile(zg_latents_fg_br, [1, 1, Gs_fcn_fg.input_shapes[0][2], Gs_fcn_fg.input_shapes[0][3]])
    interp_zg_latents_fg = interp_zg_latents_fg_ul * kernel_weight_fg_ul + interp_zg_latents_fg_ur * kernel_weight_fg_ur + interp_zg_latents_fg_bl * kernel_weight_fg_bl + interp_zg_latents_fg_br * kernel_weight_fg_br

    # interpolate zl
    interp_zl_latents_bg_ul = np.matmul(np.matmul(permutation_matrix_h_0_bg_ul, np.tile(zl_latents_bg_ul, [1,1,scale_h_bg,scale_w_bg])), permutation_matrix_w_0_bg_ul)
    interp_zl_latents_bg_ul[:, :, :Es_zl.output_shapes[0][2], :Es_zl.output_shapes[0][3]] = zl_latents_bg_ul
    interp_zl_latents_bg_ul[:, :, :Es_zl.output_shapes[0][2], -Es_zl.output_shapes[0][3]:] = zl_latents_bg_ul
    interp_zl_latents_bg_ul[:, :, -Es_zl.output_shapes[0][2]:, :Es_zl.output_shapes[0][3]] = zl_latents_bg_ul
    interp_zl_latents_bg_ul[:, :, -Es_zl.output_shapes[0][2]:, -Es_zl.output_shapes[0][3]:] = zl_latents_bg_ul
    interp_zl_latents_bg_ur = np.matmul(np.matmul(permutation_matrix_h_0_bg_ur, np.tile(zl_latents_bg_ur, [1,1,scale_h_bg,scale_w_bg])), permutation_matrix_w_0_bg_ur)
    interp_zl_latents_bg_ur[:, :, :Es_zl.output_shapes[0][2], :Es_zl.output_shapes[0][3]] = zl_latents_bg_ur
    interp_zl_latents_bg_ur[:, :, :Es_zl.output_shapes[0][2], -Es_zl.output_shapes[0][3]:] = zl_latents_bg_ur
    interp_zl_latents_bg_ur[:, :, -Es_zl.output_shapes[0][2]:, :Es_zl.output_shapes[0][3]] = zl_latents_bg_ur
    interp_zl_latents_bg_ur[:, :, -Es_zl.output_shapes[0][2]:, -Es_zl.output_shapes[0][3]:] = zl_latents_bg_ur
    interp_zl_latents_bg_bl = np.matmul(np.matmul(permutation_matrix_h_0_bg_bl, np.tile(zl_latents_bg_bl, [1,1,scale_h_bg,scale_w_bg])), permutation_matrix_w_0_bg_bl)
    interp_zl_latents_bg_bl[:, :, :Es_zl.output_shapes[0][2], :Es_zl.output_shapes[0][3]] = zl_latents_bg_bl
    interp_zl_latents_bg_bl[:, :, :Es_zl.output_shapes[0][2], -Es_zl.output_shapes[0][3]:] = zl_latents_bg_bl
    interp_zl_latents_bg_bl[:, :, -Es_zl.output_shapes[0][2]:, :Es_zl.output_shapes[0][3]] = zl_latents_bg_bl
    interp_zl_latents_bg_bl[:, :, -Es_zl.output_shapes[0][2]:, -Es_zl.output_shapes[0][3]:] = zl_latents_bg_bl
    interp_zl_latents_bg_br = np.matmul(np.matmul(permutation_matrix_h_0_bg_br, np.tile(zl_latents_bg_br, [1,1,scale_h_bg,scale_w_bg])), permutation_matrix_w_0_bg_br)
    interp_zl_latents_bg_br[:, :, :Es_zl.output_shapes[0][2], :Es_zl.output_shapes[0][3]] = zl_latents_bg_br
    interp_zl_latents_bg_br[:, :, :Es_zl.output_shapes[0][2], -Es_zl.output_shapes[0][3]:] = zl_latents_bg_br
    interp_zl_latents_bg_br[:, :, -Es_zl.output_shapes[0][2]:, :Es_zl.output_shapes[0][3]] = zl_latents_bg_br
    interp_zl_latents_bg_br[:, :, -Es_zl.output_shapes[0][2]:, -Es_zl.output_shapes[0][3]:] = zl_latents_bg_br
    interp_zl_latents_bg = interp_zl_latents_bg_ul * kernel_weight_bg_ul + interp_zl_latents_bg_ur * kernel_weight_bg_ur + interp_zl_latents_bg_bl * kernel_weight_bg_bl + interp_zl_latents_bg_br * kernel_weight_bg_br
    interp_zl_latents_fg_ul = np.matmul(np.matmul(permutation_matrix_h_0_fg_ul, np.tile(zl_latents_fg_ul, [1,1,scale_h_fg,scale_w_fg])), permutation_matrix_w_0_fg_ul)
    interp_zl_latents_fg_ul[:, :, :Es_zl.output_shapes[0][2], :Es_zl.output_shapes[0][3]] = zl_latents_fg_ul
    interp_zl_latents_fg_ul[:, :, :Es_zl.output_shapes[0][2], -Es_zl.output_shapes[0][3]:] = zl_latents_fg_ul
    interp_zl_latents_fg_ul[:, :, -Es_zl.output_shapes[0][2]:, :Es_zl.output_shapes[0][3]] = zl_latents_fg_ul
    interp_zl_latents_fg_ul[:, :, -Es_zl.output_shapes[0][2]:, -Es_zl.output_shapes[0][3]:] = zl_latents_fg_ul
    interp_zl_latents_fg_ur = np.matmul(np.matmul(permutation_matrix_h_0_fg_ur, np.tile(zl_latents_fg_ur, [1,1,scale_h_fg,scale_w_fg])), permutation_matrix_w_0_fg_ur)
    interp_zl_latents_fg_ur[:, :, :Es_zl.output_shapes[0][2], :Es_zl.output_shapes[0][3]] = zl_latents_fg_ur
    interp_zl_latents_fg_ur[:, :, :Es_zl.output_shapes[0][2], -Es_zl.output_shapes[0][3]:] = zl_latents_fg_ur
    interp_zl_latents_fg_ur[:, :, -Es_zl.output_shapes[0][2]:, :Es_zl.output_shapes[0][3]] = zl_latents_fg_ur
    interp_zl_latents_fg_ur[:, :, -Es_zl.output_shapes[0][2]:, -Es_zl.output_shapes[0][3]:] = zl_latents_fg_ur
    interp_zl_latents_fg_bl = np.matmul(np.matmul(permutation_matrix_h_0_fg_bl, np.tile(zl_latents_fg_bl, [1,1,scale_h_fg,scale_w_fg])), permutation_matrix_w_0_fg_bl)
    interp_zl_latents_fg_bl[:, :, :Es_zl.output_shapes[0][2], :Es_zl.output_shapes[0][3]] = zl_latents_fg_bl
    interp_zl_latents_fg_bl[:, :, :Es_zl.output_shapes[0][2], -Es_zl.output_shapes[0][3]:] = zl_latents_fg_bl
    interp_zl_latents_fg_bl[:, :, -Es_zl.output_shapes[0][2]:, :Es_zl.output_shapes[0][3]] = zl_latents_fg_bl
    interp_zl_latents_fg_bl[:, :, -Es_zl.output_shapes[0][2]:, -Es_zl.output_shapes[0][3]:] = zl_latents_fg_bl
    interp_zl_latents_fg_br = np.matmul(np.matmul(permutation_matrix_h_0_fg_br, np.tile(zl_latents_fg_br, [1,1,scale_h_fg,scale_w_fg])), permutation_matrix_w_0_fg_br)
    interp_zl_latents_fg_br[:, :, :Es_zl.output_shapes[0][2], :Es_zl.output_shapes[0][3]] = zl_latents_fg_br
    interp_zl_latents_fg_br[:, :, :Es_zl.output_shapes[0][2], -Es_zl.output_shapes[0][3]:] = zl_latents_fg_br
    interp_zl_latents_fg_br[:, :, -Es_zl.output_shapes[0][2]:, :Es_zl.output_shapes[0][3]] = zl_latents_fg_br
    interp_zl_latents_fg_br[:, :, -Es_zl.output_shapes[0][2]:, -Es_zl.output_shapes[0][3]:] = zl_latents_fg_br
    interp_zl_latents_fg = interp_zl_latents_fg_ul * kernel_weight_fg_ul + interp_zl_latents_fg_ur * kernel_weight_fg_ur + interp_zl_latents_fg_bl * kernel_weight_fg_bl + interp_zl_latents_fg_br * kernel_weight_fg_br

    # generate interpolated image
    images_bg = Gs_fcn_bg.run(interp_zg_latents_bg, interp_zl_latents_bg, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    images_fg = Gs_fcn_fg.run(interp_zg_latents_fg, interp_zl_latents_fg, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    images_fg = images_fg * 127.5 + 127.5; images_fg = images_fg.astype(np.uint8)
    images_fg = misc.create_image_grid(images_fg, [1,1]).transpose(1,2,0) # HWC
    images_ul = np.array(fg_ul); images_ul = images_ul * 127.5 + 127.5; images_ul = images_ul.astype(np.uint8)
    images_ur = np.array(fg_ur); images_ur = images_ur * 127.5 + 127.5; images_ur = images_ur.astype(np.uint8)
    images_bl = np.array(fg_bl); images_bl = images_bl * 127.5 + 127.5; images_bl = images_bl.astype(np.uint8)
    images_br = np.array(fg_br); images_br = images_br * 127.5 + 127.5; images_br = images_br.astype(np.uint8)
    h_plus = images_ul.shape[2]; w_plus = images_ul.shape[3]
    grid_plus = np.ones((images_fg.shape[0]+2*h_plus, images_fg.shape[1]+2*w_plus, images_fg.shape[2])).astype(np.uint8) * 255
    grid_plus[:h_plus,:w_plus,:] = misc.create_image_grid(images_ul, [1,1]).transpose(1,2,0); grid_plus[:h_plus,-w_plus:,:] = misc.create_image_grid(images_ur, [1,1]).transpose(1,2,0); grid_plus[-h_plus:,:w_plus,:] = misc.create_image_grid(images_bl, [1,1]).transpose(1,2,0); grid_plus[-h_plus:,-w_plus:,:] = misc.create_image_grid(images_br, [1,1]).transpose(1,2,0)
    grid_plus[h_plus:-h_plus, w_plus:-w_plus, :] = images_fg
    
    # sample foreground patches and tile
    interp_zg_latents_fg_1 = np.tile(np.mean(interp_zg_latents_fg[:, :, sample_fg_1[2]:sample_fg_1[2]+Gs.input_shapes[0][2], sample_fg_1[3]:sample_fg_1[3]+Gs.input_shapes[0][3]], axis=(2,3), keepdims=True), [1, 1, Gs_fcn_bg.input_shapes[0][2], Gs_fcn_bg.input_shapes[0][3]])
    interp_zl_latents_fg_1 = np.tile(interp_zl_latents_fg[:, :, sample_fg_1[2]:sample_fg_1[2]+Gs.input_shapes[0][2], sample_fg_1[3]:sample_fg_1[3]+Gs.input_shapes[0][3]], [1, 1, scale_h_bg, scale_w_bg])
    permutation_matrix_w_0_fg = permutation_matrix_w_sampler(scale_w_bg)
    permutation_matrix_h_0_fg = permutation_matrix_h_sampler(scale_h_bg)
    interp_zl_latents_fg_1 = np.matmul(np.matmul(permutation_matrix_h_0_fg, interp_zl_latents_fg_1), permutation_matrix_w_0_fg)
    images_sample_fg_1 = Gs_fcn_bg.run(interp_zg_latents_fg_1, interp_zl_latents_fg_1, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    interp_zg_latents_fg_2 = np.tile(np.mean(interp_zg_latents_fg[:, :, sample_fg_2[2]:sample_fg_2[2]+Gs.input_shapes[0][2], sample_fg_2[3]:sample_fg_2[3]+Gs.input_shapes[0][3]], axis=(2,3), keepdims=True), [1, 1, Gs_fcn_bg.input_shapes[0][2], Gs_fcn_bg.input_shapes[0][3]])
    interp_zl_latents_fg_2 = np.tile(interp_zl_latents_fg[:, :, sample_fg_2[2]:sample_fg_2[2]+Gs.input_shapes[0][2], sample_fg_2[3]:sample_fg_2[3]+Gs.input_shapes[0][3]], [1, 1, scale_h_bg, scale_w_bg])
    permutation_matrix_w_0_fg = permutation_matrix_w_sampler(scale_w_bg)
    permutation_matrix_h_0_fg = permutation_matrix_h_sampler(scale_h_bg)
    interp_zl_latents_fg_2 = np.matmul(np.matmul(permutation_matrix_h_0_fg, interp_zl_latents_fg_2), permutation_matrix_w_0_fg)
    images_sample_fg_2 = Gs_fcn_bg.run(interp_zg_latents_fg_2, interp_zl_latents_fg_2, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    interp_zg_latents_fg_3 = np.tile(np.mean(interp_zg_latents_fg[:, :, sample_fg_3[2]:sample_fg_3[2]+Gs.input_shapes[0][2], sample_fg_3[3]:sample_fg_3[3]+Gs.input_shapes[0][3]], axis=(2,3), keepdims=True), [1, 1, Gs_fcn_bg.input_shapes[0][2], Gs_fcn_bg.input_shapes[0][3]])
    interp_zl_latents_fg_3 = np.tile(interp_zl_latents_fg[:, :, sample_fg_3[2]:sample_fg_3[2]+Gs.input_shapes[0][2], sample_fg_3[3]:sample_fg_3[3]+Gs.input_shapes[0][3]], [1, 1, scale_h_bg, scale_w_bg])
    permutation_matrix_w_0_fg = permutation_matrix_w_sampler(scale_w_bg)
    permutation_matrix_h_0_fg = permutation_matrix_h_sampler(scale_h_bg)
    interp_zl_latents_fg_3 = np.matmul(np.matmul(permutation_matrix_h_0_fg, interp_zl_latents_fg_3), permutation_matrix_w_0_fg)
    images_sample_fg_3 = Gs_fcn_bg.run(interp_zg_latents_fg_3, interp_zl_latents_fg_3, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    interp_zg_latents_fg_4 = np.tile(np.mean(interp_zg_latents_fg[:, :, sample_fg_4[2]:sample_fg_4[2]+Gs.input_shapes[0][2], sample_fg_4[3]:sample_fg_4[3]+Gs.input_shapes[0][3]], axis=(2,3), keepdims=True), [1, 1, Gs_fcn_bg.input_shapes[0][2], Gs_fcn_bg.input_shapes[0][3]])
    interp_zl_latents_fg_4 = np.tile(interp_zl_latents_fg[:, :, sample_fg_4[2]:sample_fg_4[2]+Gs.input_shapes[0][2], sample_fg_4[3]:sample_fg_4[3]+Gs.input_shapes[0][3]], [1, 1, scale_h_bg, scale_w_bg])
    permutation_matrix_w_0_fg = permutation_matrix_w_sampler(scale_w_bg)
    permutation_matrix_h_0_fg = permutation_matrix_h_sampler(scale_h_bg)
    interp_zl_latents_fg_4 = np.matmul(np.matmul(permutation_matrix_h_0_fg, interp_zl_latents_fg_4), permutation_matrix_w_0_fg)
    images_sample_fg_4 = Gs_fcn_bg.run(interp_zg_latents_fg_4, interp_zl_latents_fg_4, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    
    # pre-compute movement
    print('stroke configuring...')
    im_1 = np.array(PIL.Image.open(stroke1_path)); trajectory_1 = im2trajectory(im_1)
    im_2 = np.array(PIL.Image.open(stroke2_path)); trajectory_2 = im2trajectory(im_2)
    im_3 = np.array(PIL.Image.open(stroke3_path)); trajectory_3 = im2trajectory(im_3)
    im_4 = np.array(PIL.Image.open(stroke4_path)); trajectory_4 = im2trajectory(im_4)
    
    orig_1 = [0,0]; orig_2 = [0, 128]; orig_3 = [0, 256]; orig_4 = [0, 384]
    kernel_weight_1 = [np.zeros((Gs_fcn_bg.input_shapes[0][2], Gs_fcn_bg.input_shapes[0][3]))]
    kernel_weight_2 = [np.zeros((Gs_fcn_bg.input_shapes[0][2], Gs_fcn_bg.input_shapes[0][3]))]
    kernel_weight_3 = [np.zeros((Gs_fcn_bg.input_shapes[0][2], Gs_fcn_bg.input_shapes[0][3]))]
    kernel_weight_4 = [np.zeros((Gs_fcn_bg.input_shapes[0][2], Gs_fcn_bg.input_shapes[0][3]))]
    for spot in trajectory_1:
        y = orig_1[0] + spot[0]
        x = orig_1[1] + spot[1]
        kernel_weight_1.append(np.maximum(kernel_weight_1[-1], gkern_for_weight_arbitrary_shape(out_h=Gs_fcn_bg.input_shapes[0][2], out_w=Gs_fcn_bg.input_shapes[0][3], x=x, y=y, sig_div=stroke_radius_div))) #sig_div=stroke_radius_div*4.0
        kernel_weight_2.append(kernel_weight_2[-1])
        kernel_weight_3.append(kernel_weight_3[-1])
        kernel_weight_4.append(kernel_weight_4[-1])
    for spot in trajectory_2:
        y = orig_2[0] + spot[0]
        x = orig_2[1] + spot[1]
        kernel_weight_1.append(kernel_weight_1[-1])
        kernel_weight_2.append(np.maximum(kernel_weight_2[-1], gkern_for_weight_arbitrary_shape(out_h=Gs_fcn_bg.input_shapes[0][2], out_w=Gs_fcn_bg.input_shapes[0][3], x=x, y=y, sig_div=stroke_radius_div))) #sig_div=stroke_radius_div*2.0
        kernel_weight_3.append(kernel_weight_3[-1])
        kernel_weight_4.append(kernel_weight_4[-1])
    for spot in trajectory_3:
        y = orig_3[0] + spot[0]
        x = orig_3[1] + spot[1]
        kernel_weight_1.append(kernel_weight_1[-1])
        kernel_weight_2.append(kernel_weight_2[-1])
        kernel_weight_3.append(np.maximum(kernel_weight_3[-1], gkern_for_weight_arbitrary_shape(out_h=Gs_fcn_bg.input_shapes[0][2], out_w=Gs_fcn_bg.input_shapes[0][3], x=x, y=y, sig_div=stroke_radius_div))) #sig_div=stroke_radius_div
        kernel_weight_4.append(kernel_weight_4[-1])
    for spot in trajectory_4:
        y = orig_4[0] + spot[0]
        x = orig_4[1] + spot[1]
        kernel_weight_1.append(kernel_weight_1[-1])
        kernel_weight_2.append(kernel_weight_2[-1])
        kernel_weight_3.append(kernel_weight_3[-1])
        kernel_weight_4.append(np.maximum(kernel_weight_4[-1], gkern_for_weight_arbitrary_shape(out_h=Gs_fcn_bg.input_shapes[0][2], out_w=Gs_fcn_bg.input_shapes[0][3], x=x, y=y, sig_div=stroke_radius_div))) #sig_div=stroke_radius_div/1.25

    kernel_weight_1_cur = np.tile(kernel_weight_1[-1][np.newaxis, np.newaxis, :, :], [1, Es_zl.output_shapes[0][1], 1, 1])
    kernel_weight_2_cur = np.tile(kernel_weight_2[-1][np.newaxis, np.newaxis, :, :], [1, Es_zl.output_shapes[0][1], 1, 1])
    kernel_weight_3_cur = np.tile(kernel_weight_3[-1][np.newaxis, np.newaxis, :, :], [1, Es_zl.output_shapes[0][1], 1, 1])
    kernel_weight_4_cur = np.tile(kernel_weight_4[-1][np.newaxis, np.newaxis, :, :], [1, Es_zl.output_shapes[0][1], 1, 1])
    interp_zg_brush = interp_zg_latents_fg_1 * kernel_weight_1_cur + interp_zg_latents_bg * (1.0 - kernel_weight_1_cur)
    interp_zg_brush = interp_zg_latents_fg_2 * kernel_weight_2_cur + interp_zg_brush * (1.0 - kernel_weight_2_cur)
    interp_zg_brush = interp_zg_latents_fg_3 * kernel_weight_3_cur + interp_zg_brush * (1.0 - kernel_weight_3_cur)
    interp_zg_brush = interp_zg_latents_fg_4 * kernel_weight_4_cur + interp_zg_brush * (1.0 - kernel_weight_4_cur)
    interp_zl_brush = interp_zl_latents_fg_1 * kernel_weight_1_cur + interp_zl_latents_bg * (1.0 - kernel_weight_1_cur)
    interp_zl_brush = interp_zl_latents_fg_2 * kernel_weight_2_cur + interp_zl_brush * (1.0 - kernel_weight_2_cur)
    interp_zl_brush = interp_zl_latents_fg_3 * kernel_weight_3_cur + interp_zl_brush * (1.0 - kernel_weight_3_cur)
    interp_zl_brush = interp_zl_latents_fg_4 * kernel_weight_4_cur + interp_zl_brush * (1.0 - kernel_weight_4_cur)
    images_brush = Gs_fcn_bg.run(interp_zg_brush, interp_zl_brush, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    
    # Frame generation func for moviepy.
    def make_frame(t):
        idx = int(round(t * mp4_fps))
        kernel_weight_1_cur = np.tile(kernel_weight_1[idx][np.newaxis, np.newaxis, :, :], [1, Es_zl.output_shapes[0][1], 1, 1])
        kernel_weight_2_cur = np.tile(kernel_weight_2[idx][np.newaxis, np.newaxis, :, :], [1, Es_zl.output_shapes[0][1], 1, 1])
        kernel_weight_3_cur = np.tile(kernel_weight_3[idx][np.newaxis, np.newaxis, :, :], [1, Es_zl.output_shapes[0][1], 1, 1])
        kernel_weight_4_cur = np.tile(kernel_weight_4[idx][np.newaxis, np.newaxis, :, :], [1, Es_zl.output_shapes[0][1], 1, 1])
        interp_zg_brush = interp_zg_latents_fg_1 * kernel_weight_1_cur + interp_zg_latents_bg * (1.0 - kernel_weight_1_cur)
        interp_zg_brush = interp_zg_latents_fg_2 * kernel_weight_2_cur + interp_zg_brush * (1.0 - kernel_weight_2_cur)
        interp_zg_brush = interp_zg_latents_fg_3 * kernel_weight_3_cur + interp_zg_brush * (1.0 - kernel_weight_3_cur)
        interp_zg_brush = interp_zg_latents_fg_4 * kernel_weight_4_cur + interp_zg_brush * (1.0 - kernel_weight_4_cur)
        interp_zl_brush = interp_zl_latents_fg_1 * kernel_weight_1_cur + interp_zl_latents_bg * (1.0 - kernel_weight_1_cur)
        interp_zl_brush = interp_zl_latents_fg_2 * kernel_weight_2_cur + interp_zl_brush * (1.0 - kernel_weight_2_cur)
        interp_zl_brush = interp_zl_latents_fg_3 * kernel_weight_3_cur + interp_zl_brush * (1.0 - kernel_weight_3_cur)
        interp_zl_brush = interp_zl_latents_fg_4 * kernel_weight_4_cur + interp_zl_brush * (1.0 - kernel_weight_4_cur)
        images_brush = Gs_fcn_bg.run(interp_zg_brush, interp_zl_brush, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
        images_brush = images_brush * 127.5 + 127.5; images_brush = images_brush.astype(np.uint8)
        grid = misc.create_image_grid(images_brush, [1,1]).transpose(1,2,0) # HWC
        return grid

    # Generate video.
    import moviepy.editor # pip install moviepy
    moviepy.editor.VideoClip(make_frame, duration=float(len(kernel_weight_1))/float(mp4_fps)).write_videofile(os.path.join(out_dir, 'texture_brush.mp4'), fps=mp4_fps, codec='libx264', bitrate=mp4_bitrate)
    
#----------------------------------------------------------------------------
# Generate hybridization images.
# To run, uncomment the appropriate line in config.py and launch train.py.

def path2tensors(path, sample=True):
    im = np.array(PIL.Image.open(path))
    mask = misc.adjust_dynamic_range(im[:,:,3], [0,255], [0,1])
    mask = mask > 0.5
    mask = mask.astype(np.float32)
    if not sample:
        return mask
    im = np.transpose(misc.adjust_dynamic_range(im[:,:,:3], [0,255], [-1,1]), axes=[2,0,1])
    return im, mask

def rasterize_mask(mask1, mask2, mask3, size):
    grid1 = np.zeros(mask1.shape)
    grid2 = np.zeros(mask2.shape)
    grid3 = np.zeros(mask3.shape)
    for i in range(0, mask1.shape[0], size):
        for j in range(0, mask1.shape[1], size):
            sum1 = np.sum(mask1[i:i+size, j:j+size])
            sum2 = np.sum(mask2[i:i+size, j:j+size])
            sum3 = np.sum(mask3[i:i+size, j:j+size])
            if sum1 > 0.0:
                grid1[i:i+size, j:j+size] = 1.0
            elif sum2 > 0.0 or sum3 > 0.0:
                idx = np.argmax(np.array([sum2, sum3]))
                if idx == 0:
                    grid2[i:i+size, j:j+size] = 1.0
                else:
                    grid3[i:i+size, j:j+size] = 1.0
    return grid1, grid2, grid3

def im2angle(im):
    ret, labels = cv2.connectedComponents((im>0.0).astype(np.uint8))
    num_max = 0
    for label in np.unique(labels):
        if label != 0:
            num = np.sum((labels==label).astype(np.uint8))
            if num > num_max:
                num_max = num
                label_max = label
    boundary = (labels==label_max).astype(np.float32)
    graph = im2graph(boundary)
    degree = np.sum(graph, axis=1)
    idx_valid = np.where(degree==1)[0]
    assert len(idx_valid) == 2
    h = boundary.shape[0]
    end1 = np.array([idx_valid[0] % h, idx_valid[0] // h]).astype(np.float32)
    end2 = np.array([idx_valid[1] % h, idx_valid[1] // h]).astype(np.float32)
    direction = end1 - end2
    if direction[0] < 0.0:
        direction *= -1.0
    angle = np.arctan2(-direction[0], direction[1]) / math.pi * 180.0
    return angle

def hybridization_CAF(model_path, source_dir, out_dir, rotation_enabled=True, train_size=128, weight_mode='horizontal_linear', sig_div=4.0, minibatch_size=1):
    if not os.path.isdir(out_dir): os.makedirs(out_dir)

    # read sources
    print('reading sources...')
    sample1_path = '%s/sample1.png' % source_dir
    sample2_path = '%s/sample2.png' % source_dir
    mask_path = '%s/interp_region.png' % source_dir
    sample1, sample1_mask = path2tensors(sample1_path, sample=True)
    sample2, sample2_mask = path2tensors(sample2_path, sample=True)
    interp_mask = path2tensors(mask_path, sample=False)

    # read CAF sources
    print('reading CAF sources...')
    sample1_CAF_path = '%s/sample1_CAF.png' % source_dir
    sample2_CAF_path = '%s/sample2_CAF.png' % source_dir
    sample1_CAF_fig = skimage.img_as_float(np.array(PIL.Image.open(sample1_CAF_path)))
    sample2_CAF_fig = skimage.img_as_float(np.array(PIL.Image.open(sample2_CAF_path)))

    # determine the boundary curve average direction and then rotate all the images that favor for horizontal interpolation
    if rotation_enabled:
        try:
            sample1_mask_dilated = binary_dilation(sample1_mask * (1.0-interp_mask)).astype(sample1_mask.dtype)
            sample1_boundary = sample1_mask_dilated * interp_mask
            y, x = np.where(sample1_boundary>0)
            y_min = np.amin(y); y_max = np.amax(y); x_min = np.amin(x); x_max = np.amax(x)
            sample1_boundary = sample1_boundary[y_min-1:y_max+2, x_min-1:x_max+2]
            sample1_angle = im2angle(sample1_boundary)
            sample2_mask_dilated = binary_dilation(sample2_mask * (1.0-interp_mask)).astype(sample2_mask.dtype)
            sample2_boundary = sample2_mask_dilated * interp_mask
            y, x = np.where(sample2_boundary>0)
            y_min = np.amin(y); y_max = np.amax(y); x_min = np.amin(x); x_max = np.amax(x)
            sample2_boundary = sample2_boundary[y_min-1:y_max+2, x_min-1:x_max+2]
            sample2_angle = im2angle(sample2_boundary)
            angle = (sample1_angle + sample2_angle) / 2.0
            sample1_mask = skimage.transform.rotate(sample1_mask, -90.0-angle)
            sample1_mask = (sample1_mask>0.5).astype(np.float32)
            sample2_mask = skimage.transform.rotate(sample2_mask, -90.0-angle)
            sample2_mask = (sample2_mask>0.5).astype(np.float32)
            interp_mask = skimage.transform.rotate(interp_mask, -90.0-angle)
            interp_mask = (interp_mask>0.5).astype(np.float32)
            sample1_CAF_fig = skimage.transform.rotate(sample1_CAF_fig, -90.0-angle)
            sample1_CAF_fig[sample1_CAF_fig<0.0] = 0.0; sample1_CAF_fig[sample1_CAF_fig>1.0] = 1.0
            sample2_CAF_fig = skimage.transform.rotate(sample2_CAF_fig, -90.0-angle)
            sample2_CAF_fig[sample2_CAF_fig<0.0] = 0.0; sample2_CAF_fig[sample2_CAF_fig>1.0] = 1.0
            rotation_failure = False
        except:
            rotation_failure = True

    # rasterize
    print('rasterizing...')
    sample1_CAF = np.transpose(misc.adjust_dynamic_range(sample1_CAF_fig[:,:,:3], [0,1], [-1,1]), axes=[2,0,1])
    sample2_CAF = np.transpose(misc.adjust_dynamic_range(sample2_CAF_fig[:,:,:3], [0,1], [-1,1]), axes=[2,0,1])
    interp_mask_grid, sample1_mask_grid, sample2_mask_grid = rasterize_mask(interp_mask, sample1_mask, sample2_mask, size=train_size)
    sample1_grid = np.dstack([(sample1_CAF_fig*255.0).astype(np.uint8), (sample1_mask_grid*255.0).astype(np.uint8)])
    sample1_interp_grid = np.dstack([(sample1_CAF_fig*255.0).astype(np.uint8), (interp_mask_grid*255.0).astype(np.uint8)])
    sample2_grid = np.dstack([(sample2_CAF_fig*255.0).astype(np.uint8), (sample2_mask_grid*255.0).astype(np.uint8)])
    sample2_interp_grid = np.dstack([(sample2_CAF_fig*255.0).astype(np.uint8), (interp_mask_grid*255.0).astype(np.uint8)])

    # pick up bounding source crops
    print('picking up source crops at boundaries...')
    RBF_field_weight = False
    horizontal_linear_weight = True

    if weight_mode == 'RBF':
        ul_corner_list = []
        crops_list = []
        for i in range(0, interp_mask_grid.shape[0], train_size):
            for j in range(0, interp_mask_grid.shape[1], train_size):
                if interp_mask_grid[i,j] > 0.0:
                    for di in [-train_size, 0, train_size]:
                        for dj in [-train_size, 0, train_size]:
                            if di * dj == 0 and di + dj != 0 and (sample1_mask_grid[i+di, j+dj] > 0.0 or sample2_mask_grid[i+di, j+dj] > 0.0) and [i+di, j+dj] not in ul_corner_list:
                                ul_corner_list.append([i+di, j+dj])
                                if sample1_mask_grid[i+di, j+dj] > 0.0:
                                    crops_list.append(sample1_CAF[:, i+di:i+di+train_size, j+dj:j+dj+train_size])
                                else:
                                    crops_list.append(sample2_CAF[:, i+di:i+di+train_size, j+dj:j+dj+train_size])
        crops = np.stack(crops_list, axis=0)
        row_list, col_list = zip(*ul_corner_list)
        ul_row = min(row_list)
        br_row = max(row_list)
        ul_col = min(col_list)
        br_col = max(col_list)
        scale_h = (br_row - ul_row) // train_size + 1
        scale_w = (br_col - ul_col) // train_size + 1

    elif weight_mode == 'horizontal_linear':
        ul_corner_1_list = []
        ul_corner_2_list = []
        crops_1_list = []
        crops_2_list = []
        for i in range(0, interp_mask_grid.shape[0], train_size):
            for j in range(0, interp_mask_grid.shape[1], train_size):
                if interp_mask_grid[i,j] > 0.0:
                    for di in [-train_size, 0, train_size]:
                        for dj in [-train_size, 0, train_size]:
                            if di * dj == 0 and di + dj != 0:
                                if sample1_mask_grid[i+di, j+dj] > 0.0 and [i+di, j+dj] not in ul_corner_1_list:
                                    ul_corner_1_list.append([i+di, j+dj])
                                    crops_1_list.append(sample1_CAF[:, i+di:i+di+train_size, j+dj:j+dj+train_size])
                                elif sample2_mask_grid[i+di, j+dj] > 0.0 and [i+di, j+dj] not in ul_corner_2_list:
                                    ul_corner_2_list.append([i+di, j+dj])
                                    crops_2_list.append(sample2_CAF[:, i+di:i+di+train_size, j+dj:j+dj+train_size])
        row_1_list, col_1_list = zip(*ul_corner_1_list)
        ul_row_1 = min(row_1_list)
        br_row_1 = max(row_1_list)
        ul_col_1 = min(col_1_list)
        br_col_1 = max(col_1_list)
        row_2_list, col_2_list = zip(*ul_corner_2_list)
        ul_row_2 = min(row_2_list)
        br_row_2 = max(row_2_list)
        ul_col_2 = min(col_2_list)
        br_col_2 = max(col_2_list)
        ul_row = min([ul_row_1, ul_row_2])
        br_row = max([br_row_1, br_row_2])
        ul_col = min([ul_col_1, ul_col_2])
        br_col = max([br_col_1, br_col_2])
        scale_h = (br_row - ul_row) // train_size + 1
        scale_w = (br_col - ul_col) // train_size + 1
        # only keep one patch for each row. if there are multiple patches in a row, only keep the rightmost one from the left source and keep the leftmost one from the right source
        for row in np.unique(np.array(row_1_list)):
            cols = [ul_corner[1] for ul_corner in ul_corner_1_list if ul_corner[0] == row]
            if len(cols) > 1:
                if np.mean(np.array(col_1_list)) < np.mean(np.array(col_2_list)):
                    col_ref = np.amax(np.array(cols))
                else:
                    col_ref = np.amin(np.array(cols))
                for col in cols:
                    if col != col_ref:
                        for idx in range(len(ul_corner_1_list)):
                            if row == ul_corner_1_list[idx][0] and col == ul_corner_1_list[idx][1]:
                                ul_corner_1_list.pop(idx)
                                crops_1_list.pop(idx)
                                break
        for row in np.unique(np.array(row_2_list)):
            cols = [ul_corner[1] for ul_corner in ul_corner_2_list if ul_corner[0] == row]
            if len(cols) > 1:
                if np.mean(np.array(col_2_list)) < np.mean(np.array(col_1_list)):
                    col_ref = np.amax(np.array(cols))
                else:
                    col_ref = np.amin(np.array(cols))
                for col in cols:
                    if col != col_ref:
                        for idx in range(len(ul_corner_2_list)):
                            if row == ul_corner_2_list[idx][0] and col == ul_corner_2_list[idx][1]:
                                ul_corner_2_list.pop(idx)
                                crops_2_list.pop(idx)
                                break
        # add some dummy source patches to the shorter one to align the height of the two sources
        if ul_row_1 > ul_row:
            col = col_1_list[row_1_list.index(ul_row_1)]
            for row in range(ul_row, ul_row_1, train_size):
                ul_corner_1_list.insert(0, [row, col])
                crops_1_list.insert(0, sample1_CAF[:, ul_row_1:ul_row_1+train_size, col:col+train_size])
        elif ul_row_2 > ul_row:
            col = col_2_list[row_2_list.index(ul_row_2)]
            for row in range(ul_row, ul_row_2, train_size):
                ul_corner_2_list.insert(0, [row, col])
                crops_2_list.insert(0, sample2_CAF[:, ul_row_2:ul_row_2+train_size, col:col+train_size])
        if br_row_1 < br_row:
            col = col_1_list[row_1_list.index(br_row_1)]
            for row in range(br_row_1+train_size, br_row+train_size, train_size):
                ul_corner_1_list.append([row, col])
                crops_1_list.append(sample1_CAF[:, br_row_1:br_row_1+train_size, col:col+train_size])
        elif br_row_2 < br_row:
            col = col_2_list[row_2_list.index(br_row_2)]
            for row in range(br_row_2+train_size, br_row+train_size, train_size):
                ul_corner_2_list.append([row, col])
                crops_2_list.append(sample2_CAF[:, br_row_2:br_row_2+train_size, col:col+train_size])
        ul_corner_list = ul_corner_1_list + ul_corner_2_list
        crops = np.stack(crops_1_list+crops_2_list, axis=0)

    # Load model
    E_zg, E_zl, G, D_rec, D_interp, D_blend, Es_zg, Es_zl, Gs = misc.load_pkl(model_path)
    Gs_fcn = tfutil.Network('Gs', reuse=True, num_channels=Gs.output_shape[1], resolution=Gs.output_shape[2], scale_h=scale_h, scale_w=scale_w, **config.G)

    # compute interolation weight
    if weight_mode == 'RBF':
        weights = []
        for ul_corner in ul_corner_list:
            weight = gkern_for_weight_grid_shape_hybridization(out_h=Gs_fcn.input_shapes[0][2], out_w=Gs_fcn.input_shapes[0][3], cx=float(ul_corner[1]-ul_col)/4.0, cy=float(ul_corner[0]-ul_row)/4.0, size=train_size/4.0, sig_div=sig_div)
            weights.append(weight)
        weights = np.stack(weights, axis=0)
        field = np.sum(weights, axis=0)
        field = (field - np.amin(field)) / (np.amax(field) - np.amin(field))
        field = np.dstack([field, field, field]) * 255.0
        field = field.astype(np.uint8) 
        weights /= np.sum(weights, axis=0, keepdims=True)
        weights = np.expand_dims(weights, axis=1)
        weights = np.tile(weights, [1, Gs_fcn.input_shapes[0][1], 1, 1])

    elif weight_mode == 'horizontal_linear':
        row_1_list, col_1_list = zip(*ul_corner_1_list)
        ul_row_1 = min(row_1_list)
        br_row_1 = max(row_1_list)
        ul_col_1 = min(col_1_list)
        br_col_1 = max(col_1_list)
        row_2_list, col_2_list = zip(*ul_corner_2_list)
        ul_row_2 = min(row_2_list)
        br_row_2 = max(row_2_list)
        ul_col_2 = min(col_2_list)
        br_col_2 = max(col_2_list)
        ul_row = min([ul_row_1, ul_row_2])
        br_row = max([br_row_1, br_row_2])
        ul_col = min([ul_col_1, ul_col_2])
        br_col = max([br_col_1, br_col_2])
        weights = np.zeros((crops.shape[0], Gs_fcn.input_shapes[0][2], Gs_fcn.input_shapes[0][3]))
        for row in range(ul_row, br_row+train_size, train_size):
            idx_1 = row_1_list.index(row)
            col_1 = col_1_list[idx_1]
            idx_2 = row_2_list.index(row)
            col_2 = col_2_list[idx_2]
            if col_1 < col_2:
                left = (col_1+train_size-ul_col)//4
                right = (col_2-ul_col)//4
                weights[idx_1, (row-ul_row)//4:(row+train_size-ul_row)//4, left:right] = np.tile(np.reshape(np.linspace(start=1.0, stop=0.0, num=right-left), [1,1,right-left]), [1,train_size//4,1])
                weights[idx_1, (row-ul_row)//4:(row+train_size-ul_row)//4, :left] = 1.0
                weights[idx_1, (row-ul_row)//4:(row+train_size-ul_row)//4, right:] = 0.0
                weights[len(row_1_list)+idx_2, (row-ul_row)//4:(row+train_size-ul_row)//4, left:right] = np.tile(np.reshape(np.linspace(start=0.0, stop=1.0, num=right-left), [1,1,right-left]), [1,train_size//4,1])
                weights[len(row_1_list)+idx_2, (row-ul_row)//4:(row+train_size-ul_row)//4, :left] = 0.0
                weights[len(row_1_list)+idx_2, (row-ul_row)//4:(row+train_size-ul_row)//4, right:] = 1.0
            else:
                left = (col_2+train_size-ul_col)//4
                right = (col_1-ul_col)//4
                weights[len(row_1_list)+idx_2, (row-ul_row)//4:(row+train_size-ul_row)//4, left:right] = np.tile(np.reshape(np.linspace(start=1.0, stop=0.0, num=right-left), [1,1,right-left]), [1,train_size//4,1])
                weights[len(row_1_list)+idx_2, (row-ul_row)//4:(row+train_size-ul_row)//4, :left] = 1.0
                weights[len(row_1_list)+idx_2, (row-ul_row)//4:(row+train_size-ul_row)//4, right:] = 0.0
                weights[idx_1, (row-ul_row)//4:(row+train_size-ul_row)//4, left:right] = np.tile(np.reshape(np.linspace(start=0.0, stop=1.0, num=right-left), [1,1,right-left]), [1,train_size//4,1])
                weights[idx_1, (row-ul_row)//4:(row+train_size-ul_row)//4, :left] = 0.0
                weights[idx_1, (row-ul_row)//4:(row+train_size-ul_row)//4, right:] = 1.0
        weights = np.expand_dims(weights, axis=1)
        weights = np.tile(weights, [1, Gs_fcn.input_shapes[0][1], 1, 1])
    
    # encode zg
    print('zg encoding...')
    zg_mu, zg_log_sigma = Es_zg.run(crops, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    zg_latents = np.array(zg_mu)

    # encode zl
    print('zl encoding...')
    zl_mu, zl_log_sigma = Es_zl.run(crops, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    zl_latents = np.array(zl_mu)

    def permutation_matrix_w_sampler(scale_w):
        if config.block_size == 0:
            temp_perm = np.eye(Es_zl.output_shapes[0][3]*scale_w)
            for idx in range(int(np.log2(Es_zl.output_shapes[0][3]))):
                block_size_temp = int(2**idx)
                if config.perm:
                    perm = np.random.permutation(np.eye(Es_zl.output_shapes[0][3]*scale_w//block_size_temp))
                else:
                    perm = my_swap_w(np.eye(Es_zl.output_shapes[0][3]*scale_w//block_size_temp))
                cur_perm = block_permutation(perm, block_size_temp)
                temp_perm = np.matmul(cur_perm, temp_perm)
            permutation_matrix_w = np.tile(temp_perm, [1,Es_zl.output_shapes[0][1],1,1])
        else:
            if config.perm:
                perm = np.random.permutation(np.eye(Es_zl.output_shapes[0][3]*scale_w//config.block_size))
            else:
                perm = my_swap_w(np.eye(Es_zl.output_shapes[0][3]*scale_w//config.block_size))
            permutation_matrix_w = block_permutation(perm, config.block_size)
            permutation_matrix_w = np.tile(permutation_matrix_w, [1,Es_zl.output_shapes[0][1],1,1])
        return permutation_matrix_w

    def permutation_matrix_h_sampler(scale_h):
        if config.block_size == 0:
            temp_perm = np.eye(Es_zl.output_shapes[0][2]*scale_h)
            for idx in range(int(np.log2(Es_zl.output_shapes[0][2]))):
                block_size_temp = int(2**idx)
                if config.perm:
                    perm = np.random.permutation(np.eye(Es_zl.output_shapes[0][2]*scale_h//block_size_temp))
                else:
                    perm = my_swap_h(np.eye(Es_zl.output_shapes[0][2]*scale_h//block_size_temp))
                cur_perm = block_permutation(perm, block_size_temp)
                temp_perm = np.matmul(temp_perm, cur_perm)
            permutation_matrix_h = np.tile(temp_perm, [1,Es_zl.output_shapes[0][1],1,1])
        else:
            if config.perm:
                perm = np.random.permutation(np.eye(Es_zl.output_shapes[0][2]*scale_h//config.block_size))
            else:
                perm = my_swap_h(np.eye(Es_zl.output_shapes[0][2]*scale_h//config.block_size))
            permutation_matrix_h = block_permutation(perm, config.block_size)
            permutation_matrix_h = np.tile(permutation_matrix_h, [1,Es_zl.output_shapes[0][1],1,1])
        return permutation_matrix_h

    # compute permutation matrices
    permutation_matrix_w_0 = []
    permutation_matrix_h_0 = []
    for count in range(crops.shape[0]):
        permutation_matrix_w_0.append(permutation_matrix_w_sampler(scale_w))
        permutation_matrix_h_0.append(permutation_matrix_h_sampler(scale_h))
    permutation_matrix_w_0 = np.concatenate(permutation_matrix_w_0, axis=0)
    permutation_matrix_h_0 = np.concatenate(permutation_matrix_h_0, axis=0)

    print('interpolating...')
    # interpolate zg
    interp_zg_latents = np.tile(zg_latents, [1, 1, Gs_fcn.input_shapes[0][2], Gs_fcn.input_shapes[0][3]])
    interp_zg_latents = np.sum(interp_zg_latents * weights, axis=0, keepdims=True)
    for (count, ul_corner) in enumerate(ul_corner_list):
        interp_zg_latents[0, :, (ul_corner[0]-ul_row)//4:(ul_corner[0]-ul_row)//4+train_size//4, (ul_corner[1]-ul_col)//4:(ul_corner[1]-ul_col)//4+train_size//4] = zg_latents[count,:,:,:]

    # interpolate zl
    interp_zl_latents = np.matmul(np.matmul(permutation_matrix_h_0, np.tile(zl_latents, [1,1,scale_h,scale_w])), permutation_matrix_w_0)
    interp_zl_latents = np.sum(interp_zl_latents * weights, axis=0, keepdims=True)
    for (count, ul_corner) in enumerate(ul_corner_list):
        interp_zl_latents[0, :, (ul_corner[0]-ul_row)//4:(ul_corner[0]-ul_row)//4+train_size//4, (ul_corner[1]-ul_col)//4:(ul_corner[1]-ul_col)//4+train_size//4] = zl_latents[count,:,:,:]

    # generate interpolated image
    interp = Gs_fcn.run(interp_zg_latents, interp_zl_latents, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    interp = misc.adjust_dynamic_range(np.transpose(np.squeeze(interp), [1,2,0]), [-1,1], [0,255]).astype(np.uint8)
    image = np.ones((sample1.shape[1], sample1.shape[2], 4)).astype(np.uint8) * 255
    image[ul_row:ul_row+train_size*scale_h, ul_col:ul_col+train_size*scale_w, :3] = np.array(interp)
    image[:,:,3] = interp_mask_grid.astype(np.uint8) * 255
    overlap_raw = np.ones((sample1.shape[1], sample1.shape[2], 3)).astype(np.uint8).flatten() * 255
    overlap_raw[np.dstack([sample1_mask_grid, sample1_mask_grid, sample1_mask_grid]).flatten()>0] = misc.adjust_dynamic_range(np.transpose(sample1_CAF, [1,2,0]), [-1,1], [0,255]).astype(np.uint8).flatten()[np.dstack([sample1_mask_grid, sample1_mask_grid, sample1_mask_grid]).flatten()>0]
    overlap_raw[np.dstack([sample2_mask_grid, sample2_mask_grid, sample2_mask_grid]).flatten()>0] = misc.adjust_dynamic_range(np.transpose(sample2_CAF, [1,2,0]), [-1,1], [0,255]).astype(np.uint8).flatten()[np.dstack([sample2_mask_grid, sample2_mask_grid, sample2_mask_grid]).flatten()>0]
    overlap_raw[np.dstack([interp_mask_grid, interp_mask_grid, interp_mask_grid]).flatten()>0] = image[:,:,:3].flatten()[np.dstack([interp_mask_grid, interp_mask_grid, interp_mask_grid]).flatten()>0]
    overlap_raw = np.reshape(overlap_raw, (sample1.shape[1], sample1.shape[2], 3))
    overlap = np.ones(overlap_raw.shape).astype(np.uint8).flatten() * 255
    overlap[np.dstack([sample1_mask, sample1_mask, sample1_mask]).flatten()>0] = overlap_raw.flatten()[np.dstack([sample1_mask, sample1_mask, sample1_mask]).flatten()>0]
    overlap[np.dstack([sample2_mask, sample2_mask, sample2_mask]).flatten()>0] = overlap_raw.flatten()[np.dstack([sample2_mask, sample2_mask, sample2_mask]).flatten()>0]
    overlap[np.dstack([interp_mask, interp_mask, interp_mask]).flatten()>0] = overlap_raw.flatten()[np.dstack([interp_mask, interp_mask, interp_mask]).flatten()>0]
    overlap = np.reshape(overlap, overlap_raw.shape)

    if rotation_enabled and not rotation_failure:
        overlap_rotate = (skimage.transform.rotate(overlap/255.0, 90.0+angle) * 255.0).astype(np.uint8)
        overlap_rotate[overlap_rotate<0] = 0; overlap_rotate[overlap_rotate>255] = 255
        PIL.Image.fromarray(overlap_rotate, 'RGB').save('%s/hybridization.png' % out_dir, 'png')
    else:
        PIL.Image.fromarray(overlap, 'RGB').save('%s/hybridization.png' % out_dir, 'png')

#----------------------------------------------------------------------------
# Generate horizontal interpolation over dataset.
# To run, uncomment the appropriate line in config.py and launch train.py.

def horizontal_interpolation(model_path, imageL_path, imageR_path, out_dir, scale_h=3, scale_w=8, minibatch_size=1, rotate=False, file_name='img'):
    num_images = 2
    if not os.path.isdir(out_dir): os.makedirs(out_dir)

    E_zg = config.myE_zg
    E_zl = config.myE_zl
    G = config.myG
    D_rec = config.myD_rec
    D_interp = config.myD_interp
    D_blend = config.myD_blend
    Es_zg = config.myEs_zg
    Es_zl = config.myEs_zl
    Gs = config.myGs


    Gs_fcn = tfutil.Network('Gs', reuse=True, num_channels=Gs.output_shape[1], resolution=Gs.output_shape[2], scale_h=scale_h, scale_w=scale_w, **config.G)

    # Load dataset
    reals_orig = np.zeros([num_images]+Es_zl.input_shape[1:]).astype(np.float32)
    image1 = np.array(PIL.Image.open(imageL_path)).astype(np.float32)
    if rotate:
        image1 = np.rot90(image1)
    image1 = np.transpose(misc.adjust_dynamic_range(image1, [0,255], [-1,1]), axes=[2,0,1])
    reals_orig[0,:,:,:] = image1
    image2 = np.array(PIL.Image.open(imageR_path)).astype(np.float32)
    if rotate:
        image2 = np.rot90(image2)
    image2 = np.transpose(misc.adjust_dynamic_range(image2, [0,255], [-1,1]), axes=[2,0,1])
    reals_orig[1,:,:,:] = image2

    # zg encoding 
    logger.debug('Latent zg encoding...')
    enc_zg_mu, enc_zg_log_sigma = Es_zg.run(reals_orig, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    enc_zg_mu = enc_zg_mu[:num_images,:,:,:]; enc_zg_log_sigma = enc_zg_log_sigma[:num_images,:,:,:]
    if not config.zg_enabled:
        enc_zg_mu = np.zeros(enc_zg_mu.shape); enc_zg_log_sigma = np.ones(enc_zg_log_sigma.shape)
    
    msg = "Latent vector of \"{}\" is \"{}\"\n".format(imageL_path, str(enc_zg_mu[0].reshape([1,-1])[0]))
    msg += "\tLatent vector of \"{}\" is \"{}\"".format(imageR_path, str(enc_zg_mu[1].reshape([1,-1])[0]))
    logger.debug(msg)

    # zl encoding 
    logger.debug('Latent zl encoding...')
    enc_zl_mu, enc_zl_log_sigma = Es_zl.run(reals_orig, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
    reals_orig = reals_orig[:num_images,:,:,:]; enc_zl_mu = enc_zl_mu[:num_images,:,:,:]; enc_zl_log_sigma = enc_zl_log_sigma[:num_images,:,:,:]
    
    # generating
    logger.debug('Interpolating...')
    interp_images_out = np.zeros((num_images, Gs_fcn.output_shape[1], G.output_shape[2], Gs_fcn.output_shape[3]))
    num_loop = 1
    for mb_begin in range(0, num_loop, minibatch_size):
        mb_end = min(mb_begin + minibatch_size, num_images)
        mb_size = mb_end - mb_begin
        if config.zg_interp_variational == 'hard':
            interp_zg_latents_left = np.tile(enc_zg_mu[mb_begin:mb_end,:,:,:], [1, 1, Es_zl.output_shapes[0][2]*scale_h, Es_zl.output_shapes[0][3]*scale_w])
            interp_zg_latents_right = np.tile(enc_zg_mu[num_images-mb_end:num_images-mb_begin,:,:,:][::-1,:,:,:], [1, 1, Es_zl.output_shapes[0][2]*scale_h, Es_zl.output_shapes[0][3]*scale_w])
            matt = linkern_for_weight_horizontal(interp_zg_latents_left.shape, Es_zl.output_shapes[0][2])
            interp_zg_latents = interp_zg_latents_left * matt + interp_zg_latents_right * (1.0 - matt)
        if config.zl_interp_variational == 'hard':
            interp_zl_latents_left = np.tile(enc_zl_mu[mb_begin:mb_end,:,:,:], [1, 1, scale_h, scale_w])
            interp_zl_latents_right = np.tile(enc_zl_mu[num_images-mb_end:num_images-mb_begin,:,:,:][::-1,:,:,:], [1, 1, scale_h, scale_w])
            matt = linkern_for_weight_horizontal(interp_zl_latents_left.shape, Es_zl.output_shapes[0][2])
            interp_zl_latents = interp_zl_latents_left * matt + interp_zl_latents_right * (1.0 - matt)
        elif config.zl_interp_variational == 'variational':
            interp_zl_latents = np.random.normal(size=[mb_size]+Gs_fcn.input_shape[1:]).astype(np.float32)
            scale = gkern_for_scale_horizontal(interp_zl_latents.shape, Es_zl.output_shapes[0][2])
            interp_zl_mu_left = np.tile(enc_zl_mu[mb_begin:mb_end,:,:,:], [1, 1, scale_h, scale_w])
            interp_zl_log_sigma_left = np.tile(enc_zl_log_sigma[mb_begin:mb_end,:,:,:], [1, 1, scale_h, scale_w])
            interp_zl_latents_left = interp_zl_latents * np.exp(interp_zl_log_sigma_left) * scale + interp_zl_mu_left
            interp_zl_latents = np.random.normal(size=[mb_size]+Gs_fcn.input_shape[1:]).astype(np.float32)
            interp_zl_mu_right = np.tile(enc_zl_mu[num_images-mb_end:num_images-mb_begin,:,:,:][::-1,:,:,:], [1, 1, scale_h, scale_w])
            interp_zl_log_sigma_right = np.tile(enc_zl_log_sigma[num_images-mb_end:num_images-mb_begin,:,:,:][::-1,:,:,:], [1, 1, scale_h, scale_w])
            interp_zl_latents_right = interp_zl_latents * np.exp(interp_zl_log_sigma_right[::-1,:,:,:]) * scale[:,:,:,::-1] + interp_zl_mu_right[::-1,:,:,:]
            matt = linkern_for_weight_horizontal(interp_zl_latents_left.shape, Es_zl.output_shapes[0][2])
            interp_zl_latents = interp_zl_latents_left * matt + interp_zl_latents_right * (1.0 - matt)
        elif config.zl_interp_variational == 'random':
            interp_zl_latents = np.random.normal(size=(mb_size, Es_zl.output_shapes[0][1], Es_zl.output_shapes[0][2]*scale_h, Es_zl.output_shapes[0][3]*scale_w)).astype(np.float32)
            interp_zl_latents[:,:,:,:Es_zl.output_shapes[0][3]] = enc_zl_mu[mb_begin:mb_end,:,:,:]
            interp_zl_latents[:,:,:,-Es_zl.output_shapes[0][3]:] = enc_zl_mu[num_images-mb_end:num_images-mb_begin,:,:,:][::-1,:,:,:]
        elif config.zl_interp_variational == 'permutational':
            permutation_matrix_h_0_left = np.zeros((mb_size, 1, Es_zl.output_shapes[0][2]*scale_h, Es_zl.output_shapes[0][2]*scale_h)).astype(np.float32)
            for count1 in range(mb_size):
                if config.block_size == 0:
                    temp_perm = np.eye(Es_zl.output_shapes[0][2]*scale_h)
                    for idx in range(int(np.log(Es_zl.output_shapes[0][2]))):
                        block_size_temp = int(2**idx)
                        if config.perm:
                            perm = np.random.permutation(np.eye(Es_zl.output_shapes[0][2]*scale_h//block_size_temp))
                        else:
                            perm = my_swap_h(np.eye(Es_zl.output_shapes[0][2]*scale_h//block_size_temp))
                        temp_perm = np.matmul(temp_perm, block_permutation(perm, block_size_temp))
                    permutation_matrix_h_0_left[count1,:,:,:] = np.array(temp_perm)
                else:
                    if config.perm:
                        perm = np.random.permutation(np.eye(Es_zl.output_shapes[0][2]*scale_h//config.block_size))
                    else:
                        perm = my_swap_h(np.eye(Es_zl.output_shapes[0][2]*scale_h//config.block_size))
                    permutation_matrix_h_0_left[count1,:,:,:] = block_permutation(perm, config.block_size)
            permutation_matrix_w_0_left = np.zeros((mb_size, 1, Es_zl.output_shapes[0][3]*scale_w, Es_zl.output_shapes[0][3]*scale_w)).astype(np.float32)
            for count1 in range(mb_size):
                if config.block_size == 0:
                    temp_perm = np.eye(Es_zl.output_shapes[0][3]*scale_w)
                    for idx in range(int(np.log(Es_zl.output_shapes[0][3]))):
                        block_size_temp = int(2**idx)
                        if config.perm:
                            perm = np.random.permutation(np.eye(Es_zl.output_shapes[0][3]*scale_w//block_size_temp))
                        else:
                            perm = my_swap_w(np.eye(Es_zl.output_shapes[0][3]*scale_w//block_size_temp))
                        temp_perm = np.matmul(block_permutation(perm, block_size_temp), temp_perm)
                    permutation_matrix_w_0_left[count1,:,:,:] = np.array(temp_perm)
                else:
                    if config.perm:
                        perm = np.random.permutation(np.eye(Es_zl.output_shapes[0][3]*scale_w//config.block_size))
                    else:
                        perm = my_swap_w(np.eye(Es_zl.output_shapes[0][3]*scale_w//config.block_size))
                    permutation_matrix_w_0_left[count1,:,:,:] = block_permutation(perm, config.block_size)
            permutation_matrix_h_0_right = np.zeros((mb_size, 1, Es_zl.output_shapes[0][2]*scale_h, Es_zl.output_shapes[0][2]*scale_h)).astype(np.float32)
            for count1 in range(mb_size):
                if config.block_size == 0:
                    temp_perm = np.eye(Es_zl.output_shapes[0][2]*scale_h)
                    for idx in range(int(np.log(Es_zl.output_shapes[0][2]))):
                        block_size_temp = int(2**idx)
                        if config.perm:
                            perm = np.random.permutation(np.eye(Es_zl.output_shapes[0][2]*scale_h//block_size_temp))
                        else:
                            perm = my_swap_h(np.eye(Es_zl.output_shapes[0][2]*scale_h//block_size_temp))
                        temp_perm = np.matmul(temp_perm, block_permutation(perm, block_size_temp))
                    permutation_matrix_h_0_right[count1,:,:,:] = np.array(temp_perm)
                else:
                    if config.perm:
                        perm = np.random.permutation(np.eye(Es_zl.output_shapes[0][2]*scale_h//config.block_size))
                    else:
                        perm = my_swap_h(np.eye(Es_zl.output_shapes[0][2]*scale_h//config.block_size))
                    permutation_matrix_h_0_right[count1,:,:,:] = block_permutation(perm, config.block_size)
            permutation_matrix_w_0_right = np.zeros((mb_size, 1, Es_zl.output_shapes[0][3]*scale_w, Es_zl.output_shapes[0][3]*scale_w)).astype(np.float32)
            for count1 in range(mb_size):
                if config.block_size == 0:
                    temp_perm = np.eye(Es_zl.output_shapes[0][3]*scale_w)
                    for idx in range(int(np.log(Es_zl.output_shapes[0][3]))):
                        block_size_temp = int(2**idx)
                        if config.perm:
                            perm = np.random.permutation(np.eye(Es_zl.output_shapes[0][3]*scale_w//block_size_temp))
                        else:
                            perm = my_swap_w(np.eye(Es_zl.output_shapes[0][3]*scale_w//block_size_temp))
                        temp_perm = np.matmul(block_permutation(perm, block_size_temp), temp_perm)
                    permutation_matrix_w_0_right[count1,:,:,:] = np.array(temp_perm)
                else:
                    if config.perm:
                        perm = np.random.permutation(np.eye(Es_zl.output_shapes[0][3]*scale_w//config.block_size))
                    else:
                        perm = my_swap_w(np.eye(Es_zl.output_shapes[0][3]*scale_w//config.block_size))
                    permutation_matrix_w_0_right[count1,:,:,:] = block_permutation(perm, config.block_size)
            interp_zl_latents_left = np.matmul(np.matmul(np.tile(permutation_matrix_h_0_left, [1,Es_zl.output_shapes[0][1],1,1]), np.tile(enc_zl_mu[mb_begin:mb_end,:,:,:], [1,1,scale_h,scale_w])), np.tile(permutation_matrix_w_0_left, [1,Es_zl.output_shapes[0][1],1,1]))
            interp_zl_latents_left[:, :, scale_h//2*Es_zl.output_shapes[0][2]:(scale_h//2+1)*Es_zl.output_shapes[0][2], :Es_zl.output_shapes[0][3]] = enc_zl_mu[mb_begin:mb_end,:,:,:]
            interp_zl_latents_left[:, :, scale_h//2*Es_zl.output_shapes[0][2]:(scale_h//2+1)*Es_zl.output_shapes[0][2], -Es_zl.output_shapes[0][3]:] = enc_zl_mu[mb_begin:mb_end,:,:,:]
            interp_zl_latents_right = np.matmul(np.matmul(np.tile(permutation_matrix_h_0_right, [1,Es_zl.output_shapes[0][1],1,1]), np.tile(enc_zl_mu[num_images-mb_end:num_images-mb_begin,:,:,:][::-1,:,:,:], [1,1,scale_h,scale_w])), np.tile(permutation_matrix_w_0_right, [1,Es_zl.output_shapes[0][1],1,1]))
            interp_zl_latents_right[:, :, scale_h//2*Es_zl.output_shapes[0][2]:(scale_h//2+1)*Es_zl.output_shapes[0][2], :Es_zl.output_shapes[0][3]] = enc_zl_mu[num_images-mb_end:num_images-mb_begin,:,:,:][::-1,:,:,:]
            interp_zl_latents_right[:, :, scale_h//2*Es_zl.output_shapes[0][2]:(scale_h//2+1)*Es_zl.output_shapes[0][2], -Es_zl.output_shapes[0][3]:] = enc_zl_mu[num_images-mb_end:num_images-mb_begin,:,:,:][::-1,:,:,:]
            matt = linkern_for_weight_horizontal(interp_zl_latents_left.shape, Es_zl.output_shapes[0][2])
            interp_zl_latents = interp_zl_latents_left * matt + interp_zl_latents_right * (1.0 - matt)
        if mb_size != minibatch_size:
            interp_zg_latents = np.concatenate((interp_zg_latents, np.zeros([minibatch_size-mb_size]+list(interp_zg_latents.shape[1:]))), axis=0)
            interp_zl_latents = np.concatenate((interp_zl_latents, np.zeros([minibatch_size-mb_size]+list(interp_zl_latents.shape[1:]))), axis=0)
        interp_images_out_mb = Gs_fcn.run(interp_zg_latents, interp_zl_latents, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_dtype=np.float32)
        if mb_size != minibatch_size:
            interp_images_out_mb = interp_images_out_mb[:mb_size,:,:,:]
        interp_images_out[mb_begin:mb_end,:,:,:] = interp_images_out_mb[:,:,scale_h//2*G.output_shape[2]:(scale_h//2+1)*G.output_shape[2],:]

    logger.debug('Saving interpolation and cropping results...')

    # save interpolation results
    idx1 = imageL_path.rfind('/')
    idx2 = imageR_path.rfind('/')
    #path_interp = '%s/%s-interplatingTo-%s.png' % (out_dir, imageL_path[idx1+1:-4], imageR_path[idx2+1:-4])
    #IMAGE_interp = PIL.Image.fromarray(misc.adjust_dynamic_range(np.transpose(interp_images_out[0,:,:,:], axes=[1,2,0]), [-1,1], [0,255]).astype(np.uint8), 'RGB')
    #IMAGE_interp.save(path_interp, 'png')
    #path_interp = '%s/%s-interplatingTo-%s_2.png' % (out_dir, imageL_path[idx1+1:-4], imageR_path[idx2+1:-4])
    #IMAGE_interp = PIL.Image.fromarray(misc.adjust_dynamic_range(np.transpose(interp_images_out[1,:,:,:], axes=[1,2,0]), [-1,1], [0,255]).astype(np.uint8), 'RGB')
    #IMAGE_interp.save(path_interp, 'png')

    NUM_DIVIDE = scale_w
    divided_width = interp_images_out.shape[-1]//NUM_DIVIDE
    for i in range(1, NUM_DIVIDE-1):
        path_interp = '%s/%s.png' % (out_dir, file_name+str(i))
        IMAGE_interp = PIL.Image.fromarray(misc.adjust_dynamic_range(np.transpose(interp_images_out[0,:,:,divided_width*i:divided_width*(i+1)], axes=[1,2,0]), [-1,1], [0,255]).astype(np.uint8), 'RGB')
        IMAGE_interp.save(path_interp, 'png')
        latent = ((i)*enc_zg_mu[0].reshape([1,-1])[0] + (NUM_DIVIDE-1-i)*enc_zg_mu[1].reshape([1,-1])[0])/(NUM_DIVIDE - 1)
        msg = "Interpolating latent vector of \"{}\" is \"{}\"".format(path_interp, str(latent))
        logger.debug(msg)