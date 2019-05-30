# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licen sed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import time
import numpy as np
import tensorflow as tf

import config
import tfutil
import dataset
import misc

from collections import OrderedDict
from itertools import chain

import argparse

#----------------------------------------------------------------------------
# Choose the size and contents of the image snapshot grids that are exported
# periodically during training.

def setup_snapshot_image_grid(E_zg, E_zl, G, training_set, drange_net, grid_size=None,
    size    = '1080p',      # '1080p' = to be viewed on 1080p display, '4k' = to be viewed on 4k display.
    layout  = 'random'):    # 'random' = grid contents are selected randomly, 'row_per_class' = each row corresponds to one class label.

    # Select size.
    if grid_size is None:
        if size == '1080p':
            gw = np.clip(1920 // G.output_shape[3], 3, 32)
            gh = np.clip(1080 // G.output_shape[2], 2, 32)
        if size == '4k':
            gw = np.clip(3840 // G.output_shape[3], 7, 32)
            gh = np.clip(2160 // G.output_shape[2], 4, 32)
    else:
        gw = grid_size[0]
        gh = grid_size[1]
    gw_interp = int(round(float(gw)/float(config.scale_w)))
    gh_interp = int(round(float(gh)/float(config.scale_h)))

    # Fill in reals and labels.
    reals = np.zeros([gw * gh] + training_set.shape, dtype=np.float32)
    labels = np.zeros([gw * gh, training_set.label_size], dtype=training_set.label_dtype)
    for idx in range(gw * gh):
        x = idx % gw; y = idx // gw
        while True:
            real, label = training_set.get_minibatch_np(1)
            real = real.astype(np.float32)
            real = misc.adjust_dynamic_range(real, training_set.dynamic_range, drange_net)
            if layout == 'row_per_class' and training_set.label_size > 0:
                if label[0, y % training_set.label_size] == 0.0:
                    continue
            reals[idx] = real[0]
            labels[idx] = label[0]
            break

    # Generate latents.
    zg_latents, zl_latents = misc.random_latents(gw * gh, E_zg, E_zl)
    return (gw, gh), (gw_interp, gh_interp), reals, labels, zg_latents, zl_latents

#----------------------------------------------------------------------------
# Just-in-time processing of training images before feeding them to the networks.

def process_reals(x, lod, lr_mirror_augment, ud_mirror_augment, drange_data, drange_net):
    with tf.name_scope('ProcessReals'):
        with tf.name_scope('DynamicRange'):
            x = tf.cast(x, tf.float32)
            x = misc.adjust_dynamic_range(x, drange_data, drange_net)
        if lr_mirror_augment:
            with tf.name_scope('MirrorAugment'):
                s = tf.shape(x)
                mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
                mask = tf.tile(mask, [1, s[1], s[2], s[3]])
                x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[3]))
        if ud_mirror_augment:
            with tf.name_scope('udMirrorAugment'):
                s = tf.shape(x)
                mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
                mask = tf.tile(mask, [1, s[1], s[2], s[3]])
                x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[2]))
        with tf.name_scope('FadeLOD'): # Smooth crossfade between consecutive levels-of-detail.
            s = tf.shape(x)
            y = tf.reshape(x, [-1, s[1], s[2]//2, 2, s[3]//2, 2])
            y = tf.reduce_mean(y, axis=[3, 5], keepdims=True)
            y = tf.tile(y, [1, 1, 1, 2, 1, 2])
            y = tf.reshape(y, [-1, s[1], s[2], s[3]])
            x_fade = tfutil.lerp(x, y, lod - tf.floor(lod))
            x_orig = tf.identity(x)
        with tf.name_scope('UpscaleLOD'): # Upscale to match the expected input/output size of the networks.
            s = tf.shape(x)
            factor = tf.cast(2 ** tf.floor(lod), tf.int32)
            x_fade = tf.reshape(x_fade, [-1, s[1], s[2], 1, s[3], 1])
            x_fade = tf.tile(x_fade, [1, 1, 1, factor, 1, factor])
            x_fade = tf.reshape(x_fade, [-1, s[1], s[2] * factor, s[3] * factor])
            x_orig = tf.reshape(x_orig, [-1, s[1], s[2], 1, s[3], 1])
            x_orig = tf.tile(x_orig, [1, 1, 1, factor, 1, factor])
            x_orig = tf.reshape(x_orig, [-1, s[1], s[2] * factor, s[3] * factor])
        return x_fade, x_orig

#----------------------------------------------------------------------------
# Latent tensor swapping.

def my_swap_h(mat0):
    # top down scanning
    mat = np.array(mat0)
    if mat.shape[0] > 1:
        for i in range(mat.shape[0]):
            p = np.random.uniform()
            if i == 0:
                if p < 0.5:
                    temp = np.array(mat[i,:]); mat[i,:] = np.array(mat[i+1,:]); mat[i+1,:] = np.array(temp)
            elif i == mat.shape[0]-1:
                if p < 0.5:
                    temp = np.array(mat[i,:]); mat[i,:] = np.array(mat[i-1,:]); mat[i-1,:] = np.array(temp)
            else:
                if p < 1.0/3.0:
                    temp = np.array(mat[i,:]); mat[i,:] = np.array(mat[i+1,:]); mat[i+1,:] = np.array(temp)
                elif p > 2.0/3.0:
                    temp = np.array(mat[i,:]); mat[i,:] = np.array(mat[i-1,:]); mat[i-1,:] = np.array(temp)
        # bottom up scanning
        for i in range(mat.shape[0])[::-1]:
            p = np.random.uniform()
            if i == 0:
                if p < 0.5:
                    temp = np.array(mat[i,:]); mat[i,:] = np.array(mat[i+1,:]); mat[i+1,:] = np.array(temp)
            elif i == mat.shape[0]-1:
                if p < 0.5:
                    temp = np.array(mat[i,:]); mat[i,:] = np.array(mat[i-1,:]); mat[i-1,:] = np.array(temp)
            else:
                if p < 1.0/3.0:
                    temp = np.array(mat[i,:]); mat[i,:] = np.array(mat[i+1,:]); mat[i+1,:] = np.array(temp)
                elif p > 2.0/3.0:
                    temp = np.array(mat[i,:]); mat[i,:] = np.array(mat[i-1,:]); mat[i-1,:] = np.array(temp)
    return mat

def my_swap_w(mat0):
    # top down scanning
    mat = np.array(mat0)
    if mat.shape[1] > 1:
        for j in range(mat.shape[1]):
            p = np.random.uniform()
            if j == 0:
                if p < 0.5:
                    temp = np.array(mat[:,j]); mat[:,j] = np.array(mat[:,j+1]); mat[:,j+1] = np.array(temp)
            elif j == mat.shape[1]-1:
                if p < 0.5:
                    temp = np.array(mat[:,j]); mat[:,j] = np.array(mat[:,j-1]); mat[:,j-1] = np.array(temp)
            else:
                if p < 1.0/3.0:
                    temp = np.array(mat[:,j]); mat[:,j] = np.array(mat[:,j+1]); mat[:,j+1] = np.array(temp)
                elif p > 2.0/3.0:
                    temp = np.array(mat[:,j]); mat[:,j] = np.array(mat[:,j-1]); mat[:,j-1] = np.array(temp)
        # bottom up scanning
        for j in range(mat.shape[1])[::-1]:
            p = np.random.uniform()
            if j == 0:
                if p < 0.5:
                    temp = np.array(mat[:,j]); mat[:,j] = np.array(mat[:,j+1]); mat[:,j+1] = np.array(temp)
            elif j == mat.shape[1]-1:
                if p < 0.5:
                    temp = np.array(mat[:,j]); mat[:,j] = np.array(mat[:,j-1]); mat[:,j-1] = np.array(temp)
            else:
                if p < 1.0/3.0:
                    temp = np.array(mat[:,j]); mat[:,j] = np.array(mat[:,j+1]); mat[:,j+1] = np.array(temp)
                elif p > 2.0/3.0:
                    temp = np.array(mat[:,j]); mat[:,j] = np.array(mat[:,j-1]); mat[:,j-1] = np.array(temp)
    return mat

#----------------------------------------------------------------------------
# Latent block permutation.

def block_permutation(perm, block_size):
    mat = np.zeros((perm.shape[0]*block_size, perm.shape[1]*block_size))
    for i in range(perm.shape[0]):
        for j in range(perm.shape[1]):
            if perm[i,j] == 1.0:
                mat[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = np.eye(block_size)
    return mat

#----------------------------------------------------------------------------
# Class for evaluating and storing the values of time-varying training parameters.

class TrainingSchedule:
    def __init__(
        self,
        cur_nimg,
        training_set,
        lod_initial_resolution  = 4,        # Image resolution used at the beginning.
        lod_training_kimg       = 1500,     # Thousands of real images to show before doubling the resolution.
        lod_transition_kimg     = 1500,     # Thousands of real images to show when fading in new layers.
        minibatch_base          = 16,       # Maximum minibatch size, divided evenly among GPUs.
        minibatch_dict          = {},       # Resolution-specific overrides.
        max_minibatch_per_gpu   = {},       # Resolution-specific maximum minibatch size per GPU.
        lrate_base              = 0.001,    # Learning rate for AutoEncoder.
        lrate_dict              = {},       # Resolution-specific overrides.
        tick_kimg_base          = 1,        # Default interval of progress snapshots.
        tick_kimg_dict          = {}):      # Resolution-specific overrides.

        # Training phase.
        self.kimg = cur_nimg / 1000.0
        phase_dur = lod_training_kimg + lod_transition_kimg
        phase_idx = int(np.floor(self.kimg / phase_dur)) if phase_dur > 0 else 0
        phase_kimg = self.kimg - phase_idx * phase_dur

        # Level-of-detail and resolution.
        self.lod = training_set.resolution_log2
        self.lod -= np.floor(np.log2(lod_initial_resolution))
        self.lod -= phase_idx
        if lod_transition_kimg > 0:
            self.lod -= max(phase_kimg - lod_training_kimg, 0.0) / lod_transition_kimg
        self.lod = max(self.lod, 0.0)
        self.resolution = 2 ** (training_set.resolution_log2 - int(np.floor(self.lod)))

        # Minibatch size.
        self.minibatch = minibatch_dict.get(self.resolution, minibatch_base)
        self.minibatch -= self.minibatch % config.num_gpus
        if self.resolution in max_minibatch_per_gpu:
            self.minibatch = min(self.minibatch, max_minibatch_per_gpu[self.resolution] * config.num_gpus)

        # Other parameters.
        self.lrate = lrate_dict.get(self.resolution, lrate_base)
        self.tick_kimg = tick_kimg_dict.get(self.resolution, tick_kimg_base)

#----------------------------------------------------------------------------
# Main training script.
# To run, comment/uncomment appropriate lines in config.py and launch train.py.

def train_TextureMixer(
    smoothing               = 0.999,        # Exponential running average of encoder weights.
    minibatch_repeats       = 4,            # Number of minibatches to run before adjusting training parameters.
    reset_opt_for_new_lod   = True,         # Reset optimizer internal state (e.g. Adam moments) when new layers are introduced?
    total_kimg              = 37500,        # Total length of the training, measured in thousands of real images.
    lr_mirror_augment       = False,        # Enable mirror augment?
    ud_mirror_augment       = False,        # Enable up-down mirror augment?
    drange_net              = [-1,1],       # Dynamic range used when feeding image data to the networks.
    image_snapshot_ticks    = 10,           # How often to export image snapshots?
    network_snapshot_ticks  = 100,          # How often to export network snapshots?
    save_tf_graph           = False,        # Include full TensorFlow computation graph in the tfevents file?
    save_weight_histograms  = False,        # Include weight histograms in the tfevents file?
    
    resume_training         = False,        # Resume training?
    resume_run_id           = None,         # Run ID or network pkl to resume training from, None = start from scratch.
    resume_snapshot         = None):        # Snapshot index to resume training from, None = autodetect.

    maintenance_start_time = time.time()
    training_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **config.training_set)
    val_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **config.val_set)

    # Construct networks.
    with tf.device('/gpu:0'):
        if resume_training:
            network_pkl = misc.locate_network_pkl(resume_run_id, resume_snapshot)
            resume_kimg, resume_time = misc.resume_kimg_time(network_pkl)
            print('Loading networks from "%s"...' % network_pkl)
            E_zg, E_zl, G, D_rec, D_interp, D_blend, Es_zg, Es_zl, Gs = misc.load_pkl(network_pkl)
        else:
            print('Constructing networks...')
            resume_kimg = 0.0
            resume_time = 0.0
            E_zg = tfutil.Network('E_zg', num_channels=training_set.shape[0], resolution=training_set.shape[1], **config.E_zg)
            E_zl = tfutil.Network('E_zl', num_channels=training_set.shape[0], resolution=training_set.shape[1], **config.E_zl)
            G = tfutil.Network('G', num_channels=training_set.shape[0], resolution=training_set.shape[1], scale_h=1, scale_w=1, **config.G)
            D_rec = tfutil.Network('D_rec', num_channels=training_set.shape[0], resolution=training_set.shape[1], **config.D_rec)
            D_interp = tfutil.Network('D_interp', num_channels=training_set.shape[0], resolution=training_set.shape[1], **config.D_interp)
            D_blend = tfutil.Network('D_blend', num_channels=training_set.shape[0], resolution=training_set.shape[1], **config.D_blend)
            Es_zg = E_zg.clone('Es_zg')
            Es_zl = E_zl.clone('Es_zl')
            Gs = G.clone('Gs')
        G_fcn = tfutil.Network('G', reuse=True, num_channels=training_set.shape[0], resolution=training_set.shape[1], scale_h=config.scale_h, scale_w=config.scale_w, **config.G) # The same name as G to share trainable variables
        Gs_fcn = tfutil.Network('Gs', reuse=True, num_channels=training_set.shape[0], resolution=training_set.shape[1], scale_h=config.scale_h, scale_w=config.scale_w, **config.G) # The same name as Gs to share trainable variables
        Es_zg_update_op = Es_zg.setup_as_moving_average_of(E_zg, beta=smoothing)
        Es_zl_update_op = Es_zl.setup_as_moving_average_of(E_zl, beta=smoothing)
        Gs_update_op = Gs.setup_as_moving_average_of(G, beta=smoothing)
    E_zg.print_layers(); E_zl.print_layers(); G.print_layers(); D_rec.print_layers(); D_interp.print_layers(); D_blend.print_layers()

    print('Building TensorFlow graph...')
    with tf.name_scope('Inputs'):
        lod_in          = tf.placeholder(tf.float32, name='lod_in', shape=[])
        lrate_in        = tf.placeholder(tf.float32, name='lrate_in', shape=[])
        minibatch_in    = tf.placeholder(tf.int32, name='minibatch_in', shape=[])
        minibatch_split = minibatch_in // config.num_gpus
        reals, labels   = training_set.get_minibatch_tf()
        reals_split     = tf.split(reals, config.num_gpus)
        labels_split    = tf.split(labels, config.num_gpus)
        permutation_matrix_h_forward=tf.placeholder(tf.float32, name='permutation_matrix_h_forward', shape=[config.sched.minibatch_base, 1, config.latent_res_EG*config.scale_h, config.latent_res_EG*config.scale_h])
        permutation_matrix_w_forward=tf.placeholder(tf.float32, name='permutation_matrix_w_forward', shape=[config.sched.minibatch_base, 1, config.latent_res_EG*config.scale_w, config.latent_res_EG*config.scale_w])
        permutation_matrix_h_backward=tf.placeholder(tf.float32, name='permutation_matrix_h_backward', shape=[config.sched.minibatch_base, 1, config.latent_res_EG*config.scale_h, config.latent_res_EG*config.scale_h])
        permutation_matrix_w_backward=tf.placeholder(tf.float32, name='permutation_matrix_w_backward', shape=[config.sched.minibatch_base, 1, config.latent_res_EG*config.scale_w, config.latent_res_EG*config.scale_w])
        permutation_matrix_h_forward_split=tf.split(permutation_matrix_h_forward, config.num_gpus)
        permutation_matrix_w_forward_split=tf.split(permutation_matrix_w_forward, config.num_gpus)
        permutation_matrix_h_backward_split=tf.split(permutation_matrix_h_backward, config.num_gpus)
        permutation_matrix_w_backward_split=tf.split(permutation_matrix_w_backward, config.num_gpus)
    EG_opt = tfutil.Optimizer(name='TrainEG', learning_rate=lrate_in, **config.EG_opt)
    D_rec_opt = tfutil.Optimizer(name='TrainD_rec', learning_rate=lrate_in, **config.D_rec_opt)
    D_interp_opt = tfutil.Optimizer(name='TrainD_interp', learning_rate=lrate_in, **config.D_interp_opt)
    D_blend_opt = tfutil.Optimizer(name='TrainD_blend', learning_rate=lrate_in, **config.D_blend_opt)
    for gpu in range(config.num_gpus):
        with tf.name_scope('GPU%d' % gpu), tf.device('/gpu:%d' % gpu):
            E_zg_gpu = E_zg if gpu == 0 else E_zg.clone(E_zg.name + '_shadow_%d' % gpu)
            E_zl_gpu = E_zl if gpu == 0 else E_zl.clone(E_zl.name + '_shadow_%d' % gpu)
            G_gpu = G if gpu == 0 else G.clone(G.name + '_shadow_%d' % gpu)
            D_rec_gpu = D_rec if gpu == 0 else D_rec.clone(D_rec.name + '_shadow_%d' % gpu)
            G_fcn_gpu = G_fcn if gpu == 0 else G_fcn.clone(G_fcn.name + '_shadow_%d' % gpu)
            D_interp_gpu = D_interp if gpu == 0 else D_interp.clone(D_interp.name + '_shadow_%d' % gpu)
            D_blend_gpu = D_blend if gpu == 0 else D_blend.clone(D_blend.name + '_shadow_%d' % gpu)
            lod_assign_ops = [tf.assign(E_zg_gpu.find_var('lod'), lod_in), tf.assign(E_zl_gpu.find_var('lod'), lod_in), tf.assign(G_gpu.find_var('lod'), lod_in), tf.assign(D_rec_gpu.find_var('lod'), lod_in), tf.assign(D_interp_gpu.find_var('lod'), lod_in), tf.assign(D_blend_gpu.find_var('lod'), lod_in)]
            reals_fade_gpu, reals_orig_gpu = process_reals(reals_split[gpu], lod_in, lr_mirror_augment, ud_mirror_augment, training_set.dynamic_range, drange_net)
            labels_gpu = labels_split[gpu]
            with tf.name_scope('EG_loss'), tf.control_dependencies(lod_assign_ops):
                EG_loss = tfutil.call_func_by_name(E_zg=E_zg_gpu, E_zl=E_zl_gpu, G=G_gpu, D_rec=D_rec_gpu, G_fcn=G_fcn_gpu, D_interp=D_interp_gpu, D_blend=D_blend_gpu, minibatch_size=minibatch_split, reals_fade=reals_fade_gpu, reals_orig=reals_orig_gpu, labels=labels_gpu, permutation_matrix_h_forward=permutation_matrix_h_forward_split[gpu], permutation_matrix_w_forward=permutation_matrix_w_forward_split[gpu], permutation_matrix_h_backward=permutation_matrix_h_backward_split[gpu], permutation_matrix_w_backward=permutation_matrix_w_backward_split[gpu], **config.EG_loss)
            with tf.name_scope('D_rec_loss'), tf.control_dependencies(lod_assign_ops):
                D_rec_loss = tfutil.call_func_by_name(E_zg=E_zg_gpu, E_zl=E_zl_gpu, G=G_gpu, D_rec=D_rec_gpu, D_rec_opt=D_rec_opt, minibatch_size=minibatch_split, reals_fade=reals_fade_gpu, reals_orig=reals_orig_gpu, **config.D_rec_loss)
            with tf.name_scope('D_interp_loss'), tf.control_dependencies(lod_assign_ops):
                D_interp_loss = tfutil.call_func_by_name(E_zg=E_zg_gpu, E_zl=E_zl_gpu, G_fcn=G_fcn_gpu, D_interp=D_interp_gpu, D_interp_opt=D_interp_opt, minibatch_size=minibatch_split, reals_fade=reals_fade_gpu, reals_orig=reals_orig_gpu, permutation_matrix_h_forward=permutation_matrix_h_forward_split[gpu], permutation_matrix_w_forward=permutation_matrix_w_forward_split[gpu], **config.D_interp_loss)
            with tf.name_scope('D_blend_loss'), tf.control_dependencies(lod_assign_ops):
                D_blend_loss = tfutil.call_func_by_name(E_zg=E_zg_gpu, E_zl=E_zl_gpu, G_fcn=G_fcn_gpu, D_blend=D_blend_gpu, D_blend_opt=D_blend_opt, minibatch_size=minibatch_split, reals_fade=reals_fade_gpu, reals_orig=reals_orig_gpu, permutation_matrix_h_forward=permutation_matrix_h_forward_split[gpu], permutation_matrix_w_forward=permutation_matrix_w_forward_split[gpu], permutation_matrix_h_backward=permutation_matrix_h_backward_split[gpu], permutation_matrix_w_backward=permutation_matrix_w_backward_split[gpu], **config.D_blend_loss)
            EG_opt.register_gradients(tf.reduce_mean(EG_loss), OrderedDict(chain(E_zg_gpu.trainables.items(), E_zl_gpu.trainables.items(), G_gpu.trainables.items())))
            D_rec_opt.register_gradients(tf.reduce_mean(D_rec_loss), D_rec_gpu.trainables)
            D_interp_opt.register_gradients(tf.reduce_mean(D_interp_loss), D_interp_gpu.trainables)
            D_blend_opt.register_gradients(tf.reduce_mean(D_blend_loss), D_blend_gpu.trainables)
    EG_train_op = EG_opt.apply_updates()
    D_rec_train_op = D_rec_opt.apply_updates()
    D_interp_train_op = D_interp_opt.apply_updates()
    D_blend_train_op = D_blend_opt.apply_updates()

    print('Setting up snapshot image grid...')
    grid_size, grid_size_interp, grid_reals, grid_labels, grid_zg_latents, grid_zl_latents = setup_snapshot_image_grid(E_zg, E_zl, G, val_set, drange_net, None, **config.grid)
    sched = TrainingSchedule(total_kimg * 1000, training_set, **config.sched)

    permutation_matrix_h_0 = np.zeros((np.prod(grid_size_interp), 1, config.latent_res_EG*config.scale_h, config.latent_res_EG*config.scale_h)).astype(np.float32)
    for count1 in range(np.prod(grid_size_interp)):
        if config.block_size == 0:
            temp_perm = np.eye(config.latent_res_EG*config.scale_h)
            for idx in range(int(np.log2(config.latent_res_EG))):
                block_size_temp = int(2**idx)
                if config.perm:
                    perm = np.random.permutation(np.eye(config.latent_res_EG*config.scale_h//block_size_temp))
                else:
                    perm = my_swap_h(np.eye(config.latent_res_EG*config.scale_h//block_size_temp))
                temp_perm = np.matmul(temp_perm, block_permutation(perm, block_size_temp))
            permutation_matrix_h_0[count1,:,:,:] = np.array(temp_perm)
        else:
            if config.perm:
                perm = np.random.permutation(np.eye(config.latent_res_EG*config.scale_h//config.block_size))
            else:
                perm = my_swap_h(np.eye(config.latent_res_EG*config.scale_h//config.block_size))
            permutation_matrix_h_0[count1,:,:,:] = block_permutation(perm, config.block_size)
    permutation_matrix_w_0 = np.zeros((np.prod(grid_size_interp), 1, config.latent_res_EG*config.scale_w, config.latent_res_EG*config.scale_w)).astype(np.float32)
    for count1 in range(np.prod(grid_size_interp)):
        if config.block_size == 0:
            temp_perm = np.eye(config.latent_res_EG*config.scale_w)
            for idx in range(int(np.log2(config.latent_res_EG))):
                block_size_temp = int(2**idx)
                if config.perm:
                    perm = np.random.permutation(np.eye(config.latent_res_EG*config.scale_w//block_size_temp))
                else:
                    perm = my_swap_w(np.eye(config.latent_res_EG*config.scale_w//block_size_temp))
                temp_perm = np.matmul(block_permutation(perm, block_size_temp), temp_perm)
            permutation_matrix_w_0[count1,:,:,:] = np.array(temp_perm)
        else:
            if config.perm:
                perm = np.random.permutation(np.eye(config.latent_res_EG*config.scale_w//config.block_size))
            else:
                perm = my_swap_w(np.eye(config.latent_res_EG*config.scale_w//config.block_size))
            permutation_matrix_w_0[count1,:,:,:] = block_permutation(perm, config.block_size)
    
    grid_encoded_zg_mu, grid_encoded_zg_log_sigma = Es_zg.run(grid_reals, minibatch_size=sched.minibatch//config.num_gpus)
    grid_encoded_zg_latents = np.array(grid_encoded_zg_mu)
    grid_encoded_zl_mu, grid_encoded_zl_log_sigma = Es_zl.run(grid_reals, minibatch_size=sched.minibatch//config.num_gpus)
    grid_encoded_zl_latents = np.array(grid_encoded_zl_mu)
    grid_recs = Gs.run(np.tile(grid_encoded_zg_latents, [1,1]+Es_zl.output_shapes[0][2:]), grid_encoded_zl_latents, minibatch_size=sched.minibatch//config.num_gpus)
    
    if config.zg_interp_variational == 'hard':
        grid_interp_encoded_zg_latents = np.tile(grid_encoded_zg_latents[:np.prod(grid_size_interp),:,:,:], [1, 1, Es_zl.output_shapes[0][2]*config.scale_h, Es_zl.output_shapes[0][3]*config.scale_w])
    elif config.zg_interp_variational == 'variational':
        grid_interp_encoded_zg_latents = np.random.normal(size=[np.prod(grid_size_interp)]+Es_zg.output_shapes[0][1:]).astype(np.float32)
        grid_interp_encoded_zg_latents = grid_interp_encoded_zg_latents * np.exp(grid_encoded_zg_log_sigma[:np.prod(grid_size_interp),:,:,:]) + grid_encoded_zg_mu[:np.prod(grid_size_interp),:,:,:]
        grid_interp_encoded_zg_latents = tf.tile(grid_interp_encoded_zg_latents, [1, 1, Es_zl.output_shapes[0][2]*config.scale_h, Es_zl.output_shapes[0][3]*config.scale_w])
    if config.zl_interp_variational == 'hard':
        grid_interp_encoded_zl_latents = np.tile(grid_encoded_zl_latents[:np.prod(grid_size_interp),:,:,:], [1, 1, config.scale_h, config.scale_w])
    elif config.zl_interp_variational == 'variational':
        grid_interp_encoded_zl_mu = np.tile(grid_encoded_zl_mu[:np.prod(grid_size_interp),:,:,:], [1, 1, config.scale_h, config.scale_w])
        grid_interp_encoded_zl_log_sigma = np.tile(grid_encoded_zl_log_sigma[:np.prod(grid_size_interp),:,:,:], [1, 1, config.scale_h, config.scale_w])
        grid_interp_encoded_zl_latents = np.random.normal(size=(np.prod(grid_size_interp), Es_zl.output_shapes[0][1], Es_zl.output_shapes[0][2]*config.scale_h, Es_zl.output_shapes[0][3]*config.scale_w)).astype(np.float32)
        grid_interp_encoded_zl_latents = grid_interp_encoded_zl_latents * np.exp(grid_interp_encoded_zl_log_sigma) + grid_interp_encoded_zl_mu
    elif config.zl_interp_variational == 'random':
        grid_interp_encoded_zl_latents = np.random.normal(size=(np.prod(grid_size_interp), Es_zl.output_shapes[0][1], Es_zl.output_shapes[0][2]*config.scale_h, Es_zl.output_shapes[0][3]*config.scale_w)).astype(np.float32)
        grid_interp_encoded_zl_latents[:,:,:Es_zl.output_shapes[0][2],:Es_zl.output_shapes[0][3]] = grid_encoded_zl_latents[:np.prod(grid_size_interp),:,:,:]
        grid_interp_encoded_zl_latents[:,:,-Es_zl.output_shapes[0][2]:,:Es_zl.output_shapes[0][3]] = grid_encoded_zl_latents[:np.prod(grid_size_interp),:,:,:]
        grid_interp_encoded_zl_latents[:,:,:Es_zl.output_shapes[0][2],-Es_zl.output_shapes[0][3]:] = grid_encoded_zl_latents[:np.prod(grid_size_interp),:,:,:]
        grid_interp_encoded_zl_latents[:,:,-Es_zl.output_shapes[0][2]:,-Es_zl.output_shapes[0][3]:] = grid_encoded_zl_latents[:np.prod(grid_size_interp),:,:,:]
    elif config.zl_interp_variational == 'permutational':
        grid_interp_encoded_zl_latents = np.matmul(np.matmul(np.tile(permutation_matrix_h_0, [1,Es_zl.output_shapes[0][1],1,1]), np.tile(grid_encoded_zl_latents[:np.prod(grid_size_interp),:,:,:], [1,1,config.scale_h,config.scale_w])), np.tile(permutation_matrix_w_0, [1,Es_zl.output_shapes[0][1],1,1]))
        grid_interp_encoded_zl_latents[:,:,:Es_zl.output_shapes[0][2],:Es_zl.output_shapes[0][3]] = grid_encoded_zl_latents[:np.prod(grid_size_interp),:,:,:]
        grid_interp_encoded_zl_latents[:,:,-Es_zl.output_shapes[0][2]:,:Es_zl.output_shapes[0][3]] = grid_encoded_zl_latents[:np.prod(grid_size_interp),:,:,:]
        grid_interp_encoded_zl_latents[:,:,:Es_zl.output_shapes[0][2],-Es_zl.output_shapes[0][3]:] = grid_encoded_zl_latents[:np.prod(grid_size_interp),:,:,:]
        grid_interp_encoded_zl_latents[:,:,-Es_zl.output_shapes[0][2]:,-Es_zl.output_shapes[0][3]:] = grid_encoded_zl_latents[:np.prod(grid_size_interp),:,:,:]

    grid_interps = Gs_fcn.run(grid_interp_encoded_zg_latents, grid_interp_encoded_zl_latents, minibatch_size=sched.minibatch//config.num_gpus)

    print('Setting up result dir...')
    result_subdir = misc.create_result_subdir(config.result_dir, config.desc)
    misc.save_image_grid(grid_reals, os.path.join(result_subdir, 'reals.png'), drange=drange_net, grid_size=grid_size)
    misc.save_image_grid(grid_recs, os.path.join(result_subdir, 'recs%06d.png' % 0), drange=drange_net, grid_size=grid_size)
    misc.save_image_grid(grid_interps, os.path.join(result_subdir, 'interps%06d.png' % 0), drange=drange_net, grid_size=grid_size_interp)

    summary_log = tf.summary.FileWriter(result_subdir)
    if save_tf_graph:
        summary_log.add_graph(tf.get_default_graph())
    if save_weight_histograms:
        E_zg.setup_weight_histograms(); E_zl.setup_weight_histograms(); G.setup_weight_histograms(); D_rec.setup_weight_histograms(); G_fcn.setup_weight_histograms(); D_interp.setup_weight_histograms(); D_blend.setup_weight_histograms()

    print('Training...')
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    train_start_time = tick_start_time - resume_time
    prev_lod = -1.0
    cur_time = time.time()
    total_time = cur_time - train_start_time
    while cur_nimg < total_kimg * 1000:# and total_time < 6 * 24*60*60:

        # Choose training parameters and configure training ops.
        sched = TrainingSchedule(cur_nimg, training_set, **config.sched)
        training_set.configure(sched.minibatch, sched.lod)
        if reset_opt_for_new_lod:
            if np.floor(sched.lod) != np.floor(prev_lod) or np.ceil(sched.lod) != np.ceil(prev_lod):
                EG_opt.reset_optimizer_state(); D_rec_opt.reset_optimizer_state(); D_interp_opt.reset_optimizer_state(); D_blend_opt.reset_optimizer_state()
        prev_lod = sched.lod

        permutation_matrix_h_1_forward = np.zeros((sched.minibatch, 1, config.latent_res_EG*config.scale_h, config.latent_res_EG*config.scale_h)).astype(np.float32)
        for count1 in range(sched.minibatch):
            if config.block_size == 0:
                temp_perm = np.eye(config.latent_res_EG*config.scale_h)
                for idx in range(int(np.log2(config.latent_res_EG))):
                    block_size_temp = int(2**idx)
                    if config.perm:
                        perm = np.random.permutation(np.eye(config.latent_res_EG*config.scale_h//block_size_temp))
                    else:
                        perm = my_swap_h(np.eye(config.latent_res_EG*config.scale_h//block_size_temp))
                    temp_perm = np.matmul(temp_perm, block_permutation(perm, block_size_temp))
                permutation_matrix_h_1_forward[count1,:,:,:] = np.array(temp_perm)
            else:
                if config.perm:
                    perm = np.random.permutation(np.eye(config.latent_res_EG*config.scale_h//config.block_size))
                else:
                    perm = my_swap_h(np.eye(config.latent_res_EG*config.scale_h//config.block_size))
                permutation_matrix_h_1_forward[count1,:,:,:] = block_permutation(perm, config.block_size)
        permutation_matrix_w_1_forward = np.zeros((sched.minibatch, 1, config.latent_res_EG*config.scale_w, config.latent_res_EG*config.scale_w)).astype(np.float32)
        for count1 in range(sched.minibatch):
            if config.block_size == 0:
                temp_perm = np.eye(config.latent_res_EG*config.scale_w)
                for idx in range(int(np.log2(config.latent_res_EG))):
                    block_size_temp = int(2**idx)
                    if config.perm:
                        perm = np.random.permutation(np.eye(config.latent_res_EG*config.scale_w//block_size_temp))
                    else:
                        perm = my_swap_w(np.eye(config.latent_res_EG*config.scale_w//block_size_temp))
                    temp_perm = np.matmul(block_permutation(perm, block_size_temp), temp_perm)
                permutation_matrix_w_1_forward[count1,:,:,:] = np.array(temp_perm)
            else:
                if config.perm:
                    perm = np.random.permutation(np.eye(config.latent_res_EG*config.scale_w//config.block_size))
                else:
                    perm = my_swap_w(np.eye(config.latent_res_EG*config.scale_w//config.block_size))
                permutation_matrix_w_1_forward[count1,:,:,:] = block_permutation(perm, config.block_size)
        permutation_matrix_h_1_backward = np.zeros((sched.minibatch, 1, config.latent_res_EG*config.scale_h, config.latent_res_EG*config.scale_h)).astype(np.float32)
        for count1 in range(sched.minibatch):
            if config.block_size == 0:
                temp_perm = np.eye(config.latent_res_EG*config.scale_h)
                for idx in range(int(np.log2(config.latent_res_EG))):
                    block_size_temp = int(2**idx)
                    if config.perm:
                        perm = np.random.permutation(np.eye(config.latent_res_EG*config.scale_h//block_size_temp))
                    else:
                        perm = my_swap_h(np.eye(config.latent_res_EG*config.scale_h//block_size_temp))
                    temp_perm = np.matmul(temp_perm, block_permutation(perm, block_size_temp))
                permutation_matrix_h_1_backward[count1,:,:,:] = np.array(temp_perm)
            else:
                if config.perm:
                    perm = np.random.permutation(np.eye(config.latent_res_EG*config.scale_h//config.block_size))
                else:
                    perm = my_swap_h(np.eye(config.latent_res_EG*config.scale_h//config.block_size))
                permutation_matrix_h_1_backward[count1,:,:,:] = block_permutation(perm, config.block_size)
        permutation_matrix_w_1_backward = np.zeros((sched.minibatch, 1, config.latent_res_EG*config.scale_w, config.latent_res_EG*config.scale_w)).astype(np.float32)
        for count1 in range(sched.minibatch):
            if config.block_size == 0:
                temp_perm = np.eye(config.latent_res_EG*config.scale_w)
                for idx in range(int(np.log2(config.latent_res_EG))):
                    block_size_temp = int(2**idx)
                    if config.perm:
                        perm = np.random.permutation(np.eye(config.latent_res_EG*config.scale_w//block_size_temp))
                    else:
                        perm = my_swap_w(np.eye(config.latent_res_EG*config.scale_w//block_size_temp))
                    temp_perm = np.matmul(block_permutation(perm, block_size_temp), temp_perm)
                permutation_matrix_w_1_backward[count1,:,:,:] = np.array(temp_perm)
            else:
                if config.perm:
                    perm = np.random.permutation(np.eye(config.latent_res_EG*config.scale_w//config.block_size))
                else:
                    perm = my_swap_w(np.eye(config.latent_res_EG*config.scale_w//config.block_size))
                permutation_matrix_w_1_backward[count1,:,:,:] = block_permutation(perm, config.block_size)

        # Run training ops.
        for repeat in range(minibatch_repeats):
            tfutil.run([D_rec_train_op, D_interp_train_op, D_blend_train_op], {lod_in: sched.lod, lrate_in: sched.lrate, minibatch_in: sched.minibatch, permutation_matrix_h_forward: permutation_matrix_h_1_forward, permutation_matrix_w_forward: permutation_matrix_w_1_forward, permutation_matrix_h_backward: permutation_matrix_h_1_backward, permutation_matrix_w_backward: permutation_matrix_w_1_backward})
            tfutil.run([EG_train_op], {lod_in: sched.lod, lrate_in: sched.lrate, minibatch_in: sched.minibatch, permutation_matrix_h_forward: permutation_matrix_h_1_forward, permutation_matrix_w_forward: permutation_matrix_w_1_forward, permutation_matrix_h_backward: permutation_matrix_h_1_backward, permutation_matrix_w_backward: permutation_matrix_w_1_backward})
            tfutil.run([Es_zg_update_op, Es_zl_update_op, Gs_update_op], {})
            cur_nimg += sched.minibatch

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if cur_nimg >= tick_start_nimg + sched.tick_kimg * 1000 or done:
            cur_tick += 1
            cur_time = time.time()
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = cur_time - tick_start_time
            total_time = cur_time - train_start_time
            maintenance_time = tick_start_time - maintenance_start_time
            maintenance_start_time = cur_time

            # Report progress.
            print('tick %-5d kimg %-8.1f lod %-5.2f resolution %-4d minibatch %-4d time %-12s sec/tick %-7.1f sec/kimg %-7.2f maintenance %.1f' % (
                tfutil.autosummary('Progress/tick', cur_tick),
                tfutil.autosummary('Progress/kimg', cur_nimg / 1000.0),
                tfutil.autosummary('Progress/lod', sched.lod),
                tfutil.autosummary('Progress/resolution', sched.resolution),
                tfutil.autosummary('Progress/minibatch', sched.minibatch),
                misc.format_time(tfutil.autosummary('Timing/total_sec', total_time)),
                tfutil.autosummary('Timing/sec_per_tick', tick_time),
                tfutil.autosummary('Timing/sec_per_kimg', tick_time / tick_kimg),
                tfutil.autosummary('Timing/maintenance_sec', maintenance_time)))
            tfutil.autosummary('Timing/total_hours', total_time / (60.0 * 60.0))
            tfutil.autosummary('Timing/total_days', total_time / (24.0 * 60.0 * 60.0))
            tfutil.save_summaries(summary_log, cur_nimg)

            # Save snapshots.
            if cur_tick % image_snapshot_ticks == 0 or done:
                grid_encoded_zg_mu, grid_encoded_zg_log_sigma = Es_zg.run(grid_reals, minibatch_size=sched.minibatch//config.num_gpus)
                grid_encoded_zg_latents = np.array(grid_encoded_zg_mu)
                grid_encoded_zl_mu, grid_encoded_zl_log_sigma = Es_zl.run(grid_reals, minibatch_size=sched.minibatch//config.num_gpus)
                grid_encoded_zl_latents = np.array(grid_encoded_zl_mu)
                grid_recs = Gs.run(np.tile(grid_encoded_zg_latents, [1,1]+Es_zl.output_shapes[0][2:]), grid_encoded_zl_latents, minibatch_size=sched.minibatch//config.num_gpus)
                
                if config.zg_interp_variational == 'hard':
                    grid_interp_encoded_zg_latents = np.tile(grid_encoded_zg_latents[:np.prod(grid_size_interp),:,:,:], [1, 1, Es_zl.output_shapes[0][2]*config.scale_h, Es_zl.output_shapes[0][3]*config.scale_w])
                elif config.zg_interp_variational == 'variational':
                    grid_interp_encoded_zg_latents = np.random.normal(size=[np.prod(grid_size_interp)]+Es_zg.output_shapes[0][1:]).astype(np.float32)
                    grid_interp_encoded_zg_latents = grid_interp_encoded_zg_latents * np.exp(grid_encoded_zg_log_sigma[:np.prod(grid_size_interp),:,:,:]) + grid_encoded_zg_mu[:np.prod(grid_size_interp),:,:,:]
                    grid_interp_encoded_zg_latents = tf.tile(grid_interp_encoded_zg_latents, [1, 1, Es_zl.output_shapes[0][2]*config.scale_h, Es_zl.output_shapes[0][3]*config.scale_w])
                if config.zl_interp_variational == 'hard':
                    grid_interp_encoded_zl_latents = np.tile(grid_encoded_zl_latents[:np.prod(grid_size_interp),:,:,:], [1, 1, config.scale_h, config.scale_w])
                elif config.zl_interp_variational == 'variational':
                    grid_interp_encoded_zl_mu = np.tile(grid_encoded_zl_mu[:np.prod(grid_size_interp),:,:,:], [1, 1, config.scale_h, config.scale_w])
                    grid_interp_encoded_zl_log_sigma = np.tile(grid_encoded_zl_log_sigma[:np.prod(grid_size_interp),:,:,:], [1, 1, config.scale_h, config.scale_w])
                    grid_interp_encoded_zl_latents = np.random.normal(size=(np.prod(grid_size_interp), Es_zl.output_shapes[0][1], Es_zl.output_shapes[0][2]*config.scale_h, Es_zl.output_shapes[0][3]*config.scale_w)).astype(np.float32)
                    grid_interp_encoded_zl_latents = grid_interp_encoded_zl_latents * np.exp(grid_interp_encoded_zl_log_sigma) + grid_interp_encoded_zl_mu
                elif config.zl_interp_variational == 'random':
                    grid_interp_encoded_zl_latents = np.random.normal(size=(np.prod(grid_size_interp), Es_zl.output_shapes[0][1], Es_zl.output_shapes[0][2]*config.scale_h, Es_zl.output_shapes[0][3]*config.scale_w)).astype(np.float32)
                    grid_interp_encoded_zl_latents[:,:,:Es_zl.output_shapes[0][2],:Es_zl.output_shapes[0][3]] = grid_encoded_zl_latents[:np.prod(grid_size_interp),:,:,:]
                    grid_interp_encoded_zl_latents[:,:,-Es_zl.output_shapes[0][2]:,:Es_zl.output_shapes[0][3]] = grid_encoded_zl_latents[:np.prod(grid_size_interp),:,:,:]
                    grid_interp_encoded_zl_latents[:,:,:Es_zl.output_shapes[0][2],-Es_zl.output_shapes[0][3]:] = grid_encoded_zl_latents[:np.prod(grid_size_interp),:,:,:]
                    grid_interp_encoded_zl_latents[:,:,-Es_zl.output_shapes[0][2]:,-Es_zl.output_shapes[0][3]:] = grid_encoded_zl_latents[:np.prod(grid_size_interp),:,:,:]
                elif config.zl_interp_variational == 'permutational':
                    grid_interp_encoded_zl_latents = np.matmul(np.matmul(np.tile(permutation_matrix_h_0, [1,Es_zl.output_shapes[0][1],1,1]), np.tile(grid_encoded_zl_latents[:np.prod(grid_size_interp),:,:,:], [1,1,config.scale_h,config.scale_w])), np.tile(permutation_matrix_w_0, [1,Es_zl.output_shapes[0][1],1,1]))
                    grid_interp_encoded_zl_latents[:,:,:Es_zl.output_shapes[0][2],:Es_zl.output_shapes[0][3]] = grid_encoded_zl_latents[:np.prod(grid_size_interp),:,:,:]
                    grid_interp_encoded_zl_latents[:,:,-Es_zl.output_shapes[0][2]:,:Es_zl.output_shapes[0][3]] = grid_encoded_zl_latents[:np.prod(grid_size_interp),:,:,:]
                    grid_interp_encoded_zl_latents[:,:,:Es_zl.output_shapes[0][2],-Es_zl.output_shapes[0][3]:] = grid_encoded_zl_latents[:np.prod(grid_size_interp),:,:,:]
                    grid_interp_encoded_zl_latents[:,:,-Es_zl.output_shapes[0][2]:,-Es_zl.output_shapes[0][3]:] = grid_encoded_zl_latents[:np.prod(grid_size_interp),:,:,:]

                grid_interps = Gs_fcn.run(grid_interp_encoded_zg_latents, grid_interp_encoded_zl_latents, minibatch_size=sched.minibatch//config.num_gpus)

                misc.save_image_grid(grid_recs, os.path.join(result_subdir, 'recs%06d.png' % (cur_nimg // 1000)), drange=drange_net, grid_size=grid_size)
                misc.save_image_grid(grid_interps, os.path.join(result_subdir, 'interps%06d.png' % (cur_nimg // 1000)), drange=drange_net, grid_size=grid_size_interp)
            
            if cur_tick % network_snapshot_ticks == 0 or done:
                misc.save_pkl((E_zg, E_zl, G, D_rec, D_interp, D_blend, Es_zg, Es_zl, Gs), os.path.join(result_subdir, 'network-snapshot-%06d.pkl' % (cur_nimg // 1000)))

            # Record start time of the next tick.
            tick_start_time = time.time()

    # Write final results.
    misc.save_pkl((E_zg, E_zl, G, D_rec, D_interp, D_blend, Es_zg, Es_zl, Gs), os.path.join(result_subdir, 'network-final.pkl'))
    summary_log.close()
    open(os.path.join(result_subdir, '_training-done.txt'), 'wt').close()

#----------------------------------------------------------------------------
# Main entry point.
# Calls the function indicated in config.py.

if __name__ == "__main__":
    misc.init_output_logging()
    np.random.seed(config.random_seed)
    print('Initializing TensorFlow...')
    os.environ.update(config.env)
    tfutil.init_tf(config.tf_config)
    config.env.CUDA_VISIBLE_DEVICES = '0'; config.num_gpus = 1

    parser = argparse.ArgumentParser()
    parser.add_argument('--app', type=str, default=' ')
    parser.add_argument('--model_path', type=str, default=' ')
    parser.add_argument('--out_dir', type=str, default=' ')
    #------------------- training arguments -------------------
    parser.add_argument('--train_dir', type=str, default=' ') # The prepared training dataset directory that can be efficiently called by the code
    parser.add_argument('--val_dir', type=str, default=' ') # The prepared validation dataset directory that can be efficiently called by the code
    parser.add_argument('--num_gpus', type=int, default=1) # The number of GPUs for training. Options {1, 2, 4, 8}. Using 8 NVIDIA GeForce GTX 1080 Ti GPUs, we suggest training for 3 days.
    #------------------- texture interpolation arguments -------------------
    parser.add_argument('--imageL_path', type=str, default=' ') # The left-hand side image for horizontal interpolation
    parser.add_argument('--imageR_path', type=str, default=' ') # The right-hand side image for horizontal interpolation
    #------------------- texture dissolve arguments -------------------
    parser.add_argument('--imageStartUL_path', type=str, default=' ') # The upper-left corner image in the starting frame
    parser.add_argument('--imageStartUR_path', type=str, default=' ') # The upper-right corner image in the starting frame
    parser.add_argument('--imageStartBL_path', type=str, default=' ') # The bottom-left corner image in the starting frame
    parser.add_argument('--imageStartBR_path', type=str, default=' ') # The bottom-right corner image in the starting frame
    parser.add_argument('--imageEndUL_path', type=str, default=' ') # The upper-left corner image in the ending frame
    parser.add_argument('--imageEndUR_path', type=str, default=' ') # The upper-right corner image in the ending frame
    parser.add_argument('--imageEndBL_path', type=str, default=' ') # The bottom-left corner image in the ending frame
    parser.add_argument('--imageEndBR_path', type=str, default=' ') # The bottom-right corner image in the ending frame
    #------------------- texture brush arguments -------------------
    parser.add_argument('--imageBgUL_path', type=str, default=' ') # The upper-left corner image for the background canvas
    parser.add_argument('--imageBgUR_path', type=str, default=' ') # The upper-right corner image for the background canvas
    parser.add_argument('--imageBgBL_path', type=str, default=' ') # The bottom-left corner image for the background canvas
    parser.add_argument('--imageBgBR_path', type=str, default=' ') # The bottom-right corner image for the background canvas
    parser.add_argument('--imageFgUL_path', type=str, default=' ') # The upper-left corner image for the foreground palatte
    parser.add_argument('--imageFgUR_path', type=str, default=' ') # The upper-right corner image for the foreground palatte
    parser.add_argument('--imageFgBL_path', type=str, default=' ') # The bottom-left corner image for the foreground palatte
    parser.add_argument('--imageFgBR_path', type=str, default=' ') # The bottom-right corner image for the foreground palatte
    parser.add_argument('--stroke1_path', type=str, default=' ') # The trajectory image for the 1st stroke. The stroke pattern is sampled from the [3/8, 3/8] portion of the foreground palatte
    parser.add_argument('--stroke2_path', type=str, default=' ') # The trajectory image for the 2nd stroke. The stroke pattern is sampled from the [3/8, 7/8] portion of the foreground palatte
    parser.add_argument('--stroke3_path', type=str, default=' ') # The trajectory image for the 3rd stroke. The stroke pattern is sampled from the [7/8, 3/8] portion of the foreground palatte
    parser.add_argument('--stroke4_path', type=str, default=' ') # The trajectory image for the 4th stroke. The stroke pattern is sampled from the [7/8, 7/8] portion of the foreground palatte
    #------------------- texture brush arguments -------------------
    parser.add_argument('--source_dir', type=str, default=' ') # The directory containing the hole region to be interpolated, two known source texture images adjacent to the hole, and their global Adobe Content-Aware Fill (CAF) operation results

    args = parser.parse_args()
    if args.app == 'train':
        assert args.train_dir != ' ' and args.val_dir != ' ' and args.out_dir != ' '
        config.training_set = config.EasyDict(tfrecord_dir=args.train_dir, max_label_size='full')
        config.val_set = config.EasyDict(tfrecord_dir=args.val_dir, max_label_size='full')
        config.train = config.EasyDict(func='run.train_TextureMixer', lr_mirror_augment=False, ud_mirror_augment=False, total_kimg=500000)
        config.result_dir = args.out_dir
        config.num_gpus = args.num_gpus
        if config.num_gpus == 1:
            config.env.CUDA_VISIBLE_DEVICES = '0'
            config.desc += '-preset-v2-1gpus'; config.sched.minibatch_base = 4; config.sched.lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        elif config.num_gpus == 2:
            config.env.CUDA_VISIBLE_DEVICES = '0,1'
            config.desc += '-preset-v2-2gpus'; config.sched.minibatch_base = 8; config.sched.lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        elif config.num_gpus == 4:
            config.env.CUDA_VISIBLE_DEVICES = '0,1,2,3'
            config.desc += '-preset-v2-4gpus'; config.sched.minibatch_base = 16; config.sched.lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        elif config.num_gpus == 8:
            config.env.CUDA_VISIBLE_DEVICES = '0,1,2,3,4,5,6,7'
            config.desc += '-preset-v2-8gpus'; config.sched.minibatch_base = 32; config.sched.lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        tfutil.call_func_by_name(**config.train)
    elif args.app == 'interpolation':
        assert args.model_path != ' ' and args.imageL_path != ' ' and args.imageR_path != ' ' and args.out_dir != ' '
        config.env.CUDA_VISIBLE_DEVICES = '0'; config.num_gpus = 1
        app = config.EasyDict(func='util_scripts.horizontal_interpolation', model_path=args.model_path, imageL_path=args.imageL_path, imageR_path=args.imageR_path, out_dir=args.out_dir)        
        tfutil.call_func_by_name(**app)
    elif args.app == 'dissolve':
        assert args.model_path != ' ' and args.imageStartUL_path != ' ' and args.imageStartUR_path != ' ' and args.imageStartBL_path != ' ' and args.imageStartBR_path != ' ' and args.imageEndUL_path != ' ' and args.imageEndUR_path != ' ' and args.imageEndBL_path != ' ' and args.imageEndBR_path != ' ' and args.out_dir != ' '
        config.env.CUDA_VISIBLE_DEVICES = '0'; config.num_gpus = 1
        app = config.EasyDict(func='util_scripts.texture_dissolve_video', model_path=args.model_path, imageStartUL_path=args.imageStartUL_path, imageStartUR_path=args.imageStartUR_path, imageStartBL_path=args.imageStartBL_path, imageStartBR_path=args.imageStartBR_path, imageEndUL_path=args.imageEndUL_path, imageEndUR_path=args.imageEndUR_path, imageEndBL_path=args.imageEndBL_path, imageEndBR_path=args.imageEndBR_path, out_dir=args.out_dir)
        tfutil.call_func_by_name(**app)
    elif args.app == 'brush':
        assert args.model_path != ' ' and args.imageBgUL_path != ' ' and args.imageBgUR_path != ' ' and args.imageBgBL_path != ' ' and args.imageBgBR_path != ' ' and args.imageFgUL_path != ' ' and args.imageFgUR_path != ' ' and args.imageFgBL_path != ' ' and args.imageFgBR_path != ' ' and args.stroke1_path != ' ' and args.stroke2_path != ' ' and args.stroke3_path != ' ' and args.stroke4_path != ' ' and args.out_dir != ' '
        config.env.CUDA_VISIBLE_DEVICES = '0'; config.num_gpus = 1
        app = config.EasyDict(func='util_scripts.texture_brush_video', model_path=args.model_path, imageBgUL_path=args.imageBgUL_path, imageBgUR_path=args.imageBgUR_path, imageBgBL_path=args.imageBgBL_path, imageBgBR_path=args.imageBgBR_path, imageFgUL_path=args.imageFgUL_path, imageFgUR_path=args.imageFgUR_path, imageFgBL_path=args.imageFgBL_path, imageFgBR_path=args.imageFgBR_path, stroke1_path=args.stroke1_path, stroke2_path=args.stroke2_path, stroke3_path=args.stroke3_path, stroke4_path=args.stroke4_path, out_dir=args.out_dir)
        tfutil.call_func_by_name(**app)
    elif args.app == 'hybridization':
        assert args.model_path != ' ' and args.source_dir != ' ' and args.out_dir != ' '
        config.env.CUDA_VISIBLE_DEVICES = '0'; config.num_gpus = 1
        app = config.EasyDict(func='util_scripts.hybridization_CAF', model_path=args.model_path, source_dir=args.source_dir, out_dir=args.out_dir)
        tfutil.call_func_by_name(**app)
#----------------------------------------------------------------------------
