# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to

# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import tensorflow as tf

import tfutil
import config

from custom_vgg19 import *

from networks import upscale2d

#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

# gram matrix per layer
def gram_matrix(x, data_format='NCHW'):
    if data_format == 'NCHW':
        x = tf.transpose(x, perm=[0,2,3,1])
    h = tf.shape(x)[1]; w = tf.shape(x)[2]; ch = tf.shape(x)[3]
    features = tf.reshape(x, [-1, h*w, ch])
    gram = tf.matmul(features, features, adjoint_a=True) / tf.cast(h, tf.float32) / tf.cast(w, tf.float32)
    return gram

# 2D autocorrelation matrix per layer
def autocorrelation_matrix(x, h, w, data_format='NCHW'):
    if data_format == 'NHWC':
        x = tf.transpose(x, perm=[0,3,1,2])
    x_mu = tf.tile(tf.reduce_mean(x, axis=[2,3], keepdims=True), [1,1,h,w])
    x_zero = x - x_mu
    x_std = tf.tile(tf.sqrt(tf.reduce_mean(x_zero*x_zero, axis=[2,3], keepdims=True)), [1,1,h,w])
    x_normalize = x_zero / (x_std + tf.constant(1e-8, dtype=tf.float32))
    b = tf.shape(x)[0]; ch = tf.shape(x)[1]
    x_normalize = tf.reshape(x_normalize, [1,b*ch,h,w])
    x_pad = tf.pad(x_normalize, paddings=[[0,0],[0,0],[h-1,0],[w-1,0]], mode='CONSTANT', constant_values=0)
    x_filter = tf.transpose(x_normalize, [2,3,1,0])
    autocorrelation = tf.nn.depthwise_conv2d(x_pad, x_filter, strides=[1,1,1,1], padding='VALID', data_format='NCHW')
    autocorrelation = tf.reshape(autocorrelation, [b,ch,h,w])
    div = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            div[i,j] = (i+1) * (j+1)
    div = div.astype(np.float32).flatten()
    div = tf.constant(list(div), dtype=tf.float32, shape=[1,1,h,w], verify_shape=False)
    div = tf.tile(div, [b,ch,1,1])
    autocorrelation /= div
    return autocorrelation

def loss_addup(loss, loss_):
    if loss is None:
        L = tf.identity(loss_)
    else:
        L = loss + loss_
    return L

def multi_layer_diff(feature, feature_, dtype):
    l = tf.constant(0.0, dtype)
    for f, f_ in zip(feature, feature_):
        if len(f.get_shape().as_list()) == 4:
            l += tf.reduce_mean(tf.abs(tf.cast(f, dtype) - tf.cast(f_, dtype)), axis=[1,2,3])
        elif len(f.get_shape().as_list()) == 3:
            l += tf.reduce_mean(tf.abs(tf.cast(f, dtype) - tf.cast(f_, dtype)), axis=[1,2])
    return l

# Randomly and independently crop patches from a batch of images.
def random_crop(images, minibatch_size, input_shape, output_shape):
    begin = tf.constant([0, 0], dtype=tf.int32)
    if input_shape[2] == output_shape[2]:
        rand_height = tf.constant([0], dtype=tf.int32, shape=[1])
    else:
        rand_height = tf.random_uniform(shape=[1], minval=0, maxval=input_shape[2]-output_shape[2], dtype=tf.int32)
    if input_shape[3] == output_shape[3]:
        rand_width = tf.constant([0], dtype=tf.int32, shape=[1])
    else:
        rand_width = tf.random_uniform(shape=[1], minval=0, maxval=input_shape[3]-output_shape[3], dtype=tf.int32)
    begin = tf.concat([begin, rand_height, rand_width], axis=0)
    crops = tf.slice(images, begin=begin, size=[minibatch_size, tf.shape(images)[1]] + output_shape[2:])
    return crops

def tiling_permutation(tensor, scale_h, scale_w, permutation_matrix_h, permutation_matrix_w):
    permutation_matrix_h_1 = tf.tile(permutation_matrix_h, [1, tf.shape(tensor)[1], 1, 1])
    permutation_matrix_w_1 = tf.tile(permutation_matrix_w, [1, tf.shape(tensor)[1], 1, 1])
    tensor_tiled = tf.matmul(tf.matmul(permutation_matrix_h_1, tf.tile(tensor, [1, 1, scale_h, scale_w])), permutation_matrix_w_1)
    h = tensor.get_shape().as_list()[2]; w = tensor.get_shape().as_list()[3]
    tensor1 = tf.concat([tensor, tensor_tiled[:,:,:h,w:-w], tensor], axis=3)
    tensor2 = tensor_tiled[:,:,h:-h,:]
    tensor3 = tf.concat([tensor, tensor_tiled[:,:,-h:,w:-w], tensor], axis=3)
    return tf.concat([tensor1, tensor2, tensor3], axis=2)

#----------------------------------------------------------------------------
# AutoEncoder reconstruction loss function (rec-WGAN + pixel L1 + latent L1 + KL + interp-WGAN).

def EG_wgan(E_zg, E_zl, G, D_rec, G_fcn, D_interp, D_blend, minibatch_size, reals_fade, reals_orig, labels, permutation_matrix_h_forward, permutation_matrix_w_forward, permutation_matrix_h_backward, permutation_matrix_w_backward,
    scale_h,                    # Height scale of interpolated size
    scale_w,                    # Width scale of interpolated size
    zg_interp_variational,      # Enable hard or variational or learned or random zg interpolation?
    zl_interp_variational,      # Enable hard or variational or learned or random zl interpolation?
    rec_G_weight,               # Weight of the reconstructed realism loss term.
    pixel_weight,               # Weight of the L1-based loss term in the image domain.
    gram_weight,                # Weight of the Gram matrix loss term.
    latent_weight,              # Weight of the L1-based pixel loss term in the latent domain
    kl_weight,                  # Weight of the KL divergence term
    interp_G_weight,            # Weight of the interpolated realism loss term.
    blend_interp_G_weight):     # Weight of the blended interpolatedrealism loss term.

    # zg encoding 
    enc_zg_mu, enc_zg_log_sigma = E_zg.get_output_for(reals_orig)
    if config.zg_enabled:
        enc_zg_latents = tf.identity(enc_zg_mu)
    else:
        enc_zg_latents = tf.zeros(tf.shape(enc_zg_mu))

    # zl encoding 
    enc_zl_mu, enc_zl_log_sigma = E_zl.get_output_for(reals_orig)
    enc_zl_latents = tf.identity(enc_zl_mu)

    # generating
    rec_images_out = G.get_output_for(tf.tile(enc_zg_latents, [1,1]+E_zl.output_shapes[0][2:]), enc_zl_latents)
    loss = None

    # reconstructed realism
    if rec_G_weight > 0.0:
        rec_scores_out = fp32(D_rec.get_output_for(rec_images_out))
        rec_G_loss = tf.reduce_mean(-rec_scores_out, axis=[1,2,3])
        rec_G_loss *= rec_G_weight
        rec_G_loss = tfutil.autosummary('Loss/rec_G_loss', rec_G_loss)
        loss = loss_addup(loss, rec_G_loss)

    # L1 pixel loss
    if pixel_weight > 0.0:
        rec_pixel_loss = tf.reduce_mean(tf.abs(rec_images_out - tf.cast(reals_fade, rec_images_out.dtype)), axis=[1,2,3])
        rec_pixel_loss *= pixel_weight
        rec_pixel_loss = tfutil.autosummary('Loss/rec_pixel_loss', rec_pixel_loss)
        loss = loss_addup(loss, rec_pixel_loss)

    # gram matrix loss
    if gram_weight > 0.0:
        data_dict = loadWeightsData('tensorflow_vgg/vgg19.npy')
        rec_vgg = custom_Vgg19(rec_images_out, data_dict=data_dict)
        real_vgg = custom_Vgg19(reals_fade, data_dict=data_dict)
        rec_feature = [rec_vgg.conv1_1, rec_vgg.conv2_1, rec_vgg.conv3_1, rec_vgg.conv4_1, rec_vgg.conv5_1]
        real_feature = [real_vgg.conv1_1, real_vgg.conv2_1, real_vgg.conv3_1, real_vgg.conv4_1, real_vgg.conv5_1]
        rec_gram = [gram_matrix(l, data_format='NHWC') for l in rec_feature]
        real_gram = [gram_matrix(l, data_format='NHWC') for l in real_feature]
        rec_gram_loss = multi_layer_diff(rec_gram, real_gram, dtype=rec_images_out.dtype)
        rec_gram_loss *= gram_weight
        rec_gram_loss = tfutil.autosummary('Loss/rec_gram_loss', rec_gram_loss)
        loss = loss_addup(loss, rec_gram_loss)

    # KL divergence regularization
    if kl_weight > 0.0:
        KL_zg = -0.5 * tf.reduce_mean(1+2*enc_zg_log_sigma-enc_zg_mu**2-tf.exp(2*enc_zg_log_sigma), axis=[1,2,3])
        KL_zg *= kl_weight
        KL_zg = tfutil.autosummary('Loss/KL_zg', KL_zg)
        loss = loss_addup(loss, KL_zg)
        KL_zl = -0.5 * tf.reduce_mean(1+2*enc_zl_log_sigma-enc_zl_mu**2-tf.exp(2*enc_zl_log_sigma), axis=[1,2,3])
        KL_zl *= kl_weight
        KL_zl = tfutil.autosummary('Loss/KL_zl', KL_zl)
        loss = loss_addup(loss, KL_zl)

    # interpolated realism and global/local gram matrix losses
    if interp_G_weight > 0.0 or blend_interp_G_weight > 0.0:
        if zg_interp_variational == 'hard':
            interp_enc_zg_latents = tf.tile(enc_zg_latents, [1, 1, E_zl.output_shapes[0][2]*scale_h, E_zl.output_shapes[0][3]*scale_w])
        elif zg_interp_variational == 'variational':
            interp_enc_zg_latents = tf.random_normal([minibatch_size] + E_zg.output_shapes[0][1:])
            interp_enc_zg_latents = interp_enc_zg_latents * tf.exp(enc_zg_log_sigma) + enc_zg_mu
            interp_enc_zg_latents = tf.tile(interp_enc_zg_latents, [1, 1, E_zl.output_shapes[0][2]*scale_h, E_zl.output_shapes[0][3]*scale_w])
        if zl_interp_variational == 'hard':
            interp_enc_zl_latents = tf.tile(enc_zl_latents, [1, 1, scale_h, scale_w])
        elif zl_interp_variational == 'variational':
            interp_enc_zl_mu = tf.tile(enc_zl_mu, [1, 1, scale_h, scale_w])
            interp_enc_zl_log_sigma = tf.tile(enc_zl_log_sigma, [1, 1, scale_h, scale_w])
            interp_enc_zl_latents = tf.random_normal([minibatch_size] + G_fcn.input_shapes[1][1:])
            interp_enc_zl_latents = interp_enc_zl_latents * tf.exp(interp_enc_zl_log_sigma) + interp_enc_zl_mu
        elif zl_interp_variational == 'random':
            interp_enc_zl_latents_1 = tf.concat([enc_zl_latents, tf.random_normal([minibatch_size, G_fcn.input_shapes[1][1], E_zl.output_shapes[0][2], G_fcn.input_shapes[1][3]-2*E_zl.output_shapes[0][3]]), enc_zl_latents], axis=3)
            interp_enc_zl_latents_2 = tf.random_normal([minibatch_size, G_fcn.input_shapes[1][1], G_fcn.input_shapes[1][2]-2*E_zl.output_shapes[0][2], G_fcn.input_shapes[1][3]])
            interp_enc_zl_latents_3 = tf.concat([enc_zl_latents, tf.random_normal([minibatch_size, G_fcn.input_shapes[1][1], E_zl.output_shapes[0][2], G_fcn.input_shapes[1][3]-2*E_zl.output_shapes[0][3]]), enc_zl_latents], axis=3)
            interp_enc_zl_latents = tf.concat([interp_enc_zl_latents_1, interp_enc_zl_latents_2, interp_enc_zl_latents_3], axis=2)
        elif zl_interp_variational == 'permutational':
            interp_enc_zl_latents = tiling_permutation(enc_zl_latents, scale_h, scale_w, permutation_matrix_h_forward, permutation_matrix_w_forward)

        if interp_G_weight > 0.0:
            interp_images_out = G_fcn.get_output_for(interp_enc_zg_latents, interp_enc_zl_latents)
            crop_interp_images_out = random_crop(interp_images_out, minibatch_size, G_fcn.output_shape, D_interp.input_shape)
            # interpolated realism
            crop_interp_scores_out = fp32(D_interp.get_output_for(crop_interp_images_out))
            crop_interp_G_loss = tf.reduce_mean(-crop_interp_scores_out, axis=[1,2,3])
            crop_interp_G_loss *= interp_G_weight
            crop_interp_G_loss = tfutil.autosummary('Loss/crop_interp_G_loss', crop_interp_G_loss)
            loss = loss_addup(loss, crop_interp_G_loss)
            # interpolated local gram matrix loss
            if gram_weight > 0.0:
                crop_interp_vgg = custom_Vgg19(crop_interp_images_out, data_dict=data_dict)
                crop_interp_feature = [crop_interp_vgg.conv1_1, crop_interp_vgg.conv2_1, crop_interp_vgg.conv3_1, crop_interp_vgg.conv4_1, crop_interp_vgg.conv5_1]
                crop_interp_gram = [gram_matrix(l, data_format='NHWC') for l in crop_interp_feature]
                crop_interp_gram_loss = multi_layer_diff(crop_interp_gram, real_gram, dtype=crop_interp_images_out.dtype)
                crop_interp_gram_loss *= gram_weight
                crop_interp_gram_loss = tfutil.autosummary('Loss/crop_interp_gram_loss', crop_interp_gram_loss)
                loss = loss_addup(loss, crop_interp_gram_loss)

        # multi-texture interpolated realism
        if blend_interp_G_weight > 0.0:
            if zg_interp_variational == 'hard':
                interp_enc_zg_latents_reverse = tf.tile(tf.reverse(enc_zg_latents, axis=[0]), [1, 1, E_zl.output_shapes[0][2]*scale_h, E_zl.output_shapes[0][3]*scale_w])
            elif zg_interp_variational == 'variational':
                interp_enc_zg_latents_reverse = tf.random_normal([minibatch_size] + E_zg.output_shapes[0][1:])
                interp_enc_zg_latents_reverse = interp_enc_zg_latents_reverse * tf.exp(tf.reverse(enc_zg_log_sigma, axis=[0])) + tf.reverse(enc_zg_mu, axis=[0])
                interp_enc_zg_latents_reverse = tf.tile(interp_enc_zg_latents_reverse, [1, 1, E_zl.output_shapes[0][2]*scale_h, E_zl.output_shapes[0][3]*scale_w])
            if zl_interp_variational == 'hard':
                interp_enc_zl_latents_reverse = tf.tile(tf.reverse(enc_zl_latents, axis=[0]), [1, 1, scale_h, scale_w])
            elif zl_interp_variational == 'variational':
                interp_enc_zl_mu_reverse = tf.tile(tf.reverse(enc_zl_mu, axis=[0]), [1, 1, scale_h, scale_w])
                interp_enc_zl_log_sigma_reverse = tf.tile(tf.reverse(enc_zl_log_sigma, axis=[0]), [1, 1, scale_h, scale_w])
                interp_enc_zl_latents_reverse = tf.random_normal([minibatch_size] + G_fcn.input_shapes[1][1:])
                interp_enc_zl_latents_reverse = interp_enc_zl_latents_reverse * tf.exp(interp_enc_zl_log_sigma_reverse) + interp_enc_zl_mu_reverse
            elif zl_interp_variational == 'random':
                interp_enc_zl_latents_1_reverse = tf.concat([tf.reverse(enc_zl_latents, axis=[0]), tf.random_normal([minibatch_size, G_fcn.input_shapes[1][1], E_zl.output_shapes[0][2], G_fcn.input_shapes[1][3]-2*E_zl.output_shapes[0][3]]), tf.reverse(enc_zl_latents, axis=[0])], axis=3)
                interp_enc_zl_latents_2_reverse = tf.random_normal([minibatch_size, G_fcn.input_shapes[1][1], G_fcn.input_shapes[1][2]-2*E_zl.output_shapes[0][2], G_fcn.input_shapes[1][3]])
                interp_enc_zl_latents_3_reverse = tf.concat([tf.reverse(enc_zl_latents, axis=[0]), tf.random_normal([minibatch_size, G_fcn.input_shapes[1][1], E_zl.output_shapes[0][2], G_fcn.input_shapes[1][3]-2*E_zl.output_shapes[0][3]]), tf.reverse(enc_zl_latents, axis=[0])], axis=3)
                interp_enc_zl_latents_reverse = tf.concat([interp_enc_zl_latents_1_reverse, interp_enc_zl_latents_2_reverse, interp_enc_zl_latents_3_reverse], axis=2)
            elif zl_interp_variational == 'permutational':
                interp_enc_zl_latents_reverse = tiling_permutation(tf.reverse(enc_zl_latents, axis=[0]), scale_h, scale_w, permutation_matrix_h_backward, permutation_matrix_w_backward)
            mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=enc_zg_latents.dtype)
            blend_interp_enc_zg_latents = tfutil.lerp(interp_enc_zg_latents_reverse, interp_enc_zg_latents, mixing_factors)
            blend_interp_enc_zl_latents = tfutil.lerp(interp_enc_zl_latents_reverse, interp_enc_zl_latents, mixing_factors)
            blend_interp_images_out = G_fcn.get_output_for(blend_interp_enc_zg_latents, blend_interp_enc_zl_latents)
            crop_blend_interp_images_out = random_crop(blend_interp_images_out, minibatch_size, G_fcn.output_shape, D_blend.input_shape)
            crop_blend_interp_scores_out = fp32(D_blend.get_output_for(crop_blend_interp_images_out))
            crop_blend_interp_G_loss = tf.reduce_mean(-crop_blend_interp_scores_out, axis=[1,2,3])
            crop_blend_interp_G_loss *= blend_interp_G_weight
            crop_blend_interp_G_loss = tfutil.autosummary('Loss/crop_blend_interp_G_loss', crop_blend_interp_G_loss)
            loss = loss_addup(loss, crop_blend_interp_G_loss)
            # multi-texture interpolated local gram matrix loss
            if gram_weight > 0.0:
                crop_blend_interp_vgg = custom_Vgg19(crop_blend_interp_images_out, data_dict=data_dict)
                crop_blend_interp_feature = [crop_blend_interp_vgg.conv1_1, crop_blend_interp_vgg.conv2_1, crop_blend_interp_vgg.conv3_1, crop_blend_interp_vgg.conv4_1, crop_blend_interp_vgg.conv5_1]
                crop_blend_interp_gram = [gram_matrix(l, data_format='NHWC') for l in crop_blend_interp_feature]
                real_gram_2 = [tf.reverse(mat, axis=[0]) for mat in real_gram]
                alpha = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=interp_enc_zg_latents.dtype)
                crop_blend_interp_gram_loss = (1.0 - alpha) * multi_layer_diff(crop_blend_interp_gram, real_gram_2, dtype=crop_blend_interp_images_out.dtype) + alpha * multi_layer_diff(crop_blend_interp_gram, real_gram, dtype=crop_blend_interp_images_out.dtype)
                crop_blend_interp_gram_loss *= gram_weight
                crop_blend_interp_gram_loss = tfutil.autosummary('Loss/crop_blend_interp_gram_loss', crop_blend_interp_gram_loss)
                loss = loss_addup(loss, crop_blend_interp_gram_loss)

    return loss

#----------------------------------------------------------------------------
# Random-generation-associated discriminator loss function used in the paper (WGAN-GP).

def D_gen_wgangp(E_zg, E_zl, G, D_gen, D_gen_opt, minibatch_size, reals_fade,
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0):     # Target value for gradient magnitudes.

    # random generated realism
    zg_latents = tf.random_normal([minibatch_size] + E_zg.output_shapes[0][1:])
    zl_latents = tf.random_normal([minibatch_size] + E_zl.output_shapes[0][1:])
    fake_images_out = G.get_output_for(tf.tile(zg_latents, [1,1]+E_zl.output_shapes[0][2:]), zl_latents)
    fake_scores_out = fp32(D_gen.get_output_for(fake_images_out))
    real_scores_out = fp32(D_gen.get_output_for(reals_fade))
    gen_D_loss = tf.reduce_mean(fake_scores_out - real_scores_out, axis=[1,2,3])
    gen_D_loss = tfutil.autosummary('Loss/gen_D_loss', gen_D_loss)
    loss = tf.identity(gen_D_loss)

    # gradient penalty
    with tf.name_scope('gen_GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tfutil.lerp(tf.cast(reals_fade, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out = fp32(D_gen.get_output_for(mixed_images_out))
        mixed_loss = D_gen_opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = D_gen_opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        gen_gradient_penalty = tf.square(mixed_norms - wgan_target)
        gen_gradient_penalty *= (wgan_lambda / (wgan_target**2))
        gen_gradient_penalty = tfutil.autosummary('Loss/gen_gradient_penalty', gen_gradient_penalty)
    loss += gen_gradient_penalty

    # calibration penalty
    with tf.name_scope('gen_EpsilonPenalty'):
        gen_epsilon_penalty = tf.reduce_mean(tf.square(real_scores_out), axis=[1,2,3]) * wgan_epsilon
        gen_epsilon_penalty = tfutil.autosummary('Loss/gen_epsilon_penalty', gen_epsilon_penalty)
    loss += gen_epsilon_penalty

    return loss

#----------------------------------------------------------------------------
# Reconstruction-associated discriminator loss function used in the paper (WGAN-GP).

def D_rec_wgangp(E_zg, E_zl, G, D_rec, D_rec_opt, minibatch_size, reals_fade, reals_orig,
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0):     # Target value for gradient magnitudes.

    # zg encoding 
    enc_zg_mu, enc_zg_log_sigma = E_zg.get_output_for(reals_orig)
    if config.zg_enabled:
        enc_zg_latents = tf.identity(enc_zg_mu)
    else:
        enc_zg_latents = tf.zeros(tf.shape(enc_zg_mu))

    # zl encoding 
    enc_zl_mu, enc_zl_log_sigma = E_zl.get_output_for(reals_orig)
    enc_zl_latents = tf.identity(enc_zl_mu)

    # reconstructed realism
    rec_images_out = G.get_output_for(tf.tile(enc_zg_latents, [1,1]+E_zl.output_shapes[0][2:]), enc_zl_latents)
    rec_scores_out = fp32(D_rec.get_output_for(rec_images_out))
    real_scores_out = fp32(D_rec.get_output_for(reals_fade))
    rec_D_loss = tf.reduce_mean(rec_scores_out - real_scores_out, axis=[1,2,3])
    rec_D_loss = tfutil.autosummary('Loss/rec_D_loss', rec_D_loss)
    loss = tf.identity(rec_D_loss)

    # gradient penalty
    with tf.name_scope('rec_GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=rec_images_out.dtype)
        mixed_images_out = tfutil.lerp(tf.cast(reals_fade, rec_images_out.dtype), rec_images_out, mixing_factors)
        mixed_scores_out = fp32(D_rec.get_output_for(mixed_images_out))
        mixed_loss = D_rec_opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = D_rec_opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        rec_gradient_penalty = tf.square(mixed_norms - wgan_target)
        rec_gradient_penalty *= (wgan_lambda / (wgan_target**2))
        rec_gradient_penalty = tfutil.autosummary('Loss/rec_gradient_penalty', rec_gradient_penalty)
    loss += rec_gradient_penalty

    # calibration penalty
    with tf.name_scope('rec_EpsilonPenalty'):
        rec_epsilon_penalty = tf.reduce_mean(tf.square(real_scores_out), axis=[1,2,3]) * wgan_epsilon
        rec_epsilon_penalty = tfutil.autosummary('Loss/rec_epsilon_penalty', rec_epsilon_penalty)
    loss += rec_epsilon_penalty

    return loss

#----------------------------------------------------------------------------
# Interpolation-associated discriminator loss function used in the paper (WGAN-GP).

def D_interp_wgangp(E_zg, E_zl, G_fcn, D_interp, D_interp_opt, minibatch_size, reals_fade, reals_orig, permutation_matrix_h_forward, permutation_matrix_w_forward,
    scale_h,                    # Height scale of interpolated size
    scale_w,                    # Width scale of interpolated size
    zg_interp_variational,      # Enable hard or variational or learned or random zg interpolation?
    zl_interp_variational,      # Enable hard or variational or learned or random zl interpolation?
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0):     # Target value for gradient magnitudes.

    # zg encoding 
    enc_zg_mu, enc_zg_log_sigma = E_zg.get_output_for(reals_orig)
    if config.zg_enabled:
        enc_zg_latents = tf.identity(enc_zg_mu)
    else:
        enc_zg_latents = tf.zeros(tf.shape(enc_zg_mu))

    # zl encoding 
    enc_zl_mu, enc_zl_log_sigma = E_zl.get_output_for(reals_orig)
    enc_zl_latents = tf.identity(enc_zl_mu)

    # interpolating in latent space
    if zg_interp_variational == 'hard':
        interp_enc_zg_latents = tf.tile(enc_zg_latents, [1, 1, E_zl.output_shapes[0][2]*scale_h, E_zl.output_shapes[0][3]*scale_w])
    elif zg_interp_variational == 'variational':
        interp_enc_zg_latents = tf.random_normal([minibatch_size] + E_zg.output_shapes[0][1:])
        interp_enc_zg_latents = interp_enc_zg_latents * tf.exp(enc_zg_log_sigma) + enc_zg_mu
        interp_enc_zg_latents = tf.tile(interp_enc_zg_latents, [1, 1, E_zl.output_shapes[0][2]*scale_h, E_zl.output_shapes[0][3]*scale_w])
    if zl_interp_variational == 'hard':
        interp_enc_zl_latents = tf.tile(enc_zl_latents, [1, 1, scale_h, scale_w])
    elif zl_interp_variational == 'variational':
        interp_enc_zl_mu = tf.tile(enc_zl_mu, [1, 1, scale_h, scale_w])
        interp_enc_zl_log_sigma = tf.tile(enc_zl_log_sigma, [1, 1, scale_h, scale_w])
        interp_enc_zl_latents = tf.random_normal([minibatch_size] + G_fcn.input_shapes[1][1:])
        interp_enc_zl_latents = interp_enc_zl_latents * tf.exp(interp_enc_zl_log_sigma) + interp_enc_zl_mu
    elif zl_interp_variational == 'random':
        interp_enc_zl_latents_1 = tf.concat([enc_zl_latents, tf.random_normal([minibatch_size, G_fcn.input_shapes[1][1], E_zl.output_shapes[0][2], G_fcn.input_shapes[1][3]-2*E_zl.output_shapes[0][3]]), enc_zl_latents], axis=3)
        interp_enc_zl_latents_2 = tf.random_normal([minibatch_size, G_fcn.input_shapes[1][1], G_fcn.input_shapes[1][2]-2*E_zl.output_shapes[0][2], G_fcn.input_shapes[1][3]])
        interp_enc_zl_latents_3 = tf.concat([enc_zl_latents, tf.random_normal([minibatch_size, G_fcn.input_shapes[1][1], E_zl.output_shapes[0][2], G_fcn.input_shapes[1][3]-2*E_zl.output_shapes[0][3]]), enc_zl_latents], axis=3)
        interp_enc_zl_latents = tf.concat([interp_enc_zl_latents_1, interp_enc_zl_latents_2, interp_enc_zl_latents_3], axis=2)
    elif zl_interp_variational == 'permutational':
        interp_enc_zl_latents = tiling_permutation(enc_zl_latents, scale_h, scale_w, permutation_matrix_h_forward, permutation_matrix_w_forward)

    # generating and cropping
    interp_images_out = G_fcn.get_output_for(interp_enc_zg_latents, interp_enc_zl_latents)
    crop_interp_images_out = random_crop(interp_images_out, minibatch_size, G_fcn.output_shape, D_interp.input_shape)

    # interpolated realism
    crop_interp_scores_out = fp32(D_interp.get_output_for(crop_interp_images_out))
    real_scores_out = fp32(D_interp.get_output_for(reals_fade))
    crop_interp_D_loss = tf.reduce_mean(crop_interp_scores_out - real_scores_out, axis=[1,2,3])
    crop_interp_D_loss = tfutil.autosummary('Loss/crop_interp_D_loss', crop_interp_D_loss)
    loss = tf.identity(crop_interp_D_loss)

    with tf.name_scope('interp_GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=crop_interp_images_out.dtype)
        mixed_images_out = tfutil.lerp(tf.cast(reals_fade, crop_interp_images_out.dtype), crop_interp_images_out, mixing_factors)
        mixed_scores_out = fp32(D_interp.get_output_for(mixed_images_out))
        mixed_loss = D_interp_opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = D_interp_opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        crop_interp_gradient_penalty = tf.square(mixed_norms - wgan_target)
        crop_interp_gradient_penalty *= (wgan_lambda / (wgan_target**2))
        crop_interp_gradient_penalty = tfutil.autosummary('Loss/crop_interp_gradient_penalty', crop_interp_gradient_penalty)
    loss += crop_interp_gradient_penalty

    with tf.name_scope('interp_EpsilonPenalty'):
        crop_interp_epsilon_penalty = tf.reduce_mean(tf.square(real_scores_out), axis=[1,2,3]) * wgan_epsilon
        crop_interp_epsilon_penalty = tfutil.autosummary('Loss/crop_interp_epsilon_penalty', crop_interp_epsilon_penalty)
    loss += crop_interp_epsilon_penalty

    return loss

#----------------------------------------------------------------------------
# Multi-texture interpolation-and-blending-associated discriminator loss function used in the paper (WGAN-GP).

def D_blend_wgangp(E_zg, E_zl, G_fcn, D_blend, D_blend_opt, minibatch_size, reals_fade, reals_orig, permutation_matrix_h_forward, permutation_matrix_w_forward, permutation_matrix_h_backward, permutation_matrix_w_backward,
    scale_h,                    # Height scale of interpolated size
    scale_w,                    # Width scale of interpolated size
    zg_interp_variational,      # Enable hard or variational or learned or random zg interpolation?
    zl_interp_variational,      # Enable hard or variational or learned or random zl interpolation?
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0):     # Target value for gradient magnitudes.

    # zg encoding 
    enc_zg_mu, enc_zg_log_sigma = E_zg.get_output_for(reals_orig)
    if config.zg_enabled:
        enc_zg_latents = tf.identity(enc_zg_mu)
    else:
        enc_zg_latents = tf.zeros(tf.shape(enc_zg_mu))

    # zl encoding 
    enc_zl_mu, enc_zl_log_sigma = E_zl.get_output_for(reals_orig)
    enc_zl_latents = tf.identity(enc_zl_mu)

    # interpolating in latent space
    if zg_interp_variational == 'hard':
        interp_enc_zg_latents = tf.tile(enc_zg_latents, [1, 1, E_zl.output_shapes[0][2]*scale_h, E_zl.output_shapes[0][3]*scale_w])
    elif zg_interp_variational == 'variational':
        interp_enc_zg_latents = tf.random_normal([minibatch_size] + E_zg.output_shapes[0][1:])
        interp_enc_zg_latents = interp_enc_zg_latents * tf.exp(enc_zg_log_sigma) + enc_zg_mu
        interp_enc_zg_latents = tf.tile(interp_enc_zg_latents, [1, 1, E_zl.output_shapes[0][2]*scale_h, E_zl.output_shapes[0][3]*scale_w])
    if zl_interp_variational == 'hard':
        interp_enc_zl_latents = tf.tile(enc_zl_latents, [1, 1, scale_h, scale_w])
    elif zl_interp_variational == 'variational':
        interp_enc_zl_mu = tf.tile(enc_zl_mu, [1, 1, scale_h, scale_w])
        interp_enc_zl_log_sigma = tf.tile(enc_zl_log_sigma, [1, 1, scale_h, scale_w])
        interp_enc_zl_latents = tf.random_normal([minibatch_size] + G_fcn.input_shape[1:])
        interp_enc_zl_latents = interp_enc_zl_latents * tf.exp(interp_enc_zl_log_sigma) + interp_enc_zl_mu
    elif zl_interp_variational == 'random':
        interp_enc_zl_latents_1 = tf.concat([enc_zl_latents, tf.random_normal([minibatch_size, G_fcn.input_shapes[0][1], E_zl.output_shapes[0][2], G_fcn.input_shapes[0][3]-2*E_zl.output_shapes[0][3]]), enc_zl_latents], axis=3)
        interp_enc_zl_latents_2 = tf.random_normal([minibatch_size, G_fcn.input_shapes[0][1], G_fcn.input_shapes[0][2]-2*E_zl.output_shapes[0][2], G_fcn.input_shapes[0][3]])
        interp_enc_zl_latents_3 = tf.concat([enc_zl_latents, tf.random_normal([minibatch_size, G_fcn.input_shapes[0][1], E_zl.output_shapes[0][2], G_fcn.input_shapes[0][3]-2*E_zl.output_shapes[0][3]]), enc_zl_latents], axis=3)
        interp_enc_zl_latents = tf.concat([interp_enc_zl_latents_1, interp_enc_zl_latents_2, interp_enc_zl_latents_3], axis=2)
    elif zl_interp_variational == 'permutational':
        interp_enc_zl_latents = tiling_permutation(enc_zl_latents, scale_h, scale_w, permutation_matrix_h_forward, permutation_matrix_w_forward)

    # interpolating in latent space in reverse order
    if zg_interp_variational == 'hard':
        interp_enc_zg_latents_reverse = tf.tile(tf.reverse(enc_zg_latents, axis=[0]), [1, 1, E_zl.output_shapes[0][2]*scale_h, E_zl.output_shapes[0][3]*scale_w])
    elif zg_interp_variational == 'variational':
        interp_enc_zg_latents_reverse = tf.random_normal([minibatch_size] + E_zg.output_shapes[0][1:])
        interp_enc_zg_latents_reverse = interp_enc_zg_latents_reverse * tf.exp(tf.reverse(enc_zg_log_sigma, axis=[0])) + tf.reverse(enc_zg_mu, axis=[0])
        interp_enc_zg_latents_reverse = tf.tile(interp_enc_zg_latents_reverse, [1, 1, E_zl.output_shapes[0][2]*scale_h, E_zl.output_shapes[0][3]*scale_w])
    if zl_interp_variational == 'hard':
        interp_enc_zl_latents_reverse = tf.tile(tf.reverse(enc_zl_latents, axis=[0]), [1, 1, scale_h, scale_w])
    elif zl_interp_variational == 'variational':
        interp_enc_zl_mu_reverse = tf.tile(tf.reverse(enc_zl_mu, axis=[0]), [1, 1, scale_h, scale_w])
        interp_enc_zl_log_sigma_reverse = tf.tile(tf.reverse(enc_zl_log_sigma, axis=[0]), [1, 1, scale_h, scale_w])
        interp_enc_zl_latents_reverse = tf.random_normal([minibatch_size] + G_fcn.input_shapes[1][1:])
        interp_enc_zl_latents_reverse = interp_enc_zl_latents_reverse * tf.exp(interp_enc_zl_log_sigma_reverse) + interp_enc_zl_mu_reverse
    elif zl_interp_variational == 'random':
        interp_enc_zl_latents_1_reverse = tf.concat([tf.reverse(enc_zl_latents, axis=[0]), tf.random_normal([minibatch_size, G_fcn.input_shapes[1][1], E_zl.output_shapes[0][2], G_fcn.input_shapes[1][3]-2*E_zl.output_shapes[0][3]]), tf.reverse(enc_zl_latents, axis=[0])], axis=3)
        interp_enc_zl_latents_2_reverse = tf.random_normal([minibatch_size, G_fcn.input_shapes[1][1], G_fcn.input_shapes[1][2]-2*E_zl.output_shapes[0][2], G_fcn.input_shapes[1][3]])
        interp_enc_zl_latents_3_reverse = tf.concat([tf.reverse(enc_zl_latents, axis=[0]), tf.random_normal([minibatch_size, G_fcn.input_shapes[1][1], E_zl.output_shapes[0][2], G_fcn.input_shapes[1][3]-2*E_zl.output_shapes[0][3]]), tf.reverse(enc_zl_latents, axis=[0])], axis=3)
        interp_enc_zl_latents_reverse = tf.concat([interp_enc_zl_latents_1_reverse, interp_enc_zl_latents_2_reverse, interp_enc_zl_latents_3_reverse], axis=2)
    elif zl_interp_variational == 'permutational':
        interp_enc_zl_latents_reverse = tiling_permutation(tf.reverse(enc_zl_latents, axis=[0]), scale_h, scale_w, permutation_matrix_h_backward, permutation_matrix_w_backward)
    mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=enc_zg_latents.dtype)
    blend_interp_enc_zg_latents = tfutil.lerp(interp_enc_zg_latents_reverse, interp_enc_zg_latents, mixing_factors)
    blend_interp_enc_zl_latents = tfutil.lerp(interp_enc_zl_latents_reverse, interp_enc_zl_latents, mixing_factors)

    # generating and cropping
    blend_interp_images_out = G_fcn.get_output_for(blend_interp_enc_zg_latents, blend_interp_enc_zl_latents)
    crop_blend_interp_images_out = random_crop(blend_interp_images_out, minibatch_size, G_fcn.output_shape, D_blend.input_shape)

    # interpolated realism
    crop_blend_interp_scores_out = fp32(D_blend.get_output_for(crop_blend_interp_images_out))
    real_scores_out = fp32(D_blend.get_output_for(reals_fade))
    crop_blend_interp_D_loss = tf.reduce_mean(crop_blend_interp_scores_out - real_scores_out, axis=[1,2,3])
    crop_blend_interp_D_loss = tfutil.autosummary('Loss/crop_blend_interp_D_loss', crop_blend_interp_D_loss)
    loss = tf.identity(crop_blend_interp_D_loss)

    with tf.name_scope('blend_GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=crop_blend_interp_images_out.dtype)
        mixed_images_out = tfutil.lerp(tf.cast(reals_fade, crop_blend_interp_images_out.dtype), crop_blend_interp_images_out, mixing_factors)
        mixed_scores_out = fp32(D_blend.get_output_for(mixed_images_out))
        mixed_loss = D_blend_opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = D_blend_opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        crop_blend_interp_gradient_penalty = tf.square(mixed_norms - wgan_target)
        crop_blend_interp_gradient_penalty *= (wgan_lambda / (wgan_target**2))
        crop_blend_interp_gradient_penalty = tfutil.autosummary('Loss/crop_blend_interp_gradient_penalty', crop_blend_interp_gradient_penalty)
    loss += crop_blend_interp_gradient_penalty

    with tf.name_scope('blend_EpsilonPenalty'):
        crop_blend_interp_epsilon_penalty = tf.reduce_mean(tf.square(real_scores_out), axis=[1,2,3]) * wgan_epsilon
        crop_blend_interp_epsilon_penalty = tfutil.autosummary('Loss/crop_blend_interp_epsilon_penalty', crop_blend_interp_epsilon_penalty)
    loss += crop_blend_interp_epsilon_penalty

    return loss