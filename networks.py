# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import tensorflow as tf

import config

from custom_vgg19 import *

# NOTE: Do not import any application-specific modules here!

#----------------------------------------------------------------------------

def lerp(a, b, t): return a + (b - a) * t
def lerp_clip(a, b, t): return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)
def cset(cur_lambda, new_cond, new_lambda): return lambda: tf.cond(new_cond, new_lambda, cur_lambda)

#----------------------------------------------------------------------------
# Get/create weight tensor for a convolutional or fully-connected layer.

def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in) # He init
    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
    else:
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))

#----------------------------------------------------------------------------
# Fully-connected layer.

def dense(x, fmaps, gain=np.sqrt(2), use_wscale=False):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)

#----------------------------------------------------------------------------
# Convolutional layer.

def conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    if kernel == 1:
        return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='VALID', data_format='NCHW')
    else:
        x = tf.pad(x, paddings=[[0, 0],[0, 0],[kernel//2, kernel//2],[kernel//2, kernel//2]], mode='REFLECT')
        return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='VALID', data_format='NCHW')

#----------------------------------------------------------------------------
# Apply bias to the given activation tensor.

def apply_bias(x):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros())
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        return x + tf.reshape(b, [1, -1, 1, 1])

#----------------------------------------------------------------------------
# Leaky ReLU activation. Same as tf.nn.leaky_relu, but supports FP16.

def leaky_relu(x, alpha=0.2):
    with tf.name_scope('LeakyRelu'):
        alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
        return tf.maximum(x * alpha, x)

#----------------------------------------------------------------------------
# Nearest-neighbor upscaling layer.

def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        return x

#----------------------------------------------------------------------------
# Fused upscale2d + conv2d.
# Faster and uses less memory than performing the operations separately.

def upscale2d_conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, fmaps, x.shape[1].value], gain=gain, use_wscale=use_wscale, fan_in=(kernel**2)*x.shape[1].value)
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
    w = tf.cast(w, x.dtype)
    os = [tf.shape(x)[0], fmaps, x.shape[2] * 2, x.shape[3] * 2]
    return tf.nn.conv2d_transpose(x, w, os, strides=[1,1,2,2], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# upscale2d for RGB image by upsampling + Gaussian smoothing

gaussian_filter_up = tf.constant(list(np.float32([1,4,6,4,1,4,16,24,16,4,6,24,36,24,6,4,16,24,16,4,1,4,6,4,1])/256.0*4.0), dtype=tf.float32, shape=[5,5,1,1], name='GaussianFilterUp', verify_shape=False)

def upscale2d_rgb_Gaussian(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D_RGB_Gaussian'):
        for i in range(int(round(np.log2(factor)))):
            try:
                s = x.shape
                x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
            except:
                s = tf.shape(x)
                x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
            x = tf.pad(x, paddings=[[0,0],[0,0],[0,0],[0,1],[0,0],[0,1]], mode='CONSTANT')
            x = tf.reshape(x, [-1, s[1], s[2]*2, s[3]*2])
            channel_list = []
            for j in range(3):
                z = tf.pad(x[:,j:j+1,:,:], paddings=[[0,0],[0,0],[2,2],[2,2]], mode='REFLECT')
                channel_list.append(tf.nn.conv2d(z, filter=gaussian_filter_up, strides=[1,1,1,1], padding='VALID', data_format='NCHW', name='GaussianConvUp'))
            x = tf.concat(channel_list, axis=1)
        return x

#----------------------------------------------------------------------------
# Box filter downscaling layer.

def downscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Downscale2D'):
        ksize = [1, 1, factor, factor]
        return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW') # NOTE: requires tf_config['graph_options.place_pruned_graph'] = True

#----------------------------------------------------------------------------
# Fused conv2d + downscale2d.
# Faster and uses less memory than performing the operations separately.

def conv2d_downscale2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1,1,2,2], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# downscale2d for RGB image by Gaussian smoothing + downsampling

gaussian_filter_down = tf.constant(list(np.float32([1,4,6,4,1,4,16,24,16,4,6,24,36,24,6,4,16,24,16,4,1,4,6,4,1])/256.0), dtype=tf.float32, shape=[5,5,1,1], name='GaussianFilterDown', verify_shape=False)

def downscale2d_rgb_Gaussian(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Downscale2D_RGB_Gaussian'):
        for i in range(int(round(np.log2(factor)))):
            channel_list = []
            for j in range(3):
                z = tf.pad(x[:,j:j+1,:,:], paddings=[[0,0],[0,0],[2,2],[2,2]], mode='REFLECT')
                channel_list.append(tf.nn.conv2d(z, filter=gaussian_filter_down, strides=[1,1,2,2], padding='VALID', data_format='NCHW', name='GaussianConvDown'))
            x = tf.concat(channel_list, axis=1)
        return x

#----------------------------------------------------------------------------
# Pixelwise feature vector normalization.

def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)

#----------------------------------------------------------------------------
# Minibatch standard deviation.

def minibatch_stddev_layer(x, group_size=4):
    with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape                                             # [NCHW]  Input shape.
        y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])   # [GMCHW] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)                              # [GMCHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMCHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MCHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MCHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
        y = tf.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
        y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [N1HW]  Replicate over group and pixels.
        return tf.concat([x, y], axis=1)                        # [NCHW]  Append as new fmap.

#----------------------------------------------------------------------------
# zg encoder network semi-mimicing the Discriminator network in the paper.

def E_zg(
    images_in,                          # Input: Images [minibatch, channel, height, width].
    num_channels        = 3,            # Number of input color channels. Overridden based on dataset.
    resolution          = 128,          # Input resolution. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    latent_channels     = 4,            # Dimensionality of the latent vectors. None = min(fmap_base, fmap_max).
    use_wscale          = True,         # Enable equalized learning rate?
    use_pixelnorm       = False,        # Enable pixelwise feature vector normalization?
    pixelnorm_epsilon   = 1e-8,         # Constant epsilon for pixelwise feature vector normalization.
    use_leakyrelu       = True,         # True = leaky ReLU, False = ReLU.
    tanh_at_end         = False,        # Use tanh activation for the last layer?
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = False,        # True = use fused conv2d + downscale2d, False = separate downscale2d layers.
    structure           = 'recursive',  # 'linear' = human-readable, 'recursive' = efficient, None = select automatically
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    **kwargs):                          # Ignore unrecognized keyword args.
    
    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def PN(x): return pixel_norm(x, epsilon=pixelnorm_epsilon) if use_pixelnorm else x
    if latent_channels is None: latent_channels = nf(0)
    if structure is None: structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu if use_leakyrelu else tf.nn.relu

    images_in.set_shape([None, num_channels, resolution, resolution])
    images_in = tf.cast(images_in, dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)

    # Building blocks.
    def fromrgb(x, res): # res = 2..resolution_log2
        with tf.variable_scope('FromRGB_lod%d' % (resolution_log2 - res)):
            return act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=1, use_wscale=use_wscale)))
    def block(x, res): # res = 2..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res >= 3: # 8x8 and up
                with tf.variable_scope('Conv0'):
                    x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
                if fused_scale:
                    with tf.variable_scope('Conv1_down'):
                        x = PN(act(apply_bias(conv2d_downscale2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale))))
                else:
                    with tf.variable_scope('Conv1'):
                        x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale))))
                    x = downscale2d(x)
                return x
            else: # 4x4
                with tf.variable_scope('Conv0'):
                    x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
                if fused_scale:
                    with tf.variable_scope('zg_Conv1_down'):
                        x = PN(act(apply_bias(conv2d_downscale2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale))))
                    with tf.variable_scope('zg_Conv2_down'):
                        x = apply_bias(conv2d_downscale2d(x, fmaps=latent_channels*2, kernel=3, gain=1, use_wscale=use_wscale))
                else:
                    with tf.variable_scope('zg_Conv1'):
                        x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale))))
                    x = downscale2d(x)
                    with tf.variable_scope('zg_Conv2'):
                        x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-3), kernel=3, use_wscale=use_wscale))))
                    x = downscale2d(x)
                    with tf.variable_scope('zg_Conv3'):
                        x = apply_bias(conv2d(x, fmaps=latent_channels*2, kernel=1, gain=1, use_wscale=use_wscale))
                return x
    
    # Linear structure: simple but inefficient.
    if structure == 'linear':
        img = images_in
        x = fromrgb(img, resolution_log2)
        for res in range(resolution_log2, 2, -1):
            lod = resolution_log2 - res
            x = block(x, res)
            img = downscale2d(img)
            y = fromrgb(img, res - 1)
            with tf.variable_scope('Grow_lod%d' % lod):
                x = lerp_clip(x, y, lod_in - lod)
        zg_latents_out = block(x, 2)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def grow(res, lod):
            x = lambda: fromrgb(downscale2d(images_in, 2**lod), res)
            if lod > 0: x = cset(x, (lod_in < lod), lambda: grow(res + 1, lod - 1))
            x = block(x(), res); y = lambda: x
            if res > 2: y = cset(y, (lod_in > lod), lambda: lerp(x, fromrgb(downscale2d(images_in, 2**(lod+1)), res - 1), lod_in - lod))
            return y()
        zg_latents_out = grow(2, resolution_log2 - 2)

    assert zg_latents_out.dtype == tf.as_dtype(dtype)
    if tanh_at_end:
        zg_latents_out = tf.nn.tanh(zg_latents_out, name='zg_latents_out')
    else:
        zg_latents_out = tf.identity(zg_latents_out, name='zg_latents_out')
    zg_mu = tf.identity(zg_latents_out[:, :latent_channels, :, :], name='zg_mu')
    zg_log_sigma = tf.identity(zg_latents_out[:, latent_channels:, :, :], name='zg_log_sigma')
    return zg_mu, zg_log_sigma

#----------------------------------------------------------------------------
# zl encoder network semi-mimicing the Discriminator network in the paper.

def E_zl(
    images_in,                          # Input: Images [minibatch, channel, height, width].
    num_channels        = 3,            # Number of input color channels. Overridden based on dataset.
    resolution          = 128,          # Input resolution. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    latent_res          = 4,            # Spatial dimension of the latent vectors.
    latent_channels     = 512,          # Number of channels of the latent vectors. None = min(fmap_base, fmap_max).
    use_wscale          = True,         # Enable equalized learning rate?
    use_pixelnorm       = False,        # Enable pixelwise feature vector normalization?
    pixelnorm_epsilon   = 1e-8,         # Constant epsilon for pixelwise feature vector normalization.
    use_leakyrelu       = True,         # True = leaky ReLU, False = ReLU.
    tanh_at_end         = False,        # Use tanh activation for the last layer?
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = False,        # True = use fused conv2d + downscale2d, False = separate downscale2d layers.
    structure           = 'recursive',  # 'linear' = human-readable, 'recursive' = efficient, None = select automatically
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    **kwargs):                          # Ignore unrecognized keyword args.
    
    resolution_log2 = int(np.log2(resolution))
    latent_res_log2 = int(np.log2(latent_res))
    assert resolution == 2**resolution_log2 and latent_res == 2**latent_res_log2 and resolution >= latent_res
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def PN(x): return pixel_norm(x, epsilon=pixelnorm_epsilon) if use_pixelnorm else x
    if latent_channels is None: latent_channels = nf(0)
    if structure is None: structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu if use_leakyrelu else tf.nn.relu

    images_in.set_shape([None, num_channels, resolution, resolution])
    images_in = tf.cast(images_in, dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)

    # Building blocks.
    def fromrgb(x, res): # res = latent_res_log2..resolution_log2
        with tf.variable_scope('FromRGB_lod%d' % (resolution_log2 - res)):
            return act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=1, use_wscale=use_wscale)))
    def block(x, res): # res = latent_res_log2..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res > latent_res_log2:
                with tf.variable_scope('Conv0'):
                    x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
                if fused_scale:
                    with tf.variable_scope('Conv1_down'):
                        x = PN(act(apply_bias(conv2d_downscale2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale))))
                else:
                    with tf.variable_scope('Conv1'):
                        x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale))))
                    x = downscale2d(x)
                return x
            else:
                with tf.variable_scope('Conv0'):
                    x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
                with tf.variable_scope('z_Conv1'):
                    x = apply_bias(conv2d(x, fmaps=latent_channels*2, kernel=1, gain=1, use_wscale=use_wscale))
                return x
    
    # Linear structure: simple but inefficient.
    if structure == 'linear':
        img = images_in
        x = fromrgb(img, resolution_log2)
        for res in range(resolution_log2, latent_res_log2, -1):
            lod = resolution_log2 - res
            x = block(x, res)
            img = downscale2d(img)
            y = fromrgb(img, res - 1)
            with tf.variable_scope('Grow_lod%d' % lod):
                x = lerp_clip(x, y, lod_in - lod)
        z_latents_out = block(x, latent_res_log2)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def grow(res, lod):
            x = lambda: fromrgb(downscale2d(images_in, 2**lod), res)
            if lod > 0: x = cset(x, (lod_in < lod), lambda: grow(res + 1, lod - 1))
            x = block(x(), res); y = lambda: x
            if res > latent_res_log2: y = cset(y, (lod_in > lod), lambda: lerp(x, fromrgb(downscale2d(images_in, 2**(lod+1)), res - 1), lod_in - lod))
            return y()
        z_latents_out = grow(latent_res_log2, resolution_log2 - latent_res_log2)

    assert z_latents_out.dtype == tf.as_dtype(dtype)
    if tanh_at_end:
        z_latents_out = tf.nn.tanh(z_latents_out, name='z_latents_out')
    else:
        z_latents_out = tf.identity(z_latents_out, name='z_latents_out')
    z_mu = tf.identity(z_latents_out[:, :latent_channels, :, :], name='z_mu')
    z_log_sigma = tf.identity(z_latents_out[:, latent_channels:, :, :], name='z_log_sigma')
    return z_mu, z_log_sigma

#----------------------------------------------------------------------------
# Residual Generator network.

def G_res(
    zg_latents_in,                      # Second input: zg latent vectors [minibatch, latent_channels, latent_res x scale, latent_res x scale].
    zl_latents_in,                      # Second input: zl latent vectors [minibatch, latent_channels, latent_res x scale, latent_res x scale].
    num_channels        = 3,            # Number of output color channels. Overridden based on dataset.
    resolution          = 128,          # Output resolution. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    latent_res          = 4,            # Spatial dimension of the latent vectors.
    latent_channels     = 512,          # Dimensionality of the latent vectors. None = min(fmap_base, fmap_max).
    use_wscale          = True,         # Enable equalized learning rate?
    use_pixelnorm       = False,        # Enable pixelwise feature vector normalization?
    pixelnorm_epsilon   = 1e-8,         # Constant epsilon for pixelwise feature vector normalization.
    use_leakyrelu       = True,         # True = leaky ReLU, False = ReLU.
    tanh_at_end         = False,        # Use tanh activation for the last layer?
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = False,        # True = use fused upscale2d + conv2d, False = separate upscale2d layers.
    structure           = 'recursive',  # 'linear' = human-readable, 'recursive' = efficient, None = select automatically.
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    scale_h             = 1,            # Height scale up of the fully-convolutional version based on the training scale
    scale_w             = 1,            # Width scale up of the fully-convolutional version based on the training scale
    **kwargs):                          # Ignore unrecognized keyword args.
    
    resolution_log2 = int(np.log2(resolution))
    latent_res_log2 = int(np.log2(latent_res))
    assert resolution == 2**resolution_log2 and latent_res == 2**latent_res_log2 and resolution >= latent_res
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def PN(x): return pixel_norm(x, epsilon=pixelnorm_epsilon) if use_pixelnorm else x
    if latent_channels is None: latent_channels = nf(0)
    if structure is None: structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu if use_leakyrelu else tf.nn.relu
    
    zg_latents_in.set_shape([None, latent_channels, latent_res*scale_h, latent_res*scale_w])
    zl_latents_in.set_shape([None, latent_channels, latent_res*scale_h, latent_res*scale_w])

    combo_in = tf.cast(tf.concat([zg_latents_in, zl_latents_in], axis=1), dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)

    # Building blocks.
    def block(x, res): # res = latent_res_log2..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res == latent_res_log2:
                # start: 5 residual blocks
                for count in range(5):
                    x0 = tf.identity(x)
                    with tf.variable_scope('Residual%d_0' % count):
                        x = PN(act(apply_bias(conv2d(x, fmaps=latent_channels*2, kernel=3, use_wscale=use_wscale))))
                    with tf.variable_scope('Residual%d_1' % count):
                        x = apply_bias(conv2d(x, fmaps=latent_channels*2, kernel=3, gain=1, use_wscale=use_wscale))
                    x = x0 + x
                # end: 5 residual blocks
                with tf.variable_scope('Conv0'):
                    x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, gain=np.sqrt(2)/4, use_wscale=use_wscale))))
                with tf.variable_scope('Conv1'):
                    x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
            else:
                if fused_scale:
                    with tf.variable_scope('Conv0_up'):
                        x = PN(act(apply_bias(upscale2d_conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
                else:
                    x = upscale2d(x)
                    with tf.variable_scope('Conv0'):
                        x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
                with tf.variable_scope('Conv1'):
                    x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
            return x
    def torgb(x, res): # res = latent_res_log2..resolution_log2
        lod = resolution_log2 - res
        with tf.variable_scope('ToRGB_lod%d' % lod):
            return apply_bias(conv2d(x, fmaps=num_channels, kernel=1, gain=1, use_wscale=use_wscale))

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        x = block(combo_in, latent_res_log2)
        images_out = torgb(x, latent_res_log2)
        for res in range(latent_res_log2+1, resolution_log2 + 1):
            lod = resolution_log2 - res
            x = block(x, res)
            img = torgb(x, res)
            images_out = upscale2d(images_out)
            with tf.variable_scope('Grow_lod%d' % lod):
                images_out = lerp_clip(img, images_out, lod_in - lod)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def grow(x, res, lod):
            y = block(x, res)
            img = lambda: upscale2d(torgb(y, res), 2**lod)
            if res > latent_res_log2: img = cset(img, (lod_in > lod), lambda: upscale2d(lerp(torgb(y, res), upscale2d(torgb(x, res - 1)), lod_in - lod), 2**lod))
            if lod > 0: img = cset(img, (lod_in < lod), lambda: grow(y, res + 1, lod - 1))
            return img()
        images_out = grow(combo_in, latent_res_log2, resolution_log2 - latent_res_log2)
        
    assert images_out.dtype == tf.as_dtype(dtype)
    if tanh_at_end:
        images_out = tf.nn.tanh(images_out, name='images_out')
    else:
        images_out = tf.identity(images_out, name='images_out')
    return images_out

#----------------------------------------------------------------------------
# Patch-based Discriminator network used in the paper without the auxiliary classifier.

def D_patch(
    images_in,                          # Input: Images [minibatch, channel, height, width].
    num_channels        = 3,            # Number of input color channels. Overridden based on dataset.
    resolution          = 128,          # Input resolution. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    latent_res          = 4,            # Spatial dimension of the latent vectors.
    use_wscale          = True,         # Enable equalized learning rate?
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = False,        # True = use fused conv2d + downscale2d, False = separate downscale2d layers.
    structure           = 'recursive',  # 'linear' = human-readable, 'recursive' = efficient, None = select automatically
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    **kwargs):                          # Ignore unrecognized keyword args.
    
    resolution_log2 = int(np.log2(resolution))
    latent_res_log2 = 2 if latent_res == -1 else int(np.log2(latent_res))
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    if structure is None: structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu

    images_in.set_shape([None, num_channels, resolution, resolution])
    images_in = tf.cast(images_in, dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)

    # Building blocks.
    def fromrgb(x, res): # res = latent_res_log2..resolution_log2
        with tf.variable_scope('FromRGB_lod%d' % (resolution_log2 - res)):
            return act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=1, use_wscale=use_wscale)))
    def block(x, res): # res = latent_res_log2..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res > latent_res_log2:
                with tf.variable_scope('Conv0'):
                    x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))
                if fused_scale:
                    with tf.variable_scope('Conv1_down'):
                        x = act(apply_bias(conv2d_downscale2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale)))
                else:
                    with tf.variable_scope('Conv1'):
                        x = act(apply_bias(conv2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale)))
                    x = downscale2d(x)
            else:
                if mbstd_group_size > 1:
                    x = minibatch_stddev_layer(x, mbstd_group_size)
                with tf.variable_scope('Conv0'):
                    x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))
                # fully connected
                if latent_res == -1:
                    with tf.variable_scope('Dense1'):
                        x = act(apply_bias(dense(x, fmaps=nf(res-2), use_wscale=use_wscale)))
                    with tf.variable_scope('Dense2'):
                        x = apply_bias(dense(x, fmaps=1, gain=1, use_wscale=use_wscale))
                    x = tf.expand_dims(tf.expand_dims(x, axis=2), axis=3)
                # fully convolutional
                else:
                    with tf.variable_scope('Conv1'):
                        x = act(apply_bias(conv2d(x, fmaps=nf(res-2), kernel=1, use_wscale=use_wscale)))
                    with tf.variable_scope('Conv2'):
                        x = apply_bias(conv2d(x, fmaps=1, gain=1, kernel=1, use_wscale=use_wscale))
            return x
    
    # Linear structure: simple but inefficient.
    if structure == 'linear':
        img = images_in
        x = fromrgb(img, resolution_log2)
        for res in range(resolution_log2, latent_res_log2, -1):
            lod = resolution_log2 - res
            x = block(x, res)
            img = downscale2d(img)
            y = fromrgb(img, res - 1)
            with tf.variable_scope('Grow_lod%d' % lod):
                x = lerp_clip(x, y, lod_in - lod)
        scores_out = block(x, latent_res_log2)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def grow(res, lod):
            x = lambda: fromrgb(downscale2d(images_in, 2**lod), res)
            if lod > 0: x = cset(x, (lod_in < lod), lambda: grow(res + 1, lod - 1))
            x = block(x(), res); y = lambda: x
            if res > latent_res_log2: y = cset(y, (lod_in > lod), lambda: lerp(x, fromrgb(downscale2d(images_in, 2**(lod+1)), res - 1), lod_in - lod))
            return y()
        scores_out = grow(latent_res_log2, resolution_log2 - latent_res_log2)

    assert scores_out.dtype == tf.as_dtype(dtype)
    return scores_out

#----------------------------------------------------------------------------
# Vgg19 network used for gram matrix and autocorrelation matrices calculation.

def Vgg19_gram_autocorrelation(
    images_in,                          # Input: Images [minibatch, channel, height, width].
    num_channels        = 3,            # Number of input color channels. Overridden based on dataset.
    resolution          = 128,          # Input resolution. Overridden based on dataset.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    **kwargs):         

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4

    images_in.set_shape([None, num_channels, resolution, resolution])
    images_in = tf.cast(images_in, dtype)

    data_dict = loadWeightsData('tensorflow_vgg/vgg19.npy')
    vgg = custom_Vgg19(images_in, data_dict=data_dict)
    gram_conv1_1 = gram_matrix(vgg.conv1_1, data_format='NHWC')
    gram_conv2_1 = gram_matrix(vgg.conv2_1, data_format='NHWC')
    gram_conv3_1 = gram_matrix(vgg.conv3_1, data_format='NHWC')
    gram_conv4_1 = gram_matrix(vgg.conv4_1, data_format='NHWC')
    gram_conv5_1 = gram_matrix(vgg.conv5_1, data_format='NHWC')
    autocorrelation_pool2 = autocorrelation_matrix(vgg.pool2, h=resolution//4, w=resolution//4, data_format='NHWC')

    return gram_conv1_1, gram_conv2_1, gram_conv3_1, gram_conv4_1, gram_conv5_1, autocorrelation_pool2

# gram matrix per layer
def gram_matrix(x, data_format='NCHW'):
    if data_format == 'NCHW':
        x = tf.transpose(x, perm=[0,2,3,1])
    h = x.get_shape().as_list()[1]; w = x.get_shape().as_list()[2]; ch = x.get_shape().as_list()[3]
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
    b = tf.shape(x)[0]; ch = x.get_shape().as_list()[1]
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