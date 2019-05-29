# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

#----------------------------------------------------------------------------
# Convenience class that behaves exactly like dict(), but allows accessing
# the keys and values using the attribute syntax, i.e., "mydict.key = value".

class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]

#----------------------------------------------------------------------------
# Paths.

data_dir = 'datasets'

#----------------------------------------------------------------------------
# TensorFlow options.

tf_config = EasyDict()  # TensorFlow session config, set by tfutil.init_tf().
env = EasyDict()        # Environment variables, set by the main program in train.py.

tf_config['allow_soft_placement'] 				= True 		# Uncomment when using tf.mod or tf.random_crop that are not enabled on GPU
tf_config['graph_options.place_pruned_graph']   = True      # False (default) = Check that all ops are available on the designated device. True = Skip the check for ops that are not used.
tf_config['gpu_options.allow_growth']          	= True     # False (default) = Allocate all GPU memory at the beginning. True = Allocate only as much GPU memory as needed.
env.CUDA_VISIBLE_DEVICES                       	= '0,1,2,3,4,5,6,7'       # Unspecified (default) = Use all available GPUs. List of ints = CUDA device numbers to use.
env.TF_CPP_MIN_LOG_LEVEL                        = '1'       # 0 (default) = Print all available debug info from TensorFlow. 1 = Print warnings and errors, but disable debug info.

#----------------------------------------------------------------------------
# Official training configs, targeted mainly for CelebA-HQ.
# To run, comment/uncomment the lines as appropriate and launch train.py.

train_size = 128

# Training and validation datasets
training_data = 'animal_texture_train_aug_with_labels'; val_data = 'animal_texture_test_aug_with_labels'
#training_data = 'earth_texture_train_aug_with_labels'; val_data = 'earth_texture_test_aug_with_labels'
#training_data = 'plant_texture_train_aug_with_labels'; val_data = 'plant_texture_test_aug_with_labels'

training_set = EasyDict(tfrecord_dir=training_data);
val_set = EasyDict(tfrecord_dir=val_data);

# Config encoding methods
zg_enabled = True
zg_interp_variational = 'hard'
zl_interp_variational = 'permutational'
fmap_base = 1024#1024#2048#8192
fmap_max = 512
latent_res_EG = train_size//4
latent_channels = 128
latent_res_DC = -1
scale_h = 3
scale_w = 3

# Config training losses.
rec_G_weight = 1.0
pixel_weight = 200.0
gram_weight = 0.002#0.002 # 0.00001x pixel loss weight
latent_weight = 0.0#10.0 # In BicycleGAN paper pixel weight is 20x latent weight
kl_weight = 0.0#1.0#1.0#10.0#100.0 # A larger value (than BicycleGAN) guarantees randomness for zl, s.t., zg controls most thing / everything; The smallest possible value is 10.0 to guarantee smooth spatial interpolation
interp_G_weight = 1.0#1.0
blend_interp_G_weight = 1.0#1.0
block_size = 0
perm = False

result_dir = 'models/new_swap_%d_latentResEG_%d_latentResDC_%d_latentChannels_%d_recG_%.2f_pixel_%.2f_gram_%.4f_KL_%.2f_interpG_%.2f_blendG_%.2f_%s' % (block_size, latent_res_EG, latent_res_DC, latent_channels, rec_G_weight, pixel_weight, gram_weight, kl_weight, interp_G_weight, blend_interp_G_weight, training_data)

desc        = 'pgan'                                        # Description string included in result subdir name.
random_seed = 1000                                          # Global random seed.
train       = EasyDict(func='train.train_progressive_vae_interp_gans')  # Options for main training func.
E_zg        = EasyDict(func='networks.E_zg', fmap_base=fmap_base, fmap_max=fmap_max, latent_channels=latent_channels, use_pixelnorm=False, tanh_at_end=False)             # Options for encoder network.
E_zl        = EasyDict(func='networks.E_zl', fmap_base=fmap_base, fmap_max=fmap_max, latent_res=latent_res_EG, latent_channels=latent_channels, use_pixelnorm=False, tanh_at_end=False)             # Options for encoder network.
G           = EasyDict(func='networks.G_res', fmap_base=fmap_base, fmap_max=fmap_max, latent_res=latent_res_EG, latent_channels=latent_channels, use_pixelnorm=False, tanh_at_end=True)             # Options for generator network.
D_rec       = EasyDict(func='networks.D_patch', fmap_base=fmap_base, fmap_max=fmap_max, latent_res=latent_res_DC)             # Options for reconstruction-associated discriminator network.
D_interp    = EasyDict(func='networks.D_patch', fmap_base=fmap_base, fmap_max=fmap_max, latent_res=latent_res_DC)             # Options for interpolation-associated discriminator network.
D_blend    	= EasyDict(func='networks.D_patch', fmap_base=fmap_base, fmap_max=fmap_max, latent_res=latent_res_DC)             # Options for blending-and-interpolation-associated discriminator network.
Vgg19_gram_autocorrelation = EasyDict(func='networks.Vgg19_gram_autocorrelation')             # Options for Vgg19_gram_autocorrelation network.
EG_opt   	= EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8) # Options for AE optimizer.
D_rec_opt   = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8) # Options for reconstruction-associated discriminator optimizer.
D_interp_opt   = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8) # Options for interpolation-associated discriminator optimizer.
D_blend_opt = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8) # Options for blending-and-interpolation-associated discriminator optimizer.
EG_loss 	= EasyDict(func='loss.EG_wgan', scale_h=scale_h, scale_w=scale_w, zg_interp_variational=zg_interp_variational, zl_interp_variational=zl_interp_variational, rec_G_weight=rec_G_weight, pixel_weight=pixel_weight, gram_weight=gram_weight, latent_weight=latent_weight, kl_weight=kl_weight, interp_G_weight=interp_G_weight, blend_interp_G_weight=blend_interp_G_weight)        # Options for AutoEncoder loss.
D_rec_loss  = EasyDict(func='loss.D_rec_wgangp')      # Options for reconstruction-associated discriminator loss.
D_interp_loss  = EasyDict(func='loss.D_interp_wgangp', scale_h=scale_h, scale_w=scale_w, zg_interp_variational=zg_interp_variational, zl_interp_variational=zl_interp_variational)      # Options for interpolation-associated discriminator loss.
D_blend_loss= EasyDict(func='loss.D_blend_wgangp', scale_h=scale_h, scale_w=scale_w, zg_interp_variational=zg_interp_variational, zl_interp_variational=zl_interp_variational)      # Options for interpolation-associated discriminator loss.
sched       = EasyDict()                                    # Options for train.TrainingSchedule.
grid        = EasyDict(size='1080p', layout='random')       # Options for train.setup_snapshot_image_grid().

# Dataset (choose one).
desc += '-%s' % training_data; train.lr_mirror_augment = False; train.ud_mirror_augment = False; sched.lod_initial_resolution = latent_res_EG; sched.lod_training_kimg = 1000; sched.lod_transition_kimg = 3000; train.total_kimg = 500000

# Conditioning & snapshot options.
desc += '-labels'; training_set.max_label_size = 'full'; val_set.max_label_size = 'full' # conditioned on full label

# Config presets (choose one). Note: the official settings are optimal. It is not the larger batch size the better.
#desc += '-preset-v2-1gpus'; num_gpus = 1; sched.minibatch_base = 8; sched.lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
#desc += '-preset-v2-2gpus'; num_gpus = 2; sched.minibatch_base = 16; sched.lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
#desc += '-preset-v2-4gpus'; num_gpus = 4; sched.minibatch_base = 32; sched.lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
desc += '-preset-v2-8gpus'; num_gpus = 8; sched.minibatch_base = 64; sched.lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}

# Numerical precision (choose one).
desc += '-fp32'; sched.max_minibatch_per_gpu = {256: 16, 512: 8, 1024: 4}

#----------------------------------------------------------------------------

#env.CUDA_VISIBLE_DEVICES = '0'; num_gpus = 1
#model_path = 'models/animal_texture/network-final.pkl'
#image1_path = '../../codes/adobe_stock_datasets/animal_textures_new/images/test_resize512_aug_crop_128/1000_F_99107656_XvbvoVVRintE5tmuh1MkdXqs8rkzoahB_NW_aug00000091.png'
#image2_path = '../../codes/adobe_stock_datasets/animal_textures_new/images/test_resize512_aug_crop_128/1000_F_87614886_hTyVkR2XzAJS2qRjmks6IQEAmuAzsuzP_NW_aug00000062.png'
#model_path = 'models/earth_texture/network-final.pkl'
#image1_path = '../../codes/flickr_datasets/earth_textures_new/images/test_resize512_aug_crop_128/farm9_8482_8210343099_50e43a65aa_o_aug00000007.png'
#image2_path = '../../codes/flickr_datasets/earth_textures_new/images/test_resize512_aug_crop_128/farm9_8888_17781737673_79743c6c53_o_aug00000102.png'
#model_path = 'models/plant_texture/network-final.pkl'
#image1_path = '../../codes/adobe_stock_datasets/plant_textures_new/images/test_resize512_aug_crop_128/1000_F_98832647_RaDsgFfp78ONq0RnIRYG2S8WdwiyNFPf_NW_aug00000033.png'
#image2_path = '../../codes/adobe_stock_datasets/plant_textures_new/images/test_resize512_aug_crop_128/1000_F_99674846_IpnHI1gidsXJWLvOSzuizVwhAGnlWWHf_NW_aug00000084.png'
#out_dir = 'results/plant/horizontal_interpolation_dummy'
#train = EasyDict(func='util_scripts.horizontal_interpolation', model_path=model_path, image1_path=image1_path, image2_path=image2_path, out_dir=out_dir)

'''
env.CUDA_VISIBLE_DEVICES = '0'; num_gpus = 1
model_path = 'models/animal_texture/network-final.pkl'
out_dir = 'results/animal/spatiotemporal_interpolation_video_dummy'
train = EasyDict(func='util_scripts.texture_dissolve_video', model_path=model_path, out_dir=out_dir)
'''
'''
env.CUDA_VISIBLE_DEVICES = '0'; num_gpus = 1
model_path = 'models/animal_texture/network-final.pkl'
out_dir = 'results/animal/texture_brush_video_dummy'
train = EasyDict(func='util_scripts.texture_brush_video', model_path=model_path, out_dir=out_dir, scale_h_bg=4, scale_w_bg=16, scale_h_fg=8, scale_w_fg=8, stroke_radius_div=128.0, minibatch_size=32*num_gpus)
'''
'''
env.CUDA_VISIBLE_DEVICES = '0'; num_gpus = 1
topic = 'tibra' #'leoraffe'#'tibra'
in_dir = 'hybridization_fig/%s' % topic
model_path = 'models/animal_texture/network-final.pkl'
out_dir = 'results/animal/hybridization_%s_dummy' % topic
train = EasyDict(func='util_scripts.hybridization_CAF', model_path=model_path, in_dir=in_dir, out_dir=out_dir, train_size=train_size, rotation_enabled=False, weight_mode='horizontal_linear', sig_div=4.0)
'''