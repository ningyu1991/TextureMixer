# Texture Mixer

### [Texture Mixer: A Network for Controllable Synthesis and Interpolation of Texture](https://arxiv.org/pdf/1901.03447.pdf)
[Ning Yu](https://ningyu1991.github.io/), [Connelly Barnes](http://www.connellybarnes.com/work/), [Eli Shechtman](https://research.adobe.com/person/eli-shechtman/), [Sohrab Amirghodsi](https://www.linkedin.com/in/sohrab-amirghodsi-a89548a3/), [Michal Lukáč](https://research.adobe.com/person/michal-lukac/)<br>
CVPR 2019

### [paper](https://arxiv.org/pdf/1901.03447.pdf) | [project page](https://ningyu1991.github.io/projects/TextureMixer.html) | [poster](https://ningyu1991.github.io/homepage_files/poster_TextureMixer.pdf) | [media coverage in Chinese](https://mp.weixin.qq.com/s/b_gIB_zpUCcf2a8ImKMXRA)

<img src='fig/palette_brush_teaser.png' width=400> <img src='fig/pipeline.png' width=400>

## Abstract
This paper addresses the problem of interpolating visual textures. We formulate this problem by requiring (1) by-example controllability and (2) realistic and smooth interpolation among an arbitrary number of texture samples. To solve it we propose a neural network trained simultaneously on a reconstruction task and a generation task, which can project texture examples onto a latent space where they can be linearly interpolated and projected back onto the image domain, thus ensuring both intuitive control and realistic results. We show our method outperforms a number of baselines according to a comprehensive suite of metrics as well as a user study. We further show several applications based on our technique, which include texture brush, texture dissolve, and animal hybridization.

## Texture Interpolation 128x1024 (more results are shown in the [paper](https://arxiv.org/pdf/1901.03447.pdf))
<img src='fig/qual_eval.png' width=800>

## Texture Dissolve 1024x1024
<img src='fig/texture_dissolve_animal_loop.gif' width=800>
<img src='fig/texture_dissolve_earth_loop.gif' width=800>
<img src='fig/texture_dissolve_plant_loop.gif' width=800>

## Texture Brush 512x2048
<img src='fig/texture_brush_animal.gif' width=800>
<img src='fig/texture_brush_animal_camouflage.gif' width=800>
<img src='fig/texture_brush_earth.gif' width=800>
<img src='fig/texture_brush_plant.gif' width=800>

## Animal hybridization
<img src='fig/beardog.png' width=800>
<img src='fig/leoraffe.png' width=800>
<img src='fig/ziger.png' width=800>

## Prerequisites
- Linux
- NVIDIA GPU + CUDA 10.0 + CuDNN 7.5
- Python 3.6
- tensorflow-gpu 1.12
- To install the other Python dependencies, run `pip3 install -r requirements.txt`.
- Clone the [official VGG repository](https://github.com/machrisaa/tensorflow-vgg) into the current direcotory.

## Datasets: Animal Texture, Earth Texture, Plant Texture
- Raw training and testing images of earth texture are downloaded from [Flickr](https://www.flickr.com/) under Creative Commons or public domain license. They are saved at `datasets/earth_texture/`.
- Raw training and testing images of animal texture or plant texture are copyrighted by Adobe and can be purchased from [Adobe Stock](https://stock.adobe.com/). We list the searchable image IDs at:
  - `datasets/animal_texture/train_AdobeStock_ID_list.txt`
  - `datasets/animal_texture/test_AdobeStock_ID_list.txt`
  - `datasets/plant_texture/train_AdobeStock_ID_list.txt`
  - `datasets/plant_texture/test_AdobeStock_ID_list.txt`
- Given raw images, run, e.g., the following command for data augmentation.
  ```
  python3 data_augmentation.py --iPath earth_texture/test_resize512/ --oPath earth_texture_test_aug/ --num_aug 10000
  ```
- Then follow the [official Progressive GAN repository](https://github.com/tkarras/progressive_growing_of_gans) "Preparing datasets for training" Section for dataset preparation. Use the `create_from_images` option in `dataset_tool.py`. The prepared data enables efficient streaming.
- For convenience, the prepared testing dataset of earth texture can be downloaded [here](https://drive.google.com/file/d/1A08JnZEUJGAFuLkhYtkqz7t9qnjMjVVj/view?usp=sharing). Unzip and put under `datasets/`.
  
## Training
After data preparation, run, e.g.,
```
python3 run.py \
--app train \
--train_dir earth_texture_train_aug_with_labels/ \
--val_dir earth_texture_test_aug_with_labels/ \
--out_dir models/earth_texture/ \
--num_gpus 8
```
where
- `train_dir`: The prepared training dataset directory that can be efficiently called by the code.
- `val_dir`: The prepared validation dataset directory that can be efficiently called by the code.
- `num_gpus`: The number of GPUs for training. Options {1, 2, 4, 8}. Using 8 NVIDIA GeForce GTX 1080 Ti GPUs, we suggest training for 3 days.

## Pre-Trained Models
- The pre-trained Texture Mixer models can be downloaded from:
  - [Animal texture](https://drive.google.com/file/d/1zTRwT5W8ExfnPRUZQ5kcu70c_BYzT9u2/view?usp=sharing)
  - [Earth texture](https://drive.google.com/file/d/1ObAFBPGaRJFo11LUa0qNhRX14nTEWKC1/view?usp=sharing)
  - [Plant texture](https://drive.google.com/file/d/1lAMZyXy9wYzAjseeBLw6XWq1XY9FE9SV/view?usp=sharing)
  - Unzip and put under `models/`.

## Applications
### Texture Interpolation
Run, e.g.,
```
python3 run.py \
--app interpolation \
--model_path models/animal_texture/network-final.pkl \
--imageL_path examples/animal_texture/1000_F_99107656_XvbvoVVRintE5tmuh1MkdXqs8rkzoahB_NW_aug00000094.png \
--imageR_path examples/animal_texture/1000_F_109464954_aBfyWSbdZt5PNpUo7hOqDRPWmmvQj3v9_NW_aug00000092.png \
--out_dir results/animal_texture/horizontal_interpolation/
```
where
- `imageL_path`: The left-hand side image for horizontal interpolation.
- `imageR_path`: The right-hand side image for horizontal interpolation.
### Texture Dissolve
Run, e.g.,
```
python3 run.py \
--app dissolve \
--model_path models/animal_texture/network-final.pkl \
--imageStartUL_path examples/animal_texture/1000_F_23745067_w0GkcAIQG2C4hxOelI1aQYZglEXggGRS_NW_aug00000000.png \
--imageStartUR_path examples/animal_texture/1000_F_44327602_E6wl8FNihQON8c704fE5DEY2LOJPQQ1V_NW_aug00000018.png \
--imageStartBL_path examples/animal_texture/1000_F_66218716_rxcsXWQzYpIWVB8a09ZcuNzE7qAJ3HEk_NW_aug00000098.png \
--imageStartBR_path examples/animal_texture/1000_F_40846588_APKTS3BpiRvR1nvUx0FRa7qjjR788zt8_NW_aug00000001.png \
--imageEndUL_path examples/animal_texture/1000_F_40300952_3dgaCtcLCrhzU0r6HEfnwr7nujDWXbSQ_NW_aug00000029.png \
--imageEndUR_path examples/animal_texture/1000_F_44119648_3vskjRwVc4NVT1Pf0l3RlvIFUemo8TM1_NW_aug00000001.png \
--imageEndBL_path examples/animal_texture/1000_F_70708482_2N8lknTCJg2Q8JQeomFYaFttxId9rulj_NW_aug00000070.png \
--imageEndBR_path examples/animal_texture/1000_F_79496236_8mxTSHy5OilHnJaAxWcw2dwC9SoBLmDK_NW_aug00000057.png \
--out_dir results/animal_texture/dissolve/
```
where
- `imageStartUL_path`: The upper-left corner image in the starting frame.
- `imageStartUR_path`: The upper-right corner image in the starting frame.
- `imageStartBL_path`: The bottom-left corner image in the starting frame.
- `imageStartBR_path`: The bottom-right corner image in the starting frame.
- `imageEndUL_path`: The upper-left corner image in the ending frame.
- `imageEndUR_path`: The upper-right corner image in the ending frame.
- `imageEndBL_path`: The bottom-left corner image in the ending frame.
- `imageEndBR_path`: The bottom-right corner image in the ending frame.
### Texture Brush
Run, e.g.,
```
python3 run.py \
--app brush \
--model_path models/animal_texture/network-final.pkl \
--imageBgUL_path examples/animal_texture/1000_F_23745067_w0GkcAIQG2C4hxOelI1aQYZglEXggGRS_NW_aug00000000.png \
--imageBgUR_path examples/animal_texture/1000_F_44327602_E6wl8FNihQON8c704fE5DEY2LOJPQQ1V_NW_aug00000018.png \
--imageBgBL_path examples/animal_texture/1000_F_66218716_rxcsXWQzYpIWVB8a09ZcuNzE7qAJ3HEk_NW_aug00000098.png \
--imageBgBR_path examples/animal_texture/1000_F_40846588_APKTS3BpiRvR1nvUx0FRa7qjjR788zt8_NW_aug00000001.png \
--imageFgUL_path examples/animal_texture/1000_F_40300952_3dgaCtcLCrhzU0r6HEfnwr7nujDWXbSQ_NW_aug00000029.png \
--imageFgUR_path examples/animal_texture/1000_F_44119648_3vskjRwVc4NVT1Pf0l3RlvIFUemo8TM1_NW_aug00000001.png \
--imageFgBL_path examples/animal_texture/1000_F_70708482_2N8lknTCJg2Q8JQeomFYaFttxId9rulj_NW_aug00000070.png \
--imageFgBR_path examples/animal_texture/1000_F_79496236_8mxTSHy5OilHnJaAxWcw2dwC9SoBLmDK_NW_aug00000057.png \
--stroke1_path stroke_fig/C_skeleton.png \
--stroke2_path stroke_fig/V_skeleton.png \
--stroke3_path stroke_fig/P_skeleton.png \
--stroke4_path stroke_fig/R_skeleton.png \
--out_dir results/animal_texture/brush/
```
where
- `imageBgUL_path`: The upper-left corner image for the background canvas.
- `imageBgUR_path`: The upper-right corner image for the background canvas.
- `imageBgBL_path`: The bottom-left corner image for the background canvas.
- `imageBgBR_path`: The bottom-right corner image for the background canvas.
- `imageFgUL_path`: The upper-left corner image for the foreground palatte.
- `imageFgUR_path`: The upper-right corner image for the foreground palatte.
- `imageFgBL_path`: The bottom-left corner image for the foreground palatte.
- `imageFgBR_path`: The bottom-right corner image for the foreground palatte.
- `stroke1_path`: The trajectory image for the 1st stroke. The stroke pattern is sampled from the (3/8, 3/8) portion of the foreground palatte.
- `stroke2_path`: The trajectory image for the 2nd stroke. The stroke pattern is sampled from the (3/8, 7/8) portion of the foreground palatte.
- `stroke3_path`: The trajectory image for the 3rd stroke. The stroke pattern is sampled from the (7/8, 3/8) portion of the foreground palatte.
- `stroke4_path`: The trajectory image for the 4th stroke. The stroke pattern is sampled from the (7/8, 7/8) portion of the foreground palatte.
### Animal hybridization
Run, e.g.,
```
python3 run.py \
--app hybridization \
--model_path models/animal_texture/network-final.pkl \
--source_dir hybridization_fig/leoraffe/ \
--out_dir results/animal_texture/hybridization/leoraffe/
```
where
- `source_dir`: The directory containing the hole region to be interpolated, two known source texture images adjacent to the hole, and their global [Content-Aware Fill](https://helpx.adobe.com/photoshop/using/content-aware-fill.html) results from Photoshop.

After that, post-process the output hybridization image by [Auto-Blend Layers](https://helpx.adobe.com/photoshop/using/combine-images-auto-blend-layers.html) with the original image (with hole) in Photoshop, so as to achieve the demo quality as shown above. 
## Citation
```
@inproceedings{yu2019texture,
    author = {Yu, Ning and Barnes, Connelly and Shechtman, Eli and Amirghodsi, Sohrab and Luk\'{a}\v{c}, Michal},
    title = {Texture Mixer: A Network for Controllable Synthesis and Interpolation of Texture},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2019}
}
```
## Acknowledgement
- This research was supported by Adobe Research Funding.
- We acknowledge the Maryland Advanced Research Computing Center for providing computing resources.
- We thank to the photographers for licensing photos under Creative Commons or public domain.
- We express gratitudes to the [Progressive GAN repository](https://github.com/tkarras/progressive_growing_of_gans) as we benefit a lot from their code.

## Note
- It is for non-commercial research purpose only. Adobe has filed a patent for this work: see [here](https://patentimages.storage.googleapis.com/43/18/50/b310bb01ca1ef7/US10818043.pdf).
