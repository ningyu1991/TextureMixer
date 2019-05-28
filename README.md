# TextureMixer

<img src='fig/palette_brush_teaser.png' width=800>

- Tensorflow implementation for our [CVPR'19 paper](https://arxiv.org/pdf/1901.03447.pdf) on controllable texture interpolation and several applications.
- Contact: Ning Yu (ningyu AT umd DOT edu)

## Texture Interpolation 128x1024 (more results are shown in the [paper](https://arxiv.org/pdf/1901.03447.pdf))
<img src='fig/qual_eval.png' width=800>

## Texture Dissolve 1024x1024
<img src='fig/texture_dissolve_animal_loop.gif' width=800>
<img src='fig/texture_dissolve_plant_loop.gif' width=800>
<img src='fig/texture_dissolve_earth_loop.gif' width=800>

## Texture Brush 512x2048
<img src='fig/texture_brush_animal.gif' width=800>
<img src='fig/texture_brush_animal_camouflage.gif' width=800>
<img src='fig/texture_brush_plant.gif' width=800>
<img src='fig/texture_brush_earth.gif' width=800>

## Animal hybridization
<img src='fig/beardog.png' width=800>
<img src='fig/leoraffe.png' width=800>
<img src='fig/ziger.png' width=800>

## Prerequisites
- Linux
- NVIDIA GPU + CUDA + CuDNN
- Python 3.6
- tensorflow-gpu
- Other Python dependencies: numpy, scipy, moviepy, Pillow, skimage, lmdb, opencv-python, cryptography, h5py, six
- Clone the [official VGG repository](https://github.com/machrisaa/tensorflow-vgg) into the current direcotory

## Datasets: Animal Texture, Earth Texture, Plant Texture
- Raw training and testing images are saved at `datasets/animal_texture/`, `datasets/earth_texture/`, and `datasets/plant_texture/`.
  - Modify `datasets/data_augmentation.py` for data augmentation: color histogram matching (only for earth texture) --> geometric transformation --> 128x128 cropping. 
  - Follow the [official Progressive GAN repository](https://github.com/tkarras/progressive_growing_of_gans) "Preparing datasets for training" Section for our dataset preparation. Use the `create_from_images` option in `dataset_tool.py`. The prepared data enables efficient streaming.
- For convenience, the prepared testing datasets can be downloaded: [animal texture](https://drive.google.com/file/d/15HGHJuEMMbaUPMyH23iQrru0teH1gJmw/view?usp=sharing), [earth texture](https://drive.google.com/file/d/1A08JnZEUJGAFuLkhYtkqz7t9qnjMjVVj/view?usp=sharing), [plant texture](https://drive.google.com/file/d/1HPTOc_10Uz1BXQK8_GrS0y9hEjnaBz-0/view?usp=sharing). Unzip and put under `datasets\`.

## Pre-Trained Models
- The pre-trained TextureMixer models can be downloaded: [animal texture](https://drive.google.com/file/d/1zTRwT5W8ExfnPRUZQ5kcu70c_BYzT9u2/view?usp=sharing), [earth texture](https://drive.google.com/file/d/1ObAFBPGaRJFo11LUa0qNhRX14nTEWKC1/view?usp=sharing), [plant texture](https://drive.google.com/file/d/1lAMZyXy9wYzAjseeBLw6XWq1XY9FE9SV/view?usp=sharing).

## Applications

### Texture Interpolation
- First download testing images from the addresses in the first column in the file `data/test/defect_testing_gt.csv` into `data/test/original_resolution/`.
  Or put customized images into that directory.
- Then run
```
python test.py -iPath data/test/original_resolution -oPath output/GoogLeNet -holisticDeployPath prototxt/GoogLeNet/holistic/deploy_holistic.prototxt -holisticWeightsPath model/GoogLeNet/weights_holistic.caffemodel -patchDeployPath prototxt/GoogLeNet/patch/deploy_patch.prototxt -patchWeightsPath model/GoogLeNet/weights_patch.caffemodel -gpu 0
```
- The final seven defect severity prediction results are saved in the file `output/GoogLeNet/defect_scores_combined.csv`.
- Testing images are sorted in the descent order according to each defect severity prediction and visualized correspondingly to the file `output/GoogLeNet/defect_scores_combined_*.html`.

## Evaluation
- We use the cross-class ranking correlation (proposed in Section 3.2 in our [paper](https://arxiv.org/pdf/1612.01635.pdf)) to evaluate the testing results. 
- Assuming the rows and columns of `data/test/defect_testing_gt.csv` and `output/GoogLeNet/defect_scores_combined.csv` align to each other, run
```
python evaluate.py -gtPath data/test/defect_testing_gt.csv -predPath output/GoogLeNet/defect_scores_combined.csv -oPath output/GoogLeNet
```
- The evaluation measures are saved in the file `output/GoogLeNet/evaluation.csv`.

## Citation
```
@inproceedings{yu2018learning,
    author = {Yu, Ning and Shen, Xiaohui and Lin, Zhe and Měch, Radomír and Barnes, Connelly},
    title = {Learning to Detect Multiple Photographic Defects},
    booktitle = {IEEE Winter Conference on Applications of Computer Vision (WACV)},
    year = {2018}
}
```
## Acknowledgement
- This research is supported by Adobe Research Funding.
- We thank to the photographers for licensing photos under Creative Commons or public domain.
- We express gratitudes to the popular [caffe-googlenet-bn](https://github.com/lim0606/caffe-googlenet-bn) repository as we benefit a lot from their code.

## Note
- It is for non-commercial research purpose only. Adobe has been filing a patent for this work.
