# TextureMixer

<img src='fig/teaser.jpg' width=800>

- Caffe implementation for our [WACV'18 paper](https://arxiv.org/pdf/1612.01635.pdf) ([supplemental](https://drive.google.com/file/d/1J3suq5UnSjCZGEkrqCHTnKUPlg505p7f/view?usp=sharing)) on multiple photographic defect detection.
- Contact: Ning Yu (ningyu AT umd DOT edu)

## Abstract
In this paper, we introduce the problem of simultaneously detecting multiple photographic defects. We aim at detecting the existence, severity, and potential locations of common photographic defects related to color, noise, blur and composition. The automatic detection of such defects could be used to provide users with suggestions for how to improve photos without the need to laboriously try various correction methods. Defect detection could also help users select photos of higher quality while filtering out those with severe defects in photo curation and summarization.

To investigate this problem, we collected a large-scale dataset of user annotations on seven common photographic defects, which allows us to evaluate algorithms by measuring their consistency with human judgments. Our new dataset enables us to formulate the problem as a multi-task learning problem and train a multi-column deep convolutional neural network (CNN) to simultaneously predict the severity of all the defects. Unlike some existing single-defect estimation methods that rely on low-level statistics and may fail in many cases on natural photographs, our model is able to understand image contents and quality at a higher level. As a result, in our experiments, we show that our model has predictions with much higher consistency with human judgments than low-level methods as well as several baseline CNN models. Our model also performs better than an average human from our user study.

## Prerequisites
- Linux
- NVIDIA GPU + CUDA CuDNN
- Caffe

## Dataset
- Training image addresses and seven defect severity ground truth are in the file `data/train/defect_training_gt.csv`.
- Testing image addresses and seven defect severity ground truth are in the file `data/test/defect_testing_gt.csv`.

## Network Architectures (visualize from [ethereon](http://ethereon.github.io/netscope/quickstart.html))
- Multi-column holistic-input GoogLeNet is in the file `prototxt/GoogLeNet/holistic/deploy_holistic.prototxt`.
- Multi-column patch-input GoogLeNet is in the file `prototxt/GoogLeNet/patch/deploy_patch.prototxt`.

## Pre-trained Models
- [Multi-column holistic-input GoogLeNet model](https://drive.google.com/file/d/1rW_ZmRXQasjiGt9gCAKBn7xpG_7GgnMR/view?usp=sharing) (download and put it under `model/GoogLeNet/`)
- [Multi-column patch-input GoogLeNet model](https://drive.google.com/file/d/1xsx2aRc-PIscKTMWzjWOHCkCyG0h-0vA/view?usp=sharing) (download and put it under `model/GoogLeNet/`)

## Infogain Weights
- If users launch their own training or testing with the [infogain loss](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1InfogainLossLayer.html) as Eq. 1 in our [paper](https://arxiv.org/pdf/1612.01635.pdf), they can use our pre-computed weights at `data/infogain_mat/`, which follows the formulation as Eq. 4 in our [supplemental material](https://drive.google.com/file/d/1J3suq5UnSjCZGEkrqCHTnKUPlg505p7f/view?usp=sharing).

## Testing
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
