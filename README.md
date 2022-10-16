# The Bangkok Urbanscapes Dataset for Semantic Urban Scene Understanding Using Enhanced Encoder-Decoder with Atrous Depthwise Separable A1 Convolutional Neural Networks 

## (IEEE Access'22, Accepted!!)

[KITSAPHON THITISIRIWECH](https://th.linkedin.com/in/kitsaphon-thitisiriwech),
[TEERAPONG PANBOONYUEN](https://kaopanboonyuen.github.io/),
[PITTIPOL KANTAVAT](https://cms.kapook.com/uploads/tag/21/ID_20885_5770e7b6aabd1.jpg),
[YUJI IWAHORI](https://cms.kapook.com/uploads/tag/21/ID_20885_5770e7b6aabd1.jpg),
[BOONSERM KIJSIRIKUL](https://www.cp.eng.chula.ac.th/about/faculty/boonsermk)

**[Paper Link](https://ieeexplore.ieee.org/document/9779212?fbclid=IwAR0s80z1OUgIAdDN9OljB8h6GXTuv6WV_tYFE3NmGD4i6fbyAGslbZqOVgE)** | **[Project Page](https://kaopanboonyuen.github.io/bkkurbanscapes/)** 


> **Abstract:**
>*Semantic segmentation is one of the computer vision tasks which is widely researched at present. It plays an essential role to adapt and apply for real-world use-cases, including the application with autonomous driving systems. To further study self-driving cars in Thailand, we provide both the proposed methods and the proposed dataset in this paper. In the proposed method, we contribute Deeplab-V3-A1 with Xception, which is an extension of DeepLab-V3+ architecture. Our proposed method as DeepLab-V3-A1 with Xception is enhanced by the different number of
1×1 convolution layers on the decoder side and refining the image classification backbone with modification of the Xception model. The experiment was conducted on four datasets: the proposed dataset and three public datasets i.e., the CamVid, the cityscapes, and IDD datasets, respectively. The results show that our proposed strategy as DeepLab-V3-A1 with Xception performs comparably to the baseline methods for all corpora including measurement units such as mean IoU, F1 score, Precision, and Recall. In addition, we benchmark DeepLab-V3-A1 with Xception on the validation set of the cityscapes dataset with a mean IoU of 78.86%. For our proposed dataset, we first contribute the Bangkok Urbanscapes dataset, the urban scenes in Southeast Asia. This dataset contains the pair of input images and annotated labels for 701 images. Our dataset consists of various driving environments in Bangkok, as shown for eleven semantic classes (Road, Building, Tree, Car, Footpath, Motorcycle, Pole, Person, Trash, Crosswalk, and Misc). We hope that our architecture and our dataset would help self-driving autonomous developers improve systems for driving in many cities with unique traffic and driving conditions similar to Bangkok and elsewhere in Thailand. Our implementation codes and dataset are available at https://kaopanboonyuen.github.io/bkkurbanscapes.*


<!-- <p align="center">
  <img alt="intro_image" src="image/sudchung_method.png" width="650"/>
</p> -->

![](image/sudchung_method.png)

## Bangkok Urbanscapes Data Set

The resolutions of all the images within our Bangkok urbanscapes dataset are configurated at 521 × 544 pixels.

**[Download](https://drive.google.com/file/d/1AwvuiVnzg-UZp7zzepF2G5Ck8uNZPTG6/view?fbclid=IwAR0oB-e1qQGTOgghFT_xTI6Mxlun-eoUG5HS7zwPzaNMsELmvhE7jxeABMA)**

If you're going to use this dataset, please cite the tech report at the bottom of this page.


## Usage & Data
Refer to `requirements.txt` for installing all python dependencies. We use python 3.7 with pytorch 1.7.1. 

We download the official version of CityScapes from [here](https://www.cityscapes-dataset.com/) and images are resized using code [here](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics).


## Model Training
For pre-training on models on the CityScapes dataset, use the scripts in the `scripts` directory as follows. Change the paths to dataset as required. 

```
./scripts/train.sh
``` 


## Downstream Evaluation
Scripts to perform evaluation (linear or knn) on selected downstream tasks are as below. Paths to datasets and pre-trained models must be set appropriately. Note that in the case of linear evaluation, a linear layer will be fine-tuned on the new dataset and this training can be time-consuming on a single GPU.  

```
./scripts/eval_linear.sh
./scripts/eval_knn.sh
``` 


## Pretrained Models
Our pre-trained models can be found under [releases](https://github.com/bkkurbanscapes/).

## Results

![](image/DecoupleSegNet-BKK-inference.png)
![](image/baseline-old-2-original-results-paper.png)
![](image/baseline-old-1-original-results-paper.png)
![](image/baseline-new2-original-results-paper.png)
![](image/baseline-new-1-original-results-paper.png)


## Citation

```bibtex
@article{thitisiriwech2022bangkok,
  title={The Bangkok Urbanscapes Dataset for Semantic Urban Scene Understanding Using Enhanced Encoder-Decoder with Atrous Depthwise Separable A1 Convolutional Neural Networks},
  author={Thitisiriwech, Kitsaphon and Panboonyuen, Teerapong and Kantavat, Pittipol and Iwahori, Yuji and Kijsirikul, Boonserm},
  journal={IEEE Access},
  year={2022},
  publisher={IEEE}
}
```

## Acknowledgements
Our code is based on [TensorFLow](https://www.tensorflow.org/tutorials/images/segmentation) and [SegmentationModels](https://github.com/qubvel/segmentation_models) repositories. We thank the authors for releasing their code. If you use our model, please consider citing these works as well.
