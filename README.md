# DarSwin-Unet
[WACV2025] The codes for the work "DarSwin-Unet: Distortion Aware Encoder Decoder Architecture". Our paper has been accepted by WACV 2025.

## 1. Download pre-trained swin transformer model 
* [Get pre-trained model in this link for grp1] (https://hdrdb-public.s3.valeria.science/darswin-unet//home-local2/akath.extra.nobkp/DarSwin-Unet/gr1/epoch_499_1.pth)
* [Get pre-trained model in this link for grp2] (https://hdrdb-public.s3.valeria.science/darswin-unet//home-local2/akath.extra.nobkp/DarSwin-Unet/gr2/epoch_499_2.pth)
* [Get pre-trained model in this link for grp3] (https://hdrdb-public.s3.valeria.science/darswin-unet//home-local2/akath.extra.nobkp/DarSwin-Unet/gr3/epoch_499_3.pth)
* [Get pre-trained model in this link for grp4] (https://hdrdb-public.s3.valeria.science/darswin-unet//home-local2/akath.extra.nobkp/DarSwin-Unet/gr4/epoch_499_4.pth)


## 2. Environment

- Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

## 2. Training

- Knn matrix generation : Use matrix.sh to generate a KNN matrix for your desired dataset (with distortion parameter) to determine inverse projection from polar to cartesian
- Run ./train.sh

## 2. Evaluation

- run the evaluation script with desired level of distortion (low, vlow, med, high) using the checkpoint (grp1, grp2, grp3, grp4) respectivly.
