# Carvana Image Masking Challenge Solution
This challenge is to create a deep learning model that segment car and background

Challenge information [Here](https://www.kaggle.com/c/carvana-image-masking-challenge)

# Data
This solution uses **train, train_masks, test** directory

Data download [Here](https://www.kaggle.com/c/carvana-image-masking-challenge/data)

# Requirements
- Tensorflow 2.3.0
- Keras 2.4.3
- sklearn
- OpenCV
- Python 3.7

# Environment
- OS : Windows 10
-   Hardware:
    -   CPU : i5-7th
    -   32 GB RAM
    -   GPU : RTX 2080 Super
- Software
    - Cuda 10.1
    - cuDNN 7.6
    - Jupyter Notebook

# Train and Test
This solution uses U Net for segmentation

There are U Net model and loss function in model dir

**main.ipynb** contains data extraction, training and testing

**rle_mask.ipynb** is for submission to Kaggle

# Model Weights and Submission Data
Model Weights [Here](https://drive.google.com/drive/folders/1TJJ6ydneODjaVlfALsJn0Wt2Ks0hvv7O?usp=sharing)

Submission Data(RLE Mask) [Here](https://drive.google.com/drive/folders/1TJJ6ydneODjaVlfALsJn0Wt2Ks0hvv7O?usp=sharing)
