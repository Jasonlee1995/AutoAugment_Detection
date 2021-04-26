# AutoAugment for Detection Implementation with Pytorch
- Unofficial implementation of the paper *Learning Data Augmentation Strategies for Object Detection*


## 0. Develop Environment
```
Docker Image
- tensorflow/tensorflow:tensorflow:2.4.0-gpu-jupyter

Library
- Pytorch : Stable (1.7.1) - Linux - Python - CUDA (11.0)
```


## 1. Implementation Details
- augmentation.py : augmentation class with probability included
- dataset.py : COCO pytorch dataset
- functional.py : augmentation functions
- policy.py : augmentation policy v0, v1, v2, v3, vtest
- Visualize - Bounding Box Geometric Augmentation.ipynb : experiments of bounding box geometric augmentation
- Visualize - Color Augmentation.ipynb : experiments of color augmentation
- Visualize - Geometric Augmentation.ipynb : experiments of geometric augmentation
- Visualize - Other Augmentation.ipynb : experiments of left augmentation
- Visualize - Policy.ipynb : experiments of policy
- Details
  * range are different so just followed the official code not the paper
  * do not use numpy nor opencv for speed and preventing version crashes
  * similar design pattern following torchvision transforms code
  * policy v2, v3 do not seem good for training models (checkout results)


## 2. Results
#### 2.1. Color Augmentation
![Color Augmentation](./Figures/Color.png)

#### 2.2. Geometric Augmentation
![Geometric Augmentation](./Figures/Geometric.png)

#### 2.3. Bounding Box Augmentation
![Bounding Box Augmentation](./Figures/Bbox_Geometric.png)
![Bounding Box Augmentation](./Figures/Other.png)

#### 2.4. Policy
![Policy](./Figures/Policy.png)


## 3. Reference
- Learning Data Augmentation Strategies for Object Detection [[paper]](https://arxiv.org/pdf/1906.11172.pdf), [[official code]](https://github.com/tensorflow/tpu/blob/master/models/official/detection/utils/autoaugment_utils.py)
