## Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set ##

<p align="center"> 
<img src="/images/example.gif">
</p>



This is a tensorflow implementation of the following paper:

Y. Deng, J. Yang, S. Xu, D. Chen, Y. Jia, and X. Tong, [Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set](https://arxiv.org/abs/1903.08527), IEEE Computer Vision and Pattern Recognition Workshop (CVPRW) on Analysis and Modeling of Faces and Gestures (AMFG), 2019. (**_Best Paper Award!_**)

The method enforces a hybrid-level weakly-supervised training to for CNN-based 3D face reconstruction. It is fast, accurate, and robust to pose and occlussions. It achieves state-of-the-art performance on multiple datasets such as FaceWarehouse, MICC Florence and BU-3DFE.

**_Training code is available now!_**

## Features

### ● Accurate shapes
The method reconstructs faces with high accuracy. Quantitative evaluations (shape errors in mm) on several benchmarks show its state-of-the-art performance:


|Method|FaceWareHouse|Florence|BU3DFE|
|:---:|:---:|:---:|:---:|
|[Tewari et al. 17](https://arxiv.org/abs/1703.10580)</center>|2.19±0.54|-|-|
|[Tewari et al. 18](https://arxiv.org/abs/1712.02859)|1.84±0.38|-|-|
|[Genova et al. 18](https://arxiv.org/abs/1806.06098)|-|1.77±0.53|-|
|[Sela et al. 17](https://arxiv.org/abs/1703.10131)|-|-|2.91±0.60|
|[PRN 18](https://arxiv.org/abs/1803.07835)|-|-|1.86±0.47|
|Ours|**1.81±0.50**|**1.67±0.50**|**1.40±0.31**|

(Please refer to our paper for more details about these results)

### ● High fidelity textures
The method produces high fidelity face textures meanwhile preserves identity information of input images. Scene illumination is also disentangled to generate a pure albedo.
<p align="center"> 
<img src="/images/albedo.png">
</p>

### ● Robust
The method can provide reasonable results under extreme conditions such as large pose and occlusions.
<p align="center"> 
<img src="/images/extreme.png">
</p>

### ● Aligned with images
Our method aligns reconstruction faces with input images. It provides face pose estimation and 68 facial landmarks which are useful for other tasks. We conduct an experiment on AFLW_2000 dataset (NME) to evaluate the performance, as shown in the table below:
<p align="center"> 
<img src="/images/alignment.png">
</p>

|Method|[0°,30°]|[30°,60°]|[60°,90°]|Overall|
|:---:|:---:|:---:|:---:|:---:|
|[3DDFA 16](https://arxiv.org/abs/1511.07212)</center>|3.78|4.54|7.93|5.42|
|[3DDFA+SDM 16](https://arxiv.org/abs/1511.07212)|3.43|4.24|7.17|4.94|
|[Bulat et al. 17](https://arxiv.org/abs/1703.00862)|**2.47**|**3.01**|**4.31**|**3.26**|
|[PRN 18](https://arxiv.org/abs/1803.07835)|2.75|3.51|4.61|3.62|
|Ours|2.56|3.11|4.45|3.37|


### ● Easy and Fast
Faces are represented with Basel Face Model 2009, which is easy for further manipulations (e.g expression transfer). ResNet-50 is used as backbone network to achieve over 50 fps (on GTX 1080) for reconstructions.


## Getting Started
### Testing Requirements ###

- Reconstructions can be done on both Windows and Linux. However, we suggest running on Linux because the rendering process is only supported on Linux.
- Python 3.6 (numpy, scipy, pillow, argparse).
- Tensorflow 1.12.
- [Basel Face Model 2009 (BFM09)](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model). 
- [Expression Basis (transferred from Facewarehouse by Guo et al.)](https://github.com/Juyong/3DFace). The original BFM09 model does not handle expression variations so extra expression basis are needed. 
- [tf mesh renderer (an older version)](https://github.com/google/tf_mesh_renderer/tree/ba27ea1798f6ee8d03ddbc52f42ab4241f9328bb).  We use the library to render reconstruction images. **Note that the rendering tool can only be used on Linux.**

### Install Dependencies ###
#### 1. Set up the python environment

If you use anaconda, run the following (make sure /usr/local/cuda link to cuda-9.0):
```bash
conda create -n deep3d python=3.6
source activate deep3d
pip install tensorflow-gpu==1.12.0
pip install pillow argparse scipy
```

Alternatively, you can install tensorflow via conda install (no need to set cuda version in this way):
```bash
conda install tensorflow-gpu==1.12.0
```
#### 2. Compile tf_mesh_renderer

If you install tensorflow using pip,  we provide a [pre-compiled binary file (rasterize_triangles_kernel.so)](https://drive.google.com/file/d/1VUtJPdg0UiJkKWxkACs8ZTf5L7Y4P9Wj/view?usp=sharing) of the library. **Note that the pre-compiled file can only be run with tensorflow 1.12.**

If you install tensorflow using conda, you have to compile tf_mesh_renderer from sources. Compile [tf_mesh_renderer](https://github.com/google/tf_mesh_renderer) with Bazel. We use its [older version](https://github.com/google/tf_mesh_renderer/tree/ba27ea1798f6ee8d03ddbc52f42ab4241f9328bb) because we find the latest version unstable during our training process:
```bash
git clone https://github.com/google/tf_mesh_renderer.git
cd tf_mesh_renderer
git checkout ba27ea1798
git checkout master WORKSPACE
bazel test ...
```
If the library is compiled correctly, there should be a file named "rasterize_triangles_kernel.so" in ./bazel-bin/mesh_renderer/kernels. **Set -D_GLIBCXX_USE_CXX11_ABI=1 in ./mesh_renderer/kernels/BUILD before the compilation.**



### Testing with pre-trained network ###

1. Clone the repository 

```bash
git clone https://github.com/Microsoft/Deep3DFaceReconstruction
cd Deep3DFaceReconstruction
```

2. Download the Basel Face Model. Due to the license agreement of Basel Face Model, you have to download the BFM09 model after submitting an application on its [home page](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads). After getting the access to BFM data, download "01_MorphableModel.mat" and put it into ./BFM subfolder.

3. Download the Expression Basis provided by [Guo et al.](https://github.com/Juyong/3DFace) You can find a link named "CoarseData" in the first row of Introduction part in their repository. Download and unzip the Coarse_Dataset.zip. Put "Exp_Pca.bin" into ./BFM subfolder. The expression basis are constructed using [Facewarehouse](http://kunzhou.net/zjugaps/facewarehouse/) data and transferred to BFM topology.

4. Put the compiled rasterize_triangles_kernel.so into ./renderer folder.

5. Download the pre-trained [reconstruction network](https://drive.google.com/file/d/176LCdUDxAj7T2awQ5knPMPawq5Q2RUWM/view?usp=sharing), unzip it and put "FaceReconModel.pb" into ./network subfolder.

6. Run the demo code.

```
python demo.py
```

7. ./input subfolder contains several test images and ./output subfolder stores their reconstruction results. For each input test image, two output files can be obtained after running the demo code:
	- "xxx.mat" : 
		- cropped_img: an RGB image after alignment, which is the input to the R-Net
		- recon_img: an RGBA reconstruction image aligned with the input image (only on Linux).
		- coeff: output coefficients of R-Net.
		- face_shape: vertex positions of 3D face in the world coordinate.
		- face_texture: vertex texture of 3D face, which excludes lighting effect.
		- face_color: vertex color of 3D face, which takes lighting into consideration.
		- lm\_68p: 68 2D facial landmarks derived from the reconstructed 3D face. The landmarks are aligned with cropped_img.
		- lm\_5p: 5 detected landmarks aligned with cropped_img. 
	- "xxx_mesh.obj" : 3D face mesh in the world coordinate (best viewed in MeshLab).

### Training requirements ###

- Training is only supported on Linux. To train new model from scratch, more requirements are needed on top of the requirements listed in the testing stage.
- [Facenet](https://github.com/davidsandberg/facenet) provided by 
Sandberg et al. In our paper, we use a network to exrtact perceptual face features. This network model cannot be publicly released. As an alternative, we recommend using the Facenet from Sandberg et al. This repo uses the version [20170512-110547](https://github.com/davidsandberg/facenet/blob/529c3b0b5fc8da4e0f48d2818906120f2e5687e6/README.md) trained on MS-Celeb-1M. Training process has been tested with this model to ensure similar results.
- [Resnet50-v1](https://github.com/tensorflow/models/blob/master/research/slim/README.md) pre-trained on ImageNet from Tensorflow Slim. We use the version resnet_v1_50_2016_08_28.tar.gz as an initialization of the face reconstruction network.
- [68-facial-landmark detector](https://drive.google.com/file/d/1KYFeTb963jg0F47sTiwqDdhBIvRlUkPa/view?usp=sharing). We use 68 facial landmarks for loss calculation during training. To make the training process reproducible, we provide a lightweight detector that produce comparable results to [the method of Bulat et al.](https://github.com/1adrianb/2D-and-3D-face-alignment). The detector is trained on [300WLP](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm), [LFW](http://vis-www.cs.umass.edu/lfw/), and [LS3D-W](https://www.adrianbulat.com/face-alignment).

### Training preparation ###

1. Download the [pre-trained weights](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk/edit) of Facenet provided by Sandberg et al., unzip it and put all files in ./weights/id_net.
2. Download the [pre-trained weights](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz) of Resnet_v1_50 provided by Tensorflow Slim, unzip it and put resnet_v1_50.ckpt in ./weights/resnet.
3. Download the [68 landmark detector](https://drive.google.com/file/d/1KYFeTb963jg0F47sTiwqDdhBIvRlUkPa/view?usp=sharing), put the file in ./network.

### Data pre-processing ###
1. To train our model with custom images，5 facial landmarks of each image are needed in advance for an image pre-alignment process. We recommend using [dlib](http://dlib.net/) or [MTCNN](https://github.com/ipazc/mtcnn). Use these public face detectors to get 5 landmarks, and save all images and corresponding landmarks in <raw_img_path>. Note that an image and its detected landmark file should have same name.
2. Align images and generate 68 landmarks as well as skin masks for training: 

```
# Run following command for data pre-processing. By default, the code uses example images in ./input and saves the processed data in ./processed_data
python preprocess_img.py

# Alternatively, you can set your custom image path and save path
python preprocess_img.py --img_path <raw_img_path> --save_path <save_path_for_processed_data>

```

### Training networks ###
1. Train the reconstruction network with the following command:
```
# By default, the code uses the data in ./processed_data as training data as well as validation data
python train.py

# Alternatively, you can set your custom data path
python train.py --data_path <custom_data_path> --val_data_path <custom_val_data_path> --model_name <custom_model_name>

```
2. Monitoring the training process via tensorboard:
```
tensorboard --logdir=result/<custom_model_name> --port=10001
```
3. Evaluating trained model:
```
python demo.py --use_pb 0 --pretrain_weights <custom_weights>.ckpt
```
Training a model with a batchsize of 16 and 200K iterations takes 20 hours on a single Tesla M40 GPU.

## Latest Update

### 2020.4 ###
The face reconstruction process is totally transferred to tensorflow version while the old version uses numpy. We have also integrated the rendering process into the framework. As a result, reconstruction images aligned with the input can be easily obtained without extra efforts. The whole process is tensorflow-based which allows gradient back-propagation for other tasks.
### 2020.6 ###
Upload a [pre-trained model](https://drive.google.com/file/d/1fPsvLKghlCK8rknb9GPiKwIq9HIqWWwV/view?usp=sharing) with white light assumption as described in the paper.

### 2020.12 ###
Upload the training code for single image face reconstruction.

## Note

1. An image pre-alignment with 5 facial landmarks is necessary before reconstruction. In our image pre-processing stage, we solve a least square problem between 5 facial landmarks on the image and 5 facial landmarks of the BFM09 average 3D face to cancel out face scales and misalignment. To get 5 facial landmarks, you can choose any open source face detector that returns them, such as [dlib](http://dlib.net/) or [MTCNN](https://github.com/ipazc/mtcnn). However, these traditional 2D detectors may return wrong landmarks under large poses which could influence the alignment result. Therefore, we recommend using [the method of Bulat et al.](https://github.com/1adrianb/2D-and-3D-face-alignment) to get facial landmarks (3D definition) with semantic consistency for large pose images. Note that our model is trained without position augmentation so that a bad alignment may lead to inaccurate reconstruction results. We put some examples in the ./input subfolder for reference.


2. We assume a [pinhole camera model](https://en.wikipedia.org/wiki/Pinhole_camera_model) for face projection. The camera is positioned at (0,0,10) (dm) in the world coordinate and points to the negative z axis. We set the camera fov to 12.6 empirically and fix it during training and inference time. Faces in canonical views are at the origin of the world coordinate and facing the positive z axis. Rotations and translations predicted by the R-Net are all with respect to the world coordinate.
<p align="center"> 
<img src="/images/camera.png" width="300">
</p>

3. The current model is trained using 3-channel (r,g,b) scene illumination instead of white light described in the paper. As a result, the gamma coefficient that controls lighting has a dimension of 27 instead of 9. 

4. We excluded ear and neck region of original BFM09 to allow the network concentrate on the face region. To see which vertices in the original model are preserved, check select_vertex_id.mat in the ./BFM subfolder. Note that index starts from 1.

5. Our model may give inferior results for images with severe perspetive distortions (e.g., some selfies). In addition, we cannot well handle faces with eyes closed due to the lack of these kind of images in the training data.
  
5. If you have any further questions, please contact Yu Deng (t-yudeng@microsoft.com) and Jiaolong Yang (jiaoyan@microsoft.com).


## Citation

Please cite the following paper if this model helps your research:

	@inproceedings{deng2019accurate,
	    title={Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set},
	    author={Yu Deng and Jiaolong Yang and Sicheng Xu and Dong Chen and Yunde Jia and Xin Tong},
	    booktitle={IEEE Computer Vision and Pattern Recognition Workshops},
	    year={2019}
	}
##
The face images on this page are from the public [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset released by MMLab, CUHK.
