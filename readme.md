## Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set ##

<p align="center"> 
<img src="/images/example.gif">
</p>


This is a python implementation of the following paper:

Y. Deng, J. Yang, S. Xu, D. Chen, Y. Jia, and X. Tong, [Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set](https://arxiv.org/abs/1903.08527), IEEE Computer Vision and Pattern Recognition Workshop (CVPRW) on Analysis and Modeling of Faces and Gestures (AMFG), 2019. (**_Best Paper Award!_**)

The method enforces a hybrid-level weakly-supervised training to for CNN-based 3D face reconstruction. It is fast, accurate, and robust to pose and occlussions. It achieves state-of-the-art performance on multiple datasets such as FaceWarehouse, MICC Florence and BU-3DFE.

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
### System Requirements ###

- Reconstructions can be done on both Windows and Linux. However, we suggest running on Linux because the rendering process is only supported on Linux currently. If you wish to run on Windows, you have to comment out the rendering part. 
- Python >= 3.5 (numpy, scipy, pillow, opencv).
- Tensorflow 1.4 ~ 1.12.
- [Basel Face Model 2009 (BFM09)](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model). 
- [Expression Basis (transferred from Facewarehouse by Guo et al.)](https://github.com/Juyong/3DFace). The original BFM09 model does not handle expression variations so extra expression basis are needed. 
- [tf mesh renderer](https://github.com/google/tf_mesh_renderer).  We use the library to render reconstruction images. Install the library via ```pip install mesh_renderer```. Or you can follow the instruction of tf mesh render to install it using Bazel.  Note that current rendering tool does not support tensorflow version higher than 1.13 and can only be used on Linux.
### Usage ###

1. Clone the repository 

```bash
git clone https://github.com/Microsoft/Deep3DFaceReconstruction
cd Deep3DFaceReconstruction
```

2. Download the Basel Face Model. Due to the license agreement of Basel Face Model, you have to download the BFM09 model after submitting an application on its [home page](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads). After getting the access to BFM data, download "01_MorphableModel.mat" and put it into ./BFM subfolder.

3. Download the Expression Basis provided by [Guo et al.](https://github.com/Juyong/3DFace) You can find a link named "CoarseData" in the first row of Introduction part in their repository. Download and unzip the Coarse_Dataset.zip. Put "Exp_Pca.bin" into ./BFM subfolder. The expression basis are constructed using [Facewarehouse](kunzhou.net/zjugaps/facewarehouse/) data and transferred to BFM topology.

4. Download the trained [reconstruction network](https://drive.google.com/file/d/1RSEkXwF5BGelvBaIJFtKIxjUcR5ULSK0/view?usp=sharing), and put it into ./network subfolder.

5. Run the demo code.

```
python demo.py
```

6. ./input subfolder contains several test images and ./output subfolder stores their reconstruction results. For each input test image, two output files can be obtained after running the demo code:
	- "xxx.mat" : 
		- cropped_img: an RGB image after alignment, which is the input to the R-Net
		- recon_img: an RGBA reconstruction image aligned with the input image.
		- coeff: output coefficients of R-Net.
		- face_shape: vertex positions of 3D face in the world coordinate.
		- face_texture: vertex texture of 3D face, which excludes lighting effect.
		- face_color: vertex color of 3D face, which takes lighting into consideration.
		- lm\_68p: 68 2D facial landmarks derived from the reconstructed 3D face. The landmarks are aligned with cropped_img.
		- lm\_5p: 5 detected landmarks aligned with cropped_img. 
	- "xxx_mesh.obj" : 3D face mesh in the world coordinate (best viewed in MeshLab).

### Latest Update (2020.4) ###
The face reconstruction process is totally transferred to tensorflow version while the old version uses numpy. We have also integrated the rendering process into the framework. As a result, reconstruction images aligned with the input can be easily obtained without extra efforts. The whole process is tensorflow-based which allows gradient back-propagation for other tasks.

### Note: ###

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
