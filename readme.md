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


### ● High fidelity textures
The method produces high fidelity face textures meanwhile preserves identity information of input images. Scene illumination is also disentangled to guarantee a pure albedo.
<p align="center"> 
<img src="/images/albedo.png">
</p>

### ● Robust
The method can provide reasonable results under extreme conditions such as large pose and occlusions.
<p align="center"> 
<img src="/images/extreme.png">
</p>

### ● Aligned with images
Our method aligns reconstruction faces with input images. It provides face pose estimation and 68 facial landmarks which are useful for other tasks. We conduct an experiment on AFLW_2000 dataset (NME) to evaluate the performance, as  is shown in the table below:
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
### Prerequisite ###

- Python >= 3.5 (numpy, scipy, pillow, opencv)
- Tensorflow >= 1.4
- [Basel Face Model 2009 (BFM09)](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model)
- [Expression Basis (transferred from Facewarehouse by Guo et al.)](https://github.com/Juyong/3DFace)

Optional:

- [tf mesh renderer](https://github.com/google/tf_mesh_renderer) (We use it as renderer while training. Can be used at test stage too. Only on Linux.)


### Usage ###

1. Clone the repository 

```bash
git clone https://github.com/Microsoft/Deep3DFaceReconstruction
cd Deep3DFaceReconstruction
```

2. Download the BFM09 model and put "01_MorphableModel.mat" into ./BFM subfolder.

3. Download the Expression Basis provided by Guo (You can find a link named CoarseData in the first row of Introduction part in their repository. Download and unzip the Coarse\_Dataset.zip), and put "Exp_Pca.bin" into ./BFM subfolder.

4. Download the trained model at [GoogleDrive](https://drive.google.com/file/d/1RSEkXwF5BGelvBaIJFtKIxjUcR5ULSK0/view?usp=sharing), and put it into ./network subfolder.

5. Run the demo code.

```
python demo.py
```

6. To check the results, see ./output subfolder which contains:
	- "xxx.mat" : consists of cropped input image, corresponding 5p and 68p landmarks, and output coefficients of R-Net.
	- "xxx_mesh.obj" : 3D face mesh in canonical view (best viewed in MeshLab).

### Tips ###

1. The model is trained without augmentation so that a pre-alignment with 5 facial landmarks is necessary. We put some examples in the ./input subfolder for reference.

2. Current model is trained under the assumption of 3-channel scene illumination (instead of white light described in the paper).  

3. We exclude ear and neck region of original BFM09. To see which vertices are preserved, check select_vertex_id.mat in the ./BFM subfolder. Note that index starts from 1.
  
4. If you have any questions, please contact Yu Deng (v-denyu@microsoft.com) or Jiaolong Yang (jiaoyan@microsoft.com).


## Citation

Please cite the following paper if this model helps your research:

	@inproceedings{deng2019accurate,
	    title={Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set},
	    author={Yu Deng and Jiaolong Yang and Sicheng Xu and Dong Chen and Yunde Jia and Xin Tong},
	    booktitle={IEEE Computer Vision and Pattern Recognition Workshops},
	    year={2019}
	}
