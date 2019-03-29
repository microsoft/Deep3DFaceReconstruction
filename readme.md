## Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set ##

<p align="center"> 
<img src="/images/movie_6.gif">
<img src="/images/movie_17.gif">
<img src="/images/movie_23.gif">
</p>


This is an python implement of [*Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set*](https://arxiv.org/abs/1903.08527).

The method enforces a hybrid-level weakly-supervised training to achieve accurate CNN-based face reconstruction.

## Features

### ● Accurate shapes
The method reconstructs faces with high accuracy. Quantitative evaluations (shape errors in mm) on several benchmarks show its state-of-the-art performance:

<p align="center">
|Method|FaceWareHouse|Florence|BU3DFE|
|:---:|:---:|:---:|:---:|
|[Tewari et al. 17](https://arxiv.org/abs/1703.10580)</center>|2.19±0.54|-|-|
|[Tewari et al. 18](https://arxiv.org/abs/1712.02859)|1.84±0.38|-|-|
|[Genova et al. 18](https://arxiv.org/abs/1806.06098)|-|1.77±0.53|-|
|[Sela et al. 17](https://arxiv.org/abs/1703.10131)|-|-|2.91±0.60|
|[PRN 18](https://arxiv.org/abs/1803.07835)|-|-|1.86±0.47|
|Ours|**1.81±0.50**|**1.67±0.50**|**1.40±0.31**|
</p>


### ● High fidelity textures
The method is able to produce face textures with high identity similarity to input images. Lighting information is also disentangled to get a pure albedo.
<p align="center"> 
<img src="/images/albedo.png">
</p>

### ● Robust
The method can provide reasonable results under extreme conditions such as large pose and occlusions.
<p align="center"> 
<img src="/images/extreme.png">
</p>

### ● Aligned with images
Our method aligns reconstruction faces with input images. It provides face pose information and 68 facial landmarks which are useful for other tasks.
<p align="center"> 
<img src="/images/alignment.png">
</p>

### ● Easy and Fast
Faces are represented with Basel Face Model 2009, which is easy for further manipulations (e.g expression transfer). ResNet-50 is used as backbone network to achieve over 50 fps (on GTX 1080) for reconstructions.


## Getting Started
### Prerequisite ###

- Python >= 3.5 (numpy, scipy, pillow, opencv)
- Tensorflow >= 1.4
- [Basel Face Model 2009 (BFM09)](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model)
- [Expression Basis (transferred from Facewarehouse by Yudong et al.)](https://github.com/Juyong/3DFace)


### Usage ###

1. Clone the repository 

```bash
git clone https://github.com/Microsoft/Deep3DFaceReconstruction
cd Deep3DFaceReconstruction
```

2. Download the BFM09 model and put "01_MorphableModel.mat" into ./BFM subfolder.

3. Download the Expression Basis (which is inside "CoarseData.zip"), and put "Exp_Pca.bin" into ./BFM subfolder.

4. Download the trained model at [GoogleDrive](https://drive.google.com/file/d/1RSEkXwF5BGelvBaIJFtKIxjUcR5ULSK0/view?usp=sharing), and put it into ./network subfolder.

5. Run the demo code.

```
python demo.py
```

6. To check the results, see ./output subfolder which contains:
	- "xxx.mat" : consists of cropped input image, corresponding 5p and 68p landmarks, and output coefficients of R-Net.
	- "xxx_mesh.obj" : Reconstructed 3D face mesh in canonical view (best viewed in MeshLab).

### Tips ###

1. The model is trained without augmentation so that a pre-alignment with 5 facial landmarks is necessary. We put some examples in the ./input subfolder for reference.
2. Current model is trained under the assumption of 3-channel scene illumination (instead of monochromatic lights described in the paper).    


## Citation

Please cite the following paper if this model helps your research:

	@misc{deng2019accurate,
	    title={Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set},
	    author={Yu Deng and Jiaolong Yang and Sicheng Xu and Dong Chen and Yunde Jia and Xin Tong},
	    year={2019},
	    eprint={1903.08527},
	    archivePrefix={arXiv},
	    primaryClass={cs.CV}
	}