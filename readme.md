## Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set ##

<p align="center"> 
<img src="/images/example.png">
</p>


This is an python implement of *Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set*.

The method enforces a hybrid-level weakly-supervised training to achieve accurate CNN-based face reconstruction.

## Features

### Accurate shapes
The method reconstructs faces with high accuracy. Quantitative evaluations on several benchmarks show its state-of-the-art performance. 

|Method       |FaceWareHouse|Florence|BU3DFE|
|-------------|-------------|--------|------|
|[Tewari 17]()|2.19         |-       |-     |
|[Tewari 18]()|1.84         |-       |-     |
|[Genova 18]()|-            |1.77    |-     |
|[Sela 18]()  |-            |-       |2.91  |
|[PRN]()	  |-            |-       |1.86  |
|Ours         |1.81         |1.67    |1.40  |

### High fidelity textures

### Robust

### Aligned with images

### Easy and Fast
Faces are represented with Basel Face Model 2009, which is easy for further manipulations (e.g expression transfer). ResNet-50 is used as backbone network to achieve over 50 fps (on GTX 1080) for reconstructions.


## Getting Started
### Prerequisite ###

- Python >= 3.5 (numpy, scipy, pillow, opencv)
- Tensorflow >= 1.4
- [Basel Face Model 2009 (BFM09)](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model)
- [Expression Basis (transferred from Facewarehouse by Yudong et al.)](https://github.com/Juyong/3DFace)

Optional:

- [tf mesh renderer (only for rendering reconstruction images)](https://github.com/google/tf_mesh_renderer)

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
