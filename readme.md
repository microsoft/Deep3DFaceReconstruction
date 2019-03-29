## Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set ##

This is an python implement of *Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set*.

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
