import tensorflow as tf 
import numpy as np
import cv2
from PIL import Image
import os
import glob
from scipy.io import loadmat,savemat

from preprocess_img import Preprocess
from load_data import *
from reconstruct_mesh import Reconstruction

def load_graph(graph_filename):
	with tf.gfile.GFile(graph_filename,'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	return graph_def

def demo():
	# input and output folder
	image_path = 'input'
	save_path = 'output'	
	img_list = glob.glob(image_path + '/' + '*.png')

	# read BFM face model
	# transfer original BFM model to our model
	if not os.path.isfile('./BFM/BFM_model_front.mat'):
		transferBFM09()

	# read face model
	facemodel = BFM()
	# read standard landmarks for preprocessing images
	lm3D = load_lm3d()
	n = 0

	# build reconstruction model
	with tf.Graph().as_default() as graph,tf.device('/cpu:0'):

		images = tf.placeholder(name = 'input_imgs', shape = [None,224,224,3], dtype = tf.float32)
		graph_def = load_graph('network/FaceReconModel.pb')
		tf.import_graph_def(graph_def,name='resnet',input_map={'input_imgs:0': images})

		# output coefficients of R-Net (dim = 257) 
		coeff = graph.get_tensor_by_name('resnet/coeff:0')

		with tf.Session() as sess:
			print('reconstructing...')
			for file in img_list:
				n += 1
				print(n)
				# load images and corresponding 5 facial landmarks
				img,lm = load_img(file,file.replace('png','txt'))
				# preprocess input image
				input_img,lm_new,transform_params = Preprocess(img,lm,lm3D)

				coef = sess.run(coeff,feed_dict = {images: input_img})

				# reconstruct 3D face with output coefficients and face model
				face_shape,face_texture,face_color,tri,face_projection,z_buffer,landmarks_2d = Reconstruction(coef,facemodel)

				# reshape outputs
				input_img = np.squeeze(input_img)
				shape = np.squeeze(face_shape, (0))
				color = np.squeeze(face_color, (0))
				landmarks_2d = np.squeeze(landmarks_2d, (0))

				# save output files
				# cropped image, which is the direct input to our R-Net
				# 257 dim output coefficients by R-Net
				# 68 face landmarks of cropped image
				savemat(os.path.join(save_path,file.split(os.path.sep)[-1].replace('.png','.mat')),{'cropped_img':input_img[:,:,::-1],'coeff':coef,'landmarks_2d':landmarks_2d,'lm_5p':lm_new})
				save_obj(os.path.join(save_path,file.split(os.path.sep)[-1].replace('.png','_mesh.obj')),shape,tri,np.clip(color,0,255)/255) # 3D reconstruction face (in canonical view)

if __name__ == '__main__':
	demo()
