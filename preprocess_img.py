import numpy as np 
from scipy.io import loadmat,savemat
from PIL import Image
from skin import skinmask
import argparse
from utils import *
import os
import glob
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#calculating least square problem
def POS(xp,x):
	npts = xp.shape[1]

	A = np.zeros([2*npts,8])

	A[0:2*npts-1:2,0:3] = x.transpose()
	A[0:2*npts-1:2,3] = 1

	A[1:2*npts:2,4:7] = x.transpose()
	A[1:2*npts:2,7] = 1;

	b = np.reshape(xp.transpose(),[2*npts,1])

	k,_,_,_ = np.linalg.lstsq(A,b)

	R1 = k[0:3]
	R2 = k[4:7]
	sTx = k[3]
	sTy = k[7]
	s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
	t = np.stack([sTx,sTy],axis = 0)

	return t,s

# resize and crop images
def resize_n_crop_img(img,lm,t,s,target_size = 224.):
	w0,h0 = img.size
	w = (w0/s*102).astype(np.int32)
	h = (h0/s*102).astype(np.int32)
	img = img.resize((w,h),resample = Image.BICUBIC)

	left = (w/2 - target_size/2 + float((t[0] - w0/2)*102/s)).astype(np.int32)
	right = left + target_size
	up = (h/2 - target_size/2 + float((h0/2 - t[1])*102/s)).astype(np.int32)
	below = up + target_size

	img = img.crop((left,up,right,below))
	img = np.array(img)
	img = img[:,:,::-1] #RGBtoBGR
	img = np.expand_dims(img,0)
	lm = np.stack([lm[:,0] - t[0] + w0/2,lm[:,1] - t[1] + h0/2],axis = 1)/s*102
	lm = lm - np.reshape(np.array([(w/2 - target_size/2),(h/2-target_size/2)]),[1,2])

	return img,lm


# resize and crop input images before sending to the R-Net
def align_img(img,lm,lm3D):

	w0,h0 = img.size

	# change from image plane coordinates to 3D sapce coordinates(X-Y plane)
	lm = np.stack([lm[:,0],h0 - 1 - lm[:,1]], axis = 1)

	# calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
	t,s = POS(lm.transpose(),lm3D.transpose())

	# processing the image
	img_new,lm_new = resize_n_crop_img(img,lm,t,s)
	lm_new = np.stack([lm_new[:,0],223 - lm_new[:,1]], axis = 1)
	trans_params = np.array([w0,h0,102.0/s,t[0],t[1]])

	return img_new,lm_new,trans_params

# detect 68 face landmarks for aligned images
def get_68landmark(img,detector,sess):

	input_img = detector.get_tensor_by_name('input_imgs:0')
	lm = detector.get_tensor_by_name('landmark:0')

	landmark = sess.run(lm,feed_dict={input_img:img})
	landmark = np.reshape(landmark,[68,2])
	landmark = np.stack([landmark[:,1],223-landmark[:,0]],axis=1)

	return landmark

# get skin attention mask for aligned images
def get_skinmask(img):

	img = np.squeeze(img,0)
	skin_img = skinmask(img)
	return skin_img

def parse_args():
    desc = "Data preprocessing for Deep3DRecon."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--img_path', type=str, default='./input', help='original images folder')
    parser.add_argument('--save_path', type=str, default='./processed_data', help='custom path to save proccessed images and labels')


    return parser.parse_args()

# training data pre-processing
def preprocessing():

	args = parse_args()
	image_path = args.img_path
	save_path = args.save_path
	if not os.path.isdir(save_path):
		os.makedirs(save_path)
	if not os.path.isdir(os.path.join(save_path,'lm')):
		os.makedirs(os.path.join(save_path,'lm'))
	if not os.path.isdir(os.path.join(save_path,'lm_bin')):
		os.makedirs(os.path.join(save_path,'lm_bin'))
	if not os.path.isdir(os.path.join(save_path,'mask')):
		os.makedirs(os.path.join(save_path,'mask'))

	img_list = sorted(glob.glob(image_path + '/' + '*.png'))
	img_list += sorted(glob.glob(image_path + '/' + '*.jpg'))

	lm3D = load_lm3d()

	with tf.Graph().as_default() as graph, tf.device('/gpu:0'):
		lm_detector = load_graph(os.path.join('network','landmark68_detector.pb'))
		tf.import_graph_def(lm_detector,name='')
		sess = tf.InteractiveSession()

		for file in img_list:

			print(file)
			name = file.split('/')[-1].replace('.png','').replace('.jpg','')
			img,lm5p = load_img(file,file.replace('png','txt').replace('jpg','txt'))
			img_align,_,_ = align_img(img,lm5p,lm3D)  # [1,224,224,3] BGR image

			lm68p = get_68landmark(img_align,graph,sess)
			lm68p = lm68p.astype(np.float64)
			skin_mask = get_skinmask(img_align)

			Image.fromarray(img_align.squeeze(0)[:,:,::-1].astype(np.uint8),'RGB').save(os.path.join(save_path,name+'.png'))
			Image.fromarray(skin_mask.astype(np.uint8)).save(os.path.join(save_path,'mask',name+'.png'))

			np.savetxt(os.path.join(save_path,'lm',name+'.txt'),lm68p)
			lm_bin = np.reshape(lm68p,[-1])
			lm_bin.tofile(os.path.join(save_path,'lm_bin',name+'.bin'))	

if __name__ == '__main__':
	preprocessing()