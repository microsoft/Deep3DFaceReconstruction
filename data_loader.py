import tensorflow as tf
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
import os
import glob
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
###############################################################################################
# data loader for training stage
###############################################################################################
def _parse_function(image_path,lm_path,mask_path):

	# input image
	x = tf.read_file(image_path)
	img = tf.image.decode_png(x, channels=3)
	img = tf.cast(img,tf.float32)
	img = img[:,:,::-1]

	# ground truth landmark
	x2 = tf.read_file(lm_path)
	lm = tf.decode_raw(x2,tf.float64)
	lm = tf.cast(lm,tf.float32)
	lm = tf.reshape(lm,[68,2])

	# skin mask
	x3 = tf.read_file(mask_path)
	mask = tf.image.decode_png(x3, channels=3)
	mask = tf.cast(mask,tf.float32)

	return img,lm,mask

def check_lm_bin(dataset,lm_path):
	if not os.path.isdir(os.path.join(dataset,'lm_bin')):
		os.makedirs(os.path.join(dataset,'lm_bin'))
		for i in range(len(lm_path)):
			lm = np.loadtxt(lm_path[i])
			lm = np.reshape(lm,[-1])
			lm.tofile(os.path.join(dataset,'lm_bin',lm_path[i].split('/')[-1].replace('txt','bin')))	

def load_dataset(opt,train=True):
	if train:
		data_path = opt.data_path
	else:
		data_path = opt.val_data_path
	image_path_all = []
	lm_path_all = []
	mask_path_all = []

	for dataset in data_path:
		image_path = glob.glob(dataset + '/' + '*.png')
		image_path.sort()
		lm_path_ = [os.path.join(dataset,'lm',f.split('/')[-1].replace('png','txt')) for f in image_path]
		lm_path_.sort()
		mask_path = [os.path.join(dataset,'mask',f.split('/')[-1]) for f in image_path]
		mask_path.sort()

		# check if landmark binary files exist
		check_lm_bin(dataset,lm_path_)

		lm_path = [os.path.join(dataset,'lm_bin',f.split('/')[-1].replace('png','bin')) for f in image_path]
		lm_path.sort()

		image_path_all += image_path
		mask_path_all += mask_path
		lm_path_all += lm_path

	dataset_num = len(image_path_all)

	dataset = tf.data.Dataset.from_tensor_slices((image_path_all,lm_path_all,mask_path_all))
	dataset = dataset. \
	apply(shuffle_and_repeat(dataset_num)). \
	apply(map_and_batch(_parse_function, opt.batch_size, num_parallel_batches=4, drop_remainder=True)). \
	apply(prefetch_to_device('/gpu:0', None)) # When using dataset.prefetch, use buffer_size=None to let it detect optimal buffer size

	inputs_iterator = dataset.make_one_shot_iterator()
	return inputs_iterator
