import numpy as np 
import tensorflow as tf
import os

# training options

class Option():
	def __init__(self,model_name=None,is_train=True):
		#--------------------------------------------------------------------------------------
		self.is_train = is_train
		self.model_dir = 'result'
		if model_name is None:
			self.model_name = 'model_test'
		else:
			self.model_name = model_name
		self.data_path = ['./processed_data']
		self.val_data_path = ['./processed_data']

		self.model_save_path = os.path.join(self.model_dir,self.model_name)
		if self.is_train:
			if not os.path.exists(self.model_save_path):
				os.makedirs(self.model_save_path)

		self.summary_dir = os.path.join(self.model_save_path,'summary')

		self.train_summary_path = os.path.join(self.summary_dir, 'train')
		self.val_summary_path = os.path.join(self.summary_dir, 'val')
		#---------------------------------------------------------------------------------------
		# visible gpu settings
		self.config = tf.ConfigProto()
		self.config.gpu_options.visible_device_list = '0'
		self.use_pb = True
		#---------------------------------------------------------------------------------------
		# training parameters

		self.w_photo = 1.92
		self.w_lm = 1.6e-3
		self.w_id = 0.2

		self.w_reg = 3.0e-4
		self.w_ref = 5.0

		self.w_gamma = 10.0

		self.w_ex = 0.8
		self.w_tex = 1.7e-2

		self.batch_size = 16
		self.boundaries = [100000]
		lr = [1e-4,2e-5]
		self.global_step = tf.Variable(0,name='global_step',trainable = False)
		self.lr = tf.train.piecewise_constant(self.global_step,self.boundaries,lr)
		self.augment = True
		self.train_maxiter = 200000
		self.train_summary_iter = 50
		self.image_summary_iter = 200
		self.val_summary_iter = 1000
		self.save_iter = 10000
		#---------------------------------------------------------------------------------------
		# initial weights for resnet and facenet
		self.R_net_weights = os.path.join('./weights/resnet','resnet_v1_50.ckpt')
		self.Perceptual_net_weights = './weights/id_net/model-20170512-110547.ckpt-250000'
		self.pretrain_weights = os.path.join('train/model_test','iter_100000.ckpt')
