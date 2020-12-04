import tensorflow as tf
import face_decoder
import networks
import losses
from utils import *
###############################################################################################
# model for single image face reconstruction
###############################################################################################
class Reconstruction_model():
	# initialization
	def __init__(self,opt):
		self.Face3D = face_decoder.Face3D() #analytic 3D face object
		self.opt = opt # training options
		self.Optimizer = tf.train.AdamOptimizer(learning_rate = opt.lr) # optimizer

	# load input data from queue
	def set_input(self,input_iterator):
		self.imgs,self.lm_labels,self.attention_masks = input_iterator.get_next()

	# forward process of the model
	def forward(self,is_train = True):

		with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
			self.coeff = networks.R_Net(self.imgs,is_training=is_train)

			self.Face3D.Reconstruction_Block(self.coeff,self.opt)

			self.id_labels = networks.Perceptual_Net(self.imgs)
			self.id_features = networks.Perceptual_Net(self.Face3D.render_imgs)

			self.photo_loss = losses.Photo_loss(self.imgs,self.Face3D.render_imgs,self.Face3D.img_mask_crop*self.attention_masks)
			self.landmark_loss = losses.Landmark_loss(self.Face3D.landmark_p,self.lm_labels)
			self.perceptual_loss = losses.Perceptual_loss(self.id_features,self.id_labels)

			self.reg_loss = losses.Regulation_loss(self.Face3D.id_coeff,self.Face3D.ex_coeff,self.Face3D.tex_coeff,self.opt)
			self.reflect_loss = losses.Reflectance_loss(self.Face3D.face_texture,self.Face3D.facemodel)
			self.gamma_loss = losses.Gamma_loss(self.Face3D.gamma)


			self.loss = self.opt.w_photo*self.photo_loss + self.opt.w_lm*self.landmark_loss + self.opt.w_id*self.perceptual_loss\
			+ self.opt.w_reg*self.reg_loss + self.opt.w_ref*self.reflect_loss + self.opt.w_gamma*self.gamma_loss

	# backward process
	def backward(self,is_train = True):
		if is_train:
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			var_list = tf.trainable_variables()
			update_var_list = [v for v in var_list if 'resnet_v1_50' in v.name or 'fc-' in v.name]
			grads = tf.gradients(self.loss,update_var_list)
			# get train_op with update_ops to ensure updating for bn parameters
			with tf.control_dependencies(update_ops):
				self.train_op = self.Optimizer.apply_gradients(zip(grads,update_var_list),global_step = self.opt.global_step)

		# if not training stage, avoid updating variables 
		else:
			pass

	# forward and backward
	def step(self, is_train = True):
		with tf.variable_scope(tf.get_variable_scope()) as scope:
			self.forward(is_train = is_train)
		self.backward(is_train = is_train)

	# statistics summarization
	def summarize(self):

		# scalar and histogram stats
		stat = [
		tf.summary.scalar('reflect_error',self.reflect_loss),
		tf.summary.scalar('gamma_error',self.gamma_loss),
		tf.summary.scalar('id_sim_error',self.perceptual_loss),
		tf.summary.scalar('lm_error',tf.sqrt(self.landmark_loss)),
		tf.summary.scalar('photo_error',self.photo_loss),
		tf.summary.scalar('train_error',self.loss),
		tf.summary.histogram('id_coeff',self.Face3D.id_coeff),
		tf.summary.histogram('ex_coeff',self.Face3D.ex_coeff),
		tf.summary.histogram('tex_coeff',self.Face3D.tex_coeff)]

		self.summary_stat = tf.summary.merge(stat)
		# combine face region of reconstruction images with input images
		render_imgs = self.Face3D.render_imgs[:,:,:,::-1]*self.Face3D.img_mask + tf.cast(self.imgs[:,:,:,::-1],tf.float32)*(1-self.Face3D.img_mask)
		render_imgs = tf.clip_by_value(render_imgs,0,255)
		render_imgs = tf.cast(render_imgs,tf.uint8)
		# image stats
		img_stat = [tf.summary.image('imgs',tf.concat([tf.cast(self.imgs[:,:,:,::-1],tf.uint8),render_imgs],axis = 2), max_outputs = 8)]
		self.summary_img = tf.summary.merge(img_stat) 