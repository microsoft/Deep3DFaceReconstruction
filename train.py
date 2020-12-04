import tensorflow as tf 
import numpy as np 
import os
from options import Option
from reconstruction_model import *
from data_loader import *
from utils import *
import argparse
###############################################################################################
# training stage
###############################################################################################


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# training data and validation data
def parse_args():
    desc = "Deep3DFaceReconstruction"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--data_path', type=str, default='./processed_data', help='training data folder')
    parser.add_argument('--val_data_path', type=str, default='./processed_data', help='validation data folder')
    parser.add_argument('--model_name', type=str, default='./model_test', help='model name')


    return parser.parse_args()

# initialize weights for resnet and facenet
def restore_weights_and_initialize(opt):
	var_list = tf.trainable_variables()
	g_list = tf.global_variables()

	# add batch normalization params into trainable variables 
	bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
	bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
	var_list +=bn_moving_vars

	# create saver to save and restore weights
	resnet_vars = [v for v in var_list if 'resnet_v1_50' in v.name]
	facenet_vars = [v for v in var_list if 'InceptionResnetV1' in v.name]
	saver_resnet = tf.train.Saver(var_list = resnet_vars)
	saver_facenet = tf.train.Saver(var_list = facenet_vars)

	saver = tf.train.Saver(var_list = resnet_vars + [v for v in var_list if 'fc-' in v.name],max_to_keep = 50)

	# create session
	sess = tf.InteractiveSession(config = opt.config)

	# create summary op
	train_writer = tf.summary.FileWriter(opt.train_summary_path, sess.graph)
	val_writer = tf.summary.FileWriter(opt.val_summary_path, sess.graph)

	# initialization
	tf.global_variables_initializer().run()
	tf.local_variables_initializer().run()

	saver_resnet.restore(sess,opt.R_net_weights)
	saver_facenet.restore(sess,opt.Perceptual_net_weights)

	return saver, train_writer,val_writer, sess


# main function for training
def train():

	# read BFM face model
	# transfer original BFM model to our model
	if not os.path.isfile('./BFM/BFM_model_front.mat'):
		transferBFM09()

	with tf.Graph().as_default() as graph:

		# training options
		args = parse_args()
		opt = Option(model_name=args.model_name)
		opt.data_path = [args.data_path]
		opt.val_data_path = [args.val_data_path]

		# load training data into queue
		train_iterator = load_dataset(opt)
		# create reconstruction model
		model = Reconstruction_model(opt)
		# send training data to the model
		model.set_input(train_iterator)
		# update model variables with training data
		model.step(is_train = True)
		# summarize training statistics
		model.summarize()

		# several training stattistics to be saved
		train_stat = model.summary_stat
		train_img_stat = model.summary_img
		train_op = model.train_op
		photo_error = model.photo_loss
		lm_error = model.landmark_loss
		id_error = model.perceptual_loss

		# load validation data into queue
		val_iterator = load_dataset(opt,train=False)
		# send validation data to the model
		model.set_input(val_iterator)
		# only do foward pass without updating model variables
		model.step(is_train = False)
		# summarize validation statistics
		model.summarize()
		val_stat = model.summary_stat
		val_img_stat = model.summary_img

		# initialization
		saver, train_writer,val_writer, sess = restore_weights_and_initialize(opt)

		# freeze the graph to ensure no new op will be added during training
		sess.graph.finalize()

		# training loop
		for i in range(opt.train_maxiter):
			_,ph_loss,lm_loss,id_loss = sess.run([train_op,photo_error,lm_error,id_error])
			print('Iter: %d; lm_loss: %f ; photo_loss: %f; id_loss: %f\n'%(i,np.sqrt(lm_loss),ph_loss,id_loss))
			# summarize training stats every <train_summary_iter> iterations
			if np.mod(i,opt.train_summary_iter) == 0:
				train_summary = sess.run(train_stat)
				train_writer.add_summary(train_summary,i)

			# summarize image stats every <image_summary_iter> iterations
			if np.mod(i,opt.image_summary_iter) == 0:
				train_img_summary = sess.run(train_img_stat)
				train_writer.add_summary(train_img_summary,i)

			# summarize validation stats every <val_summary_iter> iterations	
			if np.mod(i,opt.val_summary_iter) == 0:
				val_summary,val_img_summary = sess.run([val_stat,val_img_stat])
				val_writer.add_summary(val_summary,i)
				val_writer.add_summary(val_img_summary,i)

			# # save model variables every <save_iter> iterations	
			if np.mod(i,opt.save_iter) == 0:
				saver.save(sess,os.path.join(opt.model_save_path,'iter_%d.ckpt'%i))


if __name__ == '__main__':
	train()