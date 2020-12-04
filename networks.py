import tensorflow as tf 
from tensorflow.contrib.slim.nets import resnet_v1
slim = tf.contrib.slim
from inception_resnet_v1 import inception_resnet_v1
###############################################################################################
#Define R-Net and Perceptual-Net for 3D face reconstruction
###############################################################################################

def R_Net(inputs,is_training=True):
	#input: [Batchsize,H,W,C], 0-255, BGR image
	inputs = tf.cast(inputs,tf.float32)
	# standard ResNet50 backbone (without the last classfication FC layer)
	with slim.arg_scope(resnet_v1.resnet_arg_scope()):
		net,end_points = resnet_v1.resnet_v1_50(inputs,is_training = is_training ,reuse = tf.AUTO_REUSE)

	# Modified FC layer with 257 channels for reconstruction coefficients
	net_id = slim.conv2d(net, 80, [1, 1],
		activation_fn=None,
		normalizer_fn=None,
		weights_initializer = tf.zeros_initializer(),
		scope='fc-id')
	net_ex = slim.conv2d(net, 64, [1, 1],
		activation_fn=None,
		normalizer_fn=None,
		weights_initializer = tf.zeros_initializer(),
		scope='fc-ex')
	net_tex = slim.conv2d(net, 80, [1, 1],
		activation_fn=None,
		normalizer_fn=None,
		weights_initializer = tf.zeros_initializer(),
		scope='fc-tex')
	net_angles = slim.conv2d(net, 3, [1, 1],
		activation_fn=None,
		normalizer_fn=None,
		weights_initializer = tf.zeros_initializer(),
		scope='fc-angles')
	net_gamma = slim.conv2d(net, 27, [1, 1],
		activation_fn=None,
		normalizer_fn=None,
		weights_initializer = tf.zeros_initializer(),
		scope='fc-gamma')
	net_t_xy = slim.conv2d(net, 2, [1, 1],
		activation_fn=None,
		normalizer_fn=None,
		weights_initializer = tf.zeros_initializer(),
		scope='fc-XY')
	net_t_z = slim.conv2d(net, 1, [1, 1],
		activation_fn=None,
		normalizer_fn=None,
		weights_initializer = tf.zeros_initializer(),
		scope='fc-Z')

	net_id = tf.squeeze(net_id, [1,2], name='fc-id/squeezed')
	net_ex = tf.squeeze(net_ex, [1,2], name='fc-ex/squeezed')
	net_tex = tf.squeeze(net_tex, [1,2],name='fc-tex/squeezed')
	net_angles = tf.squeeze(net_angles,[1,2], name='fc-angles/squeezed')
	net_gamma = tf.squeeze(net_gamma,[1,2], name='fc-gamma/squeezed')
	net_t_xy = tf.squeeze(net_t_xy,[1,2], name='fc-XY/squeezed')
	net_t_z = tf.squeeze(net_t_z,[1,2], name='fc-Z/squeezed')

	net_ = tf.concat([net_id,net_ex,net_tex,net_angles,net_gamma,net_t_xy,net_t_z], axis = 1)

	return net_


def Perceptual_Net(input_imgs):
    #input_imgs: [Batchsize,H,W,C], 0-255, BGR image

    input_imgs = tf.reshape(input_imgs,[-1,224,224,3])
    input_imgs = tf.cast(input_imgs,tf.float32)
    input_imgs = tf.clip_by_value(input_imgs,0,255)
    input_imgs = (input_imgs - 127.5)/128.0

    #standard face-net backbone
    batch_norm_params = {
    'decay': 0.995,
    'epsilon': 0.001,
    'updates_collections': None}

    with slim.arg_scope([slim.conv2d, slim.fully_connected],weights_initializer=slim.initializers.xavier_initializer(), 
        weights_regularizer=slim.l2_regularizer(0.0),
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params):
        feature_128,_ = inception_resnet_v1(input_imgs, bottleneck_layer_size=128, is_training=False, reuse=tf.AUTO_REUSE)

    # output the last FC layer feature(before classification) as identity feature
    return feature_128