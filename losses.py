import tensorflow as tf
from scipy.io import loadmat,savemat
###############################################################################################
# Define losses for training
###############################################################################################

# photometric loss
# input_imgs and render_imgs are [batchsize,h,w,3] BGR images
# img_mask are [batchsize,h,w,1] attention masks
def Photo_loss(input_imgs,render_imgs,img_mask):

	input_imgs = tf.cast(input_imgs,tf.float32)

	# img_mask = tf.squeeze(img_mask,3)
	img_mask = tf.stop_gradient(img_mask[:,:,:,0])

	# photo loss with skin attention
	photo_loss = tf.sqrt(tf.reduce_sum(tf.square(input_imgs - render_imgs),axis = 3))*img_mask/255
	photo_loss = tf.reduce_sum(photo_loss) / tf.maximum(tf.reduce_sum(img_mask),1.0)

	return photo_loss

# perceptual loss
# id_feature and id_label are [batchsize, c] identity features for reconstruction images and input images
def Perceptual_loss(id_feature,id_label):
	id_feature = tf.nn.l2_normalize(id_feature, dim = 1)
	id_label = tf.nn.l2_normalize(id_label, dim = 1)
	# cosine similarity
	sim = tf.reduce_sum(id_feature*id_label,1)
	loss = tf.reduce_sum(tf.maximum(0.0,1.0 - sim))/tf.cast(tf.shape(id_feature)[0],tf.float32)

	return loss

# landmark loss
# landmark_p and landmark_label are [batchsize, 68, 2] landmark projections for reconstruction images and input images
def Landmark_loss(landmark_p,landmark_label):

	# we set higher weights for landmarks around the mouth and nose regions
	landmark_weight = tf.concat([tf.ones([1,28]),20*tf.ones([1,3]),tf.ones([1,29]),20*tf.ones([1,8])],axis = 1)
	landmark_weight = tf.tile(landmark_weight,[tf.shape(landmark_p)[0],1])

	landmark_loss = tf.reduce_sum(tf.reduce_sum(tf.square(landmark_p-landmark_label),2)*landmark_weight)/(68.0*tf.cast(tf.shape(landmark_p)[0],tf.float32))

	return landmark_loss

# coefficient regularization to ensure plausible 3d faces
def Regulation_loss(id_coeff,ex_coeff,tex_coeff,opt):
	w_ex = opt.w_ex
	w_tex = opt.w_tex

	regulation_loss = tf.nn.l2_loss(id_coeff) + w_ex * tf.nn.l2_loss(ex_coeff) + w_tex * tf.nn.l2_loss(tex_coeff)
	regulation_loss = 2*regulation_loss/ tf.cast(tf.shape(id_coeff)[0],tf.float32)

	return regulation_loss 

# albedo regularization to ensure an uniform skin albedo
def Reflectance_loss(face_texture,facemodel):
	skin_mask = facemodel.skin_mask
	skin_mask = tf.reshape(skin_mask,[1,tf.shape(skin_mask)[0],1])

	texture_mean = tf.reduce_sum(face_texture*skin_mask,1)/tf.reduce_sum(skin_mask)
	texture_mean = tf.expand_dims(texture_mean,1)

	# minimize texture variance for pre-defined skin region  
	reflectance_loss = tf.reduce_sum(tf.square((face_texture - texture_mean)*skin_mask/255.0))/(tf.cast(tf.shape(face_texture)[0],tf.float32)*tf.reduce_sum(skin_mask))

	return reflectance_loss

# gamma regularization to ensure a nearly-monochromatic light
def Gamma_loss(gamma):
	gamma = tf.reshape(gamma,[-1,3,9])
	gamma_mean = tf.reduce_mean(gamma,1, keep_dims = True)

	gamma_loss = tf.reduce_mean(tf.square(gamma - gamma_mean))

	return gamma_loss