import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorflow as tf
import numpy as np
import policies.tf_utils as tfu
import process_mujoco_inputs as pmi 
import hyperparams_new as hyp 

OBS_KEYS = [	
			# 'observation',
			# 'observation_with_orientation',
			# 'desired_goal',
			# 'achieved_goal',
			'observation_abs',
			'desired_goal_abs',
			# 'achieved_goal_abs',
		]


KEYS = [	'observation',
			'observation_with_orientation',
			'observation_abs',
			'desired_goal',
			'desired_goal_abs',
			'achieved_goal',
			'achieved_goal_abs',
			'image_observation',
			'object_pose',
			# 'object_pos',
			# 'image_desired_goal', 
			# 'image_achieved_goal', 
			'depth_observation',
			# 'depth_desired_goal', 
			'cam_info_observation',
			]
class PointNet_XYZ_Policy:

	def __init__(self,
				 name,
				 env,
				 is_training,
				 hidden_sizes=[64, 64, 64]):
		self.obs_dim = sum([env.observation_space.spaces[key].shape[-1] for key in OBS_KEYS])
		self.act_dim = env.action_space.shape[0]
		self.hidden_sizes = hidden_sizes
		self.KEYS = KEYS
		
		with tf.variable_scope(name):
			self.scope = tf.get_variable_scope().name
			self.build(hidden_sizes, is_training)

	# Use a simple pointnet model to extract features
	# From https://github.com/charlesq34/pointnet/blob/master/models/pointnet_cls_basic.py
	def pointnet(self, point_cloud, is_training, bn_decay=None):
	    """ Classification PointNet, input is BxNx3, output Bx40 """
	    batch_size = point_cloud.get_shape()[0].value
	    num_point = point_cloud.get_shape()[1].value
	    end_points = {}
	    input_image = tf.expand_dims(point_cloud, -1)
	    
	    # Point functions (MLP implemented as conv2d)
	    net = tfu.conv2d_bn(input_image, 64, [1,3],
	                         padding='VALID', stride=[1,1],
	                         bn=True, is_training=is_training,
	                         scope='conv1', bn_decay=bn_decay)
	    net = tfu.conv2d_bn(net, 64, [1,1],
	                         padding='VALID', stride=[1,1],
	                         bn=True, is_training=is_training,
	                         scope='conv2', bn_decay=bn_decay)
	    net = tfu.conv2d_bn(net, 64, [1,1],
	                         padding='VALID', stride=[1,1],
	                         bn=True, is_training=is_training,
	                         scope='conv3', bn_decay=bn_decay)
	    net = tfu.conv2d_bn(net, 128, [1,1],
	                         padding='VALID', stride=[1,1],
	                         bn=True, is_training=is_training,
	                         scope='conv4', bn_decay=bn_decay)
	    net = tfu.conv2d_bn(net, 1024, [1,1],
	                         padding='VALID', stride=[1,1],
	                         bn=True, is_training=is_training,
	                         scope='conv5', bn_decay=bn_decay)

	    # Symmetric function: max pooling
	    net = tfu.max_pool2d(net, [num_point,1],
	                             padding='VALID', scope='maxpool')
	    # MLP on global point cloud vector
	    net = tf.reshape(net, [hyp.B, 1024])

	    net = tfu.dense_bn(net, 512, bn=True, is_training=is_training,
	                                  scope='fc1', bn_decay=bn_decay)
	    net = tfu.dense_bn(net, 256, bn=True, is_training=is_training,
	                                  scope='fc2', bn_decay=bn_decay)
	    net = tfu.dropout(net, keep_prob=0.7, is_training=is_training,
	                          scope='dp1')
	    net = tfu.dense_bn(net, 40, activation_fn=None, scope='fc3')

	    return net, end_points

	def build(self, hidden_sizes, is_training):
		self.ob = tfu.get_placeholder(name="ob",
							dtype=tf.float32,
							shape=[hyp.B, self.obs_dim])
		
		# preprocess inputs so that this point cloud is cropped
		self.pointcloud = tfu.get_placeholder(name="pointcloud",
					dtype=tf.float32,
					shape=[None, hyp.S, hyp.V, 3])
		
		pointnet_feats = []
		with tf.variable_scope("pointnet", reuse=tf.AUTO_REUSE):
			for s in range(hyp.S):
				feats, _ = self.pointnet(self.pointcloud[:, s], is_training)
				pointnet_feats.append(feats)

		pointnet_feats_flattened = tf.concat((pointnet_feats), axis=1)
		out = tf.concat((pointnet_feats_flattened, self.ob), axis=1)

		for i, hidden_size in enumerate(hidden_sizes):
			out = tfu.dense_bn(out, hidden_size, is_training=is_training,
									scope="policyfc%i" % (i+1))

		# Use linear activation at the output layer
		action = tfu.dense_bn(out, self.act_dim, scope="policyfinal", activation_fn=None)

		self.ac = action
		self._act = tfu.function([self.ob], self.ac)

		self.flatvars = tfu.GetFlat(self.get_trainable_variables())
		self.unflatvars = tfu.SetFromFlat(self.get_trainable_variables())

	def train_process_observation(self, data, idx):
		data = {key: data[key][idx] for key in data.keys()}
		env_ob = [data[key] for key in OBS_KEYS]
		env_ob = np.hstack(env_ob)
		#assert env_ob.shape[-1] == self.obs_dim
		batch_dict, _ = pmi.get_inputs(data, data['puck_zs'])
		feed = {}
		feed.update({self.pointcloud:batch_dict['xyz_camXs']})
		feed.update({self.ob:env_ob})
		return feed

	def process_observation(self, ob):
		ob = [ob[key] for key in self.KEYS]
		ob = np.concatenate(ob)
		return ob

	def act(self, ob):
		ob = self.process_observation(ob)
		ac = self._act(ob[None])
		return ac[0]

	def get_variables(self, scope=None):
		return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
								self.scope if scope is None else scope)

	def get_trainable_variables(self, scope=None):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
								self.scope if scope is None else scope)

	def __getstate__(self):
		d = {}
		d['scope'] = self.scope
		d['obs_dim'] = self.obs_dim
		d['act_dim'] = self.act_dim
		d['hidden_sizes'] = self.hidden_dims
		d['net_params'] = self.flatvars()
		return d

	def __setstate__(self, dict_):
		self.scope = dict_['scope']
		self.obs_dim = dict_['obs_dim']
		self.act_dim = dict_['act_space']
		self.hidden_sizes = dict_['hidden_sizes']
		self.unflatvars(dict_['net_params'])