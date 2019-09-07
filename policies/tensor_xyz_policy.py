import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import tf_utils as tfu
import os
import sys
import getpass
sys.path.append("/home/{}".format(getpass.getuser()))
from discovery.test_model_loading import MUJOCO_ONLINE


class Tensor_XYZ_Policy:

	def __init__(self,
				 name,
				 env,
				 hidden_sizes=[64, 32]):

		assert 'image_observation' in env.observation_space.spaces.keys()
		self.img_obs_dim = env.observation_space.spaces['image_observation'].shape
		self.depth_obs_dim = env.observation_space.spaces['depth_observation'].shape
		self.cam_obs_dim = env.observation_space.spaces['cam_info_observation'].shape
		self.state_obs_dim = env.observation_space.spaces['state_observation'].shape[0]
		self.state_desired_dim = env.observation_space.spaces['state_desired_goal'].shape[0]
		self.flatten_tensor_dim = 256  # Need to find a way to make this variable.
		obj_size = env._env.env.sim.model.geom_size[env._env.env.sim.model.geom_name2id('puckbox')]
		puck_z = env._env.env.init_puck_z + \
				env._env.env.sim.model.geom_pos[env._env.env.sim.model.geom_name2id('puckbox')][-1]

		# self.obs_dim = self.flatten_tensor_dim + self.state_obs_dim + self.state_desired_dim
		self.act_dim = env.action_space.shape[0]
		self.hidden_sizes = hidden_sizes
		self.KEYS = env.observation_space.spaces.keys()

		# Define map3D model here
		# Writing skeleton below FOR NOW!!!!!
		name = "01_m64x64x64_p32x32_1e-3_F32_Oc_c1_s.1_Ve_d32_E32_a.8_i.2_push_and_reach_random_data_train_a35"
		checkpoint_dir_ = os.path.join("checkpoints", name)
		log_dir_ = os.path.join("logs_mujoco_offline", name)
		
		self.map3D = MUJOCO_ONLINE(obj_size,
								puck_z,
								checkpoint_dir=checkpoint_dir_,
								log_dir=log_dir_)
		self.map3D.build_main_graph()
		# self.map3D.build_graph() # Ensure no initialization or summaries are made.

		with tf.variable_scope(name):
				self.scope = tf.get_variable_scope().name
				self.build(hidden_sizes)


	def build(self, hidden_sizes):

		# CREATE TWO OBSERVATION PLACEHOLDERS : (1) for 3D tensor, (2) for other vectors
		goal_obs = tfu.get_placeholder(name="goal_obs",
							dtype=tf.float32,
							shape=[None, self.state_obs_dim + self.state_desired_dim])

		crop = tfu.get_placeholder(name="crop",
							dtype=tf.float32,
							shape=[None, 16, 16, 8, 32])

		bn = True

		with slim.arg_scope([slim.conv3d, slim.conv3d_transpose],
							activation_fn=tf.nn.relu,
							normalizer_fn=slim.batch_norm if bn else None,
							):
			d0 = 16
			dims = [d0, 2*d0, 4*d0, 8*d0]
			ksizes = [4, 4, 4, 4]
			strides = [2, 2, 2, 2]
			paddings = ['SAME'] * 4

			# ksizes[-1] = 2
			net = crop
			for i, (dim, ksize, stride, padding) in enumerate(zip(dims, ksizes, strides, paddings)):
				net = slim.conv3d(net, dim, ksize, stride=1, padding=padding)
				net = tf.nn.pool(net, [3,3,3], 'MAX', 'SAME', strides = [2,2,2])

			net = tf.layers.flatten(net)


		out = tf.concat([net, goal_obs], -1)
		for i, hidden_size in enumerate(hidden_sizes):

			out = tf.nn.tanh(tfu.dense(out,
									hidden_size,
									"policyfc%i" % (i+1),
									weight_init=tfu.normc_initializer(1.0)))

		action = tfu.dense(out,
						self.act_dim,
						"policyfinal",
						weight_init=tfu.normc_initializer(0.01))

		self.ac = action
		self._act = tfu.function([goal_obs, crop], self.ac)

		self.flatvars = tfu.GetFlat(self.get_trainable_variables())
		self.unflatvars = tfu.SetFromFlat(self.get_trainable_variables())

	def train_process_observation(self, data, idx):
		featRs = []
		for i in range(0,len(idx),16):
			datas = {key: data[key][idx[i:i+16]] for key in self.KEYS}
			featRs.append(self.map3D.forward(datas))

		featRs = np.vstack(featRs)
		data = {key: data[key][idx] for key in self.KEYS}
		ob_tensor = np.hstack([data['state_desired_goal'],
								data['state_observation']])

		#assert (featRs.shape[-1] + ob_tensor.shape[-1]) == self.obs_dim
		return featRs, ob_tensor

	def process_observation(self, ob):
		ob = {key: np.expand_dims(ob[key], axis=0) for key in ob.keys()}
		featRs = self.map3D.forward(ob)
		ob_tensor = np.concatenate((ob['state_desired_goal'],ob['state_observation']))
		return featRs, ob_tensor

	def act(self, ob):
		crop, ob = self.process_observation(ob)
		ac = self._act(ob,crop)
		return ac[0]

	def get_variables(self, scope=None):
		return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
								self.scope if scope is None else scope)

	def get_trainable_variables(self, scope=None):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
								self.scope if scope is None else scope)

	###### THESE FUNCTIONS NEED SOME WORK ######
	def __getstate__(self):
		d = {}
		d['scope'] = self.scope
		# d['obs_dim'] = self.obs_dim
		d['act_dim'] = self.act_dim
		d['hidden_sizes'] = self.hidden_dims
		d['net_params'] = self.flatvars()
		return d

	def __setstate__(self, dict_):
		self.scope = dict_['scope']
		# self.obs_dim = dict_['obs_dim']
		self.act_dim = dict_['act_space']
		self.hidden_sizes = dict_['hidden_sizes']
		self.unflatvars(dict_['net_params'])
