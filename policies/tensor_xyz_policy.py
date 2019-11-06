import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import tf_utils as tfu
import os
import sys
import getpass
sys.path.append("/home/{}".format(getpass.getuser()))
from discovery.test_model_loading import MUJOCO_ONLINE
import discovery.backend.mujoco_online_inputs as mujoco_online_inputs 


class Tensor_XYZ_Policy:

	def __init__(self,
				 name,
				 env,
				 hidden_sizes=[64, 32]):

		assert 'image_observation' in env.observation_space.spaces.keys()
		self.img_obs_dim = env.observation_space.spaces['image_observation'].shape
		self.depth_obs_dim = env.observation_space.spaces['depth_observation'].shape
		self.cam_obs_dim = env.observation_space.spaces['cam_info_observation'].shape
		self.state_obs_dim = env.observation_space.spaces['achieved_goal'].shape[0]
		self.state_desired_dim = env.observation_space.spaces['desired_goal'].shape[0]
		self.flatten_tensor_dim = 256  # Need to find a way to make this variable.

		# self.obs_dim = self.flatten_tensor_dim + self.state_obs_dim + self.state_desired_dim
		self.act_dim = env.action_space.shape[0]
		self.hidden_sizes = hidden_sizes
		self.KEYS = env.observation_space.spaces.keys()

		# Define map3D model here
		# Writing skeleton below FOR NOW!!!!!
		name = "01_m64x64x64_p32x32_1e-3_F32_Oc_c1_s.1_Ve_d32_E32_a.8_i.2_push_and_reach_random_data_train_a35"
		checkpoint_dir_ = os.path.join("checkpoints", name)
		log_dir_ = os.path.join("logs_mujoco_offline", name)
		self.map3D = MUJOCO_ONLINE(
								# obj_size,
								# puck_z,
								checkpoint_dir=checkpoint_dir_,
								log_dir=log_dir_)
		self.map3D.build_main_graph()
		# self.map3D.build_graph() # Ensure no initialization or summaries are made.

		with tf.variable_scope(name):
				self.scope = tf.get_variable_scope().name
				self.build(hidden_sizes)


	def build(self, hidden_sizes):
		# import ipdb;ipdb.set_trace()

		# CREATE TWO OBSERVATION PLACEHOLDERS : (1) for 3D tensor, (2) for other vectors
		self.goal_obs = tfu.get_placeholder(name="goal_obs",
							dtype=tf.float32,
							shape=[8, self.state_obs_dim + self.state_desired_dim])

		graph = tf.get_default_graph()
		crop = graph.get_tensor_by_name("crop_result:0")
		# crop = tfu.get_placeholder(name="crop",
		# 					dtype=tf.float32,
		# 					shape=[None, 16, 16, 8, 32])

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


		out = tf.concat([net, self.goal_obs], -1)


		# batch = 8
		# lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=96, forget_bias=1.0, state_is_tuple=True)
		# init_state = lstm_cell.zero_state(batch,dtype=tf.float32)
		# out = tf.expand_dims(out, axis=1)
		# outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, out, dtype=tf.float32, time_major=False)
		# outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))

		
		# out = outputs[-1]
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
		# self._act = tfu.function([goal_obs, crop], self.ac)

		self.flatvars = tfu.GetFlat(self.get_trainable_variables())
		self.unflatvars = tfu.SetFromFlat(self.get_trainable_variables())

	def train_process_observation(self, data, idx):

		# obj_size = env._env.env.sim.model.geom_size[env._env.env.sim.model.geom_name2id('puckbox')]
		# obj_size = 2 * obj_size
		# puck_z = env._env.env.init_puck_z + \
		# 		env._env.env.sim.model.geom_pos[env._env.env.sim.model.geom_name2id('puckbox')][-1]
		data = {key: data[key][idx] for key in data.keys()}
		ob_tensor = np.hstack([data['desired_goal'],
								data['achieved_goal']])
		batch_dict = mujoco_online_inputs.get_inputs(data, data['puck_zs'])
		feed = {}
		feed.update({self.map3D.rgb_camXs: batch_dict['rgb_camXs']})
		feed.update({self.map3D.xyz_camXs: batch_dict['xyz_camXs']})
		feed.update({self.map3D.pix_T_cams: batch_dict['pix_T_cams']})
		feed.update({self.map3D.origin_T_camRs: batch_dict['origin_T_camRs']})
		feed.update({self.map3D.origin_T_camXs: batch_dict['origin_T_camXs']})
		feed.update({self.map3D.puck_xyz_camRs: batch_dict['puck_xyz_camRs']})
		feed.update({self.map3D.camRs_T_puck: batch_dict['camRs_T_puck']})
		feed.update({self.map3D.obj_size: data['obj_sizes']})
		feed.update({self.goal_obs: ob_tensor})
		
		return feed

	def process_observation(self, ob, obj_size, puck_z):

		# obj_size = env._env.env.sim.model.geom_size[env._env.env.sim.model.geom_name2id('puckbox')]
		# obj_size = 2 * obj_size
		# puck_z = env._env.env.init_puck_z + \
		# 		env._env.env.sim.model.geom_pos[env._env.env.sim.model.geom_name2id('puckbox')][-1]

		ob = {key: np.repeat(np.expand_dims(ob[key], axis=0), 8, axis=0) for key in ob.keys()}
		ob_tensor = np.hstack([ob['desired_goal'],ob['achieved_goal']])
		batch_dict = mujoco_online_inputs.get_inputs(ob, np.repeat(puck_z,8))
		feed = {}
		feed.update({self.map3D.rgb_camXs: batch_dict['rgb_camXs']})
		feed.update({self.map3D.xyz_camXs: batch_dict['xyz_camXs']})
		feed.update({self.map3D.pix_T_cams: batch_dict['pix_T_cams']})
		feed.update({self.map3D.origin_T_camRs: batch_dict['origin_T_camRs']})
		feed.update({self.map3D.origin_T_camXs: batch_dict['origin_T_camXs']})
		feed.update({self.map3D.puck_xyz_camRs: batch_dict['puck_xyz_camRs']})
		feed.update({self.map3D.camRs_T_puck: batch_dict['camRs_T_puck']})
		feed.update({self.map3D.obj_size: np.repeat(obj_size.reshape(-1,3),8,0)})
		feed.update({self.goal_obs: ob_tensor})
		return feed

	def act(self, ob , obj_size, puck_z):
		feed = self.process_observation(ob, obj_size, puck_z)
		# ac = self._act(ob,crop)
		sess = tf.get_default_session()
		ac = sess.run(self.ac, feed_dict=feed)
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
