import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import policies.tf_utils as tfu


KEYS = [#'observation_with_orientation',
		'desired_goal',
		#'achieved_goal_abs',
		'observation',
		# 'state_observation',
		# 'state_desired_goal',
		# 'state_achieved_goal',
		# 'proprio_observation',
		# 'proprio_desired_goal',
		# 'proprio_achieved_goal'
		]

class XYZ_XYZ_Policy:

	def __init__(self,
				 name,
				 env,
				 hidden_sizes=[64, 64, 64]):
		self.obs_dim = sum([env.observation_space.spaces[key].shape[0] for key in KEYS])
		# self.obs_dim = sum([env.observation_space.spaces[key].shape[0] for key in KEYS])
		self.act_dim = env.action_space.shape[0]
		self.hidden_sizes = hidden_sizes
		self.KEYS = KEYS

		with tf.variable_scope(name):
			self.scope = tf.get_variable_scope().name
			self.build(hidden_sizes)

	def build(self, hidden_sizes):

		self.ob = tfu.get_placeholder(name="ob",
							dtype=tf.float32,
							shape=[None, self.obs_dim])
		
		self.img_ph = tfu.get_placeholder(name="img_ph",dtype=tf.float32,shape=[None,64,64,1])
		bn = True
		with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
							activation_fn=tf.nn.relu,
							normalizer_fn=slim.batch_norm if bn else None,
							):
			d0 = 8
			dims = [d0, 2*d0, 4*d0, 8*d0]
			ksizes = [4, 4, 4, 4]
			strides = [2, 2, 2, 2]
			paddings = ['SAME'] * 4

			# ksizes[-1] = 2
			net = self.img_ph
			for i, (dim, ksize, stride, padding) in enumerate(zip(dims, ksizes, strides, paddings)):
				net = slim.conv2d(net, dim, ksize, stride=1, padding=padding)
				net = tf.nn.pool(net, [3,3], 'MAX', 'SAME', strides = [3,3])
			net = tf.layers.flatten(net)


		out = tf.concat([net, self.ob], -1)

		for i, hidden_size in enumerate(hidden_sizes):
			out = tf.nn.relu(tfu.dense(out,
									hidden_size,
									"policyfc%i" % (i+1),
									weight_init=tfu.normc_initializer(1)))

		action = tfu.dense(out,
						self.act_dim,
						"policyfinal",
						weight_init=tfu.normc_initializer(1))

		self.ac = action
		self._act = tfu.function([self.ob,self.img_ph], self.ac)

		self.flatvars = tfu.GetFlat(self.get_trainable_variables())
		self.unflatvars = tfu.SetFromFlat(self.get_trainable_variables())

	def train_process_observation(self, data, idx):
		data = {key: data[key][idx] for key in data.keys()}
		ob = [data[key] for key in self.KEYS]
		ob = np.hstack(ob)
		assert ob.shape[-1] == self.obs_dim
		feed = {}
		feed.update({self.ob: ob})
		feed.update({self.img_ph: data['img']})
		return feed

	def process_observation(self, ob):
		ob = [ob[key] for key in self.KEYS]
		ob = np.concatenate(ob)
		return ob

	def act(self, ob, img):
		ob = self.process_observation(ob)
		ac = self._act(ob[None],img[None])
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