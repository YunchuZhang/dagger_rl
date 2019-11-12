import tensorflow as tf
import numpy as np
import tf_utils as tfu


KEYS = [#'observation_with_orientation',
		'desired_goal',
		'achieved_goal',
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
		self.act_dim = env.action_space.shape[0]
		self.hidden_sizes = hidden_sizes
		self.KEYS = KEYS

		with tf.variable_scope(name):
			self.scope = tf.get_variable_scope().name
			self.build(hidden_sizes)


	def build(self, hidden_sizes):

		ob = tfu.get_placeholder(name="ob",
							dtype=tf.float32,
							shape=[None, self.obs_dim])
		
		out = ob
		for i, hidden_size in enumerate(hidden_sizes):
			#import pdb; pdb.set_trace()
			out = tf.nn.relu(tfu.dense(out,
									hidden_size,
									"policyfc%i" % (i+1),
									weight_init=tfu.normc_initializer(1.0)))

		action = tfu.dense(out,
						self.act_dim,
						"policyfinal",
						weight_init=tfu.normc_initializer(0.01))

		self.ac = action
		self._act = tfu.function([ob], self.ac)

		self.flatvars = tfu.GetFlat(self.get_trainable_variables())
		self.unflatvars = tfu.SetFromFlat(self.get_trainable_variables())

	def train_process_observation(self, data, idx):
		data = {key: data[key][idx] for key in self.KEYS}
		ob = [data[key] for key in self.KEYS]
		ob = np.hstack(ob)
		assert ob.shape[-1] == self.obs_dim
		return ob

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