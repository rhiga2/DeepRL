import pickle, tensorflow as tf, tf_util, numpy as np
import pdb

# def load_policy(filename):
#     with open(filename, 'rb') as f:
#         data = pickle.loads(f.read())
#
#     # assert len(data.keys()) == 2
#     nonlin_type = data['nonlin_type']
#     policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]
#
#     assert policy_type == 'GaussianPolicy', 'Policy type {} not supported'.format(policy_type)
#     policy_params = data[policy_type]
#
#     assert set(policy_params.keys()) == {'logstdevs_1_Da', 'hidden', 'obsnorm', 'out'}
#
#     # Keep track of input and output dims (i.e. observation and action dims) for the user
#
#     def build_policy(obs_bo):
#         def read_layer(l):
#             assert list(l.keys()) == ['AffineLayer']
#             assert sorted(l['AffineLayer'].keys()) == ['W', 'b']
#             return l['AffineLayer']['W'].astype(np.float32), l['AffineLayer']['b'].astype(np.float32)
#
#         def apply_nonlin(x):
#             if nonlin_type == 'lrelu':
#                 return tf_util.lrelu(x, leak=.01) # openai/imitation nn.py:233
#             elif nonlin_type == 'tanh':
#                 return tf.tanh(x)
#             else:
#                 raise NotImplementedError(nonlin_type)
#
#         # Build the policy. First, observation normalization.
#         assert list(policy_params['obsnorm'].keys()) == ['Standardizer']
#         obsnorm_mean = policy_params['obsnorm']['Standardizer']['mean_1_D']
#         obsnorm_meansq = policy_params['obsnorm']['Standardizer']['meansq_1_D']
#         obsnorm_stdev = np.sqrt(np.maximum(0, obsnorm_meansq - np.square(obsnorm_mean)))
#         print('obs', obsnorm_mean.shape, obsnorm_stdev.shape)
#         normedobs_bo = (obs_bo - obsnorm_mean) / (obsnorm_stdev + 1e-6) # 1e-6 constant from Standardizer class in nn.py:409 in openai/imitation
#
#         curr_activations_bd = normedobs_bo
#
#         # Hidden layers next
#         assert list(policy_params['hidden'].keys()) == ['FeedforwardNet']
#         layer_params = policy_params['hidden']['FeedforwardNet']
#         for layer_name in sorted(layer_params.keys()):
#             l = layer_params[layer_name]
#             W, b = read_layer(l)
#             curr_activations_bd = apply_nonlin(tf.matmul(curr_activations_bd, W) + b)
#
#         # Output layer
#         W, b = read_layer(policy_params['out'])
#         output_bo = tf.matmul(curr_activations_bd, W) + b
#         return output_bo, obsnorm_mean.shape
#
#     obs_bo = tf.placeholder(tf.float32, [None, None])
#     a_ba, obs_shape = build_policy(obs_bo)
#     policy_fn = tf_util.function([obs_bo], a_ba)
#     return policy_fn, obs_shape

class Policy():
    def __init__(self, name, input_dim, output_dim, param_dict=None, layers=None,
                 target_net=False, learning_rate=1e-2):
        self.param_dict = param_dict
        self.assign_nodes = []
        self.assign_ph = {}
        self.layers = layers
        self.target_net = target_net

        self.input_ph = tf.placeholder(tf.float32, (None, input_dim), name='obs')
        x = self.input_ph
        params = {}
        with tf.variable_scope(name):
            for i, value in enumerate(self.layers):
                names, sizes = value
                w_name, b_name = names
                w_size, b_size = sizes
                params[w_name] = tf.get_variable(w_name, w_size)
                params[b_name] = tf.get_variable(b_name, b_size)
                x = tf.matmul(x, params[w_name]) + params[b_name]
                if i != len(self.layers) - 1:
                    x = tf.nn.relu(x)
            self.output = x

            if self.target_net:
                # Initalize placeholders and assignment operators
                for param_name in params:
                    size = params[param_name].shape
                    ph = tf.placeholder(tf.float32, size)
                    self.assign_ph[param_name] = ph
                    self.assign_nodes.append(tf.assign(params[param_name], ph))
            else:
                # define output placeholders
                self.output_ph = tf.placeholder(tf.float32, [None, output_dim])

                # define loss function
                self.loss = tf.losses.mean_squared_error(self.output_ph, self.output)

                # define optimizier
                self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

                # initialize only variables in agent policy network
                var_list = [var for var in tf.global_variables() if name in var.name]
                self.initializer = tf.variables_initializer(var_list)

    def initialize(self):
        '''
        sess : tf Session
        '''
        if self.target_net:
            feed_dict = {}
            for param_name in self.param_dict:
                feed_dict[self.assign_ph[param_name]] = self.param_dict[param_name]
            tf.get_default_session().run(self.assign_nodes, feed_dict)
        else:
            tf.get_default_session().run(self.initializer)

    def get_action(self, obs_data):
        a = tf.get_default_session().run(self.output,
            feed_dict={self.input_ph : obs_data})
        return a

    def train_policy(self, obs, acs, batch_size=128, num_epochs=10):
        size = obs.shape[0]
        indices = np.arange(size)
        losses = []
        for epoch in range(num_epochs):
            for batch in range(size // batch_size):
                sample = np.random.choice(indices, size=batch_size, replace=False)
                batch_obs = obs[sample]
                batch_acs = acs[sample]
                loss, _ = tf.get_default_session().run([self.loss, self.optimizer],
                          feed_dict={self.input_ph : batch_obs,
                                     self.output_ph : batch_acs})
            print('Loss : ', loss)
            losses.append(loss)
        return np.mean(np.array(losses))

def make_policy(filename, env):
    take_weights_here = {}
    exec(open(filename).read(), take_weights_here)

    dense1_w = take_weights_here["weights_dense1_w"]
    dense1_b = take_weights_here["weights_dense1_b"]
    dense2_w = take_weights_here["weights_dense2_w"]
    dense2_b = take_weights_here["weights_dense2_b"]
    final_w = take_weights_here["weights_final_w"]
    final_b = take_weights_here["weights_final_b"]

    layers = [
        [('dense1_w', 'dense1_b'), (dense1_w.shape, dense1_b.shape)],
        [('dense2_w', 'dense2_b'), (dense2_w.shape, dense2_b.shape)],
        [('final_w', 'final_b'), (final_w.shape, final_b.shape)]
    ]

    param_dict = {
        'dense1_w' : dense1_w,
        'dense1_b' : dense1_b,
        'dense2_w' : dense2_w,
        'dense2_b' : dense2_b,
        'final_w' : final_w,
        'final_b' : final_b
    }

    policy = Policy('expert_policy',
                    env.observation_space.shape[0],
                    env.action_space.shape[0],
                    param_dict=param_dict,
                    layers=layers, target_net=True)
    policy.initialize()
    return policy

def get_new_policy(env, learning_rate=1e-2, dense_dims = [1024, 1024]):
    input_dims = env.observation_space.shape[0]
    output_dims = env.action_space.shape[0]
    layers = [
        [('dense1_w', 'dense1_b'), ([input_dims, dense_dims[0]], dense_dims[0])],
        [('dense2_w', 'dense2_b'), ([dense_dims[0], dense_dims[1]], dense_dims[1])],
        [('final_w', 'final_b'), ([dense_dims[1], output_dims], output_dims)]
    ]
    policy = Policy('agent_policy', input_dims, output_dims, layers=layers, learning_rate=learning_rate)
    policy.initialize()
    return policy
