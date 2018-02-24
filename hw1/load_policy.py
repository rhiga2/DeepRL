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
    def __init__(self, name, input_dim, param_dict=None, layers=None, target_net=False):
        self.param_dict = param_dict
        self.assign_nodes = []
        self.assign_ph = {}
        self.layers = layers
        self.target_net = target_net

        obs_tuple = [
            tf.placeholder(tf.float32, (None, 1), name="obs0"),
            tf.placeholder(tf.float32, (None, input_dim), name="obs1"),
        ]
        self.obs_tuple = obs_tuple
        actions_input = []
        actions_input.append(obs_tuple[1])

        x = tf.concat( actions_input, axis=1)
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
                # stops gradient update of target network
                self.output = tf.stop_gradient(self.output)
                # Initalize placeholders and assignment operators
                for param_name in params:
                    size = params[param_name].shape
                    ph = tf.placeholder(tf.float32, size)
                    self.assign_ph[param_name] = ph
                    self.assign_nodes.append(tf.assign(params[param_name], ph))

    def assign_weights(self):
        '''
        sess : tf Session
        '''
        assert(self.target_net)
        # build feed dictionary
        feed_dict = {}
        for param_name in self.param_dict:
            feed_dict[self.assign_ph[param_name]] = self.param_dict[param_name]
        tf.get_default_session().run(self.assign_nodes, feed_dict)

    def run_policy(self, obs_data):
        obs_data = [np.ones((1,)), obs_data]
        obs_data = [obs_data[0], obs_data[1]]
        # Because we need batch dimension, data[None] changes shape from [A] to [1,A]
        a = tf.get_default_session().run(self.output, feed_dict=dict( (ph,data[None]) for ph,data in zip(self.obs_tuple, obs_data)))
        return a[0]  # return first in batch
        
def make_policy(filename):
    take_weights_here = {}
    exec(open(filename).read(), take_weights_here)

    dense1_w = take_weights_here["weights_dense1_w"]
    dense1_b = take_weights_here["weights_dense1_b"]
    dense2_w = take_weights_here["weights_dense2_w"]
    dense2_b = take_weights_here["weights_dense2_b"]
    final_w = take_weights_here["weights_final_w"]
    final_b = take_weights_here["weights_final_b"]

    param_dict = {
        'dense1_w' : dense1_w,
        'dense1_b' : dense1_b,
        'dense2_w' : dense2_w,
        'dense2_b' : dense2_b,
        'final_w' : final_w,
        'final_b' : final_b
    }
    layers = [
        [('dense1_w', 'dense1_b'), (dense1_w.shape, dense1_b.shape)],
        [('dense2_w', 'dense2_b'), (dense2_w.shape, dense2_b.shape)],
        [('final_w', 'final_b'), (final_w.shape, final_b.shape)],
    ]

    policy = Policy('expert_policy', dense1_w.shape[0], param_dict=param_dict, layers=layers, target_net=True)
    return policy

def get_new_policy(input_dim):
    policy = Policy('agent_policy', input_dim)
    return policy
