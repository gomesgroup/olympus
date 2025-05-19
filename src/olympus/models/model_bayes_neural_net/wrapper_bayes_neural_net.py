#!/usr/bin/env python

import silence_tensorflow



silence_tensorflow.silence_tensorflow()
import tensorflow as tf
import tensorflow_probability as tfp


from olympus.models import WrapperTensorflowModel

tfd = tfp.distributions

# ===============================================================================


class BayesNeuralNet(WrapperTensorflowModel):

    ATT_KIND = {"type": "string", "default": "BayesNeuralNet"}
    ATT_TASK = {
        "type": "string",
        "default": "regression",
        "valid": ["regression", "ordinal"],
    }

    def __init__(
        self,
        scope="model",
        task="ordinal",
        hidden_depth=3,
        hidden_nodes=48,
        hidden_act="leaky_relu",
        out_act="linear",
        learning_rate=1e-3,
        pred_int=100,
        reg=0.001,
        es_patience=100,
        max_epochs=100000,
        batch_size=20,
    ):
        """Bayesian Neural Network model.

        Args:
            scope (str): TensorFlow scope.
            task (str): Predictive task, classification (ordinal objectives)
                or regression (continuous objectives)
            hidden_depth (int): Number of hidden layers.
            hidden_nodes (int): Number of hidden nodes per layer.
            hidden_act (str): Hidden activation function. Available options are 'linear', 'leaky_relu', 'relu',
                'softmax', 'softplus', 'softsign', 'sigmoid'.
            out_act (str): Output activation function. Available options are 'linear', 'leaky_relu', 'relu',
                'softmax', 'softplus', 'softsign', 'sigmoid'.
            learning_rate (float): Learning rate.
            pred_int (int): Frequency with which we make predictions on the validation/training set (in number of epochs).
            reg (float): ???
            es_patience (int): Early stopping patience.
            max_epochs (int): Maximum number of epochs allowed.
            batch_size (int): Size batches used for training.
        """

        WrapperTensorflowModel.__init__(**locals())

    def _build_inference(self):
        self.graph = tf.Graph()
        self.is_graph_constructed = True

        # Configure TensorFlow to use CPU to avoid CUDA issues
        config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})

        activation_hidden = self.act_funcs(self.hidden_act)
        activation_out = self.act_funcs(self.out_act)

        with self.graph.as_default():

            self.tf_x = tf.compat.v1.placeholder(
                tf.float32, [self.batch_size, self.features_dim]
            )
            self.tf_y = tf.compat.v1.placeholder(
                tf.float32, [self.batch_size, self.targets_dim]
            )

            with tf.name_scope(self.scope):
                # Recreate layers with exact same variable names as in checkpoint
                self.layers = []
                x = self.tf_x
                
                # Create hidden layers with identical names to the checkpoint
                for i in range(self.hidden_depth):
                    layer_name = f'dense_layer_{i}'
                    with tf.name_scope(layer_name):
                        # Create bias variables
                        bias_loc = tf.compat.v1.get_variable(
                            f"{layer_name}/bias_posterior_loc",
                            shape=[self.hidden_nodes],
                            initializer=tf.compat.v1.zeros_initializer()
                        )
                        bias_scale_untransformed = tf.compat.v1.get_variable(
                            f"{layer_name}/bias_posterior_untransformed_scale",
                            shape=[self.hidden_nodes],
                            initializer=tf.compat.v1.random_normal_initializer(mean=-3.0, stddev=0.1)
                        )
                        bias_scale = tf.nn.softplus(bias_scale_untransformed)
                        
                        # Create kernel variables
                        kernel_loc = tf.compat.v1.get_variable(
                            f"{layer_name}/kernel_posterior_loc",
                            shape=[x.get_shape().as_list()[-1], self.hidden_nodes],
                            initializer=tf.compat.v1.glorot_normal_initializer()
                        )
                        kernel_scale_untransformed = tf.compat.v1.get_variable(
                            f"{layer_name}/kernel_posterior_untransformed_scale",
                            shape=[x.get_shape().as_list()[-1], self.hidden_nodes],
                            initializer=tf.compat.v1.random_normal_initializer(mean=-3.0, stddev=0.1)
                        )
                        kernel_scale = tf.nn.softplus(kernel_scale_untransformed)
                        
                        # Sample from the variational posterior
                        kernel_noise = tf.random.normal(kernel_loc.shape)
                        kernel_sample = kernel_loc + kernel_noise * kernel_scale
                        
                        bias_noise = tf.random.normal(bias_loc.shape)
                        bias_sample = bias_loc + bias_noise * bias_scale
                        
                        # Calculate the layer output
                        preactivation = tf.matmul(x, kernel_sample) + bias_sample
                        x = activation_hidden(preactivation)
                        
                        # Calculate KL divergence (negative entropy + cross entropy)
                        kl_weight = -tf.reduce_sum(tf.math.log(kernel_scale)) + tf.reduce_sum(0.5 * (tf.square(kernel_loc) + tf.square(kernel_scale) - 1.0 - 2.0 * tf.math.log(kernel_scale)))
                        kl_bias = -tf.reduce_sum(tf.math.log(bias_scale)) + tf.reduce_sum(0.5 * (tf.square(bias_loc) + tf.square(bias_scale) - 1.0 - 2.0 * tf.math.log(bias_scale)))
                        
                        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES, kl_weight)
                        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES, kl_bias)
                
                # Output layer
                with tf.name_scope('output_layer'):
                    # Create bias variables
                    output_bias_loc = tf.compat.v1.get_variable(
                        "output_layer/bias_posterior_loc",
                        shape=[self.targets_dim],
                        initializer=tf.compat.v1.zeros_initializer()
                    )
                    output_bias_scale_untransformed = tf.compat.v1.get_variable(
                        "output_layer/bias_posterior_untransformed_scale",
                        shape=[self.targets_dim],
                        initializer=tf.compat.v1.random_normal_initializer(mean=-3.0, stddev=0.1)
                    )
                    output_bias_scale = tf.nn.softplus(output_bias_scale_untransformed)
                    
                    # Create kernel variables
                    output_kernel_loc = tf.compat.v1.get_variable(
                        "output_layer/kernel_posterior_loc",
                        shape=[x.get_shape().as_list()[-1], self.targets_dim],
                        initializer=tf.compat.v1.glorot_normal_initializer()
                    )
                    output_kernel_scale_untransformed = tf.compat.v1.get_variable(
                        "output_layer/kernel_posterior_untransformed_scale",
                        shape=[x.get_shape().as_list()[-1], self.targets_dim],
                        initializer=tf.compat.v1.random_normal_initializer(mean=-3.0, stddev=0.1)
                    )
                    output_kernel_scale = tf.nn.softplus(output_kernel_scale_untransformed)
                    
                    # Sample from the variational posterior
                    output_kernel_noise = tf.random.normal(output_kernel_loc.shape)
                    output_kernel_sample = output_kernel_loc + output_kernel_noise * output_kernel_scale
                    
                    output_bias_noise = tf.random.normal(output_bias_loc.shape)
                    output_bias_sample = output_bias_loc + output_bias_noise * output_bias_scale
                    
                    # Calculate the final output
                    preactivation = tf.matmul(x, output_kernel_sample) + output_bias_sample
                    self.y_pred = activation_out(preactivation)
                    
                    # Calculate KL divergence
                    kl_output_weight = -tf.reduce_sum(tf.math.log(output_kernel_scale)) + tf.reduce_sum(0.5 * (tf.square(output_kernel_loc) + tf.square(output_kernel_scale) - 1.0 - 2.0 * tf.math.log(output_kernel_scale)))
                    kl_output_bias = -tf.reduce_sum(tf.math.log(output_bias_scale)) + tf.reduce_sum(0.5 * (tf.square(output_bias_loc) + tf.square(output_bias_scale) - 1.0 - 2.0 * tf.math.log(output_bias_scale)))
                    
                    tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES, kl_output_weight)
                    tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES, kl_output_bias)
                
                # Observation noise parameter
                self.scale = tf.nn.softplus(
                    tf.compat.v1.get_variable(
                        'Variable',
                        shape=[self.targets_dim],
                        initializer=tf.compat.v1.ones_initializer()
                    )
                )
                
                # Create distribution for sampling
                self.y_sample = tfd.Normal(loc=self.y_pred, scale=self.scale)
            
            # Get all regularization losses for KL divergence
            reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
            self.kl = tf.reduce_sum(reg_losses) / float(self.batch_size)
            
            # Define the loss function
            self.reg_loss = -tf.reduce_sum(self.y_sample.log_prob(self.tf_y))
            self.loss = self.reg_loss + (self.reg * self.kl)

            self.optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.learning_rate
            )
            self.train_op = self.optimizer.minimize(self.loss)

            self.init_op = tf.group(
                tf.compat.v1.global_variables_initializer(),
                tf.compat.v1.local_variables_initializer(),
            )
            # Use CPU-only session
            self.sess = tf.compat.v1.Session(graph=self.graph, config=config)
            with self.sess.as_default():
                self.sess.run(self.init_op)
                self.saver = tf.compat.v1.train.Saver()

    def restore(self, model_path):
        if not self.is_graph_constructed:
            tf.compat.v1.reset_default_graph()
            self._build_inference()
            
        with self.graph.as_default():
            with self.sess.as_default():
                try:
                    # Create a new saver with the traditional approach
                    self.saver = tf.compat.v1.train.Saver()
                    self.saver.restore(self.sess, model_path + "/model.ckpt")
                    return True
                except Exception as e:
                    # If loading fails because of variable name mismatches, try a fallback approach
                    try:
                        # Create a reader to inspect the checkpoint
                        reader = tf.train.load_checkpoint(model_path + "/model.ckpt")
                        # Get the list of variable names in the checkpoint
                        var_list = reader.get_variable_to_shape_map()
                        
                        # Create a dictionary mapping current variables to checkpoint variables
                        var_map = {}
                        for var in tf.compat.v1.global_variables():
                            var_name = var.name.split(':')[0]
                            for ckpt_var in var_list:
                                # Try different patterns for matching variables
                                if ckpt_var == var_name or ckpt_var.endswith(var_name):
                                    var_map[var] = ckpt_var
                                    break
                        
                        if var_map:
                            # Create a saver with the variable map
                            custom_saver = tf.compat.v1.train.Saver(var_map)
                            custom_saver.restore(self.sess, model_path + "/model.ckpt")
                            return True
                        else:
                            return False
                    except Exception as inner_e:
                        # If both approaches fail, return False
                        tf.compat.v1.logging.warning(f"Failed to restore model: {str(inner_e)}")
                        return False
