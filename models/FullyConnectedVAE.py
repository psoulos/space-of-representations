import tensorflow as tf

from models import BaseVAE
from utils import sample_normal

Bernoulli = tf.contrib.distributions.Bernoulli
slim = tf.contrib.slim


class FullyConnectedVAE(BaseVAE):
    def build_graph(self, graph):
        with graph.as_default():
            # TODO: MNIST sizes are hard-coded in
            self.x = x = tf.placeholder(tf.float32, [None, 784])
            shape = x.get_shape().as_list()

            # This layer's size is halfway between the layer before and after.
            # TODO: allow the architecture to be specified at the command line.
            net = slim.fully_connected(x, (shape[-1] - self.hidden_size) / 2)

            # Sample from the latent distribution
            self.q_z_mean = slim.fully_connected(net, self.hidden_size, activation_fn=None)
            tf.summary.histogram('q_z_mean', self.q_z_mean)
            self.q_z_log_var = slim.fully_connected(net, self.hidden_size, activation_fn=None)
            tf.summary.histogram('q_z_log_var', self.q_z_log_var)
            z = sample_normal(self.q_z_mean, self.q_z_log_var)
            self.representation = z

            # The decoder
            net = slim.fully_connected(z, (shape[-1] - self.hidden_size) / 2)
            # TODO: figure out the whole logits and Bernoulli dist vs MSE thing
            # Do not include the batch size in creating the final layer
            net = slim.fully_connected(net, shape[-1], activation_fn=None)
            self.p_x = Bernoulli(logits=net)

            tf.summary.image('generated', tf.reshape(self.p_x.mean(), [-1, 28, 28, 1]),
                             max_outputs=1)

            self.loss = self._vae_loss()
            tf.summary.scalar('loss', self.loss)

            learning_rate = tf.Variable(self.learning_rate)
            tf.summary.scalar('learning_rate', learning_rate)

            global_step = tf.Variable(0, trainable=False, name="global_step")
            self.train_op = tf.train.AdamOptimizer(
                learning_rate=learning_rate) \
                .minimize(self.loss, var_list=slim.get_model_variables(), global_step=global_step)

        return graph
