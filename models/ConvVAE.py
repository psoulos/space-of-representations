import tensorflow as tf

from models import BaseVAE
from utils import sample_normal

Bernoulli = tf.contrib.distributions.Bernoulli
slim = tf.contrib.slim


class ConvVAE(BaseVAE):
    def build_graph(self, graph):
        with graph.as_default():
            # TODO: MNIST sizes are hard-coded in
            self.x = x = tf.placeholder(tf.float32, [None, 784])
            x_reshape = tf.reshape(x, [-1, 28, 28, 1])

            net = slim.conv2d(x_reshape, self.num_features, kernel_size=self.kernel_size)
            net = slim.flatten(net)

            # Sample from the latent distribution
            self.q_z_mean = slim.fully_connected(net, self.hidden_size, activation_fn=None)
            tf.summary.histogram('q_z_mean', self.q_z_mean)
            self.q_z_log_var = slim.fully_connected(net, self.hidden_size, activation_fn=None)
            tf.summary.histogram('q_z_log_var', self.q_z_log_var)
            z = sample_normal(self.q_z_mean, self.q_z_log_var)
            self.representation = z

            # The decoder
            net = tf.reshape(z, [-1, 1, 1, self.hidden_size])
            net = slim.conv2d_transpose(net, self.num_features, kernel_size=self.kernel_size)
            net = slim.flatten(net)
            net = slim.fully_connected(net, x.get_shape().as_list()[-1], activation_fn=None)

            # TODO: figure out the whole logits and Bernoulli dist vs MSE thing
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
