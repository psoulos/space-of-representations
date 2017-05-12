import tensorflow as tf

from models import BaseModel
from utils import kl_normal

slim = tf.contrib.slim


class BaseVAE(BaseModel):
    def _vae_loss(self):
        kl = kl_normal(self.q_z_mean, self.q_z_log_var)
        # tf.summary.scalar('kl divergence', kl)

        # Bernoulli reconstruction
        reconstruction = tf.reduce_sum(
            self.p_x.log_prob(slim.flatten(self.x)), 1)
        # tf.summary.scalar('reconstruction', reconstruction)

        # Mean-squared error reconstruction
        # d = (slim.flatten(self.input_) - self.logits)
        # d2 = tf.multiply(d, d) * 2.0
        # reconstruction = -tf.reduce_sum(d2, 1)

        elbo = reconstruction - self.beta * kl
        return tf.reduce_mean(-elbo)

    def train_epoch(self):
        """ Trains the model for a single epoch """
        for it in range(self.iter_per_epoch):
            # Get batch
            xs, _ = self.mnist.train.next_batch(100)
            _, loss, summary = self.sess.run([self.train_op, self.loss, self.summary_op],
                                             {self.x: xs})
            self.summary_writer.add_summary(summary, it)
            if it % 1000 == 0:
                print('Iteration {}\t loss: {}'.format(it, loss))
