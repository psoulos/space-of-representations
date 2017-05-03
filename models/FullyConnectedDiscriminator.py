import tensorflow as tf

from models import BasicModel

slim = tf.contrib.slim

ACTIVATIONS = {'relu': tf.nn.relu,
               'sigmoid': tf.nn.sigmoid,
               'elu': tf.nn.elu}


class FullyConnectedDiscriminator(BasicModel):
    def build_graph(self, graph):
        with graph.as_default():
            # TODO: MNIST sizes are hard-coded in
            self.x = x = tf.placeholder(tf.float32, [None, 784])

            net = slim.fully_connected(x, self.hidden_size,
                                       activation_fn=ACTIVATIONS[self.activation])
            y = slim.fully_connected(net, 10, activation_fn=None)
            self.y_ = y_ = tf.placeholder(tf.float32, [None, 10])

            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=y))
            tf.summary.scalar('cross_entropy', self.loss)

            learning_rate = tf.Variable(self.learning_rate)
            tf.summary.scalar('learning_rate', learning_rate)

            global_step = tf.Variable(0, trainable=False, name="global_step")
            self.train_op = tf.train.AdamOptimizer(
                learning_rate=learning_rate) \
                .minimize(self.loss, var_list=slim.get_model_variables(), global_step=global_step)

            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        return graph
