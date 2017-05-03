import copy
import json
import os
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


class BasicModel(object):
    def __init__(self, config):
        self.config = copy.deepcopy(config)

        self.result_dir = self.config['result_dir']
        self.log_dir = os.path.join(self.result_dir, 'logs/')
        self.checkpoint_dir = os.path.join(self.result_dir, 'checkpoints/')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # TODO: the model should be dataset agnostic
        self.mnist = input_data.read_data_sets("data/", one_hot=True)

        self.num_epochs = self.config['num_epochs']
        self.iter_per_epoch = self.config['iter_per_epoch']
        self.activation = self.config['activation']
        self.hidden_size = self.config['hidden_size']
        self.learning_rate = self.config['learning_rate']

        self.graph = self.build_graph(tf.Graph())
        with self.graph.as_default():
            self.saver = tf.train.Saver(max_to_keep=50)
            self.init_op = tf.global_variables_initializer()
            self.summary_op = tf.summary.merge_all()

        # Add all the other common code for the initialization here
        gpu_options = tf.GPUOptions(allow_growth=True)
        sessConfig = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=sessConfig, graph=self.graph)

        self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        checkpoint = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if checkpoint is None:
            print('Initializing new model')
            self.sess.run(self.init_op)
        else:
            print('Loading the model from folder: %s' % self.checkpoint_dir)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

    def build_graph(self, graph):
        raise Exception('The build_graph function must be overriden by the model')

    def train(self):
        print('Start training')
        for epoch in range(self.num_epochs):
            print('Epoch {}'.format(epoch + 1))
            for it in range(self.iter_per_epoch):
                # Get batch
                xs, ys = self.mnist.train.next_batch(100)
                _, loss, summary = self.sess.run([self.train_op, self.loss, self.summary_op],
                                                 {self.x: xs, self.y_: ys})
                self.summary_writer.add_summary(summary, it)
                if it % 1000 == 0:
                    acc = self.sess.run(self.accuracy, feed_dict={self.x: self.mnist.test.images,
                                                                  self.y_: self.mnist.test.labels})
                    print('Iteration {}'.format(it))
                    print('Accuracy {}'.format(acc))
            self.save()

    def init(self):
        # Any model specific initialization can go here
        pass

    def save(self):
        global_step_t = tf.train.get_global_step(graph=self.graph)
        global_step = self.sess.run(global_step_t)
        print('Saving to %s with global_step %d' % (self.checkpoint_dir, global_step))
        self.saver.save(self.sess, self.checkpoint_dir + 'checkpoint', global_step=global_step)

        # Save the configuration
        if not os.path.isfile(self.result_dir + '/config.json'):
            with open(self.result_dir + '/config.json', 'w') as f:
                json.dump(self.config, f)
