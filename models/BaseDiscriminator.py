from models import BaseModel


class BaseDiscriminator(BaseModel):
    def train_epoch(self):
        """ Trains the model for a single epoch """
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
