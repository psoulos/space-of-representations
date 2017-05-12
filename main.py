import os
import tensorflow as tf
import time

# See the __init__ script in the models folder
# `make_models` is a helper function to load any models you have
from models import FullyConnectedDiscriminator, ConvDiscriminator, FullyConnectedVAE, ConvVAE

dir = os.path.dirname(os.path.realpath(__file__))

flags = tf.app.flags

# Model config
flags.DEFINE_boolean('conv', True, 'Whether the hidden layer is convolution or fully connected')
flags.DEFINE_boolean('vae', True, 'Whether the model is generative (vae) or discriminative')
flags.DEFINE_string('activation', 'relu', 'The hidden layer activation function {relu|sigmoid}')
flags.DEFINE_integer('hidden_size', 10, 'The number of units in the hidden layer for fully '
                     'connected discriminator or the number of features for a convolutional '
                     'discriminator. The size of the latent vector for a VAE.')
flags.DEFINE_integer('kernel_size', 5, 'The size of the convolution patch')
flags.DEFINE_integer('num_features', 5, 'The number of features for a convolution layer')
flags.DEFINE_integer('beta', 1, 'The coefficient to impose on the VAE divergence.'
                                ' 1 is a standard VAE.')

# Training config
flags.DEFINE_float('learning_rate', 1e-3, 'The learning rate')
flags.DEFINE_integer('num_epochs', 1, 'The number of epochs to train for')
flags.DEFINE_integer('iter_per_epoch', 50000, 'The number of iterations per epoch')

# Other config
flags.DEFINE_string('model_name', 'model', 'Unique name of the model')
flags.DEFINE_string('result_dir', dir + '/results/' + flags.FLAGS.model_name + '/' +
                    time.strftime('%Y-%m-%d-%H-%M-%S'), 'Name of the directory to store/log the '
                    'model (if it exists, the model will be loaded from it)')


def main(_):
    config = flags.FLAGS.__flags.copy()

    if config['vae']:
        if config['conv']:
            print('Training a convolutional vae')
            model = ConvVAE(config)
        else:
            print('Training a fully connected vae')
            model = FullyConnectedVAE(config)
    else:
        if config['conv']:
            print('Training a convolutional discriminator')
            model = ConvDiscriminator(config)
        else:
            print('Training a fully connected discriminator')
            model = FullyConnectedDiscriminator(config)
    print('Model created')

    model.train()


if __name__ == '__main__':
    tf.app.run()
