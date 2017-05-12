import tensorflow as tf


def kl_normal(z_mean, z_log_var):
    kl = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) -
                              tf.exp(z_log_var), 1)
    return kl


def sample_normal(z_mean, z_log_var):
    """
    Sample from the normal distribution N~(z_mean, sqrt(exp(z_log_var)))
    Args:
    z_mean     the mean of the normal distribution
    z_log_var  the log variance of the normal distribution
    """
    shape = tf.shape(z_log_var)
    epsilon = tf.random_normal(shape, mean=0., stddev=1.)
    std_dev = tf.sqrt(tf.exp(z_log_var))
    return z_mean + epsilon * std_dev
