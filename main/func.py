import tensorflow as tf


def create_my_sess(percent):
    if percent < 0.3:
        exit("Need more memory")
    elif percent > 0.7:
        exit("Run out of memory")

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = percent
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)
    return session


create_my_sess(0.5)
