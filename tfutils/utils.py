import os

import GPUtil
import tensorflow as tf


def find_avaiable_gpu(max_load=0.3, max_memory=0.5):
    gpu_avail = GPUtil.getFirstAvailable(attempts=10000, maxLoad=max_load, maxMemory=max_memory, interval=199)
    return gpu_avail[0]


def gpu_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))


def clear_dir(logdir):
    import shutil
    try:
        for the_file in os.listdir(logdir):
            file_path = os.path.join(logdir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)
    except FileNotFoundError:
        pass


def obtain_log_path(path, version=None, name=''):
    import os
    #prefix = os.environ['EXP_LOG_PATH']
    prefix = '/Users/adamgronowski/Desktop'
    assert len(prefix) > 0, 'Please set environment variable EXP_LOG_PATH'
    if type(version) is int:
        path = 'v{}/'.format(version) + path
    if len(name) > 0:
        path = name + '/' + path
    return os.path.join(prefix, path)
