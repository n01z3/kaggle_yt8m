from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import Model

import tensorflow as tf

import numpy as np
import random
import os
from n00_localconfig import *

from sklearn.metrics import average_precision_score
from numba import jit
from time import time
from multiprocessing import Pool

def make_parallel(model, gpu_count):
    __author__ = "kuza55"

    # https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py

    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([shape[:1] // parts, shape[1:]], axis=0)
        stride = tf.concat([shape[:1] // parts, shape[1:] * 0], axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx': i, 'parts': gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        if len(outputs_all) == 1:
            merged.append(merge(outputs_all[0], mode='concat', concat_axis=0, name='output'))
        else:
            for outputs in outputs_all:
                merged.append(merge(outputs, mode='concat', concat_axis=0))

        return Model(input=model.inputs, output=merged)


def ap_at_n(data):
    predictions, actuals = data
    n = 20
    total_num_positives = None

    if len(predictions) != len(actuals):
        raise ValueError("the shape of predictions and actuals does not match.")

    if n is not None:
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be 'None' or a positive integer."
                             " It was '%s'." % n)

    ap = 0.0

    sortidx = np.argsort(predictions)[::-1]

    if total_num_positives is None:
        numpos = np.size(np.where(actuals > 0))
    else:
        numpos = total_num_positives

    if numpos == 0:
        return 0

    if n is not None:
        numpos = min(numpos, n)
    delta_recall = 1.0 / numpos
    poscount = 0.0

    # calculate the ap
    r = len(sortidx)
    if n is not None:
        r = min(r, n)
    for i in range(r):
        if actuals[sortidx[i]] > 0:
            poscount += 1
            ap += poscount / (i + 1) * delta_recall
    return ap


def gap(pred, actual):
    lst = zip(list(pred), list(actual))

    with Pool() as pool:
        all = pool.map(ap_at_n, lst)

    return np.mean(all)


def check_metric():
    y_prd = np.load('tmp/-lr2_0_5_0.npy')[:100000]
    print(y_prd.shape)

    prt, sfx = 'val', '0'
    y_val = np.load(os.path.join(FAST, '%s_lbs_%s.npy' % (prt, sfx)))[:100000]
    print((y_val.shape))

    # y_prd = np.load('tmp/y_prd.npy')  # [:1000]
    # y_val = np.load('tmp/y_val.npy')  # [:1000]

    print(y_prd.shape, y_val.shape)
    for i in range(10):
        t0 = time()
        print(gap(y_prd, y_val), t0 - time())

        # mtr = []
        # for i in range(5000):
        #     mtr.append(ap_at_n(y_prd[i], y_val[i]))
        #
        # print(np.mean(np.array(mtr)))


if __name__ == '__main__':
    check_metric()
