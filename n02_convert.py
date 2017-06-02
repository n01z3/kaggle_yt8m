import numpy as np
import tensorflow as tf
import os
import glob
from n00_localconfig import FAST, FOLDER

from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler


def tf2npz(tf_path, export_folder=FAST):
    vid_ids = []
    labels = []
    mean_rgb = []
    mean_audio = []
    tf_basename = os.path.basename(tf_path)
    npz_basename = tf_basename[:-len('.tfrecord')] + '.npz'
    isTrain = '/test' not in tf_path

    for example in tf.python_io.tf_record_iterator(tf_path):
        tf_example = tf.train.Example.FromString(example).features
        vid_ids.append(tf_example.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8'))
        if isTrain:
            labels.append(np.array(tf_example.feature['labels'].int64_list.value))
        mean_rgb.append(np.array(tf_example.feature['mean_rgb'].float_list.value).astype(np.float32))
        mean_audio.append(np.array(tf_example.feature['mean_audio'].float_list.value).astype(np.float32))

    save_path = export_folder + '/' + npz_basename
    np.savez(save_path,
             rgb=StandardScaler().fit_transform(np.array(mean_rgb)),
             audio=StandardScaler().fit_transform(np.array(mean_audio)),
             ids=np.array(vid_ids),
             labels=labels
             )


def main():
    for tp in ['test', 'train', 'val']:
        with Pool(8) as p:
            p.map(tf2npz, glob.glob(os.path.join(FOLDER, tp, '*tfrecord')))


if __name__ == '__main__':
    main()