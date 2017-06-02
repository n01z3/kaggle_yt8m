from n12_pepe_zoo import build_mod2, build_mod3, build_mod4, build_mod5, build_mod6, build_mod7, \
    build_mod8, build_mod9, build_mod10, build_mod11, build_mod12, build_mod13
from keras.optimizers import sgd, adam
import numpy as np
from time import time
from n00_localconfig import FOLDER, FAST, WPATH
import glob
import os
import matplotlib.pyplot as plt
import random
import argparse
import logging
from n00_utils import make_parallel, gap
from scipy.stats import gmean
from multiprocessing import Pool

random.seed(666)
np.random.seed(666)


def del_w(dst):
    for fn in sorted(glob.glob(dst + '/*h5'))[:-10]: os.remove(fn)


def enc(ys):
    out = np.zeros((len(ys), 4716), dtype=np.int8)
    for i, el in enumerate(ys):
        for k in el:
            out[i, k] = 1
    return out


def load_npz(fn):
    tpz = np.load(fn)
    return (tpz['audio'], tpz['rgb'], enc(tpz['labels']))


def get_data(fns):
    with Pool() as pool:
        all = pool.map(load_npz, fns)
    x1, x2, y = zip(*all)
    return np.vstack(x1), np.vstack(x2), np.vstack(y)


def check():
    fns = glob.glob('/home/aakuzin/dataset/yt8m/val*npz')[:610]
    t0 = time()
    x1, x2, y = get_data(fns)
    print(x1.shape, x2.shape, y.shape, time() - t0)


def get_mod(ags):
    dst = os.path.join(ags.wpath, ags.versn)
    b_scr = -1

    if ags.optim == 'adam':
        opt = adam(ags.lrate)
    elif ags.optim == 'sgd':
        opt = sgd(ags.lrate)
    else:
        opt = adam()

    lst = [build_mod2(), build_mod3(), build_mod7(), build_mod9(), build_mod11(), build_mod12(), build_mod13()]

    model = lst[ags.mtype]
    if ags.mtype == 0:
        model = build_mod2(opt)
        logging.info('start with model 2')
    elif ags.mtype == 1:
        model = build_mod3(opt)
        logging.info('start with model 3')
    elif ags.mtype == 2:
        model = build_mod7(opt)
        logging.info('start with model 7')
    elif ags.mtype == 3:
        model = build_mod9(opt)
        logging.info('start with model 9')
    elif ags.mtype == 4:
        model = build_mod11(opt)
        logging.info('start with model 11')
    elif ags.mtype == 5:
        model = build_mod12(opt)
        logging.info('start with model 12')
    elif ags.mtype == 6:
        model = build_mod13(opt)
        logging.info('start with model 13')

    if ags.begin == -1:
        fls = sorted(glob.glob(dst + '/*h5'))
        if len(fls) > 0:
            logging.info('load weights: %s' % fls[-1])
            model.load_weights(fls[-1])
            b_scr = float(os.path.basename(fls[-1]).split('_')[0])

    return model, b_scr


def main(ags):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ags.ngpus - 1)

    dst = os.path.join(ags.wpath, ags.versn)
    if not os.path.exists(dst): os.mkdir(dst)
    log_file = os.path.join(ags.wpath, ags.versn, '%s_log' % (ags.versn))
    print('log_file', log_file)
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(log_file))
    logging.info('start with arguments %s', ags)

    model, b_scr = get_mod(ags)

    x1_val, x2_val, y_val = get_data(sorted(glob.glob(FAST + '/val*npz'))[:610])
    x1_val, x2_val, y_val = x1_val[:ags.split], x2_val[:ags.split], y_val[:ags.split]
    print(x1_val.shape, x2_val.shape, y_val.shape)
    fns = glob.glob(FAST + '/train*npz')
    cnt, chunk = 0, 256
    for e in range(ags.nepoh):
        for k, i in enumerate(range(0, 4096, 256)):
            t0 = time()

            x1_trn, x2_trn, y_trn = get_data(fns[i:i + chunk])

            model.fit({'x1': x1_trn, 'x2': x2_trn}, {'output': y_trn},
                      nb_epoch=1, batch_size=ags.batch, verbose=2, shuffle=True)

            ypd = model.predict({'x1': x1_val, 'x2': x2_val}, verbose=False, batch_size=ags.batch)
            g = gap(ypd, y_val)
            if g < b_scr + 0.00001:
                cnt += 1
            elif g < 0.6:
                cnt += 1
                b_scr = g
            else:
                cnt = 0
                b_scr = g

            logging.info('%d %d %0.5f %0.6f %3.2f' % (e, k, g, g - b_scr, time() - t0))
            model.save_weights(os.path.join(ags.wpath, ags.versn, ' %0.5f_%d_%d.h5' % (g, e, k)))
            if cnt > ags.patin:
                logging.info('cancel, best score: %0.6f' % b_scr)
                return None

        del_w(dst)


def add_fit_args(train):
    train.add_argument('--ngpus', default=1, type=int, help='amount of gpus')
    train.add_argument('--versn', default='rn-21', type=str, help='version of net')
    train.add_argument('--begin', default=0, type=int, help='start epoch')

    train.add_argument('--batch', default=8000, type=int, help='the batch size')
    train.add_argument('--nepoh', default=30, type=int, help='amount of epoch')
    train.add_argument('--check', default=20, type=int, help='period of check in iteration')
    train.add_argument('--lrate', default=0.001, type=float, help='start learning rate')
    train.add_argument('--optim', default='adam', type=str, help='optimizer')
    train.add_argument('--patin', default=15, type=int, help='waiting for n iteration without improvement')

    train.add_argument('--losss', default='categorical_crossentropy', type=str, help='loss function')
    train.add_argument('--mtype', default=1, type=int, help='neurons on branch audio')

    train.add_argument('--wpath', default=WPATH, type=str, help='net symbol path')
    train.add_argument('--dpath', default=FAST, type=str, help='data_path')
    train.add_argument('--split', default=200000, type=int, help='data_path')
    return train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_fit_args(parser)
    ags = parser.parse_args()
    main(ags)
