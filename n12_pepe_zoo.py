from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout, merge
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.utils.visualize_util import plot
from keras.regularizers import l2
from keras.utils.visualize_util import plot
from keras.optimizers import sgd, adam
from keras.models import model_from_json
from n00_utils import make_parallel, gap
import os


def fc_block1(x, n=1000, d=0.5):
    x = Dense(n)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(d)(x)
    return x


def fc_identity(input_tensor, n=1000, d=0.5):
    x = fc_block1(input_tensor, n, d)
    x = Dense(int(input_tensor.shape[1]))(x)
    x = merge([x, input_tensor], mode='sum', concat_axis=1)
    x = LeakyReLU()(x)
    return x


def build_mod2(opt=adam()):
    n = 2 * 1024
    in1 = Input((128,), name='x1')
    x1 = fc_block1(in1, n)
    # x1 = fc_block1(x1, n)
    x1 = fc_identity(x1, n)

    in2 = Input((1024,), name='x2')
    x2 = fc_block1(in2, n)
    # x2 = fc_block1(x2, n)
    x2 = fc_identity(x2, n)

    x = merge([x1, x2], mode='concat', concat_axis=1)

    # x = fc_block1(x, n)
    x = fc_identity(x, n)
    x = fc_block1(x, n)

    out = Dense(4716, activation='sigmoid', name='output')(x)

    model = Model(input=[in1, in2], output=out)
    model.compile(optimizer=opt, loss='categorical_crossentropy')

    # model.summary()
    plot(model=model, show_shapes=True)
    return model


def build_mod3(opt=adam()):
    n = 2 * 1024
    in1 = Input((128,), name='x1')
    x1 = fc_block1(in1, n)
    x1 = fc_identity(x1, n)
    x1 = fc_identity(x1, n)

    in2 = Input((1024,), name='x2')
    x2 = fc_block1(in2, n)
    x2 = fc_identity(x2, n)
    x2 = fc_identity(x2, n)

    x = merge([x1, x2], mode='concat', concat_axis=1)

    x = fc_identity(x, n)
    x = fc_identity(x, n)
    x = fc_block1(x, n)

    out = Dense(4716, activation='sigmoid', name='output')(x)

    model = Model(input=[in1, in2], output=out)
    model.compile(optimizer=opt, loss='categorical_crossentropy')

    # model.summary()
    plot(model=model, show_shapes=True)
    return model


def build_mod5(opt=adam()):
    n = 3 * 1024
    in1 = Input((128,), name='x1')
    x1 = fc_block1(in1, n)
    x1 = fc_identity(x1, n)


    in2 = Input((1024,), name='x2')
    x2 = fc_block1(in2, n)
    x2 = fc_identity(x2, n)

    x = merge([x1, x2], mode='concat', concat_axis=1)

    x = fc_identity(x, n)

    out = Dense(4716, activation='sigmoid', name='output')(x)

    model = Model(input=[in1, in2], output=out)
    model.compile(optimizer=opt, loss='categorical_crossentropy')

    # model.summary()
    # plot(model=model, show_shapes=True)
    return model


def build_mod6(opt=adam()):
    in1 = Input((128,), name='x1')
    in2 = Input((1024,), name='x2')

    inp = merge([in1, in2], mode='concat', concat_axis=1)

    wide = Dense(4000)(inp)

    deep = Dense(1000, activation='sigmoid')(inp)
    deep = Dense(1000, activation='sigmoid')(deep)
    deep = Dense(4000)(deep)

    out = merge([wide, deep], mode='sum', concat_axis=1)
    out = Dense(4716, activation='sigmoid', name='output')(out)

    model = Model(input=[in1, in2], output=out)
    model.compile(optimizer=opt, loss='categorical_crossentropy')

    # model.summary()
    plot(model=model, show_shapes=True)
    return model


def build_mod7(opt=adam()):
    n = 3 * 1024
    in1 = Input((128,), name='x1')
    x1 = fc_block1(in1, n)
    x1 = fc_identity(x1, n)
    # x1 = fc_identity(x1, n)

    in2 = Input((1024,), name='x2')
    x2 = fc_block1(in2, n)
    x2 = fc_identity(x2, n)
    # x2 = fc_identity(x2, n)

    x = merge([x1, x2], mode='concat', concat_axis=1)

    x = fc_identity(x, n)
    # x = fc_identity(x, n)
    x = fc_block1(x, n)

    out = Dense(4716, activation='sigmoid', name='output')(x)

    model = Model(input=[in1, in2], output=out)
    model.compile(optimizer=opt, loss='categorical_crossentropy')

    # model.summary()
    plot(model=model, show_shapes=True)
    return model


def build_mod8(opt=adam()):
    n = 3 * 1024
    in1 = Input((128,), name='x1')
    x1 = fc_block1(in1, n)
    x1 = fc_identity(x1, n)
    # x1 = fc_identity(x1, n)

    in2 = Input((1024,), name='x2')
    x2 = fc_block1(in2, n)
    x2 = fc_identity(x2, n)
    # x2 = fc_identity(x2, n)

    x = merge([x1, x2], mode='concat', concat_axis=1)

    x = fc_identity(x, 4000)
    # x = fc_identity(x, n)
    # x = fc_block1(x, n)

    out = Dense(4716, activation='sigmoid', name='output')(x)

    model = Model(input=[in1, in2], output=out)
    model.compile(optimizer=opt, loss='categorical_crossentropy')

    # model.summary()
    plot(model=model, show_shapes=True)
    return model


def build_mod4(opt=adam()):
    n = 1500
    in1 = Input((128,), name='x1')
    x1 = fc_block1(in1, n)
    x1 = fc_identity(x1, n)
    x1 = fc_identity(x1, n)
    x1 = fc_identity(x1, n)

    in2 = Input((1024,), name='x2')
    x2 = fc_block1(in2, n)
    x2 = fc_identity(x2, n)
    x2 = fc_identity(x2, n)
    x2 = fc_identity(x2, n)

    x = merge([x1, x2], mode='concat', concat_axis=1)

    x = fc_identity(x, n)
    x = fc_identity(x, n)
    x = fc_identity(x, n)
    x = fc_identity(x, n)
    x = fc_block1(x, 2*n)

    out = Dense(4716, activation='sigmoid', name='output')(x)

    model = Model(input=[in1, in2], output=out)
    model.compile(optimizer=opt, loss='categorical_crossentropy')

    # model.summary()
    # plot(model=model, show_shapes=True)
    return model


def build_mod9(opt=adam()):
    n = int(2.2 * 1024)
    in1 = Input((128,), name='x1')
    x1 = fc_block1(in1, n, d=0.3)
    x1 = fc_identity(x1, n, d=0.3)
    x1 = fc_identity(x1, n, d=0.3)

    in2 = Input((1024,), name='x2')
    x2 = fc_block1(in2, n, d=0.3)
    x2 = fc_identity(x2, n, d=0.3)
    x2 = fc_identity(x2, n, d=0.3)

    x = merge([x1, x2], mode='concat', concat_axis=1)

    x = fc_identity(x, n, d=0.3)
    x = fc_identity(x, n, d=0.3)
    x = fc_block1(x, n)

    out = Dense(4716, activation='sigmoid', name='output')(x)

    model = Model(input=[in1, in2], output=out)
    model.compile(optimizer=opt, loss='categorical_crossentropy')

    # model.summary()
    # plot(model=model, show_shapes=True)
    return model


def fc_inception(input_tensor, n=3000, d=0.5):

    br1 = Dense(n)(input_tensor)
    br1 = LeakyReLU()(br1)
    br1 = BatchNormalization()(br1)
    br1 = Dropout(d)(br1)
    br1 = Dense(int(n/3.0))(br1)

    br2 = Dense(n)(input_tensor)
    br2 = BatchNormalization()(br2)
    br2 = ELU()(br2)
    br2 = Dropout(d)(br2)
    br2 = Dense(int(n/3.0))(br2)

    br3 = Dense(int(n/3.0))(input_tensor)
    br3 = BatchNormalization()(br3)
    br3 = PReLU()(br3)
    br3 = Dropout(d)(br3)
    br3 = Dense(int(n/3.0))(br3)
    br3 = BatchNormalization()(br3)
    br3 = PReLU()(br3)
    br3 = Dropout(d)(br3)
    br3 = Dense(int(n/3.0))(br3)
    br3 = BatchNormalization()(br3)
    br3 = PReLU()(br3)
    br3 = Dropout(d)(br3)

    x = merge([br1, br2, br3], mode='concat', concat_axis=1)
    return x


def build_mod10(opt=adam()):
    n = int(1800)
    in1 = Input((128,), name='x1')

    x1 = fc_block1(in1, n)
    x1 = fc_inception(x1, n)
    x1 = fc_inception(x1, n)

    in2 = Input((1024,), name='x2')

    x2 = fc_block1(in2, n)
    x2 = fc_inception(x2, n)
    x2 = fc_inception(x2, n)

    x = merge([x1, x2], mode='concat', concat_axis=1)

    x = fc_inception(x, n)
    x = fc_inception(x, n)
    x = fc_block1(x, 2000)

    out = Dense(4716, activation='sigmoid', name='output')(x)
    model = Model(input=[in1, in2], output=out)
    model.compile(optimizer=opt, loss='categorical_crossentropy')

    # model.summary()
    plot(model=model, show_shapes=True)
    return model

def build_mod11(opt=adam()):
    n = int(2.5 * 1024)
    in1 = Input((128,), name='x1')
    x1 = fc_block1(in1, n, d=0.2)
    x1 = fc_identity(x1, n, d=0.2)
    x1 = fc_identity(x1, n, d=0.2)

    in2 = Input((1024,), name='x2')
    x2 = fc_block1(in2, n, d=0.2)
    x2 = fc_identity(x2, n, d=0.2)
    x2 = fc_identity(x2, n, d=0.2)

    x = merge([x1, x2], mode='concat', concat_axis=1)

    x = fc_identity(x, n, d=0.2)
    x = fc_identity(x, n, d=0.2)
    x = fc_block1(x, n)

    out = Dense(4716, activation='sigmoid', name='output')(x)

    model = Model(input=[in1, in2], output=out)
    model.compile(optimizer=opt, loss='categorical_crossentropy')

    # model.summary()
    # plot(model=model, show_shapes=True)
    return model


def build_mod12(opt=adam()):
    n = int(2 * 1024)
    in1 = Input((128,), name='x1')
    x1 = fc_block1(in1, n, d=0.2)
    x1 = fc_identity(x1, n, d=0.2)
    x1 = fc_identity(x1, n, d=0.2)

    in2 = Input((1024,), name='x2')
    x2 = fc_block1(in2, n, d=0.2)
    x2 = fc_identity(x2, n, d=0.2)
    x2 = fc_identity(x2, n, d=0.2)

    x = merge([x1, x2], mode='concat', concat_axis=1)

    x = fc_identity(x, n, d=0.2)
    x = fc_identity(x, n, d=0.2)
    x = fc_identity(x, n, d=0.2)
    x = fc_block1(x, n)

    out = Dense(4716, activation='sigmoid', name='output')(x)

    model = Model(input=[in1, in2], output=out)
    model.compile(optimizer=opt, loss='categorical_crossentropy')

    model.summary()
    # plot(model=model, show_shapes=True)
    return model


def build_mod13(opt=adam()):
    n = int(2 * 1024)
    in1 = Input((128,), name='x1')
    x1 = fc_block1(in1, n, d=0.2)
    x1 = fc_identity(x1, n, d=0.2)
    x1 = fc_identity(x1, n, d=0.2)
    x1 = fc_identity(x1, n, d=0.2)

    in2 = Input((1024,), name='x2')
    x2 = fc_block1(in2, n, d=0.2)
    x2 = fc_identity(x2, n, d=0.2)
    x2 = fc_identity(x2, n, d=0.2)
    x2 = fc_identity(x2, n, d=0.2)

    x = merge([x1, x2], mode='concat', concat_axis=1)

    x = fc_identity(x, n, d=0.2)
    x = fc_identity(x, n, d=0.2)
    x = fc_block1(x, n)

    out = Dense(4716, activation='sigmoid', name='output')(x)

    model = Model(input=[in1, in2], output=out)
    model.compile(optimizer=opt, loss='categorical_crossentropy')

    model.summary()
    # plot(model=model, show_shapes=True)
    return model

if __name__ == '__main__':
    build_mod13()
