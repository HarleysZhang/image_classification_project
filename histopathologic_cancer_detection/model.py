# -*-coding-*-: utf-8
# autor: honggao.zhang
# update: 2019-07-09

import keras
from keras import backend as K
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution1D, concatenate, SpatialDropout1D, GlobalMaxPool1D, GlobalAvgPool1D, Embedding, \
    Conv2D, SeparableConv1D, Add, BatchNormalization, Activation, GlobalAveragePooling2D, LeakyReLU, Flatten
from keras.layers import Dense, Input, Dropout, MaxPooling2D, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, \
    Lambda, Multiply, LSTM, Bidirectional, PReLU, MaxPooling1D
from keras.layers.pooling import _GlobalPooling1D
from keras.losses import mae, sparse_categorical_crossentropy, binary_crossentropy
from keras.models import Model
from keras.applications.nasnet import NASNetMobile, NASNetLarge, preprocess_input

from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras_applications.resnet_common import ResNet101
from keras_applications.resnet_common import ResNet50V2
# from keras.applications.resnet_v2 import ResNet152V2
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import tensorflow as tf


def f2(y_true, y_pred):
    beta_f2=2
    # y_pred = K.round(y_pred)
    # y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), F2_THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f2 = (1+beta_f2**2)*p*r / (p*beta_f2**2 + r + K.epsilon())
    f2 = tf.where(tf.is_nan(f2), tf.zeros_like(f2), f2)
    return K.mean(f2)

gamma = 2.0
epsilon = K.epsilon()

# focal_loss function
def focal_loss(y_true, y_pred):
    pt = y_pred * y_true + (1 - y_pred) * (1 - y_true)
    pt = K.clip(pt, epsilon, 1 - epsilon)
    CE = -K.log(pt)
    FL = K.pow(1 - pt, gamma) * CE
    loss = K.sum(FL, axis=1)
    return loss

pretrained = {
    "densenet121": DenseNet121,
    "densenet169": DenseNet169,
    "resnet50": ResNet50,
    "nasnetmobile": NASNetMobile,
    "xception": Xception,
    "inception": InceptionV3
}

def build_conv_base_pretrained(model_name, learning_rate, flags, img_shape=(512, 512, 3)):
    input_tensor = Input(shape=img_shape)
    base_model = pretrained[model_name](weights='imagenet',
                                        include_top=False,
                                        input_shape=img_shape,
                                        input_tensor=input_tensor)
    x = base_model.output
    base_model.summary()
    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    out3 = Flatten()(x)
    out = Concatenate(axis=-1)([out1, out2, out3])
    out = Dropout(0.5)(out)
    out = Dense(1, activation="sigmoid", name="3_")(out)
    model = Model(input_tensor, out)

    if flags == 1:
        # 微调预训练模型
        base_model.trainble = True
        set_trainble = False
        num_layers = len(base_model.layers)
        for i, layer in enumerate(base_model.layers):
            if i >= num_layers - 5:
                set_trainble = True
            if set_trainble:
                layer.trainable = True
            else:
                layer.trainable = False
        # 编译模型
        model.compile(optimizer=Adam(learning_rate),
                      loss=binary_crossentropy,
                      metrics=['acc'])
    else:
        # 释放所有层
        for layer in model.layers:
            layer.trainable = True
        # 编译模型
        model.compile(optimizer=Adam(learning_rate / 10),
                      loss=binary_crossentropy,
                      metrics=['acc'])
    model.summary()
    return model

# classif_model = build_conv_base_pretrained("densenet121", 0.001, 0)
# train_gen = ArtworkDataset(label_path, batch_size_base, input_shape, mode="train", aug=True)
# batch_x_array,  batch_y_array = train_gen(1)
# print(batch_x_array)

# 使用预训练NASNetMobile模型
def get_model_classif_NASNetMobile(input_shape, learning_rate, num_classes, flags):
    input_tensor = Input(shape=input_shape, name='input')
    base_model = NASNetMobile(include_top=False,
                              input_shape=input_shape,
                              weights='imagenet')  # , weights=None
    bn = BatchNormalization()(input_tensor)
    base_model.summary()
    x = base_model(bn)
    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    out3 = Flatten()(x)
    out = Concatenate(axis=-1)([out1, out2, out3])
    out = Dropout(0.5)(out)
    out = Dense(num_classes, activation="sigmoid", name="3_")(out)
    model = Model(input_tensor, out)

    if flags == 1:
        # 微调预训练模型
        base_model.trainble = False
        set_trainble = False
        num_layers = len(base_model.layers)
        for i, layer in enumerate(base_model.layers):
            if i >= num_layers - 5:
                set_trainble = True
            if set_trainble:
                layer.trainable = True
            else:
                layer.trainable = False
        # 编译模型
        model.compile(optimizer=Adam(learning_rate),
                      loss=binary_crossentropy,
                      metrics=['acc'])
    else:
        # 释放所有层
        for layer in model.layers:
            layer.trainable = True
        # 编译模型
        model.compile(optimizer=Adam(learning_rate / 10),
                      loss=binary_crossentropy,
                      metrics=['acc'])

    model.summary()
    return model

# 使用预训练DenseNet201模型
def get_model_classif_DenseNet201(input_shape, learning_rate, num_classes, flags):
    """
    创建基于DenseNet201卷积基的分类模型
    # Parameters:
        @param input_shape: Input image size, tuple, such as (224, 224, 3)
        @param flags: Integer, 0 or 1
    # Return:
        A Keras model instance.
    """
    input_tensor = Input(input_shape)
    base_model = DenseNet201(include_top=False,
                             input_shape=input_shape,
                             weights='imagenet')  # , weights=None
    x = base_model(input_tensor)
    base_model.summary()
    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    out3 = Flatten()(x)
    out = Concatenate(axis=-1)([out1, out2, out3])
    out = Dropout(0.5)(out)
    out = Dense(num_classes, activation="sigmoid", name="3_")(out)
    model = Model(input_tensor, out)
    if flags == 0:
        # 微调预训练模型
        base_model.trainble = False
        set_trainble = False
        num_layers = len(base_model.layers)
        for i, layer in enumerate(base_model.layers):
            if i >= num_layers - 2:
                set_trainble = True
            if set_trainble:
                layer.trainable = True
            else:
                layer.trainable = False
        # 编译模型
        model.compile(optimizer=Adam(learning_rate),
                      loss=binary_crossentropy,
                      metrics=['acc'])
    else:
        # 释放所有层
        for layer in model.layers:
            layer.trainable = True
        # 编译模型
        model.compile(optimizer=Adam(learning_rate / 10),
                      loss=binary_crossentropy,
                      metrics=['acc'])

    model.summary()
    return model

# 使用预训练InceptionResNetV2模型
def get_model_classif_InceptionResNetV2(input_shape, learning_rate, num_classes, flags):
    input_tensor = Input(shape=input_shape, name='input')
    base_model = InceptionResNetV2(include_top=False,
                                   input_shape=input_shape,
                                   weights='imagenet')  # , weights=None
    bn = BatchNormalization()(input_tensor)
    base_model.summary()
    x = base_model(bn)
    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    out3 = Flatten()(x)
    out = Concatenate(axis=-1)([out1, out2, out3])
    out = Dropout(0.5)(out)
    out = Dense(num_classes, activation="sigmoid", name="3_")(out)
    model = Model(input_tensor, out)

    if flags == 0:
        # 微调预训练模型
        base_model.trainble = False
        set_trainble = False
        num_layers = len(base_model.layers)
        for i, layer in enumerate(base_model.layers):
            if i >= num_layers - 5:
                set_trainble = True
            if set_trainble:
                layer.trainable = True
            else:
                layer.trainable = False
        # 编译模型
        model.compile(optimizer=Adam(learning_rate),
                      loss=binary_crossentropy,
                      metrics=['acc'])
    else:
        # 释放所有层
        for layer in model.layers:
            layer.trainable = True
        # 编译模型
        model.compile(optimizer=Adam(learning_rate / 10),
                      loss=binary_crossentropy,
                      metrics=['acc'])

    model.summary()
    return model

# 使用预训练ResNet50V2模型
def get_model_classif_ResNet50V2(input_shape, learning_rate, num_classes, flags):
    input_tensor = Input(shape=input_shape, name='input')
    base_model = ResNet50V2(include_top=False,
                            input_shape=input_shape,
                            weights='imagenet',
                            backend=keras.backend,
                            layers=keras.layers,
                            models=keras.models,
                            utils=keras.utils)
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    # x = Dense(1024, activation='relu')(x)
    # x = Dropout(0.5)(x)
    final_output = Dense(num_classes, activation='sigmoid', name='final_output')(x)
    model = Model(input_tensor, final_output)

    if flags == 0:
        # 微调预训练模型
        base_model.trainble = False
        set_trainble = False
        num_layers = len(base_model.layers)
        for i, layer in enumerate(base_model.layers):
            if i >= num_layers - 2:
                set_trainble = True
            if set_trainble:
                layer.trainable = True
            else:
                layer.trainable = False
        # 编译模型
        model.compile(optimizer=Adam(learning_rate),
                      loss=binary_crossentropy,
                      metrics=['acc'])
    else:
        # 释放所有层
        for layer in model.layers:
            layer.trainable = True
        # 编译模型
        model.compile(optimizer=Adam(learning_rate / 10),
                      # loss=binary_crossentropy,
                      loss=focal_loss,
                      metrics=['acc', f2])
    model.summary()
    return model

# 使用预训练ResNet101模型
def get_model_classif_ResNet101(input_shape, learning_rate, num_classes, flags):
    input_tensor = Input(shape=input_shape, name='input')
    base_model = ResNet101(include_top=False,
                           input_shape=input_shape,
                           weights='imagenet',  # , weights=None
                           backend=keras.backend,
                           layers=keras.layers,
                           models=keras.models,
                           utils=keras.utils)
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    # x = Dense(1024, activation='relu')(x)
    # x = Dropout(0.5)(x)
    final_output = Dense(num_classes, activation='sigmoid', name='final_output')(x)
    model = Model(input_tensor, final_output)

    if flags == 0:
        # 微调预训练模型
        base_model.trainble = False
        set_trainble = False
        num_layers = len(base_model.layers)
        for i, layer in enumerate(base_model.layers):
            if i >= num_layers - 5:
                set_trainble = True
            if set_trainble:
                layer.trainable = True
            else:
                layer.trainable = False
        # 编译模型
        model.compile(optimizer=Adam(learning_rate),
                      loss=binary_crossentropy,
                      metrics=['acc'])
    else:
        # 释放所有层
        for layer in model.layers:
            layer.trainable = True
        # 编译模型
        model.compile(optimizer=Adam(learning_rate / 10),
                      # loss=binary_crossentropy,
                      loss=focal_loss,
                      metrics=['acc', f2])
    model.summary()
    return model
