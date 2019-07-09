# coding: utf-8
# filename: cancer_main.py
# function: 癌症检测, tta预测

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import time
import re
import cv2
from glob import glob
from skimage import io
from os.path import isfile
import imgaug.augmenters as iaa
import imgaug as ia

import keras
from keras.layers import Dense, Input, Dropout, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, Flatten
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.applications.densenet import DenseNet169, DenseNet201, preprocess_input
from keras.applications.nasnet import NASNetMobile, preprocess_input
from keras.optimizers import Adam

base_name = 'DenseNet201_Epoch60'
tta_folder = base_name.split('_')[0].lower() + '_tta_result'

src_dir = os.getcwd()
# str对象不可变,　所以src_dir变量重新赋值
src_dir = src_dir.replace('\\', '/')
print(os.listdir(src_dir))
src_dir = './'

SUBMIT = os.path.join(src_dir, 'submit')
TTA_OUTPUT = os.path.join(src_dir, 'submit/' + tta_folder)
DATASET = os.path.join(src_dir, 'dataset')
TRAIN = os.path.join(src_dir, 'dataset/train')
TEST = os.path.join(src_dir, 'dataset/test')
LOG = os.path.join(src_dir, 'logs')
MODEL = os.path.join(src_dir, 'logs/models')

model_path = os.path.join(MODEL, '%s.h5' % base_name)
# model_path = './log/models/%s.h5' % base_name
print(type(model_path))

# 返回符合TEST目录下*.tif特征的所有文件路径
# test_files = glob(os.path.join(TEST, '*.tif'))

test_id = pd.read_csv(os.path.join(DATASET, 'sample_submission.csv'))['id']
test_files = []
for id in test_id:
    id = id + '.tif'
    image_path = os.path.join(TEST, id)
    test_files.append(image_path)

print("test_files size :", len(test_files))


batch_size = 32
num_tta = 256
tta = True

# 判断TTA_OUTPUT目录是否存在
if os.path.isdir(TTA_OUTPUT):
    print('The directory of TTA_OUTPUT has been created')
else:
    os.mkdir(TTA_OUTPUT)


# 根据图像Id,获取文件完整路径
def expand_path(dir_flag, file):
    dir_list = ['SUBMIT', 'TTA_OUTPUT', 'DATASET', 'TRAIN', 'TEST', 'LOG', 'MODEL']
    if isfile(os.path.join(TRAIN, file)):
        return os.path.join(TRAIN, file)
    if isfile(os.path.join(TEST, file)):
        return os.path.join(TEST, file)

    return os.path.join(dir_flag, file)


def get_id_from_file_path(file_path):
    return file_path.split(os.path.sep)[-1].replace('.tif', '')


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def get_seq():
    seq = iaa.Sometimes(0.5,
                        iaa.Noop(),
                        iaa.OneOf([
                            iaa.Fliplr(0.5),
                            iaa.Flipud(0.5),
                            iaa.CropAndPad(percent=(-0.05, 0.05)),
                            # scale images to 90-110% of their size, individually per axis
                            iaa.Affine(scale=(0.9, 1.1)),
                            # iaa.CoarseDropout(0.02, size_percent=0.5),
                            # iaa.WithChannels(0, iaa.Add((10, 100))),
                            # either change the brightness of the whole image(sometimes per channel)
                            # or change the brightness of subareas
                            iaa.Multiply((0.9, 1.1)),
                            iaa.Affine(shear=(-5, 5)),    # shear by -5 to +5 degrees
                            iaa.Affine(rotate=(-5, 5)),   # rotate by -5 to +5 degrees
                            iaa.PiecewiseAffine(scale=(0.01, 0.05))
                        ]))
    return seq


# 使用预训练NASNetMobile模型
def get_model_classif_NASNetMobile():
    inputs = Input((96, 96, 3))
    base_model = NASNetMobile(include_top=False,
                              input_tensor=inputs)  # , weights=None
    x = base_model(inputs)
    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    out3 = Flatten()(x)
    out = Concatenate(axis=-1)([out1, out2, out3])
    out = Dropout(0.5)(out)
    out = Dense(1, activation="sigmoid", name="3_")(out)
    model = Model(inputs, out)
    model.compile(optimizer=Adam(0.0001), loss=binary_crossentropy, metrics=['acc'])
    model.summary()

    return model

# 使用预训练DenseNet201模型
def get_model_classif_DenseNet201(flags):
    inputs = Input((96, 96, 3))
    base_model = DenseNet201(include_top=False,
                             input_shape=(96, 96, 3),
                             weights='imagenet')        # , weights=None
    x = base_model(inputs)
    base_model.summary()
    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    out3 = Flatten()(x)
    out = Concatenate(axis=-1)([out1, out2, out3])
    out = Dropout(0.5)(out)
    out = Dense(1, activation="sigmoid", name="3_")(out)
    model = Model(inputs, out)
    if flags == 1:
        # 微调预训练模型
        base_model.trainble = True
        set_trainble = False
        for conv_layer in base_model.layers:
            if conv_layer.name == 'conv5_block29_0_bn':
                set_trainble = True
            if set_trainble:
                conv_layer.trainable = True
            else:
                conv_layer.trainable = False
        # 编译模型
        model.compile(optimizer=Adam(0.01),
                      loss=binary_crossentropy,
                      metrics=['acc'])
    else:
        # 释放所有层
        for layer in model.layers:
            layer.trainable = True
        # 编译模型
        model.compile(optimizer=Adam(0.0001),
                      loss=binary_crossentropy,
                      metrics=['acc'])

    model.summary()
    return model


model = get_model_classif_DenseNet201(0)
model.load_weights(model_path)

# tmp = keras.models.load_model(h5_path)
# model.set_weights(tmp.get_weights())
# 开始test time augmentation预测
for num in range(num_tta):
    preds = []
    ids = []
    print('Start predict  %d tta(test time augmentation)files' % num)

    tic = time.time()

    seq = get_seq()
    for batch in chunker(test_files, batch_size):
        X = [io.imread(x) for x in batch]
        if tta:
            X = seq.augment_images(X)
        X = [preprocess_input(x) for x in X]
        ids_batch = [get_id_from_file_path(x) for x in batch]
        X = np.array(X)
        preds_batch = ((model.predict(X).ravel() * model.predict(X[:, ::-1, :, :]).ravel() * model.predict(
            X[:, ::-1, ::-1, :]).ravel() * model.predict(X[:, :, ::-1, :]).ravel()) ** 0.25).tolist()
        preds += preds_batch
        ids += ids_batch
    if tta:
        name = base_name + '_tta%s' % str(num)
        print(name)
    df = pd.DataFrame({'id': ids, 'label': preds})
    df.to_csv(expand_path(TTA_OUTPUT, "%s.csv" % str(name)), index=False)
    print('tta %d have finished!' % num)
    toc = time.time()
    print("tta %d submission files take : %.3f minute" % (num, (toc - tic) / 60.))

# 重命名tta结果文件, 将文件名字符串中间出现的'NASNetMobile'替换为'DenseNet169'
sub_files = os.listdir(TTA_OUTPUT)
for i, file in enumerate(sub_files):
    if re.match(r'\w+NASNetMobile\w+', file):
        print('OK')
        dst_file = file.replace('NASNetMobile', 'DenseNet169')
        os.rename(os.path.join(TTA_OUTPUT, file), os.path.join(TTA_OUTPUT, dst_file))
        print(dst_file)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_pd(df):
    df["label"] = df["label"].apply(sigmoid)
    return df


df = pd.read_csv(expand_path(DATASET, 'sample_submission.csv'))
for tta in range(num_tta):
    print('Start ensemble')
    tta_name = base_name + '_tta%s' % str(tta)
    print(tta_name)
    df0 = pd.read_csv(expand_path(TTA_OUTPUT, '%s.csv' % tta_name))
    # df0 = pd.read_csv(expand_path(TTA_OUTPUT, base_name + '_tta_' + str(tta + 1) + '.csv'))
    # df0 = sigmoid_pd(df0)
    df['label'] += df0['label']
    df['id'] = df0['id']
    if tta + 1 in [8, 16, 32, 64, 128]:
        df_tmp = df.copy()
        df_tmp['label'] /= (tta + 1)
        df_tmp.to_csv(expand_path(SUBMIT, base_name + '_tta_' + str(tta + 1) + '.csv'), index=False)
    # if tta + 1 == 16:
    #     df_tmp = df.copy()
    #     df_tmp['label'] /= 16
    #     df_tmp.to_csv(expand_path(SUBMIT, base_name + '_tta_' + str(tta + 1) + '.csv'), index=False)
    # if tta + 1 == 32:
    #     df_tmp = df.copy()
    #     df_tmp['label'] /= 32
    #     df_tmp.to_csv(expand_path(SUBMIT, base_name + '_tta_' + str(tta + 1) + '.csv'), index=False)

df['label'] /= num_tta
df.to_csv(expand_path(SUBMIT, base_name + '_tta_' + str(num_tta) + '.csv'), index=False)


# submit = pd.read_csv(expand_path(SUBMIT, base_name + '_tta_' + str(num_tta) + '.csv'))
# new_df = pd.read_csv(expand_path(DATASET, 'sample_submission.csv'))
# labels = []
#
# # 将'Id'列作为索引,DataFrame结构
# submit = submit.set_index('id')
# # 生成新的标签列表结构, 'id'列排序和'sample_submission.csv'一样
# new_df['label'] = [submit.loc[image_id, 'label'] for image_id in new_df['id']]
# new_df.to_csv(expand_path(SUBMIT, base_name + '_tta_' + str(num_tta) + '_new' + '.csv'), index=False)
