# -*-coding-*-: utf-8
# autor: honggao.zhang
# update: 2019-07-09
# function: 癌症检测训练并测试程序
# problem: 二分类问题

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from os.path import isfile
from tqdm import tqdm
import time
import re
import sys
from glob import glob
from random import shuffle
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import keras
from keras import backend as K
from keras.utils import Sequence
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_inputt152V2
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from imgaug import augmenters as iaa
import imgaug as ia
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt

sys.path.append('./')
from model import *
# from sklearn.utils import class_weight, shuffle
# from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
# from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

base_name = 'ResNet101_Epoch40'

input_shape = (96, 96, 3)
learning_late_base = 1e-03
batch_size_base = 16
epochs_base = 2
num_classes = 1

print(os.listdir("./"))
src_dir = os.getcwd()
src_dir = src_dir.replace('\\', '/')
print(os.listdir(src_dir))
src_dir = './'

SUBMIT = os.path.join(src_dir, 'submit')
TTA_OUTPUT = os.path.join(src_dir, 'submit/tta_submissions')
DATASET = os.path.join(src_dir, 'dataset')
TRAIN = os.path.join(src_dir, 'dataset/train')
TEST = os.path.join(src_dir, 'dataset/test')
LOG = os.path.join(src_dir, 'logs')
MODEL = os.path.join(src_dir, 'logs/models')
model_path = os.path.join(MODEL, '%s.h5' % base_name)
label_path = os.path.join(DATASET, 'train_labels.csv')

df_train = pd.read_csv(os.path.join(DATASET, "train_labels.csv"))
id_label_map = {k: v for k, v in zip(df_train.id.values, df_train.label.values)}
print(df_train.head())

# 返回符合TRAIN目录下*.tif特征的所有文件路径
# labeled_files = glob(os.path.join(TRAIN, '*.tif'))
labeled_files = [os.path.join(TRAIN, x + '.tif') for x in df_train['id']]
labeled_id = df_train['id']

# 返回符合TEST目录下*.tif特征的所有文件路径
# test_files = glob(os.path.join(TEST, '*.tif'))

test_id = pd.read_csv(os.path.join(DATASET, 'sample_submission.csv'))['id']
test_files = []
for id in test_id:
    id = id + '.tif'
    image_path = os.path.join(TEST, id)
    test_files.append(image_path)

print("labeled_files size :", len(labeled_files))
print("test_files size :", len(test_files))


def get_id_from_file_path(file_path):
    return file_path.split(os.path.sep)[-1].replace('.tif', '')


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def get_seq():
    seq = iaa.Sometimes(0.2,
                        iaa.Noop(),
                        iaa.OneOf([
                            iaa.Fliplr(0.5),
                            iaa.Flipud(0.5),
                            # scale images to 90-110% of their size, individually per axis
                            iaa.Affine(scale=(0.9, 1.1)),
                            # iaa.CoarseDropout(0.02, size_percent=0.5),
                            # iaa.WithChannels(0, iaa.Add((10, 100))),
                            # either change the brightness of the whole image(sometimes per channel)
                            # or change the brightness of subareas
                            # iaa.Multiply((0.9, 1.1)),
                            # iaa.Multiply((0.5, 1.5), per_channel=0.5),
                            iaa.Affine(shear=(-5, 5)),  # shear by -5 to +5 degrees
                            iaa.Affine(rotate=(-5, 5)),  # rotate by -5 to +5 degrees
                            iaa.PiecewiseAffine(scale=(0.01, 0.05))
                        ]))
    return seq

# 将图像尺寸固定为(224, 224)
def resized_image(file_path):
    img = cv2.imread(file_path)  # cv2读取图片速度比pillow快
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2读取通道顺序为BGR，所以要转换成RGB
    img = cv2.resize(img, (96, 96))
    # img -= np.mean(img, keepdims=True)
    # img /= np.std(img, keepdims=True) + K.epsilon()
    return img

def data_gen(list_files, id_label_map, batch_size, augment=False):
    seq = get_seq()
    while True:
        shuffle(list_files)
        for batch in chunker(list_files, batch_size):
            X = [resized_image(x) for x in batch]
            Y = [id_label_map[get_id_from_file_path(x)] for x in batch]
            if augment:
                X = seq.augment_images(X)
            X = [preprocess_input(x) for x in X]

            yield np.array(X), np.array(Y)

# Cancer dataset generator
class CancerDataset(Sequence):
    """input data generator, a sequence object
    # Argument:
        label_path: The path of train_label.csv, file
        batch_size: The size of input_data's batch, int
        target_size: The image size, tuple
        mode: default is to 'train', str
        aug: default is to True, bool
    # Returns:
        A batch of input_data sequence

    """
    def __init__(self, label_path, batch_size, target_size, mode='train', aug=True, one_hot=False):
        # 初始化类实例参数
        self.label_path = label_path
        # self.df_train = df_train
        self.batch_size = batch_size
        self.target_size = target_size
        # self.num_class = num_class
        self.mode = mode
        self.aug = aug
        self.one_hot = one_hot
        if isfile(self.label_path):
            self.df_train = pd.read_csv(self.label_path)
            self.id_label_map = {k: v for k, v in zip(self.df_train.id.values, self.df_train.label.values)}
        else:
            print('The train_labels.csv is not exist!')
        self.train_id, self.val_id = train_test_split(self.df_train['id'], test_size=0.15,
                                                      random_state=8, shuffle=True)
        self.data = []
        if self.mode == "train":
            self.data = self.train_id
            self.data = [x for x in self.train_id]
        if self.mode == "val":
            self.data = self.val_id
            self.data = [x for x in self.val_id]

    def __len__(self):
        """Number of batch in the Sequence.
        """
        return len(self.data) // self.batch_size

    def __getitem__(self, index):
        """Gets batch at position `index`.
        """
        start = self.batch_size * index
        size = min(len(self.data) - start, self.batch_size)
        X = []
        Y = []
        for i in range(size):
            img = self.resize_image(self.data[start + i])
            img = self.augment_img(img)
            img = img.astype(np.uint8)
            label = self.id_label_map[self.data[start + i]]
            X.append(img)
            Y.append(label)
        # Y = [self.id_label_map[x] for x in self.data[start: start + size]]
        X = [preprocess_input(x) for x in X]

        return np.array(X), np.array(Y)
    def augment_img(self, image):
        """
        Return the array of augment image
        :param image: image id
        :return: image array
        """
        seq = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Multiply((0.8, 1.2), per_channel=0.5),
                iaa.Affine(shear=(-5, 5)),  # shear by -5 to +5 degrees
                iaa.Affine(rotate=(-5, 5)),  # rotate by -5 to +5 degrees
            ])], random_order=True)

        image_aug = seq.augment_image(image)
        return image_aug
    def resize_image(self, x):
        # 将图像尺寸固定为target_size
        try:
            x_file = self.expand_path(x)
            img = cv2.imread(x_file)  # cv2读取图片速度比pillow快
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2读取通道顺序为BGR，所以要转换成RGB
            img = cv2.resize(img, self.target_size[:2])
            # Normalize to zero mean and unit variance
            # img -= np.mean(img, keepdims=True)
            # img /= np.std(img, keepdims=True) + K.epsilon()
            img = img.astype(np.uint8)

        except Exception as e:
            print(e)

        return img
    def expand_path(self, id):
        # 根据图像id,获取文件完整路径
        if isfile(os.path.join(TRAIN, id  + '.tif')):
            return os.path.join(TRAIN, id + '.tif')
        if isfile(os.path.join(TEST, id + '.tif')):
            return os.path.join(TEST, id + '.tif')

        return id

def train_model(label_path, batch_size, input_shape, epochs, model,
                model_path=os.path.join(MODEL, '%s.h5' % base_name)):
    """create model and train model
    # Parameters:
        @label_path: the path of train.csv file
        @input_shape: the shape of input image
        @epochs: Integer. Number of epochs to train the model
        @model: model object
        @model_path: the path of saved model
    # Return:
        A `History` object. Its `History.history` attribute is a record of 
        training loss values and metrics values at successive epochs, 
        as well as validation loss values and validation metrics values (if applicable).
    """
    # save the best model when train
    checkpoint = ModelCheckpoint(model_path, monitor='val_acc',
                                 verbose=1, mode='max', save_weights_only=True)
    # reduce learning rate when a metric has stopped improving
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_acc', factor=0.5,
                                       patience=2, verbose=1,
                                       mode='max', min_lr=1e-06)
    # stop training when a monitored quantity has stopped improving
    early = EarlyStopping(monitor="val_acc", mode="max",
                          patience=6, restore_best_weights=True)
    # callback tensorboard class
    tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=0,
                             write_graph=True, write_images=True)

    train_gen = CancerDataset(label_path, batch_size, input_shape, mode="train", aug=True)
    val_gen = CancerDataset(label_path, batch_size * 2, input_shape, mode="val", aug=False)

    # fit model
    history = model.fit_generator(
        generator=train_gen,
        # generator=data_gen(train, id_label_map, batch_size, augment=True),
        steps_per_epoch=len(train_gen.data) // train_gen.batch_size,
        epochs=epochs,
        verbose=2,
        callbacks=[checkpoint, reduceLROnPlat, early, tbCallBack],
        validation_data=val_gen,
        validation_steps=len(val_gen.data) // val_gen.batch_size
    )
    return history

def visualization_history(history):
    """visualization train acc and loss
    """
    # Plot training & validation accuracy values
    acc = history.history['acc']
    epochs = range(1,len(acc)+1)
    plt.figure(figsize=(16, 6), dpi=200) 
    plt.plot(epochs, history.history['acc'], 'bo', label='Train acc')
    plt.plot(epochs, history.history['val_acc'], 'b',label='Validation acc')
    plt.title('Model train and validation accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train_acc', 'val_acc'], loc='upper right')
    plt.show()
    plt.savefig('cancer_train_history_acc.jpg')

    # Plot training & validation loss values
    plt.figure(figsize=(16, 6), dpi=200) 
    plt.plot(epochs, history.history['loss'], 'bo', label='Train loss')
    plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
    plt.title('Model train and validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train_loss', 'val_loss'], loc='upper right')
    plt.show()
    plt.savefig('cancer_train_history_loss.jpg')

if __name__ == "__main__":
    start = time.time()
    # Start train model
    model = get_model_classif_ResNet101(input_shape, learning_late_base, num_classes, 0)  # Create and compile model
    _ = train_model(label_path, batch_size_base, input_shape, epochs_base, model, model_path)
    model = get_model_classif_ResNet101(input_shape, learning_late_base, num_classes, 1)
    history = train_model(label_path, batch_size_base, input_shape, epochs_base * 10, model, model_path)
    print(model.metrics_names)
    # visualization training history
    visualization_history(history)

    # Load the trained model
    model = get_model_classif_ResNet101(input_shape, learning_late_base, num_classes, 1)
    model.load_weights(model_path)

    # Start predict
    preds = []
    ids = []
    for batch in chunker(test_files, batch_size_base):
        X = [preprocess_input(resized_image(x)) for x in batch]
        ids_batch = [get_id_from_file_path(x) for x in batch]
        X = np.array(X)
        preds_batch = ((model.predict(X).ravel() * model.predict(X[:, ::-1, :, :]).ravel() * model.predict(
            X[:, ::-1, ::-1, :]).ravel() * model.predict(X[:, :, ::-1, :]).ravel()) ** 0.25).tolist()
        preds += preds_batch
        ids += ids_batch
        # print(preds)
    df = pd.DataFrame({'id': ids, 'label': preds})
    df.to_csv("./submit/%s.csv" % base_name, index=False)
    df.head()
    end = time.time()
    print("Program run %d hours" % (end-start)/60/60)
    print("Program run success!")