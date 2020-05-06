# -*- coding:utf-8 -*-
# filename: main_all.py
# author: honggao.zhang
# time: 2019-07-09
# function: Image classification problems pipline

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import sys
from os.path import isfile
from tqdm import tqdm
import time
import re
from glob import glob
from random import shuffle
import cv2
import keras
from keras import backend as K
from imgaug import augmenters as iaa
import imgaug as ia
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
from keras.utils import Sequence
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split

sys.path.append('./')
from model import *
# from sklearn.utils import class_weight, shuffle
# from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
# from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import tensorflow as tf

base_name = 'ResNet101_Epoch40'
input_shape = (384, 384, 3)
learning_late_base = 1e-03
batch_size_base = 16
epochs_base = 2
num_classes = 1103

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
label_path = os.path.join(DATASET, 'train.csv')

df_train = pd.read_csv(os.path.join(DATASET, "train.csv"))
submit = pd.read_csv(os.path.join(DATASET, "sample_submission.csv"))
id_label_map = {k: v for k, v in zip(df_train.id.values, df_train.attribute_ids.values)}
print(df_train.head())

# 返回符合TRAIN目录下*.png特征的所有文件路径
# labeled_files = glob(os.path.join(TRAIN, '*.png'))
labeled_files = [os.path.join(TRAIN, x + '.png') for x in df_train['id']]
labeled_id = df_train['id']

# 返回符合TEST目录下*.png特征的所有文件路径
# test_files = glob(os.path.join(TEST, '*.png'))

test_id = pd.read_csv(os.path.join(DATASET, 'sample_submission.csv'))['id']
test_files = []
for id in test_id:
    id = id + '.png'
    image_path = os.path.join(TEST, id)
    test_files.append(image_path)

print("labeled_files size :", len(labeled_files))
print("test_files size :", len(test_files))

def get_id_from_file_path(file_path):
    return file_path.split(os.path.sep)[-1].replace('.png', '')

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
                            iaa.Multiply((0.9, 1.1)),
                            # iaa.Multiply((0.5, 1.5), per_channel=0.5),
                            iaa.Affine(shear=(-5, 5)),  # shear by -5 to +5 degrees
                            iaa.Affine(rotate=(-5, 5)),  # rotate by -5 to +5 degrees
                            iaa.PiecewiseAffine(scale=(0.01, 0.05))
                        ]))
    return seq

# 将图像尺寸固定为(224, 224)
def resized_image(file_path, ):
    img = cv2.imread(file_path)  # cv2读取图片速度比pillow快
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2读取通道顺序为BGR，所以要转换成RGB
    img = cv2.resize(img, (384, 384))
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
class ArtworkDataset(Sequence):
    """input data generator sequence
    # Argument:
        @label_path: The path of train_label.csv, file
        @batch_size: The size of input_data's batch, int
        @target_size: The image size, tuple
        @mode: default is to 'train', str
        @aug: default is to True, bool
    # Returns:
        A batch of input_data sequence
    """
    def __init__(self, label_path, batch_size, target_size, num_classes, mode='train', aug=True, one_hot=False):
        # 初始化类实例参数
        self.label_path = label_path
        # self.df_train = df_train
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_classes = num_classes
        self.mode = mode
        self.aug = aug
        self.one_hot = one_hot
        if isfile(self.label_path):
            self.df_train = pd.read_csv(self.label_path)
            self.train_dataset_info = []
            columns = list(self.df_train.columns)
            for name, labels in zip(self.df_train[columns[0]], self.df_train[columns[1]].str.split(' ')):
                self.train_dataset_info.append({
                    'id': name,
                    'labels': np.array([int(label) for label in labels])})
            self.train_dataset_info = np.array(self.train_dataset_info)
        else:
            print('The train_labels.csv is not exist!')

        # split data into train, valid
        self.indexes = np.arange(self.train_dataset_info.shape[0])
        self.train_indexes, self.valid_indexes = train_test_split(self.indexes, test_size=0.1, random_state=8)
        # self.data = []
        if self.mode == "train":
            self.data = self.train_dataset_info[self.train_indexes]
        if self.mode == "val":
            self.data = self.train_dataset_info[self.valid_indexes]

    def __len__(self):
        """Number of batch in the Sequence.
        """
        return len(self.data) // self.batch_size

    def __getitem__(self, index):
        """Generate each batch of training data at position `index`
         : param index: batch index
         : return: training images and labels
        """
        assert self.target_size[2] == 3
        start = self.batch_size * index
        size = min(self.data.shape[0] - start, self.batch_size)
        # dataset_info = shuffle(dataset_info)
        batch_images = []
        X_train_batch = self.data[start:start + size]
        # print(X_train_batch.shape[0])
        batch_labels = np.zeros((len(X_train_batch), self.num_classes))
        assert size == X_train_batch.shape[0]
        for i in range(size):
            image = self.load_image(X_train_batch[i]['id'])
            if self.aug:
                image = self.augment_img(image)
            batch_images.append(preprocess_input(image))
            batch_labels[i][X_train_batch[i]['labels']] = 1
        assert len(batch_images) == X_train_batch.shape[0]
        return np.array(batch_images), batch_labels
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
    def load_image(self, x):
        # 将图像尺寸固定为target_size
        try:
            x_file = self.expand_path(x)
            img = cv2.imread(x_file)  # cv2读取图片速度比pillow快
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2读取通道顺序为BGR，所以要转换成RGB
            img = cv2.resize(img, self.target_size[:2])
            # print(img.shape)
            # Normalize to zero mean and unit variance
            # img -= np.mean(img, keepdims=True)
            # img /= np.std(img, keepdims=True) + K.epsilon()
            img = img.astype(np.uint8)
        except Exception as e:
            print(e)
        return img
    def expand_path(self, id):
        # 根据图像id,获取文件完整路径
        if isfile(os.path.join(TRAIN, id  + '.png')):
            return os.path.join(TRAIN, id + '.png')
        if isfile(os.path.join(TEST, id + '.png')):
            return os.path.join(TEST, id + '.png')
        return id

def train_model(label_path, batch_size, input_shape, epochs, num_classes, model,
                model_path=os.path.join(MODEL, '%s.h5' % base_name)):
    # save the best model when train
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss',
                                 verbose=1, mode='min', save_weights_only=True)
    # reduce learning rate when a metric has stopped improving
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                       patience=2, verbose=1,
                                       mode='min', min_lr=1e-06)
    # stop training when a monitored quantity has stopped improving
    early = EarlyStopping(monitor="val_loss", mode="min",
                          patience=6, restore_best_weights=True)
    # callback tensorboard class
    tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=0,
                             write_graph=True, write_images=True)

    train_gen = ArtworkDataset(label_path, batch_size, input_shape, num_classes, mode="train", aug=True)
    val_gen = ArtworkDataset(label_path, batch_size, input_shape, num_classes, mode="val", aug=False)

    # fit model
    history = model.fit_generator(
        generator=train_gen,
        # generator=data_gen(train, id_label_map, batch_size, augment=True),
        steps_per_epoch=len(train_gen.data) // train_gen.batch_size,
        epochs=epochs,
        verbose=2,
        callbacks=[checkpoint, reduceLROnPlat, early, tbCallBack],
        use_multiprocessing=True,
        validation_data=val_gen,
        validation_steps=len(val_gen.data) // val_gen.batch_size
    )
    return history

def visualization_history(history):
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
    # Create and compile model
    model = get_model_classif_ResNet50V2(input_shape, learning_late_base, num_classes, 0)
    _ = train_model(label_path, batch_size_base, input_shape, epochs_base, num_classes, model, model_path)
    model = get_model_classif_ResNet50V2(input_shape, learning_late_base, num_classes, 1)
    history = train_model(label_path, batch_size_base, input_shape, epochs_base * 10, num_classes, model, model_path)
    print(model.metrics_names)
    # visualization training history
    visualization_history(history)

    # Load the trained model
    model = get_model_classif_ResNet101(input_shape, learning_late_base, num_classes, 1)
    print(model_path)
    model.load_weights(model_path)

    predicted_post = []
    best_thr = 0.080
    for i, name in tqdm(enumerate(submit['id'])):
        path = os.path.join(TEST, name)
        image = resized_image(path)
        score_predict = model.predict(preprocess_input(image[np.newaxis]))
        thresh = np.percentile(score_predict,99.2)
        score_predict[score_predict < thresh] = 0
        # print(score_predict)
        label_predict = np.arange(num_classes)[score_predict[0]>=best_thr]
        # print(label_predict)
        str_predict_label = ' '.join(str(l) for l in label_predict)
        predicted_post.append(str_predict_label)
    submit['attribute_ids'] = predicted_post
    submit.to_csv('submission.csv', index=False)
