# image_classification_project
kaggle图像分类比赛项目，提供了项目bsaeline
## data direcrory
```Shell
├── dataset
│   ├── sample_submission.csv
│   ├── test
│   ├── train
│   └── train.csv
├── __init__.py
├── logs
│   ├── models
│   └── training_log.csv
├── main.py
├── model.py
└── submit
```
## [iMet Collection 2019 - FGVC6](https://www.kaggle.com/c/imet-2019-fgvc6)
### Data_Generator
Here is code:
```Python
# Artwork dataset generator
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
        """Gets batch at position `index`.
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
```
## [Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection)
### Data_Generator
Here are data generator code:
```Python
# Cancer dataset generator
class CancerDataset(Sequence):
    """input data generator, a sequence object
    # Argument:
        @label_path: The path of train_label.csv, file
        @batch_size: The size of input_data's batch, int
        @target_size: The image size, tuple
        @mode: default is to 'train', str
        @aug: default is to True, bool
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
```
## Usage
1. git clone 
2. Configure the environment and change data drectory into my data directory form
3. `Python the main.py`
