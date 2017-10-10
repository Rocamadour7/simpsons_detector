import os
from PIL import Image
import numpy as np
from math import floor
import pandas as pd

class Dataset:
    def __init__(self, batch_size, img_size=28, folder='D:\Code\Kaggle_Data\The_Simpsons'):
        self.batch_size = batch_size
        self.img_size = img_size
        self.folder = folder
        self.num_classes = len(os.listdir(os.path.join(folder,'simpsons_dataset')))

        self.csv_file = pd.read_csv(os.path.join(self.folder, 'number_pic_char.csv'))
        self.csv_file = self.csv_file.sort_values('name')

        train_files, self.train_targets, val_files, self.val_targets, test_files, self.test_targets = self.train_valid_test_split(
            os.listdir(os.path.join(self.folder, 'simpsons_dataset')), self.csv_file)

        self.characters_index = os.listdir(os.path.join(folder,'simpsons_dataset'))

        train_files, self.train_targets = self._shuffle_in_unison(train_files, self.train_targets)
        val_files, self.val_targets = self._shuffle_in_unison(val_files, self.val_targets)
        test_files, self.test_targets = self._shuffle_in_unison(test_files, self.test_targets)

        self.train_inputs = self.file_paths_to_images(train_files)
        self.val_inputs = self.file_paths_to_images(val_files)
        self.test_inputs = self.file_paths_to_images(test_files)

        self.pointer = 0

    def file_paths_to_images(self, files_list):
        inputs = []
        for file in files_list:
            test_image = Image.open(file).resize((self.img_size, self.img_size))
            test_image = np.array(test_image, dtype=np.float32)
            inputs.append(test_image)
        inputs = np.array(inputs)
        return inputs

    def train_valid_test_split(self, train_dir, csv):
        train_X = []
        train_y = []
        val_X = []
        val_y = []
        test_X = []
        test_y = []
        for index, each_char in enumerate(sorted(train_dir)):
            test_num = csv['test'][index]
            val_num = test_num
            train_num = csv['train'][index] - val_num
            char_dir = os.path.join(self.folder, 'simpsons_dataset', each_char)
            for i in range(train_num):
                img_name = os.listdir(char_dir)[i]
                img_path = os.path.join(char_dir, img_name)
                train_X.append(img_path)
                train_y.append(index)
            for i in range(train_num, train_num + val_num):
                img_name = os.listdir(char_dir)[i]
                img_path = os.path.join(char_dir, img_name)
                val_X.append(img_path)
                val_y.append(index)
            for i in range(train_num + val_num, train_num + val_num + test_num):
                img_name = os.listdir(char_dir)[i]
                img_path = os.path.join(char_dir, img_name)
                test_X.append(img_path)
                test_y.append(index)
        train_y = np.array(train_y)
        val_y = np.array(val_y)
        test_y = np.array(test_y)
        train_y = self._convertToOneHot(train_y, num_classes=self.num_classes)
        val_y = self._convertToOneHot(val_y, num_classes=self.num_classes)
        test_y = self._convertToOneHot(test_y, num_classes=self.num_classes)
        return (
            train_X, train_y,
            val_X, val_y,
            test_X, test_y
        )

    def _convertToOneHot(self, vector, num_classes=None):
        """
        Converts an input 1-D vector of integers into an output
        2-D array of one-hot vectors, where an i'th input value
        of j will set a '1' in the i'th row, j'th column of the
        output array.

        Created by: https://stackoverflow.com/users/4561314/stackoverflowuser2010

        Example:
            v = np.array((1, 0, 4))
            one_hot_v = convertToOneHot(v)
            print one_hot_v

            [[0 1 0 0 0]
            [1 0 0 0 0]
            [0 0 0 0 1]]
        """

        assert isinstance(vector, np.ndarray)
        assert len(vector) > 0

        if num_classes is None:
            num_classes = np.max(vector)+1
        else:
            assert num_classes > 0
            assert num_classes >= np.max(vector)

        result = np.zeros(shape=(len(vector), num_classes))
        result[np.arange(len(vector)), vector] = 1
        return result.astype(int)

    def _shuffle_in_unison(self, a, b):
        '''
        Shuffles two arrays by the same index, using Numpy permutation function.
        Created by: https://stackoverflow.com/users/190280/josh-bleecher-snyder
        '''
        assert len(a) == len(b)
        permutation = np.random.permutation(len(a))
        for old_index, new_index in enumerate(permutation):
            a[new_index] = a[old_index]
            b[new_index] = b[old_index]
        return a, b

    def num_batches_in_epoch(self):
        return int(floor(len(self.train_inputs) / self.batch_size))

    def reset_batch_pointer(self):
        permutation = np.random.permutation(len(self.train_inputs))
        self.train_inputs = [self.train_inputs[i] for i in permutation]
        self.train_targets = [self.train_targets[i] for i in permutation]

        self.pointer = 0

    @property
    def next_batch(self):
        inputs = []
        targets = []
        for i in range(self.batch_size):
            inputs.append(np.array(self.train_inputs[self.pointer + i]))
            targets.append(np.array(self.train_targets[self.pointer + i]))

        self.pointer += self.batch_size

        return np.array(inputs, dtype=np.uint8), np.array(targets, dtype=np.uint8)

    @property
    def val_set(self):
        return np.array(self.val_inputs, dtype=np.uint8), np.array(self.val_targets, dtype=np.uint8)

    @property
    def test_set(self):
        return np.array(self.test_inputs, dtype=np.uint8), np.array(self.test_targets, dtype=np.uint8)
