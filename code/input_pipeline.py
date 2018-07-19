from torch.utils.data import Dataset
import pandas as pd
import os
from skimage import io
import pickle
import torch


class FeatureExtraction(Dataset):

    def __init__(self, training_path, testing_path, root_dir, post_map_path, transform=None):

        with open(training_path, 'r') as f:
            data = [el.split(',')[1] for el in f.readlines()]
        with open(testing_path, 'r') as f:
            data.extend([el.split(',')[1] for el in f.readlines()])

        self.input_points = list(set(data))
        with open(post_map_path, 'w') as w:
            pickle.dump({self.input_points[i]: i for i in range(len(self.input_points))}, w)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.input_points)

    def __getitem__(self, idx):
        try:
            img_name = os.path.join(self.root_dir, self.input_points[idx]) + '.jpg'
            image = io.imread(img_name)

        except ValueError as e:
            print 'Value Error in Dataset Loading'
            print 'positive:', img_name
            raise e

        sample = {'code': img_name, 'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample


class BrandDataset(Dataset):

    def __init__(self, data_file, code_list, image_feature, brand_list):
        self.input_points = pd.read_csv(data_file).values
        self.code_list = code_list
        self.image_features = image_feature
        self.brand_list = brand_list

    def __len__(self):
        return len(self.input_points)

    def __getitem__(self, idx):
        try:
            img_name_p = self.input_points[idx, 1]
            index = self.code_list[img_name_p]
            image_p = torch.from_numpy(self.image_features[index, :])
        except ValueError as e:
            print 'Value Error in Dataset Loading'
            print 'positive:', img_name_p
            raise e
        try:
            img_name_n = self.input_points[idx, 2]
            index = self.code_list[img_name_n]
            image_n = torch.from_numpy(self.image_features[index, :])
        except ValueError as e:
            print 'Value Error in Dataset Loading'
            print 'negative:', img_name_n
            raise e
    
        sample = {'code_p': img_name_p,
                  'image_p': image_p,
                  'code_n': img_name_n,
                  'image_n': image_n,
                  'brand': self.brand_list.index(self.input_points[idx, 0])}

        return sample
