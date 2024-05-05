import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from perlin import rand_perlin_2d_np
import matplotlib.pyplot as plt
import random
import pandas as pd

# def get_waterbird_trainset():
#     train_set = MVTecDRAEMTrainDataset(count_train_landbg=3500, count_train_waterbg=100, mode='bg_all')
#     return train_set
# def get_waterbird_test_set():
#     test_set = MVTecDRAEMTestDataset(count_train_landbg=3500, count_train_waterbg=100, mode='bg_land')
#     return test_set
#
# def get_waterbird_just_test_shifted():
#     test_set = MVTecDRAEMTestDataset(count_train_landbg=3500, count_train_waterbg=100, mode='bg_water')
#     return test_set

class MVTecDRAEMTestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None, test_id=1, count_train_landbg=3500, count_train_waterbg=100, mode='bg_all'):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir + "/*/*.png"))
        self.resize_shape = resize_shape
        self.test_id = test_id
        copy = False

        root = '/kaggle/input/waterbird/waterbird'
        df = pd.read_csv(os.path.join(root, 'metadata.csv'))

        print(len(df))

        self.train = train
        self.df = df
        lb_on_l = df[(df['y'] == 0) & (df['place'] == 0)]
        lb_on_w = df[(df['y'] == 0) & (df['place'] == 1)]
        self.normal_paths = []
        self.labels = []

        normal_df = lb_on_l.iloc[:count_train_landbg]
        normal_df_np = normal_df['img_filename'].to_numpy()
        self.normal_paths.extend([os.path.join(root, x) for x in normal_df_np][:count_train_landbg])
        normal_df = lb_on_w.iloc[:count_train_waterbg]
        normal_df_np = normal_df['img_filename'].to_numpy()
        copy_count = 1
        if copy:
            copy_count = count_train_landbg // count_train_waterbg
        for _ in range(copy_count):
            self.normal_paths.extend([os.path.join(root, x) for x in normal_df_np][:count_train_waterbg])

        if train:
            self.image_paths = self.normal_paths
        else:
            self.image_paths = []
            if mode == 'bg_all':
                dff = df
            elif mode == 'bg_water':
                dff = df[(df['place'] == 1)]
            elif mode == 'bg_land':
                dff = df[(df['place'] == 0)]
            elif mode == 'ood':
                dff = df[(df['place'] == 0) & (df['y'] == 1)]
            else:
                print('Wrong mode!')
                raise ValueError('Wrong bg mode!')
            all_paths = dff[['img_filename', 'y']].to_numpy()
            for i in range(len(all_paths)):
                full_path = os.path.join(root, all_paths[i][0])
                if full_path not in self.normal_paths:
                    self.image_paths.append(full_path)
                    self.labels.append(all_paths[i][1])

    def __len__(self):
        return len(self.test_path)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.test_path[idx]
        image, _ = self.transform_image(img_path, None)

        has_anomaly = np.array([0], dtype=np.float32) if self.labels[idx] == 0 else np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly, 'mask': 'd', 'idx': idx}

        return sample


class MVTecDRAEMTrainDataset(Dataset):

    def __init__(self, root_dir, anomaly_source_path, resize_shape=None, count_train_landbg=3500, count_train_waterbg=100, mode='bg_all'):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape = resize_shape
        copy = False

        root = '/kaggle/input/waterbird/waterbird'
        df = pd.read_csv(os.path.join(root, 'metadata.csv'))

        print('full dataset len: ', len(df))

        self.train = train
        self.df = df
        lb_on_l = df[(df['y'] == 0) & (df['place'] == 0)]
        lb_on_w = df[(df['y'] == 0) & (df['place'] == 1)]
        self.normal_paths = []
        self.labels = []

        normal_df = lb_on_l.iloc[:count_train_landbg]
        normal_df_np = normal_df['img_filename'].to_numpy()
        self.normal_paths.extend([os.path.join(root, x) for x in normal_df_np][:count_train_landbg])
        normal_df = lb_on_w.iloc[:count_train_waterbg]
        normal_df_np = normal_df['img_filename'].to_numpy()
        copy_count = 1
        if copy:
            copy_count = count_train_landbg // count_train_waterbg
        for _ in range(copy_count):
            self.normal_paths.extend([os.path.join(root, x) for x in normal_df_np][:count_train_waterbg])

        if train:
            self.image_paths = self.normal_paths
        else:
            self.image_paths = []
            if mode == 'bg_all':
                dff = df
            elif mode == 'bg_water':
                dff = df[(df['place'] == 1)]
            elif mode == 'bg_land':
                dff = df[(df['place'] == 0)]
            elif mode == 'ood':
                dff = df[(df['place'] == 0) & (df['y'] == 1)]
            else:
                print('Wrong mode!')
                raise ValueError('Wrong bg mode!')
            all_paths = dff[['img_filename', 'y']].to_numpy()
            for i in range(len(all_paths)):
                full_path = os.path.join(root, all_paths[i][0])
                if full_path not in self.normal_paths:
                    self.image_paths.append(full_path)
                    self.labels.append(all_paths[i][1])


        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path + "/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def __len__(self):
        return len(self.image_paths)

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0], dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1 - msk) * image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly = 0.0
            return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)

    def transform_image(self, image_path, anomaly_source_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(self.image_paths[idx],
                                                                                 self.anomaly_source_paths[
                                                                                     anomaly_source_idx])
        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}

        return sample







