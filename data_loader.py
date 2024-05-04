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
import pydicom


def process_dicom_image(image_path, resize_shape):
    # Step 1: Read the DICOM file
    dicom = pydicom.dcmread(image_path)

    # Step 2: Access the pixel data
    pixel_array = dicom.pixel_array

    # Step 3: Normalize and scale pixel data if it's not in uint8 format
    if dicom.pixel_array.dtype != np.uint8:
        # Normalize the pixel values to the range [0, 255]
        pixel_array = pixel_array.astype(float)
        pixel_array -= pixel_array.min()  # Normalize to 0
        pixel_array /= pixel_array.max()  # Normalize to 1
        pixel_array *= 255.0
        pixel_array = np.uint8(pixel_array)

    # Step 4: Convert grayscale to BGR (3 channels) if needed
    if len(pixel_array.shape) == 2:
        image = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2BGR)
    else:
        image = pixel_array

    # Step 5: Resize the image
    image = cv2.resize(image, (resize_shape[1], resize_shape[0]))  # Note the order of dimensions

    return image


def preprocess_dicom_image(image_path):
    # Step 1: Read the DICOM file using pydicom
    dicom = pydicom.dcmread(image_path)

    # Step 2: Extract the pixel data
    pixel_array = dicom.pixel_array

    # Step 3: Handle different data types and normalize
    if dicom.pixel_array.dtype != np.uint8:
        # Normalize the pixel values if not already in uint8 format
        pixel_array = pixel_array.astype(float)
        pixel_array = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0
        pixel_array = np.uint8(pixel_array)

    # Step 4: Convert to 3-channel BGR image if it's a grayscale image
    if len(pixel_array.shape) == 2:
        image = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2BGR)
    else:
        # Assuming the pixel_array is already in a format that cv2 expects for color images
        # This might not always be the case, as DICOM can store color images in various formats.
        # Additional conversion may be necessary depending on the color format.
        image = pixel_array

    # Optional: Perform additional preprocessing here (e.g., resizing, normalization)
    # For example, resize the image if needed:
    # image = cv2.resize(image, (desired_width, desired_height))

    return image

class MVTecDRAEMTestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None, test_id=1):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir + "/*/*.png"))
        self.resize_shape = resize_shape
        self.test_id = test_id

        test_normal_path = glob.glob('/kaggle/working/test/normal/*')
        test_anomaly_path = glob.glob('/kaggle/working/test/anomaly/*')

        self.test_path = test_normal_path + test_anomaly_path
        self.test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)

        if self.test_id == 2:
            test_normal_path = glob.glob('/kaggle/working/chest_xray/test/NORMAL/*')
            test_anomaly_path = glob.glob('/kaggle/working/chest_xray/test/PNEUMONIA/*')

            self.test_path = test_normal_path + test_anomaly_path
            self.test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)

    def __len__(self):
        return len(self.test_path)

    def transform_image(self, image_path, mask_path, test_id=1):
        if test_id == 1:
            image = preprocess_dicom_image(image_path)
        else:
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
        image, _ = self.transform_image(img_path, None, self.test_id)

        has_anomaly = np.array([0], dtype=np.float32) if self.test_label[idx] == 0 else np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly, 'mask': 'd', 'idx': idx}

        return sample


class MVTecDRAEMTrainDataset(Dataset):

    def __init__(self, root_dir, anomaly_source_path, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape = resize_shape

        # self.image_paths = sorted(glob.glob(root_dir+"/*.png"))

        self.image_paths = glob.glob('/kaggle/working/train/normal/*')

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
        # image = cv2.imread(image_path)
        # image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        image = process_dicom_image(image_path, resize_shape=self.resize_shape)

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







