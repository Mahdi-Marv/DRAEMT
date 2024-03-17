import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from perlin import rand_perlin_2d_np
from imagenet_30 import IMAGENET30_TEST_DATASET
from torchvision import transforms
import random
from torchvision import transforms as T
from PIL import Image


def center_paste(large_img, small_img):
    # Calculate the center position
    large_width, large_height = large_img.size
    small_width, small_height = small_img.size

    # Calculate the top-left position
    left = (large_width - small_width) // 2
    top = (large_height - small_height) // 2

    # Create a copy of the large image to keep the original unchanged
    result_img = large_img.copy()

    # Paste the small image onto the large one at the calculated position
    result_img.paste(small_img, (left, top))

    return result_img


def mod(img1):
    img1 = np.array(img1)
    # img1 = img1[:, :, ::-1]

    img1 = img1 / 255.0

    if len(img1.shape) == 2:
        img1 = np.stack((img1,) * 3, axis=-1)
    elif img1.shape[2] > 3:  # For RGBA images or images with more than 3 channels
        img1 = img1[:, :, :3]  # Keep only the first 3 channels

    img1 = img1.astype(np.float32)

    img1 = np.transpose(img1, (2, 0, 1))

    return img1


class MVTecDRAEMTestDataset(Dataset):

    def __init__(self, root_dir, shrink_factor=1, resize_shape=None):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir + "/*/*.png"))
        self.resize_shape = int(resize_shape * shrink_factor)
        # self.transform = T.Compose([T.Resize(256),
        #                             T.ToTensor(),
        #                             ])

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]))
        if self.resize_shape is not None:
            image = cv2.resize(image, dsize=(256, 256))
            mask = cv2.resize(mask, dsize=(256, 256))
            # print(mask)

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def transform_and_paste(self, image_path, mask_path):
        imagenet_30 = IMAGENET30_TEST_DATASET()
        background_image_pil = imagenet_30[int(random.random() * len(imagenet_30))][0].resize((256, 256))

        # Load and transform the foreground image using OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]))
        if self.resize_shape is not None:
            image = cv2.resize(image, dsize=(256, 256))
            mask = cv2.resize(mask, dsize=(256, 256))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        # Convert the OpenCV image (BGR) to PIL format (RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed_image_pil = Image.fromarray((image_rgb * 255).astype(np.uint8))

        # Calculate the position to paste the transformed image on the background
        background_width, background_height = background_image_pil.size
        foreground_width, foreground_height = transformed_image_pil.size
        position = ((background_width - foreground_width) // 2, (background_height - foreground_height) // 2)

        # Paste the transformed image onto the background PIL image
        background_image_pil.paste(transformed_image_pil, position, transformed_image_pil)

        # Convert the modified PIL image back to a NumPy array in BGR format for OpenCV compatibility
        final_image_np = np.array(background_image_pil)
        final_image_np = cv2.cvtColor(final_image_np, cv2.COLOR_RGB2BGR)

        # Ensure the final image is in the same data type as the original OpenCV image, typically uint8
        final_image_np = final_image_np.astype(np.float32)/255

        return final_image_np, mask  # Returning the final image as a NumPy array and the mask

    def __getitem__(self, idx):
        imagenet_30 = IMAGENET30_TEST_DATASET()
        imagenet30_img = imagenet_30[int(random.random() * len(imagenet_30))][0].resize((256, 256))

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        # img1 = Image.open(img_path).convert('RGB')
        # img1 = np.array(img1)
        # img1 = img1[:, :, ::-1]
        # img1 = Image.fromarray(img1)
        # img1 = img1.resize((self.resize_shape, self.resize_shape))
        #
        # img1 = center_paste(imagenet30_img, img1)
        #
        # img1 = mod(img1)

        # if self.resize_shape is not None:
        #     resizeTransf = transforms.Resize(self.resize_shape)
        #     img1 = resizeTransf(img1)

        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_and_paste(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0] + "_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_and_paste(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        # image = center_paste(imagenet30_img, img1)
        #
        # image = np.array(image).reshape((256, 256, 3)).astype(np.float32)
        # image = image / 255.0
        # image = np.transpose(image, (2, 0, 1))

        # print(has_anomaly)



        sample = {'image': img1, 'has_anomaly': has_anomaly, 'mask': mask, 'idx': idx}

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

        self.image_paths = sorted(glob.glob(root_dir + "/*.png"))

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
