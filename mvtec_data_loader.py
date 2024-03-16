from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import torch.utils.data as data
import matplotlib.image as mpimg
from torchvision import transforms
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


class MVTEC(data.Dataset):
    """`MVTEC <https://www.mvtec.com/company/research/datasets/mvtec-ad/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directories
            ``bottle``, ``cable``, etc., exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        resize (int, optional): Desired output image size.
        interpolation (int, optional): Interpolation method for downsizing image.
        category: bottle, cable, capsule, etc.
    """

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 category='carpet', resize=None, interpolation=2, use_imagenet=False,
                 select_random_image_from_imagenet=False, shrink_factor=0.9):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize = resize
        if use_imagenet:
            self.resize = int(resize * shrink_factor)
        self.interpolation = interpolation
        self.select_random_image_from_imagenet = select_random_image_from_imagenet

        # load images for training
        if self.train:
            self.train_data = []
            self.train_labels = []
            cwd = os.getcwd()
            trainFolder = self.root + '/' + category + '/train/good/'
            os.chdir(trainFolder)
            filenames = [f.name for f in os.scandir()]
            for file in filenames:
                img = mpimg.imread(file)
                img = img * 255
                img = img.astype(np.uint8)
                self.train_data.append(img)
                self.train_labels.append(1)
            os.chdir(cwd)

            self.train_data = np.array(self.train_data)
        else:
            # load images for testing
            self.test_data = []
            self.test_labels = []

            cwd = os.getcwd()
            testFolder = self.root + '/' + category + '/test/'
            os.chdir(testFolder)
            subfolders = [sf.name for sf in os.scandir() if sf.is_dir()]
            #             print(subfolders)
            cwsd = os.getcwd()

            # for every subfolder in test folder
            for subfolder in subfolders:
                label = 0
                if subfolder == 'good':
                    label = 1
                testSubfolder = testFolder + subfolder + '/'
                #                 print(testSubfolder)
                os.chdir(testSubfolder)
                filenames = [f.name for f in os.scandir()]
                for file in filenames:
                    img = mpimg.imread(file)
                    img = img * 255
                    img = img.astype(np.uint8)
                    self.test_data.append(img)
                    self.test_labels.append(label)
                os.chdir(cwsd)
            os.chdir(cwd)

            self.test_data = np.array(self.test_data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.select_random_image_from_imagenet:
            imagenet30_img = imagenet30_testset[int(random.random() * len(imagenet30_testset))][0].resize((224, 224))
        else:
            imagenet30_img = imagenet30_testset[100][0].resize((224, 224))

        # if resizing image
        if self.resize is not None:
            resizeTransf = transforms.Resize(self.resize, self.interpolation)
            img = resizeTransf(img)

        #         print(f"imagenet30_img.size: {imagenet30_img.size}")
        #         print(f"img.size: {img.size}")
        img = center_paste(imagenet30_img, img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """
        Args:
            None
        Returns:
            int: length of array.
        """
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)