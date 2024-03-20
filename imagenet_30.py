from torch.utils.data import Dataset
from PIL import ImageFilter, Image, ImageOps
from torchvision.datasets.folder import default_loader
import os
import cv2


class IMAGENET30_TEST_DATASET(Dataset):
    def __init__(self, root_dir="/kaggle/input/imagenet30-dataset/one_class_test/one_class_test/", transform=None):
        """
        Args:
            root_dir (string): Directory with all the classes.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_path_list = []
        self.targets = []

        # Map each class to an index
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(root_dir)))}
        # print(f"self.class_to_idx in ImageNet30_Test_Dataset:\n{self.class_to_idx}")

        # Walk through the directory and collect information about the images and their labels
        for i, class_name in enumerate(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            for instance_folder in os.listdir(class_path):
                instance_path = os.path.join(class_path, instance_folder)
                if instance_path != "/kaggle/input/imagenet30-dataset/one_class_test/one_class_test/airliner/._1.JPEG":
                    for img_name in os.listdir(instance_path):
                        if img_name.endswith('.JPEG'):
                            img_path = os.path.join(instance_path, img_name)
                            # image = Image.open(img_path).convert('RGB')
                            self.img_path_list.append(img_path)
                            self.targets.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        label = self.targets[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        return img, label
