import os
import json
from PIL import Image
import numpy as np
import torch

from torch.utils.data import Dataset


class brain_CT_scan(Dataset):
    """Brain CT Scans dataset."""

    def __init__(self, json_file, root_dir, transform=None, num_channels=3):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            num_channels (int): if 1 -> one slice is the input for the model,
                if 3 -> the previous and post slices are stacked to the main slice and become a 3-channel input for the model.
        """
        with open(json_file) as f_obj:
            self.dataset_annotations = json.load(f_obj)["questions"]
        self.root_dir = root_dir

        assert transform is not None
        self.transform = transform

        self.num_channels = num_channels

    def __len__(self):
        return len(self.dataset_annotations)

    def __getitem__(self, idx):
        if self.num_channels == 1:
            img_name = os.path.join(self.root_dir, '{0:07d}.jpg'.format(self.dataset_annotations[idx]['iid']))
            image = np.array(Image.open(img_name)).astype(np.float32)
            image = torch.from_numpy(image).unsqueeze(dim=0)
        elif self.num_channels == 3:
            img_name_mid = os.path.join(self.root_dir, '{0:07d}.jpg'.format(self.dataset_annotations[idx]['iid']))

            try:
                img_name_pre = os.path.join(self.root_dir,
                                            '{0:07d}.jpg'.format(self.dataset_annotations[idx - 1]['iid']))
            except:
                # if idx == 0
                img_name_pre = os.path.join(self.root_dir,
                                            '{0:07d}.jpg'.format(self.dataset_annotations[idx]['iid']))

            try:
                img_name_post = os.path.join(self.root_dir,
                                             '{0:07d}.jpg'.format(self.dataset_annotations[idx + 1]['iid']))
            except:
                # if idx == len(self.dataset_annotations)
                img_name_post = os.path.join(self.root_dir,
                                             '{0:07d}.jpg'.format(self.dataset_annotations[idx]['iid']))

            image_mid = np.array(Image.open(img_name_mid)).astype(np.float32)
            image_pre = np.array(Image.open(img_name_pre)).astype(np.float32)
            image_post = np.array(Image.open(img_name_post)).astype(np.float32)
            image = np.dstack((image_pre, image_mid, image_post))
            image = torch.from_numpy(image).permute(2, 0, 1)

        image1 = self.transform(image)
        image2 = self.transform(image)

        return (image1, image2)
