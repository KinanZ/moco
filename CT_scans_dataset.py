import os
import json
from PIL import Image
import numpy as np
import torch

from torch.utils.data import Dataset


class brain_CT_scan(Dataset):
    """Brain CT Scans dataset."""

    def __init__(self, json_file, root_dir, transform=None, num_channels=3, moco=True, num_classes=15):
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

        self.images = {}
        for i in range(len(self.dataset_annotations)):
            img_iid = self.dataset_annotations[i]['iid']
            img_path = os.path.join(self.root_dir, '{0:07d}.jpg'.format(img_iid))
            image = np.array(Image.open(img_path)).astype(np.float32)
            self.images[img_iid] = image

        assert transform is not None
        self.transform = transform

        self.num_channels = num_channels
        self.moco = moco
        self. num_classes= num_classes

    def __len__(self):
        return len(self.dataset_annotations)

    def __getitem__(self, idx):
        if self.num_channels == 1:
            img_iid = self.dataset_annotations[idx]['iid']
            image = self.images[img_iid]
            image = torch.from_numpy(image).unsqueeze(dim=0)
        elif self.num_channels == 3:
            img_iid_mid = self.dataset_annotations[idx]['iid']
            image_mid = self.images[img_iid_mid]
            try:
                img_iid_pre = self.dataset_annotations[idx - 1]['iid']
            except:
                img_iid_pre = self.dataset_annotations[idx]['iid']
            image_pre = self.images[img_iid_pre]
            try:
                img_iid_post = self.dataset_annotations[idx + 1]['iid']
            except:
                img_iid_post = self.dataset_annotations[idx]['iid']
            image_post = self.images[img_iid_post]

            image = np.dstack((image_pre, image_mid, image_post))
            image = torch.from_numpy(image).permute(2, 0, 1)

        image1 = self.transform(image)
        if self.moco:
            image2 = self.transform(image)
            return (image1, image2)
        else:
            classes = self.dataset_annotations[idx]['labels']
            labels = np.zeros(self.num_classes).astype(np.uint8)
            labels[classes] = 1
            return image1, torch.tensor(labels, dtype=torch.float32)
