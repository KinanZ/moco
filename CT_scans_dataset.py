import os
import json
from PIL import Image
import numpy as np
import torch
import random

from torch.utils.data import Dataset


class brain_CT_scan_moco(Dataset):
    """Brain CT Scans dataset."""

    def __init__(self, root_dir, transform=None, stack_pre_post=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            stack_pre_post (bool): if True -> the previous and post slices are stacked to the main slice and become a 3-channel input for the model.
        """
        self.root_dir = root_dir
        self.img_list = sorted(os.listdir("/misc/lmbraid19/argusm/CLUSTER/multimed/NSEG2015_2/JPEGImages/"))

        assert transform is not None
        self.transform = transform

        self.stack_pre_post = stack_pre_post

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if not self.stack_pre_post:
            img_name = os.path.join(self.root_dir, self.img_list[idx])
            image = np.array(Image.open(img_name)).astype(np.float32)
            image = np.dstack((image, image, image))
            image = torch.from_numpy(image).permute(2, 0, 1)
        else:
            img_name_mid = os.path.join(self.root_dir, self.img_list[idx])
            try:
                img_name_pre = os.path.join(self.root_dir, self.img_list[idx - 1])
            except:
                # if idx == 0
                img_name_pre = img_name_mid

            try:
                img_name_post = os.path.join(self.root_dir, self.img_list[idx + 1])
            except:
                # if idx == len(self.img_list)
                img_name_post = img_name_mid

            image_mid = np.array(Image.open(img_name_mid)).astype(np.float32)
            image_pre = np.array(Image.open(img_name_pre)).astype(np.float32)
            image_post = np.array(Image.open(img_name_post)).astype(np.float32)
            image = np.dstack((image_pre, image_mid, image_post))
            image = torch.from_numpy(image).permute(2, 0, 1)

        image1 = self.transform(image)
        image2 = self.transform(image)
        return (image1, image2)


class brain_CT_scan(Dataset):
    """Brain CT Scans dataset."""

    def __init__(self, json_file, root_dir, transform=None, stack_pre_post=True, num_classes=15, bbox_aug=False):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            stack_pre_post (bool): if True -> the previous and post slices are stacked to the main slice and become a 3-channel input for the model.
            num_classes (int): number of categories in the dataset
        """
        with open(json_file) as f_obj:
            self.dataset_annotations = json.load(f_obj)["questions"]
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = num_classes
        self.stack_pre_post = stack_pre_post
        self.bbox_aug = bbox_aug

        self.y = np.zeros((len(self.dataset_annotations), self.num_classes)).astype(np.uint8)
        for i in range(len(self.dataset_annotations)):
            classes = self.dataset_annotations[i]['labels']
            self.y[i][classes] = 1


    def __len__(self):
        return len(self.dataset_annotations)

    def __getitem__(self, idx):
        if not self.stack_pre_post:
            img_name = os.path.join(self.root_dir, '{0:07d}.jpg'.format(self.dataset_annotations[idx]['iid']))
            image = np.array(Image.open(img_name)).astype(np.float32)
            image = np.dstack((image, image, image))
        else:
            img_name_mid = os.path.join(self.root_dir, '{0:07d}.jpg'.format(self.dataset_annotations[idx]['iid']))
            try:
                img_name_pre = os.path.join(self.root_dir, '{0:07d}.jpg'.format(self.dataset_annotations[idx-1]['iid']))
            except:
                # if idx == 0
                img_name_pre = img_name_mid

            try:
                img_name_post = os.path.join(self.root_dir, '{0:07d}.jpg'.format(self.dataset_annotations[idx+1]['iid']))
            except:
                # if idx == len(self.img_list)
                img_name_post = img_name_mid

            image_mid = np.array(Image.open(img_name_mid)).astype(np.float32)
            image_pre = np.array(Image.open(img_name_pre)).astype(np.float32)
            image_post = np.array(Image.open(img_name_post)).astype(np.float32)
            image = np.dstack((image_pre, image_mid, image_post))

        classes = self.dataset_annotations[idx]['labels']
        labels = np.zeros(self.num_classes).astype(np.uint8)
        labels[classes] = 1
        bboxes = self.dataset_annotations[idx]['bboxes']

        if self.bbox_aug:
            image = crop_show_augment(image, labels, bboxes)

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels, dtype=torch.float32)


def clip(box):
    new_box = (max(min(int(round(int(round(box[0])))), 512), 0),
               max(min(int(round(int(round(box[1])))), 512), 0))
    return new_box


def stretch(bbox, factor=.2):
    # Arguments:
    bbox2 = []
    for dim in ((bbox[0], bbox[2]), (bbox[1], bbox[3])):
        cur_min, cur_max = dim
        rnd_min, rnd_max = clip((cur_min - np.random.chisquare(df=3) / 8 * cur_min,
                                 cur_max + np.random.chisquare(df=3) / 8 * (512 - cur_max)))

        bbox2.append((rnd_min, rnd_max))
    return (bbox2[0][0], bbox2[1][0], bbox2[0][1], bbox2[1][1])


def crop_show_augment(image, labels, bboxes):
    print('bboxes', bboxes)
    # show the diseased areas based on bounding boxes
    tmp = np.zeros((512, 512, 3), dtype=np.uint8)
    if labels[0] == 1:
        bboxes = random.sample(range(48, 464), 2)
        bboxes.append(random.randint(bboxes[0], 464))
        bboxes.append(random.randint(bboxes[1], 464))
        bboxes = [bboxes]
    for b in bboxes:
        b = stretch(b)
        tmp[b[1]:b[3], b[0]:b[2], :] = np.asarray(image)[b[1]:b[3], b[0]:b[2], :]
    return tmp
