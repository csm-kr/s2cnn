from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import os
import glob
import torch
import numpy as np
from examples.mnist.gendata import get_projection_grid, project_2d_on_sphere_sun360, rand_rotation_matrix, rotate_grid
import cv2
from utils import rotate_map_given_R, calculate_Rmatrix_from_phi_theta, show_spheres


class SUN360Dataset(Dataset):

    # class_names = ('bathroom', 'beach', 'cave', 'church',
    #                'desert', 'field ', 'forest', 'mountain', 'theater',
    #                'train_interior')

    def __init__(self, root, split, vis=False, rotate=True):
        # indoor vs outdoor classification
        self.root = os.path.join(root, "pano1024x512")
        # self.img_indoor_path = glob.glob(os.path.join(self.root, 'indoor', '*/*.jpg'))
        # self.img_outdoor_path = glob.glob(os.path.join(self.root, 'outdoor', '*/*.jpg'))

        self.img_indoor_path = glob.glob(os.path.join(self.root, 'indoor_sample', '*/*.jpg'))
        self.img_outdoor_path = glob.glob(os.path.join(self.root, 'outdoor_sample', '*/*.jpg'))

        self.img_path = self.img_indoor_path + self.img_outdoor_path
        # self.img_others_path = glob.glob(os.path.join(self.root, 'others', '*.jpg'))

        ratio = 0.7
        num_train_data = int(ratio * len(self.img_path))
        np.random.seed(1)
        train_data_path = sorted(np.random.choice(self.img_path, num_train_data, replace=False))
        test_data_path = sorted(list(set(self.img_path) - set(train_data_path)))

        assert len(train_data_path) + len(test_data_path) == len(self.img_path)

        self.split = split
        if self.split == 'train':
            self.img_path = train_data_path
        elif self.split == 'test':
            self.img_path = test_data_path

        self.rotate = rotate
        self.vis = vis
        super().__init__()

    def __getitem__(self, idx):

        img = cv2.imread(self.img_path[idx])         # BGR
        img = cv2.imread("D:\data\SUN360_panoramas_1024x512\pano1024x512\outdoor\others\pano_aaartbimirvryq.jpg")         # BGR
        # print(self.img_path[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # RGB
        img_np = cv2.resize(img, (224, 224))

        # FIXME
        bandwidth = 112
        grid = get_projection_grid(b=bandwidth)
        if self.rotate:
            rot = rand_rotation_matrix()
            rotated_grid = rotate_grid(rot, grid)
            map_x, map_y = rotate_map_given_R(rot, bandwidth * 2, bandwidth * 2)
            img_np = cv2.remap(img_np, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
        else:
            rotated_grid = grid

        if self.vis:
            img_np_vis = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)   # RGB
            cv2.imshow('rotated_img', img_np_vis)
            cv2.waitKey(0)
            img_np_ = np.transpose(img_np, (2, 0, 1))
            show_spheres(scale=2, points=rotated_grid, rgb=img_np_)
        # R = calculate_Rmatrix_from_phi_theta(0, 0)

        img_np = np.transpose(img_np, (2, 0, 1))                # [3, 224, 224]
        img_torch = torch.FloatTensor(img_np)                   # [3, 224, 224]

        if "indoor" in self.img_path[idx]:
            label = torch.zeros(1).type(torch.long)
        elif "outdoor" in self.img_path[idx]:
            label = torch.ones(1).type(torch.long)

        return img_torch, label

    def __len__(self):
        return len(self.img_path)


if __name__ == '__main__':
    root = "D:\data\SUN360_panoramas_1024x512"
    dataset = SUN360Dataset(root, 'train')
    dataloader = DataLoader(dataset=dataset,
                            batch_size=4,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=4)

    for img, label in dataloader:
        print(label)
