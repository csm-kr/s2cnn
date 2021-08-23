from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import os
import glob
from PIL import Image
import numpy as np
from examples.mnist.gendata import get_projection_grid, project_2d_on_sphere_sun360
import cv2
from utils import rotate_map_given_R, calculate_Rmatrix_from_phi_theta, show_spheres


class SUN360Dataset(nn.Module):
    def __init__(self, root):
        self.root = os.path.join(root, "pano1024x512")
        self.img_indoor_path = glob.glob(os.path.join(self.root, 'indoor', '*/*.jpg'))
        self.img_outdoor_path = glob.glob(os.path.join(self.root, 'outdoor', '*/*.jpg'))
        self.img_others_path = glob.glob(os.path.join(self.root, 'others', '*.jpg'))

        print(self.img_indoor_path)
        print(self.img_outdoor_path)
        print(self.img_others_path)
        super().__init__()

    def __getitem__(self, idx):
        img = cv2.imread(self.img_indoor_path[idx])  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # RGB
        img_np = cv2.resize(img, (224, 224))
        # cv2.imshow('input', img)
        # cv2.waitKey(0)

        # img = Image.open(self.img_indoor_path[idx]).convert('RGB')
        # img_np = np.array(img)
        # img_np = np.resize(img_np, (224, 224)).transpose(2, 0, 1)
        # img_np = np.transpose(img_np_, (2, 0, 1))

        # FIXME
        bandwidth = 112
        grid = get_projection_grid(b=bandwidth)

        R = calculate_Rmatrix_from_phi_theta(0, 0)
        map_x, map_y = rotate_map_given_R(R, bandwidth * 2, bandwidth * 2)
        img_np = cv2.remap(img_np, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
        img_np = np.transpose(img_np, (2, 0, 1))
        show_spheres(scale=2, points=grid, rgb=img_np)

        # project_2d_on_sphere_sun360(img_np, grid)
        print(img_np.shape)

    def __len__(self):
        return len(self.img_indoor_path)


if __name__ == '__main__':
    root = "D:\data\SUN360_panoramas_1024x512"
    dataset = SUN360Dataset(root)
    dataset.__getitem__(22)