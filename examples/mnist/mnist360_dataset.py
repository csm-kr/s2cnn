from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
import warnings
from typing import Dict, IO, Union
import os
import numpy as np
import torch
import codecs
import gzip
import lzma
from torchvision.datasets.utils import download_and_extract_archive
import cv2
from examples.mnist.gendata import get_projection_grid, rand_rotation_matrix, rotate_grid, rotate_map_given_R
from utils import show_spheres


SN3_PASCALVINCENT_TYPEMAP = {
    8: (torch.uint8, np.uint8, np.uint8),
    9: (torch.int8, np.int8, np.int8),
    11: (torch.int16, np.dtype('>i2'), 'i2'),
    12: (torch.int32, np.dtype('>i4'), 'i4'),
    13: (torch.float32, np.dtype('>f4'), 'f4'),
    14: (torch.float64, np.dtype('>f8'), 'f8')
}


def get_int(b: bytes) -> int:
    return int(codecs.encode(b, 'hex'), 16)


def open_maybe_compressed_file(path: Union[str, IO]) -> Union[IO, gzip.GzipFile]:
    """Return a file object that possibly decompresses 'path' on the fly.
       Decompression occurs when argument `path` is a string and ends with '.gz' or '.xz'.
    """
    if not isinstance(path, torch._six.string_classes):
        return path
    if path.endswith('.gz'):
        return gzip.open(path, 'rb')
    if path.endswith('.xz'):
        return lzma.open(path, 'rb')
    return open(path, 'rb')


def read_sn3_pascalvincent_tensor(path: Union[str, IO], strict: bool = True) -> torch.Tensor:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
       Argument may be a filename, compressed filename, or file object.
    """
    # read
    with open_maybe_compressed_file(path) as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert nd >= 1 and nd <= 3
    assert ty >= 8 and ty <= 14
    m = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)


def read_label_file(path: str) -> torch.Tensor:
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 1)
    return x.long()


def read_image_file(path: str) -> torch.Tensor:
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 3)
    return x


class MNIST360Dataset(Dataset):

    # set for mnist
    resources = [
        ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]

    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
            self,
            root: str,
            split: str,
            download: bool = False,
            rotate: bool = True,
            vis: bool = False,
            bandwidth: int = 30
    ) -> None:
        super().__init__()

        self.root = root
        self.split = split  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.split == 'train':
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

        self.bandwidth = bandwidth
        self.rotate = rotate
        self.vis = vis
        super().__init__()

    def __getitem__(self, idx):

        img = self.data[idx]                                # tensor
        img_np = img.numpy()
        # img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)    # RGB
        img_np = cv2.resize(img_np, (self.bandwidth * 2, self.bandwidth * 2))
        img_np = img_np[:, :, np.newaxis]                    # [self.bandwidth * 2, self.bandwidth * 2, 1]

        grid = get_projection_grid(b=self.bandwidth)
        if self.rotate:
            rot = rand_rotation_matrix()
            rotated_grid = rotate_grid(rot, grid)
            map_x, map_y = rotate_map_given_R(rot, self.bandwidth * 2, self.bandwidth * 2)
            img_np = cv2.remap(img_np, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
            img_np = img_np[:, :, np.newaxis]
        else:
            rotated_grid = grid

        if self.vis:
            img_np_vis = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)   # RGB
            cv2.imshow('rotated_img', img_np_vis)
            cv2.waitKey(0)
            img_np_ = np.transpose(img_np_vis, (2, 0, 1))
            show_spheres(scale=2, points=rotated_grid, rgb=img_np_)
        # R = calculate_Rmatrix_from_phi_theta(0, 0)

        img_np = np.transpose(img_np, (2, 0, 1))                # [3, 224, 224]
        img_torch = torch.FloatTensor(img_np)                   # [3, 224, 224]

        label = int(self.targets[idx])
        return img_torch, label

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')


if __name__ == '__main__':
    root = "MNIST_data"
    dataset = MNIST360Dataset(root, 'test', download=True, rotate=False, vis=True, bandwidth=30)
    print(len(dataset))
    dataloader = DataLoader(dataset=dataset,
                            batch_size=1,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=0)

    for img, label in dataloader:
        print(label)

