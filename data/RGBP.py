import torch.utils.data as data
import os.path
import cv2
import numpy as np
from data import common

def default_loader(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0]]


def npy_loader(path):
    return np.load(path)


IMG_EXTENSIONS = [
    '.png', '.npy',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


class RgbP(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.scale = self.opt.scale
        self.root = self.opt.root
        self.ext = self.opt.ext  # '.png' or '.npy'(default)
        self.train = True if self.opt.phase == 'train' else False
        self.repeat = self.opt.test_every // (self.opt.n_train // self.opt.batch_size)
        self._set_filesystem(self.root)
        self.images_hr, self.images_lr, self.images_plr = self._scan()

    def _set_filesystem(self, dir_data):
        self.root = dir_data + '/P_all_demosaic_decoded_X2'
        self.dir_hr = os.path.join(self.root, 'target')
        self.dir_lr = os.path.join(self.root, 'input')
        self.dir_plr = os.path.join(self.root, 'DOP')

    def __getitem__(self, idx):
        lr, hr, plr = self._load_file(idx)
        lr, hr, plr = common.set_channel(lr, hr, plr, n_channels=self.opt.n_colors)
        lr, hr, plr = self._get_patch(lr, hr, plr)
        lr_tensor, hr_tensor, plr_tensor = common.np2Tensor(lr, hr, plr, rgb_range=self.opt.rgb_range)
        return lr_tensor, hr_tensor, plr_tensor

    def __len__(self):
        if self.train:
            return self.opt.n_train * self.repeat

    def _get_index(self, idx):
        if self.train:
            return idx % self.opt.n_train
        else:
            return idx

    def _get_patch(self, img_in, img_tar, img_plr):
        patch_size = self.opt.patch_size
        scale = self.scale
        if self.train:
            img_in, img_plr, img_tar= common.get_patch(
                img_in, img_plr, img_tar,  patch_size=patch_size, scale=scale)
            img_in, img_tar, img_plr = common.augment(img_in, img_tar, img_plr)
        else:
            ih, iw = img_in.shape[:2]
            img_tar = img_tar[0:ih * scale, 0:iw * scale, :]
        return img_in, img_tar, img_plr

    def _scan(self):
        list_hr = sorted(make_dataset(self.dir_hr))
        list_lr = sorted(make_dataset(self.dir_lr))
        list_plr = sorted(make_dataset(self.dir_plr))
        return list_hr, list_lr, list_plr

    def _load_file(self, idx):
        idx = self._get_index(idx)
        if self.ext == '.npy':
            lr = npy_loader(self.images_lr[idx])
            hr = npy_loader(self.images_hr[idx])
            plr = npy_loader(self.images_plr[idx])
        else:
            lr = default_loader(self.images_lr[idx])
            hr = default_loader(self.images_hr[idx])
            plr = default_loader(self.images_plr[idx])
        return lr, hr, plr



class RgbAP(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.scale = self.opt.scale
        self.root = self.opt.root
        self.ext = self.opt.ext  # '.png' or '.npy'(default)
        self.train = True if self.opt.phase == 'train' else False
        self.repeat = self.opt.test_every // (self.opt.n_train // self.opt.batch_size)
        self._set_filesystem(self.root)
        self.images_hr, self.images_lr, self.images_plr = self._scan()

    def _set_filesystem(self, dir_data):
        self.root = dir_data + '/P_Our_decoded_X4'
        self.dir_hr = os.path.join(self.root, 'target')
        self.dir_lr = os.path.join(self.root, 'input')
        self.dir_plr = os.path.join(self.root, 'AOP')

    def __getitem__(self, idx):
        lr, hr, plr = self._load_file(idx)
        lr, hr, plr = common.set_channel(lr, hr, plr, n_channels=self.opt.n_colors)
        lr, hr, plr = self._get_patch(lr, hr, plr)
        lr_tensor, hr_tensor, plr_tensor = common.np2Tensor(lr, hr, plr, rgb_range=self.opt.rgb_range)
        return lr_tensor, hr_tensor, plr_tensor

    def __len__(self):
        if self.train:
            return self.opt.n_train * self.repeat

    def _get_index(self, idx):
        if self.train:
            return idx % self.opt.n_train
        else:
            return idx

    def _get_patch(self, img_in, img_tar, img_plr):
        patch_size = self.opt.patch_size
        scale = self.scale
        if self.train:
            img_in, img_plr, img_tar= common.get_patch(
                img_in, img_plr, img_tar,  patch_size=patch_size, scale=scale)
            img_in, img_tar, img_plr = common.augment(img_in, img_tar, img_plr)
        else:
            ih, iw = img_in.shape[:2]
            img_tar = img_tar[0:ih * scale, 0:iw * scale, :]
        return img_in, img_tar, img_plr

    def _scan(self):
        list_hr = sorted(make_dataset(self.dir_hr))
        list_lr = sorted(make_dataset(self.dir_lr))
        list_plr = sorted(make_dataset(self.dir_plr))
        return list_hr, list_lr, list_plr

    def _load_file(self, idx):
        idx = self._get_index(idx)
        if self.ext == '.npy':
            lr = npy_loader(self.images_lr[idx])
            hr = npy_loader(self.images_hr[idx])
            plr = npy_loader(self.images_plr[idx])
        else:
            lr = default_loader(self.images_lr[idx])
            hr = default_loader(self.images_hr[idx])
            plr = default_loader(self.images_plr[idx])
        return lr, hr, plr


class RgbADP(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.scale = self.opt.scale
        self.root = self.opt.root
        self.ext = self.opt.ext  # '.png' or '.npy'(default)
        self.train = True if self.opt.phase == 'train' else False
        self.repeat = self.opt.test_every // (self.opt.n_train // self.opt.batch_size)
        self._set_filesystem(self.root)
        self.images_hr, self.images_lr, self.images_plr ,self.images_Aplr= self._scan()

    def _set_filesystem(self, dir_data):
        self.root = dir_data + '/P_Our_decoded_X4'
        self.dir_hr = os.path.join(self.root, 'target')
        self.dir_lr = os.path.join(self.root, 'input')
        self.dir_plr = os.path.join(self.root, 'DOP')
        self.dir_Aplr = os.path.join(self.root, 'AOP')

    def __getitem__(self, idx):
        lr, hr, plr,Aplr = self._load_file(idx)
        lr, hr, plr ,Aplr= common.set_channel(lr, hr, plr, Aplr,n_channels=self.opt.n_colors)
        lr, hr, plr ,Aplr= self._get_patch(lr, hr, plr, Aplr)
        lr_tensor, hr_tensor, plr_tensor,Aplr_tensor = common.np2Tensor(lr, hr, plr, Aplr,rgb_range=self.opt.rgb_range)
        return lr_tensor, hr_tensor, plr_tensor, Aplr_tensor

    def __len__(self):
        if self.train:
            return self.opt.n_train * self.repeat

    def _get_index(self, idx):
        if self.train:
            return idx % self.opt.n_train
        else:
            return idx

    def _get_patch(self, img_in, img_tar, img_plr, img_Aplr):
        patch_size = self.opt.patch_size
        scale = self.scale
        if self.train:
            img_in, img_plr,img_Aplr, img_tar= common.get_patch2(
                img_in, img_plr,img_Aplr, img_tar , patch_size=patch_size, scale=scale)
            img_in, img_tar, img_plr,img_Aplr = common.augment(img_in, img_tar, img_plr,img_Aplr)
        else:
            ih, iw = img_in.shape[:2]
            img_tar = img_tar[0:ih * scale, 0:iw * scale, :]
        return img_in, img_tar, img_plr,img_Aplr

    def _scan(self):
        list_hr = sorted(make_dataset(self.dir_hr))
        list_lr = sorted(make_dataset(self.dir_lr))
        list_plr = sorted(make_dataset(self.dir_plr))
        list_Aplr = sorted(make_dataset(self.dir_Aplr))
        return list_hr, list_lr, list_plr,list_Aplr

    def _load_file(self, idx):
        idx = self._get_index(idx)
        if self.ext == '.npy':
            lr = npy_loader(self.images_lr[idx])
            hr = npy_loader(self.images_hr[idx])
            plr = npy_loader(self.images_plr[idx])
            Aplr = npy_loader(self.images_Aplr[idx])
        else:
            lr = default_loader(self.images_lr[idx])
            hr = default_loader(self.images_hr[idx])
            plr = default_loader(self.images_plr[idx])
            Aplr = default_loader(self.images_Aplr[idx])
        return lr, hr, plr,Aplr

