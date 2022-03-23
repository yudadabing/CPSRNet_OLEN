import torch.utils.data as data
from os.path import join
from os import listdir
from torchvision.transforms import Compose, ToTensor
from PIL import Image
import numpy as np
from data import common
import torchvision.transforms.functional as TF


def img_modcrop(image, modulo):
    sz = image.size
    w = np.int32(sz[0] / modulo) * modulo
    h = np.int32(sz[1] / modulo) * modulo
    out = image.crop((0, 0, w, h))
    return out


def np2tensor():
    return Compose([
        ToTensor(),
    ])


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".bmp", ".png", ".jpg"])


def load_image(filepath):
    return Image.open(filepath).convert('RGB')


##修改

class DatasetFromFolderVal(data.Dataset):
    def __init__(self, hr_dir, lr_dir, plr_dir, upscale):
        super(DatasetFromFolderVal, self).__init__()
        self.hr_filenames = sorted([join(hr_dir, x) for x in listdir(hr_dir) if is_image_file(x)])
        self.lr_filenames = sorted([join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)])
        self.plr_filenames = sorted([join(plr_dir, x) for x in listdir(plr_dir) if is_image_file(x)])
        self.upscale = upscale

    def __getitem__(self, index):
        input = load_image(self.lr_filenames[index])

        imname= self.lr_filenames[index].split('/')[-1]

        target = load_image(self.hr_filenames[index])
        img_plr = load_image(self.plr_filenames[index])
        input = np2tensor()(input)
        img_plr = np2tensor()(img_plr)
        target = np2tensor()(img_modcrop(target, self.upscale))

        # tar_img = TF.to_tensor(target)
        # H = tar_img.size(1)
        # W = tar_img.size(2)
        # ps = 1024
        # target = TF.center_crop(target, (ps,ps))
        # input = TF.center_crop(input, (int(ps/4),int(ps/4)))
        # img_plr = TF.center_crop(img_plr, (int(ps/4),int(ps/4)))
        return input, target, img_plr, imname

    def __len__(self):
        return len(self.lr_filenames)

# class DatasetFromFolderVal(data.Dataset):
#     def __init__(self, hr_dir, lr_dir, upscale):
#         super(DatasetFromFolderVal, self).__init__()
#         self.hr_filenames = sorted([join(hr_dir, x) for x in listdir(hr_dir) if is_image_file(x)])
#         self.lr_filenames = sorted([join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)])
#         self.upscale = upscale

#     def __getitem__(self, index):
#         # input = load_image(self.lr_filenames[index])
#         target = load_image(self.hr_filenames[index])


#         # tar_img = TF.to_tensor(tar_img)
#         # H = tar_img.size(1)
#         # W = tar_img.size(2)
#         ps=256
#         tar_img = TF.center_crop(target, (ps,ps))


#         input = TF.resize(tar_img, (int(ps/4), int(ps/4)), Image.BICUBIC)

#         input = np2tensor()(input)
#         target = np2tensor()(img_modcrop(tar_img, self.upscale))

#         return input, target

#     def __len__(self):
#         return len(self.lr_filenames)



class DatasetFromFolderVal_ADI(data.Dataset):
    def __init__(self, hr_dir, lr_dir, plr_dir,Aplr_dir, upscale):
        super(DatasetFromFolderVal, self).__init__()
        self.hr_filenames = sorted([join(hr_dir, x) for x in listdir(hr_dir) if is_image_file(x)])
        self.lr_filenames = sorted([join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)])
        self.plr_filenames = sorted([join(plr_dir, x) for x in listdir(plr_dir) if is_image_file(x)])
        self.Aplr_filenames = sorted([join(Aplr_dir, x) for x in listdir(Aplr_dir) if is_image_file(x)])
        self.upscale = upscale

    def __getitem__(self, index):
        input = load_image(self.lr_filenames[index])

        imname= self.lr_filenames[index].split('/')[-1]

        target = load_image(self.hr_filenames[index])
        img_plr = load_image(self.plr_filenames[index])
        img_Aplr = load_image(self.Aplr_filenames[index])
        input = np2tensor()(input)
        img_plr = np2tensor()(img_plr)
        img_Aplr = np2tensor()(img_Aplr)
        target = np2tensor()(img_modcrop(target, self.upscale))

        # tar_img = TF.to_tensor(target)
        # H = tar_img.size(1)
        # W = tar_img.size(2)
        # ps = 1024
        # target = TF.center_crop(target, (ps,ps))
        # input = TF.center_crop(input, (int(ps/4),int(ps/4)))
        # img_plr = TF.center_crop(img_plr, (int(ps/4),int(ps/4)))
        return input, target, img_plr, img_Aplr,imname

    def __len__(self):
        return len(self.lr_filenames)


class Dataset_psnr(data.Dataset):
    def __init__(self, hr_dir, lr_dir, upscale):
        super(Dataset_psnr, self).__init__()
        self.hr_filenames = sorted([join(hr_dir, x) for x in listdir(hr_dir) if is_image_file(x)])
        self.lr_filenames = sorted([join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)])

        self.upscale = upscale

    def __getitem__(self, index):
        input = load_image(self.lr_filenames[index])

        imname= self.lr_filenames[index].split('/')[-1]

        target = load_image(self.hr_filenames[index])
 
 
        input = np2tensor()(input)

        target = np2tensor()(target)

        # tar_img = TF.to_tensor(target)
        # H = tar_img.size(1)
        # W = tar_img.size(2)
        # ps = 1024
        # target = TF.center_crop(target, (ps,ps))
        # input = TF.center_crop(input, (int(ps/4),int(ps/4)))
        # img_plr = TF.center_crop(img_plr, (int(ps/4),int(ps/4)))
        return input, target,imname

    def __len__(self):
        return len(self.lr_filenames)