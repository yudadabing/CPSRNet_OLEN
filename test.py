import os

import cv2
import skimage.color as sc
import torch
from torch.utils.data import DataLoader

import utils
from data.RGBPTest import DatasetFromFolderVal
from model.PRNet_ACOS import PRNet

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# 测试80数据集
testset = DatasetFromFolderVal("./image/HR/Intensity", "./image/LR/Intensity", "./image/LR/DoLP", 4)
# testset = DatasetFromFolderVal("./Polarity/Simeng/HR/Intensity", "./Polarity/Simeng/LR/X8/Intensity", "./Polarity/Simeng/LR/X8/DOP", 4)
# testset = DatasetFromFolderVal("./P_test_our/HR/Intensity", "./P_test_our/LR/X8/Intensity", "./P_test_our/LR/X8/DOP", 4)



# 测试RGBP数据集
# testset = DatasetFromFolderVal("RGBP/val/target", "RGBP/val/input", "RGBP/val/DOP", 4)
# 测试水下偏振数据
# testset = DatasetFromFolderVal("./underwater_sr/S0", "./underwater_sr/X8/S0", "./underwater_sr/X8/DOP", 4)
# 测试Polarity各偏振方向图像数据
# testset = DatasetFromFolderVal("./Orientation_Test/target", "./Orientation_Test/input", "./Orientation_Test/DOP", 4)

testing_data_loader = DataLoader(dataset=testset, num_workers=0, batch_size=1, shuffle=False)


# 测试结果存放文件夹
dir_output = './results/X4/CPSRNett'
# dir_output = './results/EDSRNet/RGBP_Epoch_100'
if not os.path.exists(dir_output):
    os.makedirs(dir_output)

model = PRNet(upscale=4)
model_dict = utils.load_state_dict('checkpoint/epoch_X4.pth' )
model.load_state_dict(model_dict, strict=True)

print("===> Setting GPU")
device = torch.device('cuda')
model = model.to(device)

model.eval()

avg_psnr, avg_ssim = 0, 0

for batch in testing_data_loader:

    lr_tensor, hr_tensor, plr_tensor, imname = batch[0], batch[1], batch[2], batch[3][0]

    lr_tensor = lr_tensor.to(device)
    hr_tensor = hr_tensor.to(device)
    plr_tensor = plr_tensor.to(device)
    # lr_img = utils.tensor2np(lr_tensor.detach()[0])
    with torch.no_grad():
        pre = model(lr_tensor,plr_tensor)

    sr_img = utils.tensor2np(pre[0].detach()[0])
    gt_img = utils.tensor2np(hr_tensor.detach()[0])
    crop_size = 4
    cropped_sr_img = utils.shave(sr_img, crop_size)
    cropped_gt_img = utils.shave(gt_img, crop_size)
    # if args.isY is True:
    im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
    im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
    # else:
    #     im_label = cropped_gt_img
    #     im_pre = cropped_sr_img
    avg_psnr = utils.compute_psnr(im_pre, im_label)
    avg_ssim = utils.compute_ssim(im_pre, im_label)

    output_folder = os.path.join(dir_output, imname.split("\\")[-1])
    # input_folder = os.path.join('./results/PPRNet/RGBP_input', imname.split("\\")[-1])

    cv2.imwrite(output_folder, sr_img[:, :, [2, 1, 0]])
    # cv2.imwrite(input_folder, lr_img[:, :, [2, 1, 0]])
    print("===> Valid\n "
      "psnr: {:.4f}, ssim: {:.4f}".format(avg_psnr, avg_ssim ))
    print(imname)

# print("===> Valid\n "
#       "psnr: {:.4f}, ssim: {:.4f}".format(avg_psnr / len(testing_data_loader), avg_ssim / len(testing_data_loader)))

