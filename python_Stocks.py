from glob import glob
from natsort import natsorted
import cv2
import os
import numpy as np

# save_S0_path = "./result/HANet/Orientation_Epoch_62_Best/S0/"
save_dop_path = "./results/Orientation_epoch_PRNet_acos/epoch_best/DOP/"
save_aop_path = "./results/Orientation_epoch_PRNet_acos/epoch_best/AOP/"

# if not os.path.exists(save_S0_path):
#     os.makedirs(save_S0_path)
if not os.path.exists(save_dop_path):
    os.makedirs(save_dop_path)
if not os.path.exists(save_aop_path):
    os.makedirs(save_aop_path)


imgs_paths = natsorted(glob("./results/Orientation_epoch_PRNet_acos/epoch_best/*.png"))

for i in range(0,  len(imgs_paths)//4):
    img0_path = imgs_paths[i*4]
    img45_path = imgs_paths[i*4+1]
    img90_path = imgs_paths[i*4+2]
    img135_path = imgs_paths[i*4+3]

    img0 = cv2.imread(img0_path, 1)
    img45 = cv2.imread(img45_path, 1)
    img90 = cv2.imread(img90_path, 1)
    img135 = cv2.imread(img135_path, 1)

    name = img0_path.split("/")[-1]

    S0 = 0.5 * (np.array(img0, dtype=np.float32) + np.array(img90, np.float32))
    S1 = np.array(img0, np.float32) - np.array(img90, np.float32)
    S2 = np.array(img45, np.float32) - np.array(img135, np.float32)
    Dop = np.sqrt(np.power(S1, 2) + np.power(S2, 2)) / S0
    Aop = 0.5 * np.arctan(S2 / S1)
    # s0 = np.uint8(S0)
    dop = np.uint8(Dop * 255)
    aop = np.uint8(Aop * 255)
    cv2.imwrite(save_aop_path + name, cv2.cvtColor(aop, cv2.COLOR_BGR2GRAY))
    cv2.imwrite(save_dop_path + name, cv2.cvtColor(dop, cv2.COLOR_BGR2GRAY))
    # cv2.imwrite(save_S0_path + name, s0)
    print("Saved %s" % name)




# img0_paths = natsorted(glob("./result/HANet/Orientation_Epoch_62_Best/*"))
# img45_paths = natsorted(glob("./RGBP/45/*"))
# img90_paths = natsorted(glob("./RGBP/90/*"))
# img135_paths = natsorted(glob("./RGBP/135/*"))


# for img0_path, img45_path, img90_path, img135_path in zip(img0_paths, img45_paths, img90_paths, img135_paths):
#     img0 = cv2.imread(img0_path, 1)
#     img45 = cv2.imread(img45_path, 1)
#     img90 = cv2.imread(img90_path, 1)
#     img135 = cv2.imread(img135_path, 1)
#     name = img0_path.split("\\")[-1]
#     S0 = 0.5 * (np.array(img0, dtype=np.float32) + np.array(img90, np.float32))
#     S1 = np.array(img0, np.float32) - np.array(img90, np.float32)
#     S2 = np.array(img45, np.float32) - np.array(img135, np.float32)
#     Dop = np.sqrt(np.power(S1, 2) + np.power(S2, 2)) / S0
#     Aop = 0.5 * np.arctan(S2 / S1)
#     s0 = np.uint8(S0)
#     dop = np.uint8(Dop * 255)
#     aop = np.uint8(Aop * 255)
#     cv2.imwrite(save_aop_path + name, cv2.cvtColor(aop, cv2.COLOR_BGR2GRAY))
#     cv2.imwrite(save_dop_path + name, cv2.cvtColor(dop, cv2.COLOR_BGR2GRAY))
#     cv2.imwrite(save_S0_path + name, s0)
#     print("Saved %s" % name)

# print("Process Finish")
