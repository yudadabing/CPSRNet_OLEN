import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.PRNet_ACOS import PRNet
# from model.loss import  CharbonnierLoss
from data import RGBP
from model.loss import AcosLoss
from data.RGBPTest import DatasetFromFolderVal
import utils
import torch.nn.functional as F
import skimage.color as sc
import random
from collections import OrderedDict
from warmup_scheduler import GradualWarmupScheduler
import torch.utils.data as Data
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Training settings
parser = argparse.ArgumentParser(description="PRNet")
parser.add_argument("--batch_size", type=int, default=8,
                    help="training batch size")
parser.add_argument("--testBatchSize", type=int, default=1,
                    help="testing batch size")
parser.add_argument("-nEpochs", type=int, default=400,
                    help="number of epochs to train")
parser.add_argument("--lr", type=float, default=2e-4,
                    help="Learning Rate. Default=2e-4")
parser.add_argument("--step_size", type=int, default=200,
                    help="learning rate decay per N epochs")
parser.add_argument("--gamma", type=int, default=0.5,
                    help="learning rate decay factor for step decay")
parser.add_argument("--cuda", action="store_true", default=True,
                    help="use cuda")
parser.add_argument("--resume", default="", type=str,
                    help="path to checkpoint")
parser.add_argument("--start-epoch", default=1, type=int,
                    help="manual epoch number")
parser.add_argument("--threads", type=int, default=8,
                    help="number of threads for data loading")
parser.add_argument("--root", type=str, default="./P_Our/",
                    help='dataset directory')
parser.add_argument("--n_train", type=int, default=114,
                    help="number of training set")
parser.add_argument("--n_val", type=int, default=80,
                    help="number of validation set")
parser.add_argument("--test_every", type=int, default=1000)
parser.add_argument("--scale", type=int, default=4,
                    help="super-resolution scale")
parser.add_argument("--patch_size", type=int, default=192, ## X2 96   X4 192  X4 384
                    help="output patch size")
parser.add_argument("--rgb_range", type=int, default=1, # 1?255
                    help="maxium value of RGB")
parser.add_argument("--n_colors", type=int, default=3,
                    help="number of color channels to use")
parser.add_argument("--pretrained", default="", type=str,
                    help="path to pretrained models")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--isY", action="store_true", default=True)
parser.add_argument("--ext", type=str, default='.npy')
parser.add_argument("--phase", type=str, default='train')

args = parser.parse_args()
print(args)
torch.backends.cudnn.benchmark = True
# random seed
seed = args.seed
if seed is None:
    seed = random.randint(1, 10000)
print("Ramdom Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)

cuda = args.cuda
device = torch.device('cuda' if cuda else 'cpu')

print("===> Loading datasets")

trainset = RGBP.RgbP(args)
# testset = DatasetFromFolderVal("RGBP/val/target/", "RGBP/val/input/", "RGBP/val/DOP/", args.scale)
# testset = DatasetFromFolderVal("./Polarity/HR/Intensity", "./Polarity/LR/X4/Intensity", "./Polarity/LR/X4/DOP",args.scale)
# testset = DatasetFromFolderVal("./Polarity/HR/Intensity", "./Polarity/LR/X2/Intensity", "./Polarity/LR/X2/DOP", args.scale)
testset1 = DatasetFromFolderVal("./Polarity/Miki/HR/Intensity", "./Polarity/Miki/LR/X4/Intensity", "./Polarity/Miki/LR/X4/DOP", args.scale)
testset2 = DatasetFromFolderVal("./Polarity/Simeng/HR/Intensity", "./Polarity/Simeng/LR/X4/Intensity", "./Polarity/Simeng/LR/X4/DOP", args.scale)
testset3 = DatasetFromFolderVal("./P_test_our/HR/Intensity", "./P_test_our/LR/X4/Intensity", "./P_test_our/LR/X4/DOP", args.scale)

training_data_loader = DataLoader(dataset=trainset, num_workers=args.threads, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
testing_data_loader1 = DataLoader(dataset=testset1, num_workers=args.threads, batch_size=args.testBatchSize,
                                 shuffle=False)
testing_data_loader2 = DataLoader(dataset=testset2, num_workers=args.threads, batch_size=args.testBatchSize,
                                 shuffle=False)
testing_data_loader3 = DataLoader(dataset=testset3, num_workers=args.threads, batch_size=args.testBatchSize,
                                 shuffle=False)


print("===> Building models")
args.is_train = True
model = PRNet(upscale=4)

l1_criterion = nn.L1Loss()
l2_criterion = AcosLoss()     ##cosine loss（第一种）
# l3_criterion = torch.cosine_similarity()
# l1_criterion = CharbonnierLoss()    ##更换为charbonnierloss 
# l1_criterion = F.smooth_l1_loss()  ##更换为smoth_l1loss 

print("===> Setting GPU")
if cuda:
    model = model.to(device)
    # l1_criterion = l1_criterion.to(device)

if args.pretrained:

    if os.path.isfile(args.pretrained):
        print("===> loading models '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        new_state_dcit = OrderedDict()
        for k, v in checkpoint.items():
            if 'module' in k:
                name = k[7:]
            else:
                name = k
            new_state_dcit[name] = v
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dcit.items() if k in model_dict}

        for k, v in model_dict.items():
            if k not in pretrained_dict:
                print(k)
        model.load_state_dict(pretrained_dict, strict=True)

    else:
        print("===> no models found at '{}'".format(args.pretrained))

print("===> Setting Optimizer")


##lr

# new_lr = 2e-4
optimizer = optim.Adam(model.parameters(), lr=args.lr)
# optimizer = optim.Adam(model.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8)

######### Scheduler ###########
# warmup_epochs = 3
# scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, 300-warmup_epochs, eta_min=1e-6)
# scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
# scheduler.step()

logger = utils.get_logger('exp/demosaic/CPSNet2.log')

# 构建 SummaryWriter
writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")


logger.info('start training!')
def train(epoch):
    model.train()
    loss_sr1=0
    loss_sr2=0
    loss_sr3=0
    # new_lr = scheduler.get_lr()[0]
    utils.adjust_learning_rate(optimizer, epoch, args.step_size, args.lr, args.gamma)
    print('epoch =', epoch, 'lr = ', optimizer.param_groups[0]['lr'])

    for iteration, (lr_tensor, hr_tensor, plr_tensor) in enumerate(training_data_loader, 1):
      
        if args.cuda:
            lr_tensor = lr_tensor.to(device)  # ranges from [0, 1]
            hr_tensor = hr_tensor.to(device)  # ranges from [0, 1]
            plr_tensor = plr_tensor.to(device)  # ranges from [0, 1]

        optimizer.zero_grad()
        sr_tensor = model(lr_tensor, plr_tensor)
        # hr_tensor1 = F.interpolate(hr_tensor, scale_factor=0.5, mode='bilinear')
        sr_stage0 = sr_tensor[0]
        sr_stage1 = sr_tensor[1]
        sr_stage2 = sr_tensor[2]
        # sr_stage1 = sr_tensor[1]

        # loss_l1 = F.smooth_l1_loss(sr_stage0, hr_tensor)
        loss_l1 = l1_criterion(sr_stage0, hr_tensor)

     ##cosine loss（第二种）
        loss_l3= torch.cosine_similarity(sr_stage1,sr_stage2,dim=1,eps=1e-8)
        loss_l3=(1-torch.mean(loss_l3))

        loss_sr=loss_l1+(0.1*loss_l3)
        # loss_sr=loss_l1+loss_l3
        loss_sr.backward()
        loss_sr3+=loss_sr.item()

        loss_sr1+=loss_l1.item()
        loss_sr2+=loss_l3.item()        
        # loss_sr+=loss_l1.item()
        nn.utils.clip_grad_norm_(model.parameters(),0.01)
        optimizer.step()
        if iteration % 100 == 0:
            logger.info("===> Epoch[{}]({}/{}): Loss_l1: {:.5f} Loss_l2: {:.5f} Loss_sr: {:.5f}".format(epoch, iteration, len(training_data_loader),
                                                                 loss_l1.item(),loss_l3.item(),loss_sr.item()))
            # print("===> Epoch[{}]({}/{}): Loss_l1: {:.5f} Loss_l2: {:.5f} Loss_sr: {:.5f}".format(epoch, iteration, len(training_data_loader),
            #                                                       loss_l1.item(),loss_l2.item(),loss_sr.item()))
    aver_loss1 = loss_sr1/len(training_data_loader)
    aver_loss2 = loss_sr2/len(training_data_loader)
    aver_loss3 = loss_sr3/len(training_data_loader)
        #########记录数据，保存于event file，这里记录了每一个epoch的损失和准确度########
    writer.add_scalars("Loss_SR", {"Train": aver_loss3}, epoch)
    writer.add_scalars("Loss_acos", {"Train": aver_loss2}, epoch)
    writer.add_scalars("Loss", {"Train": aver_loss1}, epoch)


def valid(epoch):
    model.eval()
    # best_psnr=0
    # best_epoch=0
    avg_psnr, avg_ssim = 0,0
    for batch in testing_data_loader1:
        lr_tensor, hr_tensor, plr_tensor = batch[0], batch[1], batch[2]
        if args.cuda:
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)
            plr_tensor = plr_tensor.to(device)

        with torch.no_grad():
            pre = model(lr_tensor, plr_tensor)

        sr_img = utils.tensor2np(pre[0].detach()[0])
        gt_img = utils.tensor2np(hr_tensor.detach()[0])
        crop_size = args.scale
        cropped_sr_img = utils.shave(sr_img, crop_size)
        cropped_gt_img = utils.shave(gt_img, crop_size)
        if args.isY is True:
            im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
            im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
        else:
            im_label = cropped_gt_img
            im_pre = cropped_sr_img
        avg_psnr += utils.compute_psnr(im_pre, im_label)

        # avg_psnr=avg_psnr / len(testing_data_loader)

        avg_ssim += utils.compute_ssim(im_pre, im_label)

    # best_psnr=0,
    avg_psnr=avg_psnr/ len(testing_data_loader1)
    avg_ssim=avg_ssim/ len(testing_data_loader1)
 
    return avg_psnr, avg_ssim

def valid2(epoch):
    model.eval()
    # best_psnr=0
    # best_epoch=0
    avg_psnr2, avg_ssim2 = 0,0
    for batch in testing_data_loader2:
        lr_tensor, hr_tensor, plr_tensor = batch[0], batch[1], batch[2]
        if args.cuda:
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)
            plr_tensor = plr_tensor.to(device)

        with torch.no_grad():
            pre = model(lr_tensor, plr_tensor)

        sr_img = utils.tensor2np(pre[0].detach()[0])
        gt_img = utils.tensor2np(hr_tensor.detach()[0])
        crop_size = args.scale
        cropped_sr_img = utils.shave(sr_img, crop_size)
        cropped_gt_img = utils.shave(gt_img, crop_size)
        if args.isY is True:
            im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
            im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
        else:
            im_label = cropped_gt_img
            im_pre = cropped_sr_img
        avg_psnr2 += utils.compute_psnr(im_pre, im_label)

        # avg_psnr=avg_psnr / len(testing_data_loader)

        avg_ssim2 += utils.compute_ssim(im_pre, im_label)

    # best_psnr=0,
    avg_psnr2=avg_psnr2/ len(testing_data_loader2)
    avg_ssim2=avg_ssim2/ len(testing_data_loader2)
 
    return avg_psnr2, avg_ssim2

def valid3(epoch):
    model.eval()
    # best_psnr=0
    # best_epoch=0
    avg_psnr3, avg_ssim3 = 0,0
    for batch in testing_data_loader3:
        lr_tensor, hr_tensor, plr_tensor = batch[0], batch[1], batch[2]
        if args.cuda:
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)
            plr_tensor = plr_tensor.to(device)

        with torch.no_grad():
            pre = model(lr_tensor, plr_tensor)

        sr_img = utils.tensor2np(pre[0].detach()[0])
        gt_img = utils.tensor2np(hr_tensor.detach()[0])
        crop_size = args.scale
        cropped_sr_img = utils.shave(sr_img, crop_size)
        cropped_gt_img = utils.shave(gt_img, crop_size)
        if args.isY is True:
            im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
            im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
        else:
            im_label = cropped_gt_img
            im_pre = cropped_sr_img
        avg_psnr3 += utils.compute_psnr(im_pre, im_label)

        # avg_psnr=avg_psnr / len(testing_data_loader)

        avg_ssim3 += utils.compute_ssim(im_pre, im_label)

    # best_psnr=0,
    avg_psnr3=avg_psnr3/ len(testing_data_loader2)
    avg_ssim3=avg_ssim3/ len(testing_data_loader2)
 
    return avg_psnr3, avg_ssim3


def save_checkpoint(epoch):
    model_folder = "CPSRNet_demosaic_x{}/".format(args.scale)
    model_out_path = model_folder + "epoch_{}.pth".format(epoch)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path)
    print("===> Checkpoint saved to {}".format(model_out_path))




def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)



print("===> Training")
print_network(model)
best_psnr=0
best_epoch = 0
for epoch in range(args.start_epoch, args.nEpochs + 1):

    
    e_psnr1,e_ssim1 = valid(epoch)
    e_psnr2, e_ssim2 = valid2(epoch)
    e_psnr3, e_ssim3 = valid3(epoch)
    e_psnr=e_psnr1+e_psnr2

    if e_psnr > best_psnr:
        best_psnr = e_psnr
        best_epoch = epoch
    
    print("===> Valid. psnr1: {:.4f}, ssim1: {:.4f}".format(e_psnr1, e_ssim1))
    print("===> Valid. psnr2: {:.4f}, ssim2: {:.4f}".format(e_psnr2, e_ssim2))
    print("===> Valid. psnr3: {:.4f}, ssim3: {:.4f}".format(e_psnr3, e_ssim3))
    print("===> Valid. best_epoch: {:.4f}, Best_PSNR: {:.4f}".format((best_epoch-1), e_psnr))
    # print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, e_psnr, best_epoch, best_psnr))
    # logger.info("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f SSIM %.4f]" % ((epoch-1), e_psnr, (best_epoch-1), e_psnr1,e_ssim1))
    # logger.info("[epoch %d PSNR2: %.4f --- best_epoch %d Best_PSNR %.4f SSIM2 %.4f]" % ((epoch-1), e_psnr, (best_epoch-1), e_psnr2,e_ssim2))
    logger.info("[epoch %d PSNR1: %.4f --- best_epoch %d Best_PSNR %.4f SSIM1 %.4f]" % ((epoch-1), e_psnr, (best_epoch-1), e_psnr1,e_ssim1))
    logger.info("[epoch %d PSNR2: %.4f --- best_epoch %d Best_PSNR %.4f SSIM2 %.4f]" % ((epoch-1), e_psnr, (best_epoch-1), e_psnr2,e_ssim2))
    logger.info("[epoch %d PSNR3: %.4f --- best_epoch %d Best_PSNR %.4f SSIM3 %.4f]" % ((epoch-1), e_psnr, (best_epoch-1), e_psnr3,e_ssim3))
    # print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % ((epoch-1), e_psnr, (best_epoch-1), best_psnr))
    train(epoch)
    save_checkpoint(epoch)


