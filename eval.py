import os
import yaml
import shutil
import torch
import data.util as util
from data.dataloader import OriDataset, AugDataset, OpticalDataset
from network.DeblurNet import WholeNet
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import matplotlib.pyplot as plot
import argparse
import numpy as np
import torch.nn.functional as F


parser = argparse.ArgumentParser(description="yaml file path")
parser.add_argument('-c', '--config', default='yaml file path')
args = parser.parse_args()

CONFIG_PATH = args.config
print(CONFIG_PATH)



with open(CONFIG_PATH,'r') as f:
    config = yaml.load(f)

    ONLYTRAINMASK = "onlyTrainMask" if config['onlyTrainMask'] else ""
    USEMASK = "useMask" if config['useMask']  else "notUseMask"
    FIXMASK = "fixMask" if config['fixMask']  else "notFixMask"
    USINGSA = "usingSA" if config['usingSA']  else "notUsingSA"
    USINGMASKLOSS = "usingMaskLoss" if config['usingMaskLoss'] else "notUsingMaskLoss"
    USINGSALOSS = "usingSALoss" if config['usingSALoss'] else "notUsingSALoss"

    saveName = "%s-%s-%s-%s-%s-%s-%s-%s" % (
        config['date'], ONLYTRAINMASK, USEMASK, FIXMASK, USINGSA,
        USINGMASKLOSS, USINGSALOSS,config['dataset'])
    print(saveName)

os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_available']
device_ids = range(config['gpu_num'])
if not os.path.exists('./config/'+saveName+'.yaml'):
    shutil.copy(CONFIG_PATH, './config/'+saveName+'.yaml')
else:
    print('already exists')
#shutil.copy(CONFIG_PATH, './config/'+saveName+'.yaml')
if config['local']:
    print('test Local')
    testSet = OpticalDataset(sharp_root1=config['test_sharp'], blur_root1=config['test_blur'],
                         mask_root1=config['test_mask'],
                         patch_size=config['crop_size'], resize_size=config['resize_size'], phase='test')
else:
    testSet = OriDataset(sharp_root1=config['test_sharp'], blur_root1=config['test_blur'],
                             sharp_root2=None, blur_root2=None,
                             patch_size=config['crop_size'], resize_size=config['resize_size'], phase='test')
test_loader = DataLoader(testSet, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)

def testMaskNet():
    model = WholeNet(inChannels=3, outChannels=3, bilinear=True,
                     onlyTrainMask=config['onlyTrainMask'],
                     usingMask=config['useMask'],
                     fixMask=config['fixMask'],
                     usingSA=config['usingSA'],
                     usingMaskLoss=config['usingMaskLoss'],
                     usingSALoss=config['usingSALoss'])

    model.load_state_dict(torch.load(config['pretrained_model']))

    model = model.cuda()
    idx = 0
    if not os.path.exists(config['result_dir']):
        os.mkdir(config['result_dir'])
    with torch.no_grad():
        model.eval()
        for test_data in test_loader:

            idx += 1
            test_data['L'] = test_data['L'].cuda()

            mask = model(test_data['L'])
            if idx % 100 == 0:
                print("tested {}".format(idx))

            mask = F.softmax(mask,dim=1)
            mask = mask.detach().cpu()
            savePath = config['result_dir']
            imgName = test_data['L_path'][0].split('/')[-1].split('.')[0]+'.png'
            imgName2 = test_data['L_path'][0].split('/')[-1].split('.')[0] + 'bg.png'
            outmask = np.zeros(mask[0,1,:,:].shape)

            plot.matshow(mask[0,1,:,:])
            plot.savefig(os.path.join(savePath, imgName))
            plot.matshow(mask[0, 0, :, :])
            plot.savefig(os.path.join(savePath, imgName2))

def testSimpleNet():
    model = WholeNet(inChannels=3, outChannels=3, bilinear=True,
                     onlyTrainMask=config['onlyTrainMask'],
                     usingMask=config['useMask'],
                     fixMask=config['fixMask'],
                     usingSA=config['usingSA'],
                     usingMaskLoss=config['usingMaskLoss'],
                     usingSALoss=config['usingSALoss'])

    model.load_state_dict(torch.load(config['pretrained_model']))
    avg_PSNR = 0
    avg_PSNR_BELOW_THRES = 0
    avg_PSNR_ABOVE_THRES = 0
    avg_SSIM = 0
    avg_SSIM_BELOW_THRES = 0
    avg_SSIM_ABOVE_THRES = 0
    model = model.cuda()
    idx = 0
    if not os.path.exists(config['result_dir']):
        os.mkdir(config['result_dir'])
    with torch.no_grad():
        model.eval()
        for test_data in test_loader:


            test_data['L'] = test_data['L'].cuda()

            sharp = model(test_data['L'])
            if idx % 100 == 0:
                print("tested {}".format(idx))
            sharp = sharp.detach().float().cpu()
            sharp = util.tensor2uint(sharp)
            test_data['H'] = util.tensor2uint(test_data['H'])
            test_data['L'] = test_data['L'].detach().cpu()
            test_data['L'] = util.tensor2uint(test_data['L'])
            f = open("./{}.txt".format(saveName), "a")
            if config['local']:

                test_data['mask'] = test_data['mask'].detach().cpu()[0]

                current_psnr_below = util.calculate_local_psnr_below(sharp.copy(), test_data['H'].copy(),
                                                                     test_data['mask'], border=0)
                current_ssim_below = util.calculate_local_ssim_below(sharp.copy(), test_data['H'].copy(),
                                                                     test_data['mask'], border=0)
                current_psnr_above = util.calculate_local_psnr_above(sharp.copy(), test_data['H'].copy(), test_data['mask'], border=0)
                current_ssim_above = util.calculate_local_ssim_above(sharp.copy(), test_data['H'].copy(), test_data['mask'], border=0)

                current_ssim = util.calculate_ssim(sharp, test_data['H'], border=0)
                current_psnr = util.calculate_psnr(sharp, test_data['H'], border=0)
                pre_psnr = util.calculate_psnr(test_data['L'], test_data['H'], border=0)
                if current_psnr_above == float('inf') or current_psnr_below == float('inf'):
                    print('get not so blured')
                    continue

                f.writelines(
                    "{} {:<4.4f} {:<4.4f} {:<4.4f} {:<4.4f} {:<4.4f} {:<4.4f} {:<4.4f}\n".format(test_data['L_path'],
                                                                                               current_psnr,
                                                                                               current_ssim,
                                                                                               pre_psnr,
                                                                                               current_psnr_below,
                                                                                               current_psnr_above,
                                                                                               current_ssim_below,
                                                                                               current_ssim_above))

            else:
                current_psnr = util.calculate_psnr(sharp, test_data['H'], border=0)
                current_ssim = util.calculate_ssim(sharp, test_data['H'], border=0)
                f.writelines(
                    "{} {:<4.4f} {:<4.4f}\n".format(test_data['L_path'],current_psnr,current_ssim))
            idx += 1
            avg_SSIM += current_ssim
            avg_PSNR += current_psnr
            f.close()
            if config['local']:
                avg_PSNR_BELOW_THRES += current_psnr_below
                avg_PSNR_ABOVE_THRES += current_psnr_above
                avg_SSIM_BELOW_THRES += current_ssim_below
                avg_SSIM_ABOVE_THRES += current_ssim_above
            #savePath = config['result_dir']
            #imgName = test_data['L_path'][0].split('/')[-1].split('.')[0] + '.png'
            #util.imsave(sharp, os.path.join(savePath, imgName))
        avg_SSIM = avg_SSIM / idx
        avg_PSNR = avg_PSNR / idx
        if config['local']:
            avg_PSNR_BELOW_THRES = avg_PSNR_BELOW_THRES / idx
            avg_PSNR_ABOVE_THRES = avg_PSNR_ABOVE_THRES / idx
            avg_SSIM_BELOW_THRES = avg_SSIM_BELOW_THRES / idx
            avg_SSIM_ABOVE_THRES = avg_SSIM_ABOVE_THRES / idx
            print(
                "avg_PSNR_BELOW_THRES : {:<4.2f}, avg_PSNR_ABOVE_THRES : {:<4.2f}, avg_SSIM_BELOW_THRES : {:<4.4f}, avg_SSIM_ABOVE_THRES : {:<4.4f}".format(
                    avg_PSNR_BELOW_THRES, avg_PSNR_ABOVE_THRES, avg_SSIM_BELOW_THRES, avg_SSIM_ABOVE_THRES))
        print("total loss : {:<4.2f}, total SSIM : {:<4.4f}".format(
            avg_PSNR, avg_SSIM))

def testWholeNetFixNoSA():
    model = WholeNet(inChannels=3, outChannels=3, bilinear=True,
                     onlyTrainMask=config['onlyTrainMask'],
                     usingMask=config['useMask'],
                     fixMask=config['fixMask'],
                     usingSA=config['usingSA'],
                     usingMaskLoss=config['usingMaskLoss'],
                     usingSALoss=config['usingSALoss'])
    model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)
    model.load_state_dict(torch.load(config['pretrained_model']))


    idx = 0
    if not os.path.exists(config['result_dir']):
        os.mkdir(config['result_dir'])
    with torch.no_grad():
        model.eval()
        avg_PSNR = 0
        avg_PSNR_BELOW_THRES = 0
        avg_PSNR_ABOVE_THRES = 0
        avg_SSIM = 0
        avg_SSIM_BELOW_THRES = 0
        avg_SSIM_ABOVE_THRES = 0
        for test_data in test_loader:


            test_data['L'] = test_data['L'].cuda()

            sharp, _ = model(test_data['L'])
            if idx % 100 == 0:
                print("tested {}".format(idx))
            sharp = util.tensor2uint(sharp)
            test_data['H'] = util.tensor2uint(test_data['H'])
            test_data['L'] = test_data['L'].detach().cpu()
            test_data['L'] = util.tensor2uint(test_data['L'])
            f = open("./{}.txt".format(saveName), "a")
            if config['local']:

                test_data['mask'] = test_data['mask'].detach().cpu()[0]

                current_psnr_below = util.calculate_local_psnr_below(sharp.copy(), test_data['H'].copy(),
                                                                     test_data['mask'], border=0)
                current_ssim_below = util.calculate_local_ssim_below(sharp.copy(), test_data['H'].copy(),
                                                                     test_data['mask'], border=0)
                current_psnr_above = util.calculate_local_psnr_above(sharp.copy(), test_data['H'].copy(), test_data['mask'], border=0)
                current_ssim_above = util.calculate_local_ssim_above(sharp.copy(), test_data['H'].copy(), test_data['mask'], border=0)

                current_ssim = util.calculate_ssim(sharp, test_data['H'], border=0)
                current_psnr = util.calculate_psnr(sharp, test_data['H'], border=0)
                pre_psnr = util.calculate_psnr(test_data['L'], test_data['H'], border=0)
                if current_psnr_above == float('inf') or current_psnr_below == float('inf'):
                    print('get not so blured')
                    continue

                f.writelines("{} {:<4.4f} {:<4.4f} {:<4.4f} {:<4.4f} {:<4.4f} {:<4.4f} {:<4.4f}\n".format(test_data['L_path'],
                                                                                                     current_psnr,
                                                                                                     current_ssim,
                                                                                                     pre_psnr,
                                                                                                     current_psnr_below,
                                                                                                     current_psnr_above,
                                                                                                     current_ssim_below,
                                                                                                     current_ssim_above))

            else:
                current_psnr = util.calculate_psnr(sharp, test_data['H'], border=0)
                current_ssim = util.calculate_ssim(sharp, test_data['H'], border=0)
                #f.writelines(
                #    "{} {:<4.4f} {:<4.4f}\n".format(test_data['L_path'], current_psnr, current_ssim))
            idx += 1
            avg_SSIM += current_ssim
            avg_PSNR += current_psnr
            f.close()
            if config['local']:
                avg_PSNR_BELOW_THRES += current_psnr_below
                avg_PSNR_ABOVE_THRES += current_psnr_above
                avg_SSIM_BELOW_THRES += current_ssim_below
                avg_SSIM_ABOVE_THRES += current_ssim_above
            #savePath = config['result_dir']
            #imgName = test_data['L_path'][0].split('/')[-1].split('.')[0] + '.png'
            #util.imsave(sharp, os.path.join(savePath, imgName))
        avg_SSIM = avg_SSIM / idx
        avg_PSNR = avg_PSNR / idx
        if config['local']:
            avg_PSNR_BELOW_THRES = avg_PSNR_BELOW_THRES / idx
            avg_PSNR_ABOVE_THRES = avg_PSNR_ABOVE_THRES / idx
            avg_SSIM_BELOW_THRES = avg_SSIM_BELOW_THRES / idx
            avg_SSIM_ABOVE_THRES = avg_SSIM_ABOVE_THRES / idx
            print("avg_PSNR_BELOW_THRES : {:<4.2f}, avg_PSNR_ABOVE_THRES : {:<4.2f}, avg_SSIM_BELOW_THRES : {:<4.4f}, avg_SSIM_ABOVE_THRES : {:<4.4f}".format(
                avg_PSNR_BELOW_THRES, avg_PSNR_ABOVE_THRES, avg_SSIM_BELOW_THRES, avg_SSIM_ABOVE_THRES))
        print("total loss : {:<4.2f}, total SSIM : {:<4.4f}".format(
            avg_PSNR, avg_SSIM))

def testWholeNet():
    model = WholeNet(inChannels=3, outChannels=3, bilinear=True,
                     onlyTrainMask=config['onlyTrainMask'],
                     usingMask=config['useMask'],
                     fixMask=config['fixMask'],
                     usingSA=config['usingSA'],
                     usingMaskLoss=config['usingMaskLoss'],
                     usingSALoss=config['usingSALoss'])
    #model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)
    model.load_state_dict(torch.load(config['pretrained_model']))
    model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)
    #gpu_model = model.module  # GPU-version
    # 载入为cpu模型
    #model = WholeNet(inChannels=3, outChannels=3, bilinear=True,
    #                 onlyTrainMask=config['onlyTrainMask'],
    #                 usingMask=config['useMask'],
    #                 fixMask=config['fixMask'],
    #                 usingSA=config['usingSA'],
    #                 usingMaskLoss=config['usingMaskLoss'],
    #                 usingSALoss=config['usingSALoss'])
    #model.load_state_dict(gpu_model.state_dict())

    idx = 0
    if not os.path.exists(config['result_dir']):
        os.mkdir(config['result_dir'])
    with torch.no_grad():
        model.eval()
        avg_PSNR = 0
        avg_PSNR_BELOW_THRES = 0
        avg_PSNR_ABOVE_THRES = 0
        avg_SSIM = 0
        avg_SSIM_BELOW_THRES = 0
        avg_SSIM_ABOVE_THRES = 0
        for test_data in test_loader:


            test_data['L'] = test_data['L'].cuda()

            sharp, _, _ = model(test_data['L'])
            if idx % 100 == 0:
                print("tested {}".format(idx))
            sharp = util.tensor2uint(sharp)
            test_data['H'] = util.tensor2uint(test_data['H'])
            test_data['L'] = test_data['L'].detach().cpu()
            test_data['L'] = util.tensor2uint(test_data['L'])
            f = open("./{}.txt".format(saveName), "a")
            if config['local']:

                test_data['mask'] = test_data['mask'].detach().cpu()[0]

                current_psnr_below = util.calculate_local_psnr_below(sharp.copy(), test_data['H'].copy(),
                                                                     test_data['mask'], border=0)
                current_ssim_below = util.calculate_local_ssim_below(sharp.copy(), test_data['H'].copy(),
                                                                     test_data['mask'], border=0)
                current_psnr_above = util.calculate_local_psnr_above(sharp.copy(), test_data['H'].copy(), test_data['mask'], border=0)
                current_ssim_above = util.calculate_local_ssim_above(sharp.copy(), test_data['H'].copy(), test_data['mask'], border=0)

                current_ssim = util.calculate_ssim(sharp, test_data['H'], border=0)
                current_psnr = util.calculate_psnr(sharp, test_data['H'], border=0)
                pre_psnr = util.calculate_psnr(test_data['L'], test_data['H'], border=0)
                if current_psnr_above == float('inf') or current_psnr_below == float('inf'):
                    print('get not so blured')
                    continue

                f.writelines("{} {:<4.4f} {:<4.4f} {:<4.4f} {:<4.4f} {:<4.4f} {:<4.4f} {:<4.4f}\n".format(test_data['L_path'],
                                                                                                     current_psnr,
                                                                                                     current_ssim,
                                                                                                     pre_psnr,
                                                                                                     current_psnr_below,
                                                                                                     current_psnr_above,
                                                                                                     current_ssim_below,
                                                                                                     current_ssim_above))

            else:
                current_psnr = util.calculate_psnr(sharp, test_data['H'], border=0)
                current_ssim = util.calculate_ssim(sharp, test_data['H'], border=0)
                f.writelines(
                    "{} {:<4.4f} {:<4.4f}\n".format(test_data['L_path'], current_psnr, current_ssim))
            idx += 1
            avg_SSIM += current_ssim
            avg_PSNR += current_psnr
            f.close()
            if config['local']:
                avg_PSNR_BELOW_THRES += current_psnr_below
                avg_PSNR_ABOVE_THRES += current_psnr_above
                avg_SSIM_BELOW_THRES += current_ssim_below
                avg_SSIM_ABOVE_THRES += current_ssim_above
            #savePath = config['result_dir']
            #imgName = test_data['L_path'][0].split('/')[-1].split('.')[0] + '.png'
            #util.imsave(sharp, os.path.join(savePath, imgName))
        avg_SSIM = avg_SSIM / idx
        avg_PSNR = avg_PSNR / idx
        if config['local']:
            avg_PSNR_BELOW_THRES = avg_PSNR_BELOW_THRES / idx
            avg_PSNR_ABOVE_THRES = avg_PSNR_ABOVE_THRES / idx
            avg_SSIM_BELOW_THRES = avg_SSIM_BELOW_THRES / idx
            avg_SSIM_ABOVE_THRES = avg_SSIM_ABOVE_THRES / idx
            print("avg_PSNR_BELOW_THRES : {:<4.2f}, avg_PSNR_ABOVE_THRES : {:<4.2f}, avg_SSIM_BELOW_THRES : {:<4.4f}, avg_SSIM_ABOVE_THRES : {:<4.4f}".format(
                avg_PSNR_BELOW_THRES, avg_PSNR_ABOVE_THRES, avg_SSIM_BELOW_THRES, avg_SSIM_ABOVE_THRES))
        print("total loss : {:<4.2f}, total SSIM : {:<4.4f}".format(
            avg_PSNR, avg_SSIM))

def testSANetSALoss():
    model = WholeNet(inChannels=3, outChannels=3, bilinear=True,
                     onlyTrainMask=config['onlyTrainMask'],
                     usingMask=config['useMask'],
                     fixMask=config['fixMask'],
                     usingSA=config['usingSA'],
                     usingMaskLoss=config['usingMaskLoss'],
                     usingSALoss=config['usingSALoss'])

    model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)

    model.load_state_dict(torch.load(config['pretrained_model']))
    idx = 0
    if not os.path.exists(config['result_dir']):
        os.mkdir(config['result_dir'])
    with torch.no_grad():
        model.eval()
        avg_PSNR = 0
        avg_PSNR_BELOW_THRES = 0
        avg_PSNR_ABOVE_THRES = 0
        avg_SSIM = 0
        avg_SSIM_BELOW_THRES = 0
        avg_SSIM_ABOVE_THRES = 0
        for test_data in test_loader:


            test_data['L'] = test_data['L'].cuda()

            sharp, SASum = model(test_data['L'])
            if idx % 100 == 0:
                print("tested {}".format(idx))

            sharp = sharp.detach().float().cpu()
            SASum = SASum.detach().float().cpu()[0,0,:,:]
            sharp = util.tensor2uint(sharp)
            test_data['H'] = util.tensor2uint(test_data['H'])
            test_data['L'] = test_data['L'].detach().cpu()
            test_data['L'] = util.tensor2uint(test_data['L'])
            f = open("./{}.txt".format(saveName), "a")
            if config['local']:
                test_data['mask'] = test_data['mask'].detach().cpu()[0]

                current_psnr_below = util.calculate_local_psnr_below(sharp.copy(), test_data['H'].copy(),
                                                                     test_data['mask'], border=0)
                current_ssim_below = util.calculate_local_ssim_below(sharp.copy(), test_data['H'].copy(),
                                                                     test_data['mask'], border=0)
                current_psnr_above = util.calculate_local_psnr_above(sharp.copy(), test_data['H'].copy(), test_data['mask'], border=0)
                current_ssim_above = util.calculate_local_ssim_above(sharp.copy(), test_data['H'].copy(), test_data['mask'], border=0)

                current_ssim = util.calculate_ssim(sharp, test_data['H'], border=0)
                current_psnr = util.calculate_psnr(sharp, test_data['H'], border=0)
                pre_psnr = util.calculate_psnr(test_data['L'], test_data['H'], border=0)


                if current_psnr_above == float('inf') or current_psnr_below == float('inf'):
                    print('get not so blured')
                    continue

                f.writelines(
                    "{} {:<4.4f} {:<4.4f} {:<4.4f} {:<4.4f} {:<4.4f} {:<4.4f} {:<4.4f}\n".format(test_data['L_path'],
                                                                                               current_psnr,
                                                                                               current_ssim,
                                                                                               pre_psnr,
                                                                                               current_psnr_below,
                                                                                               current_psnr_above,
                                                                                               current_ssim_below,
                                                                                               current_ssim_above))

            else:
                current_psnr = util.calculate_psnr(sharp, test_data['H'], border=0)
                current_ssim = util.calculate_ssim(sharp, test_data['H'], border=0)
                f.writelines(
                    "{} {:<4.4f} {:<4.4f}\n".format(test_data['L_path'], current_psnr, current_ssim))
            idx += 1
            avg_SSIM += current_ssim
            avg_PSNR += current_psnr
            f.close()
            if config['local']:
                avg_PSNR_BELOW_THRES += current_psnr_below
                avg_PSNR_ABOVE_THRES += current_psnr_above
                avg_SSIM_BELOW_THRES += current_ssim_below
                avg_SSIM_ABOVE_THRES += current_ssim_above
            #savePath = config['result_dir']
            #sasavePath = config['result_dir']+'_SA'
            #imgName = test_data['L_path'][0].split('/')[-1].split('.')[0] + '.png'
            #util.imsave(sharp, os.path.join(savePath, imgName))
            #plot.matshow(SASum)
            #plot.savefig(sasavePath+'/'+imgName)
            #plot.close()

        avg_SSIM = avg_SSIM / idx
        avg_PSNR = avg_PSNR / idx
        if config['local']:
            avg_PSNR_BELOW_THRES = avg_PSNR_BELOW_THRES / idx
            avg_PSNR_ABOVE_THRES = avg_PSNR_ABOVE_THRES / idx
            avg_SSIM_BELOW_THRES = avg_SSIM_BELOW_THRES / idx
            avg_SSIM_ABOVE_THRES = avg_SSIM_ABOVE_THRES / idx
            print(
                "avg_PSNR_BELOW_THRES : {:<4.2f}, avg_PSNR_ABOVE_THRES : {:<4.2f}, avg_SSIM_BELOW_THRES : {:<4.4f}, avg_SSIM_ABOVE_THRES : {:<4.4f}".format(
                    avg_PSNR_BELOW_THRES, avg_PSNR_ABOVE_THRES, avg_SSIM_BELOW_THRES, avg_SSIM_ABOVE_THRES))
        print("total loss : {:<4.2f}, total SSIM : {:<4.4f}".format(
            avg_PSNR, avg_SSIM))

if __name__ == '__main__':
    if (config['onlyTrainMask']) and (config['useMask']) and (not config['fixMask']) \
        and (not config['usingSA']) and (not config['usingMaskLoss']) and (not config['usingSALoss']):
        print("testMaskNet")
        testMaskNet()

    if ( not config['onlyTrainMask']) and (not config['useMask']) and (not config['fixMask']) \
            and (not config['usingSA']) and (not config['usingMaskLoss']) and (not config['usingSALoss']):
        print("trainSimpleUNet")
        testSimpleNet()

    if (not config['onlyTrainMask']) and (not config['useMask']) and (not config['fixMask']) \
        and (config['usingSA']) and (not config['usingMaskLoss']) and (config['usingSALoss']):
        print("trainSANetSALoss")
        testSANetSALoss()

    if (not config['onlyTrainMask']) and (not config['useMask']) and (not config['fixMask']) \
        and (config['usingSA']) and (not config['usingMaskLoss']) and (not config['usingSALoss']):
        print("trainSANetNoSALoss")
        #trainSANetNoSALoss()

    if (not config['onlyTrainMask']) and (config['useMask']) and (not config['fixMask']) \
        and (not config['usingSA']) and (not config['usingMaskLoss']) and (not config['usingSALoss']):
        print("testWholeNetFixNoSA")
        testWholeNetFixNoSA()

    if (not config['onlyTrainMask']) and (config['useMask']) and (not config['fixMask']) \
        and (config['usingSA']) and (config['usingMaskLoss']) and (config['usingSALoss']):
        print("testWholeNet")
        testWholeNet()
