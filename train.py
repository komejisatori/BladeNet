import os
import yaml
import shutil
import torch
import data.util as util
from data.dataloader import OriDataset, AugDataset, MaskDataset
from network.DeblurNet import WholeNet
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from visdom import Visdom
from criterion import *

import argparse

parser = argparse.ArgumentParser(description="yaml file path")
parser.add_argument('-c', '--config', default='yaml file path')
args = parser.parse_args()

CONFIG_PATH = args.config
print(CONFIG_PATH)

with open(CONFIG_PATH, 'r') as f:
    config = yaml.load(f)

    ONLYTRAINMASK = "onlyTrainMask" if config['onlyTrainMask'] else ""
    USEMASK = "useMask" if config['useMask'] else "notUseMask"
    FIXMASK = "fixMask" if config['fixMask'] else "notFixMask"
    USINGSA = "usingSA" if config['usingSA'] else "notUsingSA"
    USINGMASKLOSS = "usingMaskLoss" if config['usingMaskLoss'] else "notUsingMaskLoss"
    USINGSALOSS = "usingSALoss" if config['usingSALoss'] else "notUsingSALoss"

    saveName = "%s-%s-%s-%s-%s-%s-%s-%s" % (
        config['date'], ONLYTRAINMASK, USEMASK, FIXMASK, USINGSA,
        USINGMASKLOSS, USINGSALOSS, config['dataset'])
    print(saveName)
    print(config['train_sharp2'])
os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_available']
device_ids = range(config['gpu_num'])
if not os.path.exists('./config/' + saveName + '.yaml'):
    shutil.copy(CONFIG_PATH, './config/' + saveName + '.yaml')
else:
    print('already exists')


def trainMaskNet():
    trainSet = MaskDataset(sharp_root=config['train_sharp'], blur_root=config['train_blur'],
                           mask_root=config['train_mask'], resize_size=config['resize_size'],
                           patch_size=config['crop_size'], phase='train')
    testSet = MaskDataset(sharp_root=config['test_sharp'], blur_root=config['test_blur'],
                          mask_root=config['test_mask'], resize_size=config['resize_size'],
                          patch_size=config['crop_size'], phase='test')

    train_loader = DataLoader(trainSet,
                              batch_size=config['batchsize'],
                              shuffle=True,
                              num_workers=4,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(testSet, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)
    model = WholeNet(inChannels=3, outChannels=3, bilinear=True,
                     onlyTrainMask=config['onlyTrainMask'],
                     usingMask=config['useMask'],
                     fixMask=config['fixMask'],
                     usingSA=config['usingSA'],
                     usingMaskLoss=config['usingMaskLoss'],
                     usingSALoss=config['usingSALoss'])

    for name, value in model.deblurNet.named_parameters():
        value.requires_grad = False

    model = model.cuda()
    curStep = 0
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'])
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config['step'], gamma=0.5)  # learning rates
    criterion = CrossEntropyLoss2d()
    criterion_test = CrossEntropyLoss2d()
    viz = Visdom(env=saveName)
    bestMSE = 100000

    for epoch in range(10000000):
        avg_loss_1 = 0.0
        idx = 0
        model.train()
        for i, train_data in enumerate(train_loader):
            idx += 1
            train_data['L'] = train_data['L'].cuda()
            train_data['mask'] = torch.tensor(train_data['mask'], dtype=torch.long)
            train_data['mask'] = train_data['mask'].cuda()
            curStep += 1
            optimizer.zero_grad()
            mask = model(train_data['L'])
            loss = criterion(mask, train_data['mask'])

            loss.backward()
            optimizer.step()

            avg_loss_1 += loss.item()
            if idx % 100 == 0:
                print("epoch {}: trained {}".format(epoch, idx))

        scheduler.step()
        avg_loss_1 = avg_loss_1
        print("epoch {}: total loss : {:<4.2f}, lr : {}".format(
            epoch, avg_loss_1, scheduler.get_lr()[0]))
        viz.line(
            X=[epoch],
            Y=[avg_loss_1],
            win='masknetLoss',
            opts=dict(title='ce', legend=['train_ce']),
            update='append')

        if epoch % config['save_epoch'] == 0:
            with torch.no_grad():
                model.eval()
                avg_mse = 0
                idx = 0
                for test_data in test_loader:
                    idx += 1
                    test_data['L'] = test_data['L'].cuda()
                    test_data['mask'] = torch.tensor(test_data['mask'], dtype=torch.long)
                    test_data['mask'] = test_data['mask'].cuda()
                    mask = model(test_data['L'])
                    testMSE = criterion_test(mask, test_data['mask'])
                    avg_mse += testMSE.item()
                    if idx % 100 == 0:
                        print("epoch {}: tested {}".format(epoch, idx))
                avg_mse = avg_mse / idx
                print("total loss : {:<4.2f}".format(
                    avg_mse))
                viz.line(
                    X=[epoch],
                    Y=[avg_mse],
                    win='masknettestmse',
                    opts=dict(title='ce', legend=['valid_ce']),
                    update='append')
                if avg_mse < bestMSE:
                    bestMSE = avg_mse
                    save_path = os.path.join(config['model_dir'], saveName)
                    if not os.path.exists(config['model_dir']):
                        os.mkdir(config['model_dir'])
                    state_dict = model.state_dict()
                    for key, param in state_dict.items():
                        state_dict[key] = param.cpu()
                    torch.save(state_dict, save_path)


def trainSimpleUNet():
    trainSet = OriDataset(sharp_root1=config['train_sharp'], blur_root1=config['train_blur'],
                          sharp_root2=None, blur_root2=None,
                          patch_size=config['crop_size'], resize_size=config['resize_size'], phase='train')
    testSet = OriDataset(sharp_root1=config['test_sharp'], blur_root1=config['test_blur'],
                         sharp_root2=None, blur_root2=None,
                         patch_size=config['crop_size'], resize_size=config['resize_size'], phase='test')

    train_loader = DataLoader(trainSet,
                              batch_size=config['batchsize'],
                              shuffle=True,
                              num_workers=4,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(testSet, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)
    model = WholeNet(inChannels=3, outChannels=3, bilinear=True,
                     onlyTrainMask=config['onlyTrainMask'],
                     usingMask=config['useMask'],
                     fixMask=config['fixMask'],
                     usingSA=config['usingSA'],
                     usingMaskLoss=config['usingMaskLoss'],
                     usingSALoss=config['usingSALoss'])
    if config['pretrained_model'] != 'None':
        print('loading Pretrained {}'.format(config['pretrained_model']))
        model.load_state_dict(torch.load(config['pretrained_model']))

    startEpoch = config['startEpoch']

    model = model.cuda()
    curStep = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config['step'], gamma=0.5)  # learning rates
    mse = torch.nn.L1Loss()

    viz = Visdom(env=saveName)
    bestPSNR = config['best_psnr']

    for epoch in range(startEpoch, 10000000):
        avg_loss_1 = 0.0
        idx = 0
        model.train()
        for i, train_data in enumerate(train_loader):
            idx += 1
            train_data['L'] = train_data['L'].cuda()
            train_data['H'] = train_data['H'].cuda()
            curStep += 1
            optimizer.zero_grad()
            sharp = model(train_data['L'])
            loss = mse(sharp, train_data['H'])
            loss.backward()
            optimizer.step()
            avg_loss_1 += loss.item()
            if idx % 100 == 0:
                print("epoch {}: trained {}".format(epoch, idx))

        scheduler.step()
        avg_loss_1 = avg_loss_1 / idx
        print("epoch {}: total loss : {:<4.2f}, lr : {}".format(
            epoch, avg_loss_1, scheduler.get_lr()[0]))
        viz.line(
            X=[epoch],
            Y=[avg_loss_1],
            win='masknetLoss',
            opts=dict(title='mse', legend=['train_mse']),
            update='append')

        if epoch % config['save_epoch'] == 0:
            with torch.no_grad():
                model.eval()
                avg_PSNR = 0
                idx = 0
                for test_data in test_loader:
                    idx += 1
                    test_data['L'] = test_data['L'].cuda()

                    sharp = model(test_data['L'])
                    sharp = sharp.detach().float().cpu()
                    sharp = util.tensor2uint(sharp)
                    test_data['H'] = util.tensor2uint(test_data['H'])
                    current_psnr = util.calculate_psnr(sharp, test_data['H'], border=0)

                    avg_PSNR += current_psnr
                    if idx % 100 == 0:
                        print("epoch {}: tested {}".format(epoch, idx))
                avg_PSNR = avg_PSNR / idx
                print("total loss : {:<4.2f}".format(
                    avg_PSNR))
                viz.line(
                    X=[epoch],
                    Y=[avg_PSNR],
                    win='unetpsnr',
                    opts=dict(title='psnr', legend=['valid_psnr']),
                    update='append')
                if avg_PSNR > bestPSNR:
                    bestPSNR = avg_PSNR
                    save_path = os.path.join(config['model_dir'], saveName)
                    if not os.path.exists(config['model_dir']):
                        os.mkdir(config['model_dir'])
                    state_dict = model.state_dict()
                    for key, param in state_dict.items():
                        state_dict[key] = param.cpu()
                    torch.save(state_dict, save_path)


def trainSANetNoSALoss():
    trainSet = OriDataset(sharp_root1=config['train_sharp'], blur_root1=config['train_blur'],
                          sharp_root2=config['train_sharp2'], blur_root2=config['train_blur2'],
                          patch_size=config['crop_size'], resize_size=config['resize_size'], phase='train')
    testSet = OriDataset(sharp_root1=config['test_sharp'], blur_root1=config['test_blur'],
                         sharp_root2=None, blur_root2=None,
                         patch_size=config['crop_size'], resize_size=config['resize_size'], phase='test')

    train_loader = DataLoader(trainSet,
                              batch_size=config['batchsize'],
                              shuffle=True,
                              num_workers=8,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(testSet, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)
    model = WholeNet(inChannels=3, outChannels=3, bilinear=True,
                     onlyTrainMask=config['onlyTrainMask'],
                     usingMask=config['useMask'],
                     fixMask=config['fixMask'],
                     usingSA=config['usingSA'],
                     usingMaskLoss=config['usingMaskLoss'],
                     usingSALoss=config['usingSALoss'])

    if config['pretrained_model'] != 'None':
        print('loading Pretrained {}'.format(config['pretrained_model']))
        model.load_state_dict(torch.load(config['pretrained_model']))

    startEpoch = config['startEpoch']
    model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)

    # model = model.cuda()
    curStep = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config['step'], gamma=0.5)  # learning rates
    mse = torch.nn.L1Loss()
    viz = Visdom(env=saveName)
    bestPSNR = config['best_psnr']

    for epoch in range(startEpoch, 10000000):
        avg_loss_1 = 0.0
        idx = 0
        model.train()
        for i, train_data in enumerate(train_loader):
            idx += 1

            train_data['L'] = train_data['L'].cuda()
            train_data['H'] = train_data['H'].cuda()

            curStep += 1
            optimizer.zero_grad()
            sharp, SASum = model(train_data['L'])
            loss = mse(sharp, train_data['H'])

            loss.backward()
            optimizer.step()
            avg_loss_1 += loss.item()

            if idx % 100 == 0:
                print("epoch {}: trained {}".format(epoch, idx))

        scheduler.step()
        avg_loss_1 = avg_loss_1 / idx

        print("epoch {}: total loss image: {:<4.2f}, lr : {}".format(
            epoch, avg_loss_1, scheduler.get_lr()[0]))
        viz.line(
            X=[epoch],
            Y=[avg_loss_1],
            win='sanetloss',
            opts=dict(title='mse', legend=['train_image_mse']),
            update='append')

        if epoch % config['save_epoch'] == 0:
            with torch.no_grad():
                model.eval()
                avg_PSNR = 0
                idx = 0
                for test_data in test_loader:
                    idx += 1
                    test_data['L'] = test_data['L'].cuda()

                    sharp, _ = model(test_data['L'])
                    sharp = sharp.detach().float().cpu()
                    sharp = util.tensor2uint(sharp)
                    test_data['H'] = util.tensor2uint(test_data['H'])
                    current_psnr = util.calculate_psnr(sharp, test_data['H'], border=0)

                    avg_PSNR += current_psnr
                    if idx % 100 == 0:
                        print("epoch {}: tested {}".format(epoch, idx))
                avg_PSNR = avg_PSNR / idx
                print("total loss : {:<4.2f}".format(
                    avg_PSNR))
                viz.line(
                    X=[epoch],
                    Y=[avg_PSNR],
                    win='sanetpsnr',
                    opts=dict(title='psnr', legend=['valid_psnr']),
                    update='append')
                if avg_PSNR < bestPSNR:
                    bestPSNR = avg_PSNR
                    save_path = os.path.join(config['model_dir'], saveName)
                    if not os.path.exists(config['model_dir']):
                        os.mkdir(config['model_dir'])
                    state_dict = model.state_dict()
                    for key, param in state_dict.items():
                        state_dict[key] = param.cpu()
                    torch.save(state_dict, save_path)


def trainSANetSALoss():
    trainSet = OriDataset(sharp_root1=config['train_sharp'], blur_root1=config['train_blur'],
                          sharp_root2=None, blur_root2=None,
                          resize_size=config['resize_size'],
                          patch_size=config['crop_size'], phase='train')
    testSet = OriDataset(sharp_root1=config['test_sharp'], blur_root1=config['test_blur'],
                         sharp_root2=None, blur_root2=None,
                         resize_size=config['resize_size'],
                         patch_size=config['crop_size'], phase='test')
    trainSet_m = AugDataset(sharp_root=config['train_sharp2'], blur_root=config['train_blur2'],
                            mask_root=config['train_mask'], resize_size=config['resize_size'],
                            patch_size=config['crop_size'], phase='train')
    train_loader = DataLoader(trainSet,
                              batch_size=config['batchsize'],
                              shuffle=True,
                              num_workers=8,
                              drop_last=True,
                              pin_memory=True)
    train_loader_m = DataLoader(trainSet_m,
                                batch_size=config['batchsize'],
                                shuffle=True,
                                num_workers=8,
                                drop_last=True,
                                pin_memory=True)
    test_loader = DataLoader(testSet, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)

    model = WholeNet(inChannels=3, outChannels=3, bilinear=True,
                     onlyTrainMask=config['onlyTrainMask'],
                     usingMask=config['useMask'],
                     fixMask=config['fixMask'],
                     usingSA=config['usingSA'],
                     usingMaskLoss=config['usingMaskLoss'],
                     usingSALoss=config['usingSALoss'])
    model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)
    if config['pretrained_model'] != 'None':
        print('loading Pretrained {}'.format(config['pretrained_model']))
        model.load_state_dict(torch.load(config['pretrained_model']))

    startEpoch = config['startEpoch']



    # model = model.cuda()
    curStep = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config['step'], gamma=0.5)  # learning rates
    mse = torch.nn.L1Loss()
    dice = DiceLoss()
    viz = Visdom(env=saveName)
    bestPSNR = config['best_psnr']

    for epoch in range(startEpoch, 10000000):
        avg_loss_1 = 0.0
        avg_loss_m_1 = 0.0
        avg_loss_m_2 = 0.0
        idx = 0
        model.train()
        for i, train_data in enumerate(train_loader):
            idx += 1
            train_data['L'] = train_data['L'].cuda()
            train_data['H'] = train_data['H'].cuda()
            curStep += 1
            optimizer.zero_grad()
            sharp, mask = model(train_data['L'])
            loss = mse(sharp, train_data['H'])
            loss.backward()
            optimizer.step()
            avg_loss_1 += loss.item()
            if idx % 100 == 0:
                print("epoch {}: trained {}".format(epoch, idx))
        avg_loss_1 = avg_loss_1 / idx
        print("epoch {}: ori loss image: {:<4.2f}, lr : {}".format(
            epoch, avg_loss_1, scheduler.get_lr()[0]))

        idx = 0
        for i, train_data in enumerate(train_loader_m):
            idx += 1

            train_data['L'] = train_data['L'].cuda()
            train_data['H'] = train_data['H'].cuda()
            train_data['mask'] = train_data['mask'].cuda()
            curStep += 1
            optimizer.zero_grad()
            sharp, SASum = model(train_data['L'])
            loss1 = mse(sharp, train_data['H'])
            loss2 = dice(SASum, train_data['mask']) * 0.01 #0.01
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            avg_loss_m_1 += loss1.item()
            avg_loss_m_2 += loss2.item()
            if idx % 100 == 0:
                print("epoch {}: trained {}".format(epoch, idx))

        scheduler.step()
        avg_loss_m_1 = avg_loss_m_1 / idx
        avg_loss_m_2 = avg_loss_m_2 / idx
        print("epoch {}: aug loss image: {:<4.2f}, aug loss mask: {:<4.2f}, lr : {}".format(
            epoch, avg_loss_m_1, avg_loss_m_2, scheduler.get_lr()[0]))
        viz.line(
            X=[epoch],
            Y=[[avg_loss_1, avg_loss_m_1, avg_loss_m_2]],
            win='sanetloss',
            opts=dict(title='mse', legend=['train_ori_image_mse', 'train_aug_image_mse', 'train_aug_mask_dice']),
            update='append')

        if epoch % config['save_epoch'] == 0:
            with torch.no_grad():
                model.eval()
                avg_PSNR = 0
                idx = 0
                for test_data in test_loader:
                    idx += 1
                    test_data['L'] = test_data['L'].cuda()

                    sharp, _ = model(test_data['L'])
                    sharp = sharp.detach().float().cpu()
                    sharp = util.tensor2uint(sharp)
                    test_data['H'] = util.tensor2uint(test_data['H'])
                    current_psnr = util.calculate_psnr(sharp, test_data['H'], border=0)

                    avg_PSNR += current_psnr
                    if idx % 100 == 0:
                        print("epoch {}: tested {}".format(epoch, idx))
                avg_PSNR = avg_PSNR / idx
                print("total loss : {:<4.2f}".format(
                    avg_PSNR))
                viz.line(
                    X=[epoch],
                    Y=[avg_PSNR],
                    win='sanetpsnr',
                    opts=dict(title='psnr', legend=['valid_psnr']),
                    update='append')
                if avg_PSNR > bestPSNR:
                    bestPSNR = avg_PSNR
                    save_path = os.path.join(config['model_dir'], saveName)
                    if not os.path.exists(config['model_dir']):
                        os.mkdir(config['model_dir'])
                    state_dict = model.state_dict()
                    for key, param in state_dict.items():
                        state_dict[key] = param.cpu()
                    torch.save(state_dict, save_path)


def trainWholeNetMaskLossNoFixNoSA():
    trainSet = OriDataset(sharp_root1=config['train_sharp'], blur_root1=config['train_blur'],
                          sharp_root2=None, blur_root2=None,
                          resize_size=config['resize_size'],
                          patch_size=config['crop_size'], phase='train')
    testSet = OriDataset(sharp_root1=config['test_sharp'], blur_root1=config['test_blur'],
                         sharp_root2=None, blur_root2=None,
                         resize_size=config['resize_size'],
                         patch_size=config['crop_size'], phase='test')
    trainSet_m = AugDataset(sharp_root=config['train_sharp2'], blur_root=config['train_blur2'],
                            mask_root=config['train_mask'], resize_size=config['resize_size'],
                            patch_size=config['crop_size'], phase='train')

    train_loader = DataLoader(trainSet,
                              batch_size=config['batchsize'],
                              shuffle=True,
                              num_workers=8,
                              drop_last=True,
                              pin_memory=True)
    train_loader_m = DataLoader(trainSet_m,
                                batch_size=config['batchsize'],
                                shuffle=True,
                                num_workers=8,
                                drop_last=True,
                                pin_memory=True)
    test_loader = DataLoader(testSet, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)
    model = WholeNet(inChannels=3, outChannels=3, bilinear=True,
                     onlyTrainMask=config['onlyTrainMask'],
                     usingMask=config['useMask'],
                     fixMask=config['fixMask'],
                     usingSA=config['usingSA'],
                     usingMaskLoss=config['usingMaskLoss'],
                     usingSALoss=config['usingSALoss'])

    model.load_state_dict(torch.load(config['mask_pretrained_model']))

    if config['pretrained_model'] != 'None':
        print('loading Pretrained {}'.format(config['pretrained_model']))
        model.load_state_dict(torch.load(config['pretrained_model']))

    startEpoch = config['startEpoch']
    optimizer1 = torch.optim.Adam(model.deblurNet.parameters(), lr=config['lr'])
    optimizer2 = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler1 = lr_scheduler.MultiStepLR(optimizer1, milestones=config['step'], gamma=0.5)  # learning rates
    scheduler2 = lr_scheduler.MultiStepLR(optimizer2, milestones=config['step'], gamma=0.5)  # learning rates

    model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)
    curStep = 0
    mse = torch.nn.L1Loss()
    criterion = CrossEntropyLoss2d()
    viz = Visdom(env=saveName)
    bestPSNR = config['best_psnr']

    for epoch in range(startEpoch, 10000000):
        avg_loss_1 = 0.0
        avg_loss_m_1 = 0.0
        avg_loss_m_2 = 0.0
        idx = 0
        model.train()
        for i, train_data in enumerate(train_loader):
            idx += 1
            train_data['L'] = train_data['L'].cuda()
            train_data['H'] = train_data['H'].cuda()
            curStep += 1
            optimizer1.zero_grad()
            sharp, mask = model(train_data['L'])
            loss = mse(sharp, train_data['H'])
            loss.backward()
            optimizer1.step()
            avg_loss_1 += loss.item()
            if idx % 100 == 0:
                print("epoch {}: trained {}".format(epoch, idx))
        avg_loss_1 = avg_loss_1 / idx
        print("epoch {}: ori loss image: {:<4.2f}, lr : {}".format(
            epoch, avg_loss_1, scheduler1.get_lr()[0]))
        scheduler1.step()

        idx = 0
        for i, train_data in enumerate(train_loader_m):
            idx += 1
            train_data['L'] = train_data['L'].cuda()
            train_data['H'] = train_data['H'].cuda()
            train_data['mask'] = torch.tensor(train_data['mask'], dtype=torch.long)
            train_data['mask'] = train_data['mask'].cuda()
            curStep += 1
            optimizer2.zero_grad()
            sharp, mask = model(train_data['L'])
            loss1 = mse(sharp, train_data['H'])
            loss2 = criterion(mask, train_data['mask']) * 0.25
            loss = loss1 + loss2
            loss.backward()
            optimizer2.step()
            avg_loss_m_1 += loss1.item()
            avg_loss_m_2 += loss2.item()
            if idx % 100 == 0:
                print("epoch {}: trained {}".format(epoch, idx))

        scheduler2.step()
        avg_loss_m_1 = avg_loss_m_1 / idx
        avg_loss_m_2 = avg_loss_m_2 / idx

        print("epoch {}: aug loss image: {:<4.2f}, aug loss mask: {:<4.2f}, lr : {}".format(
            epoch, avg_loss_m_1, avg_loss_m_2, scheduler2.get_lr()[0]))

        viz.line(
            X=[epoch],
            Y=[[avg_loss_1, avg_loss_m_1, avg_loss_m_2]],
            win='wholenetloss',
            opts=dict(title='mse', legend=['train_ori_image_mse', 'train_aug_image_mse', 'train_aug_mask_dice']),
            update='append')

        if epoch % config['save_epoch'] == 0:
            with torch.no_grad():
                model.eval()
                avg_PSNR = 0
                idx = 0
                for test_data in test_loader:
                    idx += 1
                    test_data['L'] = test_data['L'].cuda()
                    sharp, _ = model(test_data['L'])
                    sharp = sharp.detach().float().cpu()
                    sharp = util.tensor2uint(sharp)
                    test_data['H'] = util.tensor2uint(test_data['H'])
                    current_psnr = util.calculate_psnr(sharp, test_data['H'], border=0)

                    avg_PSNR += current_psnr
                    if idx % 100 == 0:
                        print("epoch {}: tested {}".format(epoch, idx))
                avg_PSNR = avg_PSNR / idx
                print("total loss : {:<4.2f}".format(
                    avg_PSNR))
                viz.line(
                    X=[epoch],
                    Y=[avg_PSNR],
                    win='wholenetpsnr',
                    opts=dict(title='psnr', legend=['valid_psnr']),
                    update='append')
                if avg_PSNR > bestPSNR:
                    bestPSNR = avg_PSNR
                    save_path = os.path.join(config['model_dir'], saveName)
                    if not os.path.exists(config['model_dir']):
                        os.mkdir(config['model_dir'])
                    state_dict = model.state_dict()
                    for key, param in state_dict.items():
                        state_dict[key] = param.cpu()
                    torch.save(state_dict, save_path)


def trainWholeNetFixNoSA():
    trainSet = OriDataset(sharp_root1=config['train_sharp'], blur_root1=config['train_blur'],
                          sharp_root2=None, blur_root2=None,
                          patch_size=256, resize_size=config['resize_size'], phase='train')
    testSet = OriDataset(sharp_root1=config['test_sharp'], blur_root1=config['test_blur'],
                         sharp_root2=None, blur_root2=None,
                         patch_size=config['crop_size'], resize_size=config['resize_size'], phase='test')
    trainSet = OriDataset(sharp_root1=config['train_sharp'], blur_root1=config['train_blur'],
                          sharp_root2=None, blur_root2=None,
                          patch_size=config['crop_size'], resize_size=config['resize_size'], phase='train')
    testSet = OriDataset(sharp_root1=config['test_sharp'], blur_root1=config['test_blur'],
                         sharp_root2=None, blur_root2=None,
                         patch_size=config['crop_size'], resize_size=config['resize_size'], phase='test')

    train_loader = DataLoader(trainSet,
                              batch_size=16,
                              shuffle=True,
                              num_workers=8,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(testSet, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)
    model = WholeNet(inChannels=3, outChannels=3, bilinear=True,
                     onlyTrainMask=config['onlyTrainMask'],
                     usingMask=config['useMask'],
                     fixMask=config['fixMask'],
                     usingSA=config['usingSA'],
                     usingMaskLoss=config['usingMaskLoss'],
                     usingSALoss=config['usingSALoss'])

    model.load_state_dict(torch.load(config['pretrained_model']))

    for name, value in model.maskNet.named_parameters():
        value.requires_grad = False

    model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)
    curStep = 0
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'])
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config['step'], gamma=0.5)  # learning rates
    mse = torch.nn.L1Loss()

    viz = Visdom(env=saveName)
    bestPSNR = 0

    for epoch in range(10000000):
        avg_loss_1 = 0.0
        idx = 0
        model.train()
        for i, train_data in enumerate(train_loader):
            idx += 1
            train_data['L'] = train_data['L'].cuda()
            train_data['H'] = train_data['H'].cuda()

            curStep += 1
            optimizer.zero_grad()
            sharp, mask = model(train_data['L'])
            loss = mse(sharp, train_data['H'])
            loss.backward()
            optimizer.step()
            avg_loss_1 += loss.item()
            if idx % 100 == 0:
                print("epoch {}: trained {}".format(epoch, idx))

        scheduler.step()
        avg_loss_1 = avg_loss_1 / idx
        print("epoch {}: total loss image: {:<4.2f}, lr : {}".format(
            epoch, avg_loss_1, scheduler.get_lr()[0]))
        viz.line(
            X=[epoch],
            Y=[avg_loss_1],
            win='wholenetloss',
            opts=dict(title='mse', legend=['train_image_mse', 'train_mask_mse']),
            update='append')

        if epoch % config['save_epoch'] == 0:
            with torch.no_grad():
                model.eval()
                avg_PSNR = 0
                idx = 0
                for test_data in test_loader:
                    idx += 1
                    test_data['L'] = test_data['L'].cuda()

                    sharp, _ = model(test_data['L'])
                    sharp = sharp.detach().float().cpu()
                    sharp = util.tensor2uint(sharp)
                    test_data['H'] = util.tensor2uint(test_data['H'])
                    current_psnr = util.calculate_psnr(sharp, test_data['H'], border=0)

                    avg_PSNR += current_psnr
                    if idx % 100 == 0:
                        print("epoch {}: tested {}".format(epoch, idx))
                avg_PSNR = avg_PSNR / idx
                print("total loss : {:<4.2f}".format(
                    avg_PSNR))
                viz.line(
                    X=[epoch],
                    Y=[avg_PSNR],
                    win='wholenetpsnr',
                    opts=dict(title='psnr', legend=['valid_psnr']),
                    update='append')
                if avg_PSNR > bestPSNR:
                    bestPSNR = avg_PSNR
                    save_path = os.path.join(config['model_dir'], saveName)
                    if not os.path.exists(config['model_dir']):
                        os.mkdir(config['model_dir'])
                    state_dict = model.state_dict()
                    for key, param in state_dict.items():
                        state_dict[key] = param.cpu()
                    torch.save(state_dict, save_path)


def trainWholeNetMaskLossNoFixSALoss():
    trainSet = OriDataset(sharp_root1=config['train_sharp'], blur_root1=config['train_blur'],
                          sharp_root2=None, blur_root2=None,
                          resize_size=config['resize_size'],
                          patch_size=config['crop_size'], phase='train')
    testSet = OriDataset(sharp_root1=config['test_sharp'], blur_root1=config['test_blur'],
                         sharp_root2=None, blur_root2=None,
                         resize_size=config['resize_size'],
                         patch_size=config['crop_size'], phase='test')
    trainSet_m = AugDataset(sharp_root=config['train_sharp2'], blur_root=config['train_blur2'],
                            mask_root=config['train_mask'], resize_size=config['resize_size'],
                            patch_size=config['crop_size'], phase='train')
    train_loader = DataLoader(trainSet,
                              batch_size=config['batchsize'],
                              shuffle=True,
                              num_workers=8,
                              drop_last=True,
                              pin_memory=True)
    train_loader_m = DataLoader(trainSet_m,
                                batch_size=config['batchsize'],
                                shuffle=True,
                                num_workers=8,
                                drop_last=True,
                                pin_memory=True)
    test_loader = DataLoader(testSet, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)

    model = WholeNet(inChannels=3, outChannels=3, bilinear=True,
                     onlyTrainMask=config['onlyTrainMask'],
                     usingMask=config['useMask'],
                     fixMask=config['fixMask'],
                     usingSA=config['usingSA'],
                     usingMaskLoss=config['usingMaskLoss'],
                     usingSALoss=config['usingSALoss'])
    '''
    save_model = torch.load(config['mask_pretrained_model'])
    model_dict =  model.state_dict()
    state_dict = {k:v for k,v in save_model.items() if k[:7]=='maskNet'}
    print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])



    #model.maskNet.load_state_dict(torch.load(config['mask_pretrained_model']))

    if config['sanet_pretrianed_model'] != 'None':
        save_model = torch.load(config['sanet_pretrianed_model'])
        model_dict = model.state_dict()
        for (k, v) in  save_model.items():
            if k[:16] == 'module.deblurNet':

                if k[:41] == 'module.deblurNet.inc.double_conv.0.weight':
                    temp = model_dict[k[7:]]
                    temp[:,:3,:,:] = v[:,:3,:,:]
                    state_dict[k[7:]] = temp
                else:
                    state_dict[k[7:]] = v
    print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    '''

    startEpoch = config['startEpoch']

    # model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)

    optimizer1 = torch.optim.Adam(model.deblurNet.parameters(), lr=config['lr'])
    optimizer2 = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler1 = lr_scheduler.MultiStepLR(optimizer1, milestones=config['step'], gamma=0.5)  # learning rates
    scheduler2 = lr_scheduler.MultiStepLR(optimizer2, milestones=config['step'], gamma=0.5)  # learning rates
    model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)
    if config['pretrained_model'] != 'None':
        print('loading Pretrained {}'.format(config['pretrained_model']))
        model.load_state_dict(torch.load(config['pretrained_model']))

    # model = model.cuda()
    curStep = 0
    # optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config['step'], gamma=0.5)  # learning rates
    mse = torch.nn.L1Loss()
    dice = DiceLoss()
    criterion = CrossEntropyLoss2d()
    viz = Visdom(env=saveName)
    bestPSNR = config['best_psnr']

    for epoch in range(startEpoch, 10000000):
        avg_loss_1 = 0.0
        avg_loss_m_1 = 0.0
        avg_loss_m_2 = 0.0
        avg_loss_m_3 = 0.0
        idx = 0
        model.train()

        for i, train_data in enumerate(train_loader):
            idx += 1

            train_data['L'] = train_data['L'].cuda()
            train_data['H'] = train_data['H'].cuda()
            curStep += 1
            optimizer1.zero_grad()
            sharp, _, _ = model(train_data['L'])
            loss = mse(sharp, train_data['H'])
            loss.backward()
            optimizer1.step()
            avg_loss_1 += loss.item()
            if idx % 100 == 0:
                print("epoch {}: trained {}".format(epoch, idx))

        scheduler1.step()
        avg_loss_1 = avg_loss_1 / idx
        print("epoch {}: ori loss image: {:<4.2f}, lr : {}".format(
            epoch, avg_loss_1, scheduler1.get_lr()[0]))

        idx = 0
        for i, train_data in enumerate(train_loader_m):
            idx += 1

            train_data['L'] = train_data['L'].cuda()
            train_data['H'] = train_data['H'].cuda()
            # train_data['mask'] = torch.tensor(train_data['mask'], dtype=torch.long)
            maskforCE = torch.tensor(train_data['mask'], dtype=torch.long)
            train_data['mask'] = train_data['mask'].cuda()
            maskforCE = maskforCE.cuda()
            curStep += 1
            optimizer2.zero_grad()
            sharp, SASum, mask = model(train_data['L'])
            loss1 = mse(sharp, train_data['H'])
            loss2 = dice(SASum, train_data['mask']) * 0.01
            loss3 = criterion(mask, maskforCE) * 0.25
            loss = loss1 + loss2 + loss3
            loss.backward()
            optimizer2.step()
            avg_loss_m_1 += loss1.item()
            avg_loss_m_2 += loss2.item()
            avg_loss_m_3 += loss3.item()
            if idx % 100 == 0:
                print("epoch {}: trained {}".format(epoch, idx))

        scheduler2.step()
        avg_loss_m_1 = avg_loss_m_1 / idx
        avg_loss_m_2 = avg_loss_m_2 / idx
        avg_loss_m_3 = avg_loss_m_3 / idx
        print("epoch {}: aug loss image: {:<4.2f}, aug loss sa: {:<4.2f}, aug loss mask: {:<4.2f}, lr : {}".format(
            epoch, avg_loss_m_1, avg_loss_m_2, avg_loss_m_3, scheduler2.get_lr()[0]))
        viz.line(
            X=[epoch],
            Y=[[avg_loss_1, avg_loss_m_1, avg_loss_m_2, avg_loss_m_3]],
            win='sanetloss',
            opts=dict(title='mse',
                      legend=['train_ori_image_mse', 'train_aug_image_mse', 'train_aug_sa_dice', 'train_aug_mask_ce']),
            update='append')

        if epoch % config['save_epoch'] == 0:
            with torch.no_grad():
                model.eval()
                avg_PSNR = 0
                idx = 0
                for test_data in test_loader:
                    idx += 1
                    test_data['L'] = test_data['L'].cuda()

                    sharp, _, _ = model(test_data['L'])
                    sharp = sharp.detach().float().cpu()
                    sharp = util.tensor2uint(sharp)
                    test_data['H'] = util.tensor2uint(test_data['H'])
                    current_psnr = util.calculate_psnr(sharp, test_data['H'], border=0)

                    avg_PSNR += current_psnr
                    if idx % 100 == 0:
                        print("epoch {}: tested {}".format(epoch, idx))
                avg_PSNR = avg_PSNR / idx
                print("total loss : {:<4.2f}".format(
                    avg_PSNR))
                viz.line(
                    X=[epoch],
                    Y=[avg_PSNR],
                    win='sanetpsnr',
                    opts=dict(title='psnr', legend=['valid_psnr']),
                    update='append')
                if avg_PSNR > bestPSNR:
                    bestPSNR = avg_PSNR
                    save_path = os.path.join(config['model_dir'], saveName)
                    if not os.path.exists(config['model_dir']):
                        os.mkdir(config['model_dir'])
                    state_dict = model.state_dict()
                    for key, param in state_dict.items():
                        state_dict[key] = param.cpu()
                    torch.save(state_dict, save_path)


def trainWholeNetMaskLossNoFixSALossPre():
    trainSet = OriDataset(sharp_root1=config['train_sharp'], blur_root1=config['train_blur'],
                          sharp_root2=None, blur_root2=None,
                          resize_size=config['resize_size'],
                          patch_size=config['crop_size'], phase='train')
    testSet = OriDataset(sharp_root1=config['test_sharp'], blur_root1=config['test_blur'],
                         sharp_root2=None, blur_root2=None,
                         resize_size=config['resize_size'],
                         patch_size=config['crop_size'], phase='test')
    trainSet_m = AugDataset(sharp_root=config['train_sharp2'], blur_root=config['train_blur2'],
                            mask_root=config['train_mask'], resize_size=config['resize_size'],
                            patch_size=config['crop_size'], phase='train')
    train_loader = DataLoader(trainSet,
                              batch_size=config['batchsize'],
                              shuffle=True,
                              num_workers=8,
                              drop_last=True,
                              pin_memory=True)
    train_loader_m = DataLoader(trainSet_m,
                                batch_size=config['batchsize'],
                                shuffle=True,
                                num_workers=8,
                                drop_last=True,
                                pin_memory=True)
    test_loader = DataLoader(testSet, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)

    model = WholeNet(inChannels=3, outChannels=3, bilinear=True,
                     onlyTrainMask=config['onlyTrainMask'],
                     usingMask=config['useMask'],
                     fixMask=config['fixMask'],
                     usingSA=config['usingSA'],
                     usingMaskLoss=config['usingMaskLoss'],
                     usingSALoss=config['usingSALoss'])

    save_model = torch.load(config['mask_pretrained_model'])
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k[:7] == 'maskNet'}
    print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    # model.maskNet.load_state_dict(torch.load(config['mask_pretrained_model']))

    if config['pretrained_model'] != 'None':
        print('loading Pretrained {}'.format(config['pretrained_model']))
        model.load_state_dict(torch.load(config['pretrained_model']))

    startEpoch = config['startEpoch']

    # model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)

    optimizer1 = torch.optim.Adam(model.deblurNet.parameters(), lr=config['lr'])
    optimizer2 = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler1 = lr_scheduler.MultiStepLR(optimizer1, milestones=config['step'], gamma=0.5)  # learning rates
    scheduler2 = lr_scheduler.MultiStepLR(optimizer2, milestones=config['step'], gamma=0.5)  # learning rates

    model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)
    # model = model.cuda()
    curStep = 0
    # optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config['step'], gamma=0.5)  # learning rates
    mse = torch.nn.L1Loss()
    dice = DiceLoss()
    criterion = CrossEntropyLoss2d()
    viz = Visdom(env=saveName)
    bestPSNR = config['best_psnr']

    for epoch in range(startEpoch, 10000000):
        avg_loss_1 = 0.0
        avg_loss_m_1 = 0.0
        avg_loss_m_2 = 0.0
        avg_loss_m_3 = 0.0
        idx = 0
        model.train()

        for i, train_data in enumerate(train_loader):
            idx += 1
            train_data['L'] = train_data['L'].cuda()
            train_data['H'] = train_data['H'].cuda()
            curStep += 1
            optimizer1.zero_grad()
            sharp, _, _ = model(train_data['L'])
            loss = mse(sharp, train_data['H'])
            loss.backward()
            optimizer1.step()
            avg_loss_1 += loss.item()
            if idx % 100 == 0:
                print("epoch {}: trained {}".format(epoch, idx))

        scheduler1.step()
        avg_loss_1 = avg_loss_1 / idx
        print("epoch {}: ori loss image: {:<4.2f}, lr : {}".format(
            epoch, avg_loss_1, scheduler1.get_lr()[0]))

        idx = 0
        for i, train_data in enumerate(train_loader_m):
            idx += 1

            train_data['L'] = train_data['L'].cuda()
            train_data['H'] = train_data['H'].cuda()
            # train_data['mask'] = torch.tensor(train_data['mask'], dtype=torch.long)
            maskforCE = torch.tensor(train_data['mask'], dtype=torch.long)
            train_data['mask'] = train_data['mask'].cuda()
            maskforCE = maskforCE.cuda()
            curStep += 1
            optimizer2.zero_grad()
            sharp, SASum, mask = model(train_data['L'])
            loss1 = mse(sharp, train_data['H'])
            loss2 = dice(SASum, train_data['mask']) * 0.005
            loss3 = criterion(mask, maskforCE) * 0.25
            loss = loss1 + loss2 + loss3
            loss.backward()
            optimizer2.step()
            avg_loss_m_1 += loss1.item()
            avg_loss_m_2 += loss2.item()
            avg_loss_m_3 += loss3.item()
            if idx % 100 == 0:
                print("epoch {}: trained {}".format(epoch, idx))

        scheduler2.step()
        avg_loss_m_1 = avg_loss_m_1 / idx
        avg_loss_m_2 = avg_loss_m_2 / idx
        avg_loss_m_3 = avg_loss_m_3 / idx
        print("epoch {}: aug loss image: {:<4.2f}, aug loss sa: {:<4.2f}, aug loss mask: {:<4.2f}, lr : {}".format(
            epoch, avg_loss_m_1, avg_loss_m_2, avg_loss_m_3, scheduler2.get_lr()[0]))
        viz.line(
            X=[epoch],
            Y=[[avg_loss_1, avg_loss_m_1, avg_loss_m_2, avg_loss_m_3]],
            win='sanetloss',
            opts=dict(title='mse',
                      legend=['train_ori_image_mse', 'train_aug_image_mse', 'train_aug_sa_dice', 'train_aug_mask_ce']),
            update='append')

        if epoch % config['save_epoch'] == 0:
            with torch.no_grad():
                model.eval()
                avg_PSNR = 0
                idx = 0
                for test_data in test_loader:
                    idx += 1
                    test_data['L'] = test_data['L'].cuda()

                    sharp, _, _ = model(test_data['L'])
                    sharp = sharp.detach().float().cpu()
                    sharp = util.tensor2uint(sharp)
                    test_data['H'] = util.tensor2uint(test_data['H'])
                    current_psnr = util.calculate_psnr(sharp, test_data['H'], border=0)

                    avg_PSNR += current_psnr
                    if idx % 100 == 0:
                        print("epoch {}: tested {}".format(epoch, idx))
                avg_PSNR = avg_PSNR / idx
                print("total loss : {:<4.2f}".format(
                    avg_PSNR))
                viz.line(
                    X=[epoch],
                    Y=[avg_PSNR],
                    win='sanetpsnr',
                    opts=dict(title='psnr', legend=['valid_psnr']),
                    update='append')
                if avg_PSNR > bestPSNR:
                    bestPSNR = avg_PSNR
                    save_path = os.path.join(config['model_dir'], saveName)
                    if not os.path.exists(config['model_dir']):
                        os.mkdir(config['model_dir'])
                    state_dict = model.state_dict()
                    for key, param in state_dict.items():
                        state_dict[key] = param.cpu()
                    torch.save(state_dict, save_path)


def finetuneWholeNetMaskLossNoFixSALoss():
    trainSet = OriDataset(sharp_root1=config['train_sharp1'], blur_root1=config['train_blur1'],
                          sharp_root2=config['train_sharp2'], blur_root2=config['train_blur2'],
                          resize_size=config['resize_size'],
                          patch_size=config['crop_size'], phase='train')
    testSet = OriDataset(sharp_root1=config['test_sharp'], blur_root1=config['test_blur'],
                         sharp_root2=None, blur_root2=None,
                         resize_size=config['resize_size'],
                         patch_size=config['crop_size'], phase='test')
    train_loader = DataLoader(trainSet,
                              batch_size=config['batchsize'],
                              shuffle=True,
                              num_workers=8,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(testSet, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)

    model = WholeNet(inChannels=3, outChannels=3, bilinear=True,
                     onlyTrainMask=config['onlyTrainMask'],
                     usingMask=config['useMask'],
                     fixMask=config['fixMask'],
                     usingSA=config['usingSA'],
                     usingMaskLoss=config['usingMaskLoss'],
                     usingSALoss=config['usingSALoss'])

    startEpoch = config['startEpoch']

    # model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)

    optimizer1 = torch.optim.Adam(model.deblurNet.parameters(), lr=config['lr'])

    scheduler1 = lr_scheduler.MultiStepLR(optimizer1, milestones=config['step'], gamma=0.5)  # learning rates

    model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)
    if config['pretrained_model'] != 'None':
        print('loading Pretrained {}'.format(config['pretrained_model']))
        model.load_state_dict(torch.load(config['pretrained_model']))

    # model = model.cuda()
    curStep = 0
    # optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config['step'], gamma=0.5)  # learning rates
    mse = torch.nn.L1Loss()
    viz = Visdom(env=saveName)
    bestPSNR = config['best_psnr']

    for epoch in range(startEpoch, 10000000):
        avg_loss_1 = 0.0
        idx = 0
        model.train()

        for i, train_data in enumerate(train_loader):
            idx += 1
            train_data['L'] = train_data['L'].cuda()
            train_data['H'] = train_data['H'].cuda()
            curStep += 1
            optimizer1.zero_grad()
            sharp, _, _ = model(train_data['L'])
            loss = mse(sharp, train_data['H'])
            loss.backward()
            optimizer1.step()
            avg_loss_1 += loss.item()
            if idx % 100 == 0:
                print("epoch {}: trained {}".format(epoch, idx))

        scheduler1.step()
        avg_loss_1 = avg_loss_1 / idx
        print("epoch {}: ori loss image: {:<4.2f}, lr : {}".format(
            epoch, avg_loss_1, scheduler1.get_lr()[0]))

        viz.line(
            X=[epoch],
            Y=[avg_loss_1],
            win='sanetloss',
            opts=dict(title='mse',
                      legend=['train_ori_image_mse']),
            update='append')

        if epoch % config['save_epoch'] == 0:
            with torch.no_grad():
                model.eval()
                avg_PSNR = 0
                idx = 0
                for test_data in test_loader:
                    idx += 1
                    test_data['L'] = test_data['L'].cuda()

                    sharp, _, _ = model(test_data['L'])
                    sharp = sharp.detach().float().cpu()
                    sharp = util.tensor2uint(sharp)
                    test_data['H'] = util.tensor2uint(test_data['H'])
                    current_psnr = util.calculate_psnr(sharp, test_data['H'], border=0)

                    avg_PSNR += current_psnr
                    if idx % 100 == 0:
                        print("epoch {}: tested {}".format(epoch, idx))
                avg_PSNR = avg_PSNR / idx
                print("total loss : {:<4.2f}".format(
                    avg_PSNR))
                viz.line(
                    X=[epoch],
                    Y=[avg_PSNR],
                    win='sanetpsnr',
                    opts=dict(title='psnr', legend=['valid_psnr']),
                    update='append')
                if avg_PSNR > bestPSNR:
                    bestPSNR = avg_PSNR
                    save_path = os.path.join(config['model_dir'], saveName)
                    if not os.path.exists(config['model_dir']):
                        os.mkdir(config['model_dir'])
                    state_dict = model.state_dict()
                    for key, param in state_dict.items():
                        state_dict[key] = param.cpu()
                    torch.save(state_dict, save_path)


if __name__ == '__main__':
    if config['finetune']:
        print("finetune")
        if (not config['onlyTrainMask']) and (config['useMask']) and (not config['fixMask']) \
                and (config['usingSA']) and (config['usingMaskLoss']) and (config['usingSALoss']):
            print("trainWholeNetMaskLossNoFixSALoss")
            finetuneWholeNetMaskLossNoFixSALoss()

    if not config['finetune']:
        if (config['onlyTrainMask']) and (config['useMask']) and (not config['fixMask']) \
                and (not config['usingSA']) and (not config['usingMaskLoss']) and (not config['usingSALoss']):
            print("trainMaskNet")
            trainMaskNet()

        if (not config['onlyTrainMask']) and (not config['useMask']) and (not config['fixMask']) \
                and (not config['usingSA']) and (not config['usingMaskLoss']) and (not config['usingSALoss']):
            print("trainSimpleUNet")
            trainSimpleUNet()

        if (not config['onlyTrainMask']) and (not config['useMask']) and (not config['fixMask']) \
                and (config['usingSA']) and (not config['usingMaskLoss']) and (config['usingSALoss']):
            print("trainSANetSALoss")
            trainSANetSALoss()

        if (not config['onlyTrainMask']) and (not config['useMask']) and (not config['fixMask']) \
                and (config['usingSA']) and (not config['usingMaskLoss']) and (not config['usingSALoss']):
            print("trainSANetNoSALoss")
            trainSANetNoSALoss()

        if (not config['onlyTrainMask']) and (config['useMask']) and (not config['fixMask']) \
                and (not config['usingSA']) and (config['usingMaskLoss']) and (not config['usingSALoss']):
            print("trainWholeNetMaskLossNoFixNoSA")
            trainWholeNetMaskLossNoFixNoSA()

        if (not config['onlyTrainMask']) and (config['useMask']) and (config['fixMask']) \
                and (not config['usingSA']) and (not config['usingMaskLoss']) and (not config['usingSALoss']):
            print("trainWholeNetFixNoSA")
            trainWholeNetFixNoSA()

        if (not config['onlyTrainMask']) and (config['useMask']) and (not config['fixMask']) \
                and (config['usingSA']) and (config['usingMaskLoss']) and (config['usingSALoss']):
            print("trainWholeNetMaskLossNoFixSALoss")
            trainWholeNetMaskLossNoFixSALoss()

