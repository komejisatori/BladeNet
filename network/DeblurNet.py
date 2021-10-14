import torch
import torch.nn.functional as F
from network.layers import *
from network.MaskNet import MaskNet

class WholeNet(nn.Module):
    def __init__(self, inChannels, outChannels, bilinear=True,
                 onlyTrainMask=False, usingMask=True, fixMask=False, usingSA=True, usingMaskLoss=True, usingSALoss=True):
        super(WholeNet, self).__init__()

        if usingMask:
            self.maskNet = MaskNet(inChannels, 2, bilinear)

            self.deblurNet = DeblurNet(inChannels+2, outChannels, bilinear, usingSA)
        else:
            self.deblurNet = DeblurNet(inChannels, outChannels, bilinear, usingSA)

        self.fixMask = fixMask
        self.usingMaskLoss = usingMaskLoss
        self.usingMask = usingMask
        self.onlyTrainMask = onlyTrainMask
        self.usingSA = usingSA
        self.usingSALoss = usingSALoss

        self.n_channels = inChannels
        self.n_classes = outChannels
        self.bilinear = bilinear



    def SAAdd(self, x2, x1):
        x1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return x2 + x1

    def forward(self, x):
        if self.usingMask:
            blurMask = self.maskNet(x)
            if self.onlyTrainMask:
                return blurMask

            inputWithMap = torch.cat((x, blurMask), dim=1)
            if self.usingSA:
                out, mask = self.deblurNet(inputWithMap)
                
                return out, mask, blurMask

            else:

                out = self.deblurNet(inputWithMap)

                return out, blurMask

        else:
            if self.usingSA:
                out, mask = self.deblurNet(x)
                return out, mask
            else:
                out = self.deblurNet(x)
                return out


class DeblurNet(nn.Module):
    def __init__(self, inChannels, outChannels, bilinear=True,
                usingSA=True):
        super(DeblurNet, self).__init__()

        self.usingSA = usingSA

        self.n_channels = inChannels
        self.n_classes = outChannels
        self.bilinear = bilinear

        self.inc = DoubleConv(inChannels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)

        if usingSA:
            self.up2 = SAUp(512, 256 // factor, bilinear)
            self.up3 = SAUp(256, 128 // factor, bilinear)
            self.up4 = SAUp(128, 64, bilinear)
        else:
            self.up2 = Up(512, 256 // factor, bilinear)
            self.up3 = Up(256, 128 // factor, bilinear)
            self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, outChannels)

    def SAAdd(self, x2, x1):
        x1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return x2 + x1

    def forward(self, input):
        
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        if self.usingSA:
            x, map1 = self.up2(x, x3)
            x, map2 = self.up3(x, x2)
            x, map3 = self.up4(x, x1)

            saSum = self.SAAdd(map2, map1)
            saSum = self.SAAdd(map3, saSum)
            
            x = self.outc(x)
            return x, saSum

        else:
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
            return logits



