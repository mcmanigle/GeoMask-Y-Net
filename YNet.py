#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# In[2]:


def conv_layer(in_channels, out_channels):
    return [ nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True) ]

def down_layer(in_channels, out_channels, double = False):
    if double:
        return ([nn.MaxPool2d(2)] + 
                conv_layer(in_channels, out_channels) + 
                conv_layer(out_channels, out_channels)
               )
    return [nn.MaxPool2d(2)] + conv_layer(in_channels, out_channels)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        
        conv_layers = conv_layer(in_channels, out_channels) + conv_layer(out_channels, out_channels)
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# In[3]:


class ICUNet(nn.Module):
    def __init__(self, input_channels, classif_classes, segment_channels,
                 bilinear=True, pretrained=True):
        super(ICUNet, self).__init__()
        self.input_channels = input_channels
        self.classif_classes = classif_classes
        self.segment_channels = segment_channels
        self.bilinear = bilinear

        self.conv0 = nn.Sequential(*conv_layer(input_channels, 64))
        self.down1 = nn.Sequential(*down_layer(64, 128))
        self.down2 = nn.Sequential(*down_layer(128, 256, double=True))
        self.down3 = nn.Sequential(*down_layer(256, 512, double=True))
        self.down4 = nn.Sequential(*down_layer(512, 512, double=True))
        
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.segoutc = nn.Conv2d(64, segment_channels, kernel_size=1)
        
        self.avgpl = nn.Sequential(
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.clsfc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, classif_classes),
        )
        if pretrained:
            self.load_weights()

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        y = x5.clone()
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.segoutc(x)
        
        y = self.avgpl(y)
        y = torch.flatten(y, 1)
        y = self.clsfc(y)
        
        return {'cls': y, 'seg': x}
    
    def load_weights(self):
        base_model = models.vgg11_bn(pretrained=True)
        if self.input_channels == 1:
            base_sd = base_model.state_dict()
            base_f0w = base_sd['features.0.weight']
            base_sd['features.0.weight'] = base_f0w.mean(dim=1, keepdim=True)
            base_model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            base_model.load_state_dict(base_sd)
        
        self.conv0[0].load_state_dict(base_model.features[ 0].state_dict())
        self.conv0[1].load_state_dict(base_model.features[ 1].state_dict())
        self.conv0[2].load_state_dict(base_model.features[ 2].state_dict())
        self.down1[0].load_state_dict(base_model.features[ 3].state_dict())
        self.down1[1].load_state_dict(base_model.features[ 4].state_dict())
        self.down1[2].load_state_dict(base_model.features[ 5].state_dict())
        self.down1[3].load_state_dict(base_model.features[ 6].state_dict())
        self.down2[0].load_state_dict(base_model.features[ 7].state_dict())
        self.down2[1].load_state_dict(base_model.features[ 8].state_dict())
        self.down2[2].load_state_dict(base_model.features[ 9].state_dict())
        self.down2[3].load_state_dict(base_model.features[10].state_dict())
        self.down2[4].load_state_dict(base_model.features[11].state_dict())
        self.down2[5].load_state_dict(base_model.features[12].state_dict())
        self.down2[6].load_state_dict(base_model.features[13].state_dict())
        self.down3[0].load_state_dict(base_model.features[14].state_dict())
        self.down3[1].load_state_dict(base_model.features[15].state_dict())
        self.down3[2].load_state_dict(base_model.features[16].state_dict())
        self.down3[3].load_state_dict(base_model.features[17].state_dict())
        self.down3[4].load_state_dict(base_model.features[18].state_dict())
        self.down3[5].load_state_dict(base_model.features[19].state_dict())
        self.down3[6].load_state_dict(base_model.features[20].state_dict())
        self.down4[0].load_state_dict(base_model.features[21].state_dict())
        self.down4[1].load_state_dict(base_model.features[22].state_dict())
        self.down4[2].load_state_dict(base_model.features[23].state_dict())
        self.down4[3].load_state_dict(base_model.features[24].state_dict())
        self.down4[4].load_state_dict(base_model.features[25].state_dict())
        self.down4[5].load_state_dict(base_model.features[26].state_dict())
        self.down4[6].load_state_dict(base_model.features[27].state_dict())
        self.avgpl[0].load_state_dict(base_model.features[28].state_dict())
        self.avgpl[1].load_state_dict(base_model.avgpool.state_dict())
        self.clsfc[0].load_state_dict(base_model.classifier[0].state_dict())
        self.clsfc[1].load_state_dict(base_model.classifier[1].state_dict())
        self.clsfc[2].load_state_dict(base_model.classifier[2].state_dict())
        self.clsfc[3].load_state_dict(base_model.classifier[3].state_dict())
        self.clsfc[4].load_state_dict(base_model.classifier[4].state_dict())
        self.clsfc[5].load_state_dict(base_model.classifier[5].state_dict())


# In[ ]:




