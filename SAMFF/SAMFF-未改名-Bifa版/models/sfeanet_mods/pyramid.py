import torch
import torch.nn as nn
from torch.nn import functional as F
from models.sfeanet_mods.acfm_module import CAFMAttention, ACFMBlock

class Pyramid_Extraction(nn.Module):
    def __init__(self, channel, rate=1, bn_mom=0.1):
        super(Pyramid_Extraction, self).__init__()
        self.channel = channel

        r1, r2, r3, r4 = rate, 2*rate, 4*rate, 8*rate

        
        self.branch1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=r1, dilation=r1, groups=channel, bias=True),
            nn.BatchNorm2d(channel, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True),
            nn.BatchNorm2d(channel, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=r2, dilation=r2, groups=channel, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=r3, dilation=r3, groups=channel, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=r4, dilation=r4, groups=channel, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.branch5_conv = nn.Conv2d(channel, channel, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(channel, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d((channel) * 5, (channel) * 5, kernel_size=(1, 1), stride=(1, 1), padding=0,
                      groups=(channel) * 5, bias=False),
            nn.BatchNorm2d((channel) * 5, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d((channel) * 5, channel, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()

        conv1_1 = self.branch1(x)
        conv3_1 = self.branch2(x)
        conv3_2 = self.branch3(x)
        conv3_3 = self.branch4(x)

        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        feature_cat = torch.cat([conv1_1, conv3_1, conv3_2, conv3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class Pyramid_Merge_Multi(nn.Module):
    def __init__(self, in_channels):
        super(Pyramid_Merge_Multi, self).__init__()
        
        self.pyramid_extractions = nn.ModuleList([
            Pyramid_Extraction(channel) for channel in in_channels
        ])
        
        self.acfm_modules = nn.ModuleList([
            CAFMAttention(dim=channel, num_heads=2, bias=False)
            for channel in in_channels
        ])
        
        self.fusions = nn.ModuleList([
            nn.Conv2d(in_channels[i+1], in_channels[i], kernel_size=1, stride=1) 
            for i in range(len(in_channels)-1)
        ])
        
        self.acfm_fusions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels[i] * 2, in_channels[i], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(in_channels[i]),
                nn.ReLU(inplace=True)
            )
            for i in range(len(in_channels)-1)
        ])
        
    def forward(self, diff_features):
        
        context_features = []
        for i, diff in enumerate(diff_features):
            context_feat = self.pyramid_extractions[i](diff)
            context_features.append(context_feat + diff)
            
        enhanced_features = context_features.copy()
        for i in range(len(diff_features) - 2, -1, -1):
            high_feat = self.fusions[i](enhanced_features[i+1])
            h, w = enhanced_features[i].shape[2:]
            high_feat = F.interpolate(high_feat, size=(h, w), mode='bilinear', align_corners=True)
            
            acfm_low = self.acfm_modules[i](enhanced_features[i])
            acfm_high = self.acfm_modules[i](high_feat)
            
            fused_feat = torch.cat([acfm_low, acfm_high], dim=1)
            enhanced_features[i] = self.acfm_fusions[i](fused_feat)
        
        final_features = []
        for i, (enhanced, original) in enumerate(zip(enhanced_features, diff_features)):
            final_features.append(enhanced + original)
            
        return final_features