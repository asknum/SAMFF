import torch
import torch.nn as nn
from torch.nn import functional as F

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SEFIInspiredSpatialAttention(nn.Module):

    def __init__(self):
        super(SEFIInspiredSpatialAttention, self).__init__()
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, high_res, low_res_up):
        
        avg_high = torch.mean(high_res, dim=1, keepdim=True)
        max_high = torch.max(high_res, dim=1, keepdim=True)[0]
        
        avg_low = torch.mean(low_res_up, dim=1, keepdim=True)
        max_low = torch.max(low_res_up, dim=1, keepdim=True)[0]
        
        P1 = torch.cat([avg_high, max_low], dim=1)
        P2 = torch.cat([avg_low, max_high], dim=1)
        
        sefi_descriptor = torch.cat([P1, P2], dim=1)
        
        spatial_att_map = self.spatial_attention(sefi_descriptor)
        
        return spatial_att_map


class FGFM(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FGFM, self).__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.flow_make = nn.Conv2d(out_channels * 2, 4, 3, padding=1, bias=False)
        self.ca = ChannelAttentionModule(out_channels, ratio=16)
       
        self.spatial_attention = SEFIInspiredSpatialAttention()
        
        conf_mid1 = out_channels // 2
        conf_mid2 = out_channels // 4
        
        self.confidence_net = nn.ModuleDict({
            'conv1': nn.Conv2d(out_channels * 2, conf_mid1, kernel_size=3, padding=1, bias=False),
            'bn1': nn.BatchNorm2d(conf_mid1),
            'relu1': nn.ReLU(inplace=True),
            'conv2': nn.Conv2d(conf_mid1, conf_mid2, kernel_size=3, padding=1, bias=False),
            'bn2': nn.BatchNorm2d(conf_mid2),
            'relu2': nn.ReLU(inplace=True),
            'dropout': nn.Dropout2d(p=0.1),
            'conv3': nn.Conv2d(conf_mid2, 1, kernel_size=1, bias=False),
            'sigmoid': nn.Sigmoid()
        })

    def flow_confidence(self, x):
        
        x = self.confidence_net['conv1'](x)
        x = self.confidence_net['bn1'](x)
        x = self.confidence_net['relu1'](x)
        
        x = self.confidence_net['conv2'](x)
        x = self.confidence_net['bn2'](x)
        x = self.confidence_net['relu2'](x)
        
        x = self.confidence_net['dropout'](x)
        
        x = self.confidence_net['conv3'](x)
        confidence = self.confidence_net['sigmoid'](x)
        
        return confidence

    def flow_warp(self, input, flow, size):

        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output

    def forward(self, lowres_feature, highres_feature):
        h_feature = highres_feature
        h, w = highres_feature.size()[2:]
        size = (h, w)

        l_feature = self.down(lowres_feature)
        l_feature_up = F.interpolate(l_feature, size=size, mode="bilinear", align_corners=True)

        spatial_att_map = self.spatial_attention(h_feature, l_feature_up)
        
        h_feature_att = h_feature * spatial_att_map
        l_feature_up_att = l_feature_up * spatial_att_map
        
        attended_features = torch.cat([l_feature_up_att, h_feature_att], 1)
        
        flow_confidence = self.flow_confidence(attended_features)
        
        flow = self.flow_make(attended_features)
        flow_l, flow_h = flow[:, :2, :, :], flow[:, 2:, :, :]

        l_feature_warp = self.flow_warp(l_feature_up, flow_l, size=size)
        h_feature_warp = self.flow_warp(h_feature, flow_h, size=size)

        l_feature_warp_confident = l_feature_warp * flow_confidence
        h_feature_warp_confident = h_feature_warp * flow_confidence
        
        feature_sum_confident = l_feature_warp_confident + h_feature_warp_confident
        
        flow_gates = self.ca(feature_sum_confident)
        
        fuse_feature = l_feature_warp_confident * flow_gates + h_feature_warp_confident * (1 - flow_gates)
        return fuse_feature


class ConvModule(nn.Module):
    def __init__(self, in_channels):
        super(ConvModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class FGFM_Decoder(nn.Module):

    def __init__(self, in_channels=[32, 64, 160, 256], num_classes=2):
        super(FGFM_Decoder, self).__init__()
        
        self.conv1 = ConvModule(in_channels[0])
        self.conv2 = ConvModule(in_channels[1]) 
        self.conv3 = ConvModule(in_channels[2])
        self.conv4 = ConvModule(in_channels[3])
        
        self.fgfm43 = FGFM(in_channels[3], in_channels[2])
        self.fgfm32 = FGFM(in_channels[2], in_channels[1])
        self.fgfm21 = FGFM(in_channels[1], in_channels[0])
        
        self.fgfm31 = FGFM(in_channels[2], in_channels[0])
        self.fgfm41 = FGFM(in_channels[3], in_channels[0])
        
        self.out = nn.Conv2d(in_channels[0], num_classes, kernel_size=1)
        
    def forward(self, features):
        f1, f2, f3, f4 = features
        
        ff4 = self.conv4(f4)
        
        ff3 = self.fgfm43(ff4, f3)
        ff3 = self.conv3(ff3)
        
        ff2 = self.fgfm32(ff3, f2)
        ff2 = self.conv2(ff2)
        
        ff1 = self.fgfm21(ff2, f1)
        ff1 = self.conv1(ff1)
        
        ff2_up = self.fgfm21(ff2, ff1)
        ff3_up = self.fgfm31(ff3, ff1)
        ff4_up = self.fgfm41(ff4, ff1)
        
        ff = ff1 + ff2_up + ff3_up + ff4_up
        
        ff_out = F.interpolate(self.out(ff), scale_factor=4, mode='bilinear', align_corners=True)
        
        return ff_out, []