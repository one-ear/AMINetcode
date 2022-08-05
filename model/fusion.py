import torch
import torch.nn as nn
import torch.nn.functional as F
from model.HolisticAttention import HA
from network.deform_conv.deform_conv import DeformConv2D
###############3
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_r, x_d):
        avg_out_r = torch.mean(x_r, dim=1, keepdim=True)
        max_out_r = torch.max(x_r, dim=1, keepdim=True)
        avg_out_d = torch.mean(x_d, dim=1, keepdim=True)
        max_out_d = torch.max(x_d, dim=1, keepdim=True)
        x_r = torch.cat([avg_out_r, max_out_r], dim=1)
        x_r = self.conv1(x_r)
        x_d = torch.cat([avg_out_d, max_out_d], dim=1)
        x_d = self.conv1(x_d)
        return self.sigmoid(x_r), self.sigmoid(x_d)
    ########

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RFB(nn.Module):
    # RFB-like multi-scale module
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.Deform0_0 = DeformConv2D(out_channel, out_channel, 1, padding=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.Deform1_0 = DeformConv2D(out_channel, out_channel, 1, padding=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.Deform2_0 = DeformConv2D(out_channel, out_channel, 1, padding=1)
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.Deform3_0 = DeformConv2D(out_channel, out_channel, 1, padding=1)
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.Deform0_0(self.branch0(x))
        x1 = self.Deform1_0(self.branch1(x))
        x2 = self.Deform2_0(self.branch2(x))
        x3 = self.Deform3_0(self.branch3(x))

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation model, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class BMF(SpatialAttention,nn.Module):
    # Boundary-aware Multimodal Fusion Strategy
    def __init__(self):
        super(BMF, self).__init__()

        self.squeeze_rgb = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_rgb = nn.Sequential(
            nn.Conv2d(32, 32, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid())
        self.spatial_attention_rgb = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Sigmoid())


        # self.squeeze_depth = nn.AdaptiveAvgPool2d(1)
        self.squeeze_depth1 = nn.AdaptiveAvgPool2d(1)
        # self.squeeze_depth2 = nn.AdaptiveMaxPool2d(1)
        self.channel_attention_depth = nn.Sequential(
            nn.Conv2d(32, 32, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid())
        self.spatial_attention_depth = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Sigmoid())

        self.cross_conv = nn.Conv2d(32*2, 32, 1, padding=0)
        self.Deform_c = DeformConv2D(32, 32, 1, padding=1)

        self.B_conv_3x3 = nn.Conv2d(32, 32, 3, padding=1)
        self.Deform_1 = DeformConv2D(32, 32, 1, padding=1)
        self.B_conv1_Sal = nn.Conv2d(32, 1, 1)
        self.sig = nn.Sigmoid()
        self.B_conv1_Edge= nn.Conv2d(32, 1, 1)

        self.fusion_layer = nn.Conv2d(32*2 + 2, 32, 1, padding=0)
        self.Deform_2 = DeformConv2D(32, 32, 1, padding=1)
        self.conv1_sal = nn.Conv2d(32, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                # m.bias.data.fill_(0)

    def forward(self, x3_r,x3_d):

        # # RGB channel attention
        # fea_ca = self.channel_attention_rgb(self.squeeze_rgb(x3_r))
        # fea_3_o = x3_r * fea_ca.expand_as(x3_r)
        # # Depth channel attention
        # fea_d_ca = self.channel_attention_depth(self.squeeze_depth(x3_d))
        # fea_3d_o = x3_d * fea_d_ca.expand_as(x3_d)
        # # RGB & Depth multi-modal feature
        # CR_fea3 = torch.cat([fea_3_o,fea_3d_o],dim=1)
        # CR_fea3 = self.cross_conv(CR_fea3)

        #####
        SCA_ca = self.channel_attention_rgb(self.squeeze_rgb(x3_r))  # (???)->(N C 1 1)->(N 32 1 1)
        # SCA_ca2 = self.channel_attention_rgb(self.squeeze_rgb2(x3_r))  # (???)->(N C 1 1)->(N 32 1 1)
        # SCA_ca=nn.sigmoid(SCA_ca1+SCA_ca2)
        # avg_out=torch.mean(SCA_ca,dim=1,keepdim=True)
        # max_out,_=torch.max(SCA_ca,dim=1,keepdim=True)
        # SCA_ca=torch.cat([avg_out,max_out],dim=1)
        # SCA_ca=self.spatial_attention_rgb(SCA_ca)
        SCA_3_o1 = x3_r * SCA_ca.expand_as(x3_r)  # 扩充大小为X3_r大小，然后和X3相乘
        avg_out = torch.mean(SCA_3_o1, dim=1, keepdim=True)
        max_out, _ = torch.max(SCA_3_o1, dim=1, keepdim=True)
        SCA_3_o2 = torch.cat([avg_out, max_out], dim=1)
        SCA_3_o2 = self.spatial_attention_rgb(SCA_3_o2)
        SCA_3_o = SCA_3_o1 * SCA_3_o2

        SCA_d_ca = self.channel_attention_depth(self.squeeze_depth1(x3_d))  # (???)->(N C 1 1)->(N 32 1 1)
        # SCA_d_ca2 = self.channel_attention_depth(self.squeeze_depth2(x3_d))
        # SCA_d_ca=nn.sigmoid(SCA_d_ca1+SCA_d_ca2)
        # d_avg_out = torch.mean(SCA_d_ca, dim=1, keepdim=True)
        # d_max_out, _ = torch.max(SCA_d_ca, dim=1, keepdim=True)
        # SCA_d_ca = torch.cat([d_avg_out, d_max_out], dim=1)
        # SCA_d_ca = self.spatial_attention_depth(SCA_d_ca)
        SCA_3d_o1 = x3_d * SCA_d_ca.expand_as(x3_d)  # 扩充大小为X3_d大小，然后和X3相乘
        d_avg_out = torch.mean(SCA_3d_o1, dim=1, keepdim=True)
        d_max_out, _ = torch.max(SCA_3d_o1, dim=1, keepdim=True)
        SCA_3d_o2 = torch.cat([d_avg_out, d_max_out], dim=1)
        SCA_3d_o2 = self.spatial_attention_depth(SCA_3d_o2)
        SCA_3d_o = SCA_3d_o1 * SCA_3d_o2

        Co_ca3 = torch.softmax(SCA_ca + SCA_d_ca, dim=1)  # N 32 1 1,softmax 后归一化到0-1

        SCA_3_co = x3_r * Co_ca3.expand_as(x3_r)  # 扩充大小为X3_r大小，然后和X3相乘
        SCA_3d_co = x3_d * Co_ca3.expand_as(x3_d)  # 扩充大小为X3_d大小，然后和X3相乘

        CR_fea3_rgb = SCA_3_o + SCA_3_co  # softmax 之后经过扩充 与没有softmax之前相加
        CR_fea3_d = SCA_3d_o + SCA_3d_co

        CR_fea3 = torch.cat([CR_fea3_rgb, CR_fea3_d], dim=1)  # (N 64 H(X3_r) W(X3_r))
        CR_fea3 = self.cross_conv(CR_fea3)  # (N 32  H(X3_r) W(X3_r))
        CR_fea3 = self.Deform_c(CR_fea3)


        #####



        '''Boundary-aware Strategy'''
        Content_fea3 = self.B_conv_3x3(CR_fea3)
        Content_fea3 = self.Deform_1(Content_fea3)
        Sal_main_pred= self.B_conv1_Sal(Content_fea3)

        Edge_fea3= CR_fea3 * (1 - self.sig(Sal_main_pred))

        # import matplotlib.pyplot as plt
        # plt.figure()
        # # plt.imshow(Edge_fea3.detach().numpy()[0][1])
        # for i in range(1, 32+1):
        #     plt.subplot(2*4, 4, i)
        #     plt.imshow(Content_fea3.detach().numpy()[0][i-1],cmap='gray')
        #     plt.axis('off')
        #     plt.subplots_adjust(wspace=0.02,hspace=0.02)
        # plt.show()
        #
        # import matplotlib.pyplot as plt
        # plt.figure()
        # # plt.imshow(Edge_fea3.detach().numpy()[0][1])
        # for i in range(1, 32 + 1):
        #     plt.subplot(2 * 4, 4, i)
        #     plt.imshow(Edge_fea3.detach().numpy()[0][i - 1], cmap='gray')
        #     plt.axis('off')
        #     plt.subplots_adjust(wspace=0.02, hspace=0.02)
        # plt.show()


        Edge_pred = self.B_conv1_Edge(Edge_fea3)


        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(torch.sigmoid(Edge_pred).detach().numpy()[0][0],cmap='gray')
        # plt.show()
        #
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(torch.sigmoid(Sal_main_pred).detach().numpy()[0][0], cmap='gray')
        # plt.show()


        Multimodal = torch.cat([Content_fea3,Sal_main_pred,
                                    Edge_fea3,Edge_pred],dim=1)
        Multimodal_fea = self.fusion_layer(Multimodal)
        Multimodal_fea = self.Deform_2(Multimodal_fea)
        med_sal = self.conv1_sal(Multimodal_fea)

        return Multimodal_fea, Sal_main_pred, Edge_pred, med_sal



class fusion(nn.Module):
    def __init__(self, in_channel=32, out_channel=32):
        super(fusion, self).__init__()

        channel = in_channel
        self.rfb3_1 = RFB(channel, channel)
        self.rfb4_1 = RFB(channel, channel)
        self.rfb5_1 = RFB(channel, channel)
        self.agg1 = aggregation(channel)

        self.rfb3_2 = RFB(channel, channel)
        self.rfb4_2 = RFB(channel, channel)
        self.rfb5_2 = RFB(channel, channel)
        self.agg2 = aggregation(channel)
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        self.HA = HA()

        self.BMF3 = BMF()
        self.BMF4 = BMF()
        self.BMF5 = BMF()

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel,kernel_size=3,padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )

        self._init_weight()



    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x3_r,x4_r,x5_r,x3_d,x4_d,x5_d):

        # Cross Reference Module
        x3, sal_main3, edge_main3, med_sal3 = self.BMF3(x3_r, x3_d)          # b_size,32, 1/8.  1/8    (44, 44)
        x4, sal_main4, edge_main4, med_sal4 = self.BMF4(x4_r, x4_d)          # b_size,32, 1/16. 1/16   (22, 22)
        x5, sal_main5, edge_main5, med_sal5 = self.BMF5(x5_r, x5_d)          # b_size,32, 1/32. 1/32   (11, 11)


        # Decoder
        x3_1 = self.rfb3_1(x3)
        x4_1 = self.rfb4_1(x4)
        x5_1 = self.rfb5_1(x5)
        attention_map = self.agg1(x5_1, x4_1, x3_1)
        x3_2 = self.HA(attention_map.sigmoid(), x3)
        x4_2 = self.conv4(x3_2)
        x5_2 = self.conv5(x4_2)
        x3_2 = self.rfb3_2(x3_2)
        x4_2 = self.rfb4_2(x4_2)
        x5_2 = self.rfb5_2(x5_2)
        detection_map = self.agg2(x5_2, x4_2, x3_2)

        return self.upsample(attention_map), self.upsample(detection_map), \
               [self.up8(sal_main3),self.up8(edge_main3)], \
               [self.up16(sal_main4),self.up16(edge_main4)], \
               [self.up32(sal_main5), self.up32(edge_main5)], \
               [self.up8(med_sal3), self.up16(med_sal4), self.up32(med_sal5)]

