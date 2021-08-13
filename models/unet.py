#!/usr/bin/env python3
# encoding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#### Variational_Information_Distillation #####
class VIDLoss(nn.Module):
    """Variational Information Distillation for Knowledge Transfer (CVPR 2019),
    code from author: https://github.com/ssahn0215/variational-information-distillation"""
    def __init__(self,
                 num_input_channels,
                 num_mid_channel,
                 num_target_channels,
                 init_pred_var=5.0,
                 eps=1e-5):
        super(VIDLoss, self).__init__()

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv3d(
                in_channels, out_channels,
                kernel_size=1, padding=0,
                bias=False, stride=stride)

        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_target_channels),
        )
        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(init_pred_var-eps)-1.0) * torch.ones(num_target_channels)
            )
        self.eps = eps

    def forward(self, x):
        
        pred_mean = self.regressor(x)
        log_scale = self.log_scale
        
        return pred_mean, log_scale
    
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups=8):
        super(BasicBlock, self).__init__()
        self.gn1 = nn.GroupNorm(n_groups, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.gn2 = nn.GroupNorm(n_groups, in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x):
        residul = x
        x = self.relu1(self.gn1(x))
        x = self.conv1(x)

        x = self.relu2(self.gn2(x))
        x = self.conv2(x)
        x = x + residul

        return x


class VAEBranch(nn.Module):

    def __init__(self, input_shape, init_channels, out_channels, squeeze_channels=None):
        super(VAEBranch, self).__init__()
        self.input_shape = input_shape

        if squeeze_channels:
            self.squeeze_channels = squeeze_channels
        else:
            self.squeeze_channels = init_channels * 4

        self.hidden_conv = nn.Sequential(nn.GroupNorm(8, init_channels * 8),
                                         nn.ReLU(inplace=True),
                                         nn.Conv3d(init_channels * 8, self.squeeze_channels, (3, 3, 3),
                                                   padding=(1, 1, 1)),
                                         nn.AdaptiveAvgPool3d(1))

        self.mu_fc = nn.Linear(self.squeeze_channels // 2, self.squeeze_channels // 2)
        self.logvar_fc = nn.Linear(self.squeeze_channels // 2, self.squeeze_channels // 2)

        recon_shape = np.prod(self.input_shape) // (16 ** 3)

        self.reconstraction = nn.Sequential(nn.Linear(self.squeeze_channels // 2, init_channels * 8 * recon_shape),
                                            nn.ReLU(inplace=True))

        self.vconv4 = nn.Sequential(nn.Conv3d(init_channels * 8, init_channels * 8, (1, 1, 1)),
                                    nn.Upsample(scale_factor=2))

        self.vconv3 = nn.Sequential(nn.Conv3d(init_channels * 8, init_channels * 4, (3, 3, 3), padding=(1, 1, 1)),
                                    nn.Upsample(scale_factor=2),
                                    BasicBlock(init_channels * 4, init_channels * 4))

        self.vconv2 = nn.Sequential(nn.Conv3d(init_channels * 4, init_channels * 2, (3, 3, 3), padding=(1, 1, 1)),
                                    nn.Upsample(scale_factor=2),
                                    BasicBlock(init_channels * 2, init_channels * 2))

        self.vconv1 = nn.Sequential(nn.Conv3d(init_channels * 2, init_channels, (3, 3, 3), padding=(1, 1, 1)),
                                    nn.Upsample(scale_factor=2),
                                    BasicBlock(init_channels, init_channels))

        self.vconv0 = nn.Conv3d(init_channels, out_channels, (1, 1, 1))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)

    def forward(self, x):
        x = self.hidden_conv(x)
        batch_size = x.size()[0]
        x = x.view((batch_size, -1))
        mu = x[:, :self.squeeze_channels // 2]
        mu = self.mu_fc(mu)
        logvar = x[:, self.squeeze_channels // 2:]
        logvar = self.logvar_fc(logvar)
        z = self.reparameterize(mu, logvar)
        re_x = self.reconstraction(z)
        recon_shape = [batch_size,
                       self.squeeze_channels // 2 * 4,
                       self.input_shape[0] // 16,
                       self.input_shape[1] // 16,
                       self.input_shape[2] // 16]
        #print("input_shape:", self.input_shape) #(160, 192, 128)
        #print("re_x:", re_x.shape) #torch.Size([1, 122880])
        #print("recon_shape:", recon_shape)  #[1, 128, 10, 12, 8]
        re_x = re_x.view(recon_shape)
        x = self.vconv4(re_x)
        x = self.vconv3(x)
        x = self.vconv2(x)
        x = self.vconv1(x)
        vout = self.vconv0(x)

        return vout, mu, logvar


class UNet3D(nn.Module):
    """3d unet
    Ref:
        3D MRI brain tumor segmentation using autoencoder regularization. Andriy Myronenko
    Args:
        input_shape: tuple, (height, width, depth)
    """

    def __init__(self, input_shape, in_channels=4, out_channels=3, init_channels=32, p=0.2):
        super(UNet3D, self).__init__()
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_channels = init_channels
        self.make_encoder()
        self.make_decoder()
        self.dropout = nn.Dropout(p=p)

    def make_encoder(self):
        init_channels = self.init_channels
        self.conv1a = nn.Conv3d(self.in_channels, init_channels, (3, 3, 3), padding=(1, 1, 1))
        self.conv1b = BasicBlock(init_channels, init_channels)  # 32

        self.ds1 = nn.Conv3d(init_channels, init_channels * 2, (3, 3, 3), stride=(2, 2, 2),
                             padding=(1, 1, 1))  # down sampling and add channels

        self.conv2a = BasicBlock(init_channels * 2, init_channels * 2)
        self.conv2b = BasicBlock(init_channels * 2, init_channels * 2)

        self.ds2 = nn.Conv3d(init_channels * 2, init_channels * 4, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.conv3a = BasicBlock(init_channels * 4, init_channels * 4)
        self.conv3b = BasicBlock(init_channels * 4, init_channels * 4)

        self.ds3 = nn.Conv3d(init_channels * 4, init_channels * 8, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.conv4a = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4b = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4c = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4d = BasicBlock(init_channels * 8, init_channels * 8)

    def make_decoder(self):
        init_channels = self.init_channels
        self.up4conva = nn.Conv3d(init_channels * 8, init_channels * 4, (1, 1, 1))
        self.up4 = nn.Upsample(scale_factor=2)  # mode='bilinear'
        self.up4convb = BasicBlock(init_channels * 4, init_channels * 4)

        self.up3conva = nn.Conv3d(init_channels * 4, init_channels * 2, (1, 1, 1))
        self.up3 = nn.Upsample(scale_factor=2)
        self.up3convb = BasicBlock(init_channels * 2, init_channels * 2)

        self.up2conva = nn.Conv3d(init_channels * 2, init_channels, (1, 1, 1))
        self.up2 = nn.Upsample(scale_factor=2)
        self.up2convb = BasicBlock(init_channels, init_channels)

        self.up1conv = nn.Conv3d(init_channels, self.out_channels, (1, 1, 1))

    def forward(self, x):
        c1 = self.conv1a(x)
        c1 = self.conv1b(c1)
        c1d = self.ds1(c1)
        #print("c1d shape:", c1d.shape)
        
        c2 = self.conv2a(c1d)
        c2 = self.conv2b(c2)
        c2d = self.ds2(c2)
        #print("c2d shape:", c2d.shape)
        
        c3 = self.conv3a(c2d)
        c3 = self.conv3b(c3)
        c3d = self.ds3(c3)
        #print("c3d shape:", c3d.shape)

        c4 = self.conv4a(c3d)
        c4 = self.conv4b(c4)
        c4 = self.conv4c(c4)
        c4d = self.conv4d(c4) #[1, 128, 20, 24, 16]
        #print("c4d shape:", c4d.shape)

        df = c4d 
        
        c4d = self.dropout(c4d)

        u4 = self.up4conva(c4d)
        u4 = self.up4(u4)
        u4 = u4 + c3
        u4 = self.up4convb(u4)

        u3 = self.up3conva(u4)
        u3 = self.up3(u3)
        u3 = u3 + c2
        u3 = self.up3convb(u3)

        u2 = self.up2conva(u3)
        u2 = self.up2(u2)
        u2 = u2 + c1
        u2 = self.up2convb(u2)

        uout = self.up1conv(u2)
        uout = F.sigmoid(uout)

        return uout, c4d, df


class UnetVAE3D(nn.Module):

    def __init__(self, input_shape, in_channels=4, out_channels=3, init_channels=32, p=0.2):
        super(UnetVAE3D, self).__init__()
        self.unet = UNet3D(input_shape, in_channels, out_channels, init_channels, p)
        self.vae_branch = VAEBranch(input_shape, init_channels, out_channels=in_channels)
        self.VID_branch = VIDLoss(num_input_channels = init_channels * 8, num_mid_channel = init_channels * 8, num_target_channels = init_channels * 8)
        
    def forward(self, x):
        uout, c4d, df = self.unet(x)
        vout, mu, logvar = self.vae_branch(c4d)
        VID_pred_mean, VID_log_scale = self.VID_branch(df)
        return uout, vout, mu, logvar, df, VID_pred_mean, VID_log_scale
