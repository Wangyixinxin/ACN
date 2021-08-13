from torch import nn


def get_fc_discriminator(num_classes, ndf=64):
    return nn.Sequential(
        nn.Conv3d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv3d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv3d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv3d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv3d(ndf * 8, 1, kernel_size=4, stride=2, padding=1),
    )

def get_df_discriminator(num_classes, ndf=64):
    return nn.Sequential(
        nn.Conv3d(num_classes, ndf * 4, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv3d(ndf * 4, ndf * 8, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv3d(ndf * 8, 1, kernel_size=3, stride=2, padding=1),
    )