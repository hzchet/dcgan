import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        fmap_dim: int,
        out_channels: int,
        **kwargs
    ):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, fmap_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(fmap_dim * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(fmap_dim * 8, fmap_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmap_dim * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( fmap_dim * 4, fmap_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmap_dim * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( fmap_dim * 2, fmap_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmap_dim),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(fmap_dim, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        fmap_dim: int,
        **kwargs
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, fmap_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(fmap_dim, fmap_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmap_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(fmap_dim * 2, fmap_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmap_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(fmap_dim * 4, fmap_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmap_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(fmap_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, x):
        return self.net(x)
