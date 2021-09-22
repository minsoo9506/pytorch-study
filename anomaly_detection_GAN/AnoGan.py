import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=20, img_size=64):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(
                z_dim,
                img_size * 8,
                kernel_size=4,
                stride=2,
            ),
            nn.BatchNorm2d(img_size * 8),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(
                img_size * 8,
                img_size * 4,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(img_size * 4),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(
                img_size * 4,
                img_size * 2,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(img_size * 2),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(
                img_size * 2,
                img_size,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(img_size),
            nn.ReLU()
        )
        self.last = nn.Sequential(
            nn.ConvTranspose2d(
                img_size,
                1,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Tanh()
        )
    
    def forward(self, z):
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, z_dim=20, img_size=64):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                1, # 흑백이라 channel이 1
                img_size,
                kernel_size=4,
                stride=2,
                padding=1),
            nn.LeakyReLU(0.1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                img_size,
                img_size * 2,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.LeakyReLU(0.1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(
                img_size * 2,
                img_size * 4,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.LeakyReLU(0.1)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(
                img_size * 4,
                img_size * 8,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.LeakyReLU(0.1)
        )
        self.last = nn.Conv2d(
            img_size * 8,
            1,
            kernel_size=4,
            stride=1
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        feature = out
        feature = feature.view(feature.size()[0], -1) # 2차원

        out = self.last(out)
        return out, feature