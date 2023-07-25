import torch.nn as nn

from networks.unet.encoder import Encoder
from networks.unet.decoder import Decoder


class Unet(nn.Module):
    def __init__(self, num_classes=9):
        super(Unet, self).__init__()
        self.encoder = Encoder(in_channels=1)
        self.decoder = Decoder(num_classes)

    def forward(self, inputs):
        [feat1, feat2, feat3, feat4, feat5] = self.encoder(inputs)
        output = self.decoder(feat1, feat2, feat3, feat4, feat5)

        return output
