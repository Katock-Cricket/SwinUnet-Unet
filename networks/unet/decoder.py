import torch
from torch import nn


class DecoderLayer(nn.Module):
    def __init__(self, in_size, out_size):
        super(DecoderLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()

        # upSampling
        # 64,64,512
        self.up_concat4 = DecoderLayer(1024, 512)
        # 128,128,256
        self.up_concat3 = DecoderLayer(768, 256)
        # 256,256,128
        self.up_concat2 = DecoderLayer(384, 128)
        # 512,512,64
        self.up_concat1 = DecoderLayer(192, 64)

        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, feat1, feat2, feat3, feat4, feat5):
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)
        final = self.final(up1)

        return final
