import torch.nn as nn
import torch

from block import ConvBlock


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN.
    W = (W - F + 2P) /S + 1"""

    def __init__(self, num_attr=13, spec_norm=True, LR=0.02):
        super(Discriminator, self).__init__()
        self.main = list()
        self.main.append(ConvBlock(3, 16, spec_norm, stride=2, LR=LR)) # 256 -> 128
        self.main.append(ConvBlock(16, 32, spec_norm, stride=2, LR=LR)) # 128 -> 64
        self.main.append(ConvBlock(32, 64, spec_norm, stride=2, LR=LR)) # 64 -> 32
        self.main.append(ConvBlock(64, 128, spec_norm, stride=2, LR=LR)) # 32 -> 16
        self.last_conv = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.main = nn.Sequential(*self.main)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc_att = nn.Linear(128 * 16 * 16, num_attr)

    def forward(self, x):
        x = self.main(x)
        dis = self.last_conv(x)
        x = self.dropout(x.view(-1, 128 * 16 * 16))
        return dis, self.fc_att(x)


class STU(nn.Module):

    def __init__(self, feature_channel, state_channel, num_attr=13):
        super(STU, self).__init__()
        self.num_attr = num_attr
        self.feature_channel = feature_channel
        self.state_channel = state_channel

        self.trans_posed = list()
        self.trans_posed.append(nn.ConvTranspose2d(state_channel + self.num_attr, feature_channel, kernel_size=4, stride=2, padding=1))
        self.trans_posed.append(nn.Conv2d(feature_channel, feature_channel, kernel_size=3, stride=1, padding=1))
        self.trans_posed.append(nn.BatchNorm2d(feature_channel, track_running_stats=True, affine=True))
        self.trans_posed = nn.Sequential(*self.trans_posed)

        self.transfrom = list()
        self.transfrom.append(nn.Conv2d(feature_channel * 2, feature_channel, kernel_size=3, stride=1, padding=1))
        self.transfrom.append(nn.BatchNorm2d(feature_channel, track_running_stats=True, affine=True))
        self.transfrom.append(nn.Tanh())
        self.transfrom = nn.Sequential(*self.transfrom)

        self.reset_gate = list()
        self.reset_gate.append(nn.Conv2d(feature_channel * 2, feature_channel, kernel_size=3, stride=1, padding=1))
        self.reset_gate.append(nn.BatchNorm2d(feature_channel, track_running_stats=True, affine=True))
        self.reset_gate.append(nn.Sigmoid())
        self.reset_gate = nn.Sequential(*self.reset_gate)

        self.update_gate = list()
        self.update_gate.append(nn.Conv2d(feature_channel * 2, feature_channel, kernel_size=3, stride=1, padding=1))
        self.update_gate.append(nn.BatchNorm2d(feature_channel, track_running_stats=True, affine=True))
        self.update_gate.append(nn.Sigmoid())
        self.update_gate = nn.Sequential(*self.update_gate)


    def forward(self, feature, state, attribute):
        batch, _, h, w = state.size()
        attr_tensor = attribute.view(batch, self.num_attr, 1, 1).expand((batch, self.num_attr, h, w))
        state_hat = self.trans_posed(torch.cat([state, attr_tensor], dim=1))
        r = self.reset_gate(torch.cat([feature, state_hat], dim=1))
        z = self.update_gate(torch.cat([feature, state_hat], dim=1))
        new_state = r * state_hat
        transformed_feature = self.transfrom(torch.cat([feature, new_state], dim=1))
        output = (1 - z) * state_hat + z * transformed_feature
        return output, new_state


class Encoder(nn.Module):
    """Discriminator network with PatchGAN.
    W = (W - F + 2P) /S + 1"""

    def __init__(self, in_channel=3, spec_norm=False, LR=0.2):
        super(Encoder, self).__init__()

        self.layer1 = ConvBlock(in_channel, 16, stride=2, spec_norm=spec_norm, LR=LR) # 256 -> 128
        self.layer2 = ConvBlock(16, 32, stride=2, spec_norm=spec_norm, LR=LR) # 128 -> 64
        self.layer3 = ConvBlock(32, 64, stride=2, spec_norm=spec_norm, LR=LR) # 64-> 32
        self.layer4 = ConvBlock(64, 128, stride=2, spec_norm=spec_norm, LR=LR) # 32 -> 16
        self.layer5 = ConvBlock(128, 256, stride=2, spec_norm=spec_norm, LR=LR) # 16 -> 8

    def forward(self, image):
        feature1 = self.layer1(image)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        feature4 = self.layer4(feature3)
        feature5 = self.layer5(feature4)
        return [feature1, feature2, feature3, feature4, feature5]


class Generator(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, LR=0.02, spec_norm=False, num_attr=10):
        super(Generator, self).__init__()
        self.num_attr = num_attr
        self.encoder = Encoder(in_channel, spec_norm, LR)

        self.layer5 = ConvBlock(256 + num_attr, 128, spec_norm=spec_norm, LR=LR)  # 8 - > 16
        self.layer4 = ConvBlock(128 + 128, 128, spec_norm=spec_norm, LR=LR)  # 16 - > 32
        self.layer3 = ConvBlock(128 + 64, 64, spec_norm=spec_norm, LR=LR)  # 32 -> 64
        self.layer2 = ConvBlock(64 + 32, 64, spec_norm=spec_norm, LR=LR)  # 64 -> 128
        self.layer1 = ConvBlock(64 + 16, 32, spec_norm=spec_norm, LR=LR)  # 128 -> 256

        self.up5 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.up4 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)

        self.last_conv = nn.Conv2d(32, out_channel, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        self.STU_4 = STU(feature_channel=128, state_channel=256, num_attr=num_attr)
        self.STU_3 = STU(feature_channel=64, state_channel=128,  num_attr=num_attr)
        self.STU_2 = STU(feature_channel=32, state_channel=64,  num_attr=num_attr)
        self.STU_1 = STU(feature_channel=16, state_channel=32,  num_attr=num_attr)

    def forward(self, image, attr_diff):
        feature1, feature2, feature3, feature4, feature5 = self.encoder(image)
        # 128, 64, 32, 16, 8

        """
        self.layer1 = ConvBlock(in_channel, 16, stride=2, spec_norm=spec_norm, LR=LR) # 256 -> 128
        self.layer2 = ConvBlock(16, 32, stride=2, spec_norm=spec_norm, LR=LR) # 128 -> 64
        self.layer3 = ConvBlock(32, 64, stride=2, spec_norm=spec_norm, LR=LR) # 64-> 32
        self.layer4 = ConvBlock(64, 128, stride=2, spec_norm=spec_norm, LR=LR) # 32 -> 16
        self.layer5 = ConvBlock(128, 256, stride=2, spec_norm=spec_norm, LR=LR) # 16 -> 8
        """

        trans_featrue_4, new_state_4 = self.STU_4(feature=feature4, state=feature5, attribute=attr_diff) # 16
        trans_featrue_3, new_state_3 = self.STU_3(feature=feature3, state=new_state_4, attribute=attr_diff) # 32
        trans_featrue_2, new_state_2 = self.STU_2(feature=feature2, state=new_state_3, attribute=attr_diff) # 64
        trans_featrue_1, new_state_1 = self.STU_1(feature=feature1, state=new_state_2, attribute=attr_diff) # 128

        batch, _, h, w = feature5.size()
        attr_5 = attr_diff.view(batch, self.num_attr, 1, 1).expand((batch, self.num_attr, h, w))

        dec_layer5 = self.up5(self.layer5(torch.cat([feature5, attr_5], dim=1))) # 8 -> 16
        dec_layer4 = self.up4(self.layer4(torch.cat([dec_layer5, trans_featrue_4], dim=1))) # 16 -> 32
        dec_layer3 = self.up3(self.layer3(torch.cat([dec_layer4, trans_featrue_3], dim=1))) # 32 -> 64
        dec_layer2 = self.up2(self.layer2(torch.cat([dec_layer3, trans_featrue_2], dim=1))) # 64 -> 128
        dec_layer1 = self.up1(self.layer1(torch.cat([dec_layer2, trans_featrue_1], dim=1))) # 128 -> 256

        image = self.tanh(self.last_conv(dec_layer1))
        return image

