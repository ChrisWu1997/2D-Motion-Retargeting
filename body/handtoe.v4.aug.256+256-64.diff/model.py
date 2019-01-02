import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, channels, kernel_size=8, activation='lrelu', global_maxpool=False, convpool=False):
        super(Encoder, self).__init__()

        model = []
        if activation == 'lrelu':
            acti = nn.LeakyReLU(0.2)
        elif activation == 'relu':
            acti = nn.ReLU()
        elif activation == 'tanh':
            acti = nn.Tanh()
        else:
            raise NameError

        nr_layer = len(channels) - 2 if global_maxpool else len(channels) - 1

        for i in range(nr_layer):
            if not convpool:
                pad = (kernel_size - 2) // 2
                model.append(nn.ReflectionPad1d(pad))
                model.append(nn.Conv1d(channels[i], channels[i+1],
                                   kernel_size=kernel_size, stride=2))
                model.append(acti)
            else:
                pad = (kernel_size - 1) // 2
                model.append(nn.ReflectionPad1d(pad))
                model.append(nn.Conv1d(channels[i], channels[i+1],
                                       kernel_size=kernel_size, stride=1))
                model.append(acti)
                model.append(nn.MaxPool1d(kernel_size=2, stride=2))
            #odel.append(nn.Dropout(p=0.2))

        self.global_maxpool = global_maxpool

        self.model = nn.Sequential(*model)

        if self.global_maxpool:
            self.conv1x1 = nn.Conv1d(channels[-2], channels[-1], kernel_size=1)

    def forward(self, x):
        x = self.model(x)
        if self.global_maxpool:
            ks = x.shape[-1]
            x = F.max_pool1d(x, ks)
            x = self.conv1x1(x)
        return x


class Decoder(nn.Module):
    def __init__(self, channels, kernel_size=7, activation='lrelu'):
        super(Decoder, self).__init__()

        model = []
        pad = (kernel_size - 1) // 2
        if activation == 'lrelu':
            acti = nn.LeakyReLU(0.2)
        elif activation == 'relu':
            acti = nn.ReLU()
        elif activation == 'tanh':
            acti = nn.Tanh()
        else:
            raise NameError

        for i in range(len(channels) - 1):
            model.append(nn.Upsample(scale_factor=2, mode='nearest'))
            model.append(nn.ReflectionPad1d(pad))
            model.append(nn.Conv1d(channels[i], channels[i + 1],
                                            kernel_size=kernel_size, stride=1))
            if i == 0 or i == 1:
                model.append(nn.Dropout(p=0.2))
            if not i == len(channels) - 2:
                model.append(acti)          # whether to add tanh a last?
                #model.append(nn.Dropout(p=0.2))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class AutoEncoder(nn.Module):
    def __init__(self, mot_en_channels, body_en_channels, de_channels, input_size):
        super(AutoEncoder, self).__init__()
        assert mot_en_channels[0] == de_channels[-1] and \
               mot_en_channels[-1] + body_en_channels[-1] == de_channels[0]

        self.mot_encoder = Encoder(mot_en_channels)
        self.body_encoder = Encoder(body_en_channels, kernel_size=7, global_maxpool=True, convpool=True)
        self.decoder = Decoder(de_channels)

    def cross(self, x1, x2):
        m1 = self.mot_encoder(x1)
        b1 = self.body_encoder(x1[:, :-2, :])
        m2 = self.mot_encoder(x2)
        b2 = self.body_encoder(x2[:, :-2, :])

        out1 = self.decoder(torch.cat([m1, b1.repeat(1, 1, m1.shape[-1])], dim=1))
        out2 = self.decoder(torch.cat([m2, b2.repeat(1, 1, m2.shape[-1])], dim=1))
        out12 = self.decoder(torch.cat([m1, b2.repeat(1, 1, m1.shape[-1])], dim=1))
        out21 = self.decoder(torch.cat([m2, b1.repeat(1, 1, m2.shape[-1])], dim=1))

        return out1, out2, out12, out21

    def transfer(self, x1, x2):
        m1 = self.mot_encoder(x1)
        b2 = self.body_encoder(x2[:, :-2, :]).repeat(1, 1, m1.shape[-1])

        out12 = self.decoder(torch.cat([m1, b2], dim=1))

        return out12

    def forward(self, x):
        m = self.mot_encoder(x)
        b = self.body_encoder(x[:, :-2, :])
        b = b.repeat(1, 1, m.shape[-1])
        d = torch.cat([m, b], dim=1)
        d = self.decoder(d)
        return d


def test():
    net = AutoEncoder([32, 48, 96, 128], [30, 48, 96, 128, 2], [130, 128, 64, 32], 64)
    print(net)
    x = torch.ones((8, 32, 64))
    x = net(x)
    print(x.shape)

    y = torch.zeros((8, 32, 64))
    o1, o2, o12, o21 = net.cross(x, y)
    print(o1.shape)
    print(o2.shape)
    print(o12.shape)
    print(o21.shape)


if __name__ == '__main__':
    test()
