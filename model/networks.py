import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, channels, kernel_size=8, global_pool=None, convpool=None, compress=False):
        super(Encoder, self).__init__()

        model = []
        acti = nn.LeakyReLU(0.2)

        nr_layer = len(channels) - 2 if compress else len(channels) - 1

        for i in range(nr_layer):
            if convpool is None:
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
                model.append(convpool(kernel_size=2, stride=2))

        self.global_pool = global_pool
        self.compress = compress

        self.model = nn.Sequential(*model)

        if self.compress:
            self.conv1x1 = nn.Conv1d(channels[-2], channels[-1], kernel_size=1)

    def forward(self, x):
        x = self.model(x)
        if self.global_pool is not None:
            ks = x.shape[-1]
            x = self.global_pool(x, ks)
            if self.compress:
                x = self.conv1x1(x)
        return x


class Decoder(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super(Decoder, self).__init__()

        model = []
        pad = (kernel_size - 1) // 2
        acti = nn.LeakyReLU(0.2)

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


class AutoEncoder2x(nn.Module):
    def __init__(self, mot_en_channels, body_en_channels, de_channels, global_pool=None, convpool=None, compress=False):
        super(AutoEncoder2x, self).__init__()
        assert mot_en_channels[0] == de_channels[-1] and \
               mot_en_channels[-1] + body_en_channels[-1] == de_channels[0]

        self.mot_encoder = Encoder(mot_en_channels)
        self.static_encoder = Encoder(body_en_channels, kernel_size=7, global_pool=global_pool, convpool=convpool, compress=compress)
        self.decoder = Decoder(de_channels)

    def cross(self, x1, x2):
        m1 = self.mot_encoder(x1)
        b1 = self.static_encoder(x1[:, :-2, :])
        m2 = self.mot_encoder(x2)
        b2 = self.static_encoder(x2[:, :-2, :])

        out1 = self.decoder(torch.cat([m1, b1.repeat(1, 1, m1.shape[-1])], dim=1))
        out2 = self.decoder(torch.cat([m2, b2.repeat(1, 1, m2.shape[-1])], dim=1))
        out12 = self.decoder(torch.cat([m1, b2.repeat(1, 1, m1.shape[-1])], dim=1))
        out21 = self.decoder(torch.cat([m2, b1.repeat(1, 1, m2.shape[-1])], dim=1))

        return out1, out2, out12, out21

    def transfer(self, x1, x2):
        m1 = self.mot_encoder(x1)
        b2 = self.static_encoder(x2[:, :-2, :]).repeat(1, 1, m1.shape[-1])

        out12 = self.decoder(torch.cat([m1, b2], dim=1))

        return out12

    def cross_with_triplet(self, x1, x2, x12, x21):
        m1 = self.mot_encoder(x1)
        b1 = self.static_encoder(x1[:, :-2, :])
        m2 = self.mot_encoder(x2)
        b2 = self.static_encoder(x2[:, :-2, :])

        out1 = self.decoder(torch.cat([m1, b1.repeat(1, 1, m1.shape[-1])], dim=1))
        out2 = self.decoder(torch.cat([m2, b2.repeat(1, 1, m2.shape[-1])], dim=1))
        out12 = self.decoder(torch.cat([m1, b2.repeat(1, 1, m1.shape[-1])], dim=1))
        out21 = self.decoder(torch.cat([m2, b1.repeat(1, 1, m2.shape[-1])], dim=1))

        m12 = self.mot_encoder(x12)
        b12 = self.static_encoder(x12[:, :-2, :])
        m21 = self.mot_encoder(x21)
        b21 = self.static_encoder(x21[:, :-2, :])

        outputs = [out1, out2, out12, out21]
        motionvecs = [m1.reshape(m1.shape[0], -1),
                      m2.reshape(m2.shape[0], -1),
                      m12.reshape(m12.shape[0], -1),
                      m21.reshape(m21.shape[0], -1)]
        bodyvecs = [b1.reshape(b1.shape[0], -1),
                      b2.reshape(b2.shape[0], -1),
                      b21.reshape(b21.shape[0], -1),
                      b12.reshape(b12.shape[0], -1)]

        return outputs, motionvecs, bodyvecs

    def forward(self, x):
        m = self.mot_encoder(x)
        b = self.static_encoder(x[:, :-2, :])
        b = b.repeat(1, 1, m.shape[-1])
        d = torch.cat([m, b], dim=1)
        d = self.decoder(d)
        return d


class AutoEncoder3x(nn.Module):
    def __init__(self, mot_en_channels, body_en_channels, view_en_channels, de_channels):
        super(AutoEncoder3x, self).__init__()

        assert mot_en_channels[0] == de_channels[-1] and \
               mot_en_channels[-1] + body_en_channels[-1] + view_en_channels[-1] == de_channels[0]

        self.mot_encoder = Encoder(mot_en_channels)
        self.body_encoder = Encoder(body_en_channels, kernel_size=7,
                                    global_pool=F.max_pool1d, convpool=nn.MaxPool1d, compress=True)
        self.view_encoder = Encoder(view_en_channels, kernel_size=7,
                                    global_pool=F.avg_pool1d, convpool=nn.AvgPool1d, compress=True)
        self.decoder = Decoder(de_channels)

    def cross(self, x1, x2):
        m1 = self.mot_encoder(x1)
        b1 = self.body_encoder(x1[:, :-2, :])
        v1 = self.view_encoder(x1[:, :-2, :])
        m2 = self.mot_encoder(x2)
        b2 = self.body_encoder(x2[:, :-2, :])
        v2 = self.view_encoder(x2[:, :-2, :])

        out1 = self.decoder(torch.cat([m1, b1.repeat(1, 1, m1.shape[-1]), v1.repeat(1, 1, m1.shape[-1])], dim=1))
        out2 = self.decoder(torch.cat([m2, b2.repeat(1, 1, m2.shape[-1]), v2.repeat(1, 1, m2.shape[-1])], dim=1))
        out121 = self.decoder(torch.cat([m1, b2.repeat(1, 1, m1.shape[-1]), v1.repeat(1, 1, m1.shape[-1])], dim=1))
        out112 = self.decoder(torch.cat([m1, b1.repeat(1, 1, m1.shape[-1]), v2.repeat(1, 1, m1.shape[-1])], dim=1))
        out122 = self.decoder(torch.cat([m1, b2.repeat(1, 1, m1.shape[-1]), v2.repeat(1, 1, m1.shape[-1])], dim=1))
        out212 = self.decoder(torch.cat([m2, b1.repeat(1, 1, m2.shape[-1]), v2.repeat(1, 1, m2.shape[-1])], dim=1))
        out221 = self.decoder(torch.cat([m2, b2.repeat(1, 1, m2.shape[-1]), v1.repeat(1, 1, m2.shape[-1])], dim=1))
        out211 = self.decoder(torch.cat([m2, b1.repeat(1, 1, m2.shape[-1]), v1.repeat(1, 1, m2.shape[-1])], dim=1))

        return out1, out2, out121, out112, out122, out212, out221, out211

    def cross_with_triplet(self, inputs):
        x1, x2, x121, x112, x122, x212, x221, x211 = inputs
        m1 = self.mot_encoder(x1)
        b1 = self.body_encoder(x1[:, :-2, :])
        v1 = self.view_encoder(x1[:, :-2, :])
        m2 = self.mot_encoder(x2)
        b2 = self.body_encoder(x2[:, :-2, :])
        v2 = self.view_encoder(x2[:, :-2, :])

        out1 = self.decoder(torch.cat([m1, b1.repeat(1, 1, m1.shape[-1]), v1.repeat(1, 1, m1.shape[-1])], dim=1))
        out2 = self.decoder(torch.cat([m2, b2.repeat(1, 1, m2.shape[-1]), v2.repeat(1, 1, m2.shape[-1])], dim=1))
        out121 = self.decoder(torch.cat([m1, b2.repeat(1, 1, m1.shape[-1]), v1.repeat(1, 1, m1.shape[-1])], dim=1))
        out112 = self.decoder(torch.cat([m1, b1.repeat(1, 1, m1.shape[-1]), v2.repeat(1, 1, m1.shape[-1])], dim=1))
        out122 = self.decoder(torch.cat([m1, b2.repeat(1, 1, m1.shape[-1]), v2.repeat(1, 1, m1.shape[-1])], dim=1))
        out212 = self.decoder(torch.cat([m2, b1.repeat(1, 1, m2.shape[-1]), v2.repeat(1, 1, m2.shape[-1])], dim=1))
        out221 = self.decoder(torch.cat([m2, b2.repeat(1, 1, m2.shape[-1]), v1.repeat(1, 1, m2.shape[-1])], dim=1))
        out211 = self.decoder(torch.cat([m2, b1.repeat(1, 1, m2.shape[-1]), v1.repeat(1, 1, m2.shape[-1])], dim=1))

        outputs = [out1, out2, out121, out112, out122, out212, out221, out211]

        m122 = self.mot_encoder(x122)
        m211 = self.mot_encoder(x211)
        b212 = self.body_encoder(x212[:, :-2, :])
        b121 = self.body_encoder(x121[:, :-2, :])
        v221 = self.view_encoder(x221[:, :-2, :])
        v112 = self.view_encoder(x112[:, :-2, :])

        motionvecs = [m1.reshape(m1.shape[0], -1),
                      m2.reshape(m2.shape[0], -1),
                      m122.reshape(m122.shape[0], -1),
                      m211.reshape(m211.shape[0], -1)]
        bodyvecs = [b1.reshape(b1.shape[0], -1),
                    b2.reshape(b2.shape[0], -1),
                    b212.reshape(b212.shape[0], -1),
                    b121.reshape(b121.shape[0], -1)]
        viewvecs = [v1.reshape(v1.shape[0], -1),
                    v2.reshape(v2.shape[0], -1),
                    v221.reshape(v221.shape[0], -1),
                    v112.reshape(v112.shape[0], -1)]

        return outputs, motionvecs, bodyvecs, viewvecs


    def transfer_body(self, x1, x2):
        m1 = self.mot_encoder(x1)
        b2 = self.body_encoder(x2[:, :-2, :]).repeat(1, 1, m1.shape[-1])
        v1 = self.view_encoder(x1[:, :-2, :]).repeat(1, 1, m1.shape[-1])

        out12 = self.decoder(torch.cat([m1, b2, v1], dim=1))

        return out12

    def transfer_view(self, x1, x2):
        m1 = self.mot_encoder(x1)
        b1 = self.body_encoder(x1[:, :-2, :]).repeat(1, 1, m1.shape[-1])
        v2 = self.view_encoder(x2[:, :-2, :]).repeat(1, 1, m1.shape[-1])

        out12 = self.decoder(torch.cat([m1, b1, v2], dim=1))

        return out12

    def transfer_both(self, x1, x2):
        m1 = self.mot_encoder(x1)
        b2 = self.body_encoder(x2[:, :-2, :]).repeat(1, 1, m1.shape[-1])
        v2 = self.view_encoder(x2[:, :-2, :]).repeat(1, 1, m1.shape[-1])

        out12 = self.decoder(torch.cat([m1, b2, v2], dim=1))

        return out12

    def transfer_three(self, x1, x2, x3):
        m1 = self.mot_encoder(x1)
        b2 = self.body_encoder(x2[:, :-2, :]).repeat(1, 1, m1.shape[-1])
        v3 = self.view_encoder(x3[:, :-2, :]).repeat(1, 1, m1.shape[-1])

        out = self.decoder(torch.cat([m1, b2, v3], dim=1))

        return out

    def forward(self, x):
        m = self.mot_encoder(x)
        b = self.body_encoder(x[:, :-2, :]).repeat(1, 1, m.shape[-1])
        v = self.view_encoder(x[:, :-2, :]).repeat(1, 1, m.shape[-1])
        d = torch.cat([m, b, v], dim=1)
        d = self.decoder(d)
        return d
