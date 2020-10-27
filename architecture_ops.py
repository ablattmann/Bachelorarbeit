import torch
import torch.nn as nn
import torch.nn.functional as F


Pool = nn.MaxPool2d


def softmax(logit_map):
    eps = 1e-12
    exp = torch.exp(logit_map - torch.max(logit_map.view(logit_map.size(0), logit_map.size(1), -1), dim=2)[0].unsqueeze(-1).unsqueeze(-1))
    norm = torch.sum(exp, dim=[1, 2], keepdim=True) + eps
    softmax = exp / norm
    return softmax


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class Hourglass(nn.Module):
    def __init__(self, n, f):
        super(Hourglass, self).__init__()
        self.up1 = Residual(f, f)
        # Lower branch
        self.pool1 = Pool(2, 2)
        self.low1 = Residual(f, f)
        self.n = n
        # Recursive hourglass
        if self.n > 0:
            self.low2 = Hourglass(n - 1, f)
        else:
            self.low2 = Residual(f, f)
        self.low3 = Residual(f, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


class E(nn.Module):
    def __init__(self, depth, n_feature, residual_dim, sigma=True):
        super(E, self).__init__()
        self.sigma = sigma
        self.hg = Hourglass(depth, residual_dim)  # depth 4 has bottleneck of 4x4
        self.dropout = nn.Dropout()
        self.out = Conv(residual_dim, residual_dim, kernel_size=1, stride=1, bn=True, relu=True)
        self.feature = Conv(residual_dim, n_feature, kernel_size=1, stride=1, bn=False, relu=False)
        if self.sigma:
            self.preprocess_1 = Conv(3, 64, kernel_size=6, stride=2, bn=True, relu=True)  # transform to 64 x 64 for sigma
            self.preprocess_2 = Residual(64, residual_dim)
            self.map_transform = Conv(n_feature, residual_dim, 1, 1)    # channels for addition must be increased
        #if not self.sigma:
        #    self.preprocess_3 = Conv(2 * residual_dim, residual_dim, 1, 1, bn=True, relu=True)

    def forward(self, x):
        if self.sigma:
            x = self.preprocess_1(x)
            x = self.preprocess_2(x)
        #else:
        #    x = self.preprocess_3(x) # Try for concatenate instead of sum
        out = self.hg(x)
        out = self.dropout(out)
        out = self.out(out)
        # Get Normalized Feature Maps for E_sigma
        map = self.feature(out)
        if self.sigma:
            map_normalized = F.softmax(map.reshape(map.size(0), map.size(1), -1), dim=2).view_as(map)
            map_transformed = self.map_transform(map_normalized)
            #stack = torch.cat((map_transformed, x), dim=1) # Try for concatenate insteaad of sum
            stack = map_transformed + x
            return map_normalized, stack
        else:
            return map


class Nccuc(nn.Module):
    def __init__(self, in_channels, filters):
        super(Nccuc, self).__init__()
        self.in_channels = in_channels
        self.n_filters = filters
        self.down_Conv = Conv(inp_dim=in_channels, out_dim=filters[0], kernel_size=3, stride=1, bn=True, relu=True)
        self.up_Conv = nn.ConvTranspose2d(in_channels=filters[0], out_channels=filters[1], kernel_size=4, stride=2,
                                          padding=1)
        self.relu = nn.ReLU()

    def forward(self, input_A, input_B):
        down_conv = self.down_Conv(input_A)
        up_conv = self.up_Conv(down_conv)
        up_conv = self.relu(up_conv)
        out = torch.cat((up_conv, input_B), dim=1)
        return out


class Decoder(nn.Module):
    def __init__(self, nk, nf, reconstr_dim=128, n_c=3):
        super(Decoder, self).__init__()
        self.reconstr_dim = reconstr_dim
        self.out_channels = n_c
        self.conv1 = Nccuc(nf + 2, [512, 512])  # 8
        self.conv2 = Nccuc(nf + 4 + 512, [512, 256])  # 16
        self.conv3 = Nccuc(nf + nk + 256, [256, 256])  # 32
        self.conv4 = Nccuc(nk + 256, [256, 128])  # 64
        self.conv5 = Nccuc(nk + 128, [128, 64])   # 128
        self.conv6 = Conv(nk + 64, self.out_channels, kernel_size=5, stride=1, relu=False)

    def forward(self, encoding_list):
        conv1 = self.conv1(encoding_list[-1], encoding_list[-2])
        conv2 = self.conv2(conv1, encoding_list[-3])
        conv3 = self.conv3(conv2, encoding_list[-4])
        conv4 = self.conv4(conv3, encoding_list[-5])
        conv5 = self.conv5(conv4, encoding_list[-6])
        conv6 = self.conv6(conv5)
        out = torch.sigmoid(conv6)
        return out