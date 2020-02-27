import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels


# dropout and bn and residualに対応したlinear
class LinearBlock(nn.Module):
    def __init__(self, in_features=None, out_features=None, bias=True,
                 use_bn=True, activation=F.relu, dropout_ratio=-1, residual=False):
        super(LinearBlock, self).__init__()
        # validation
        if in_features is None or out_features is None:
            raise ValueError('You should set both in_features and out_features!!')
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation
        self.use_bn = use_bn
        self.dropout_ratio = dropout_ratio
        self.residual = residual

        if use_bn:
            self.bn = nn.BatchNorm1d(out_features)
        if dropout_ratio > 0.:
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout = None

    def __call__(self, x):
        h = self.linear(x)
        if self.use_bn:
            h = self.bn(h)
        if self.activation is not None:
            h = self.activation(h)
        if self.residual:
            h = self._residual_add(h, x)
        if self.dropout_ratio > 0:
            h = self.dropout(h)
        return h

    def _residual_add(lhs, rhs):
        lhs_ch, rhs_ch = lhs.shape[1], rhs.shape[1]
        if lhs_ch < rhs_ch:
            out = lhs + rhs[:, :lhs_ch]
        elif lhs_ch > rhs_ch:
            out = torch.cat([lhs[:, :rhs_ch] + rhs, lhs[:, rhs_ch:]], dim=1)
        else:
            out = lhs + rhs
        return out


class BengaliAIClassifier(nn.Module):
    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7,
                 model_name='se_resnext101_32x4d', in_channels=3,
                 hdim=1048, use_bn=True, pretrained=None):
        super(BengaliAIClassifier, self).__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.conv0 = nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.base_model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
        inch = self.base_model.last_linear.in_features
        out_dim = n_grapheme + n_vowel + n_consonant
        self.fc1 = LinearBlock(inch, hdim, use_bn=use_bn, activation=F.relu, residual=False)
        self.fc2 = LinearBlock(hdim, out_dim, use_bn=use_bn, activation=None, residual=False)

    def forward(self, x):
        h = self.conv0(x)
        h = self.base_model.features(h)
        # sum pool (batch_size × inch)
        h = torch.sum(h, dim=(-1, -2))
        h = self.fc1(h)
        h = self.fc2(h)
        # split each class
        h = torch.split(h, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1)
        return h
