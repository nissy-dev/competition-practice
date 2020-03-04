import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


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


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                            (x.size(-2), x.size(-1))).pow(1./self.p)


class BengaliBaselineClassifier(nn.Module):
    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7,
                 pretrainedmodels=None, in_channels=3,
                 hdim=1048, use_bn=True, pretrained=None):
        super(BengaliBaselineClassifier, self).__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.base_model = pretrainedmodels
        self.conv0 = nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, bias=True)
        inch = self.base_model.last_linear.in_features
        self.fc1 = LinearBlock(inch, hdim, use_bn=use_bn, activation=F.relu)
        self.logits_for_grapheme = LinearBlock(hdim, n_grapheme, use_bn=False, activation=None)
        self.logits_for_vowel = LinearBlock(hdim, n_vowel, use_bn=False, activation=None)
        self.logits_for_consonant = LinearBlock(hdim, n_consonant, use_bn=False, activation=None)

    def forward(self, x):
        h = self.conv0(x)
        h = self.base_model.features(h)
        # sum pool (batch_size × inch)
        h = torch.sum(h, dim=(-1, -2))
        h = self.fc1(h)
        logits_for_grapheme = self.logits_for_grapheme(h)
        logits_for_consonant = self.logits_for_consonant(h)
        logits_for_vowel = self.logits_for_vowel(h)
        # target_col = ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic']
        logits = (logits_for_grapheme, logits_for_consonant, logits_for_vowel)
        return logits


class CustomBengaliBaselineClassifier(nn.Module):
    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7,
                 pretrainedmodels=None, in_channels=3,
                 hdim=512, use_bn=True, pretrained=None):
        super(BengaliBaselineClassifier, self).__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.base_model = pretrainedmodels
        self.conv0 = nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, bias=True)
        inch = self.base_model.last_linear.in_features
        self.avg_pool = GeM()
        self.fc1 = LinearBlock(inch, hdim, use_bn=use_bn, activation=F.relu)
        self.logits_for_grapheme = LinearBlock(hdim, n_grapheme, use_bn=False, activation=None)
        self.logits_for_vowel = LinearBlock(hdim, n_vowel, use_bn=False, activation=None)
        self.logits_for_consonant = LinearBlock(hdim, n_consonant, use_bn=False, activation=None)

    def forward(self, x):
        h = self.conv0(x)
        h = self.base_model.features(h)
        # gem pool
        h = self.avg_pool(h)
        h = h.view(h.size(0), -1)
        h = self.fc1(h)
        logits_for_grapheme = self.logits_for_grapheme(h)
        logits_for_consonant = self.logits_for_consonant(h)
        logits_for_vowel = self.logits_for_vowel(h)
        # target_col = ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic']
        logits = (logits_for_grapheme, logits_for_consonant, logits_for_vowel)
        return logits
