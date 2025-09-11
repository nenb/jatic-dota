import numpy as np
import torch
import torch.nn as nn

from . import resnet
from .model_parts import CombinationModule


class CTRBOX(nn.Module):
    def __init__(self, down_ratio, pretrained=True, head_conv=256, final_kernel=1):
        super(CTRBOX, self).__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        self.base_network = resnet.resnet101(pretrained=pretrained)
        self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
        self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
        self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)
        self.heads = {"hm": 15, "wh": 10, "reg": 2, "cls_theta": 1}

        for head in self.heads:
            classes = self.heads[head]
            if head == "wh":
                fc = nn.Sequential(
                    nn.Conv2d(
                        channels[self.l1],
                        head_conv,
                        kernel_size=3,
                        padding=1,
                        bias=True,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes, kernel_size=3, padding=1, bias=True),
                )
            else:
                fc = nn.Sequential(
                    nn.Conv2d(
                        channels[self.l1],
                        head_conv,
                        kernel_size=3,
                        padding=1,
                        bias=True,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        head_conv,
                        classes,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        bias=True,
                    ),
                )
            if "hm" in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)

    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])

        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            if "hm" in head or "cls" in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict
