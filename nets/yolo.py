import numpy as np
import torch
import torch.nn as nn

from nets.backbone import Backbone, Multi_Concat_Block, Conv, SiLU, Transition_Block, autopad


class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        # 输出通道数为c2
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=SiLU(), deploy=False):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2
        self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        else:
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=c1, eps=0.001, momentum=0.03) if c2 == c1 and s == 1 else None)
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )

    def fuse_conv_bn(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         padding=conv.padding,
                         dilation=conv.dilation,
                         groups=conv.groups,
                         bias=True,
                         padding_mode=conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):
        if self.deploy:
            return
        print(f"RepConv.fuse_repvgg_block")
        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])

        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])

        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity,
                                                                        nn.modules.batchnorm.SyncBatchNorm)):
            identity_conv_1x1 = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=self.groups,
                bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)

            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])
        else:
            bias_identity_expanded = torch.nn.Parameter(torch.zeros_like(rbr_1x1_bias))
            weight_identity_expanded = torch.nn.Parameter(torch.zeros_like(weight_1x1_expanded))

        self.rbr_dense.weight = torch.nn.Parameter(
            self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)

        self.rbr_reparam = self.rbr_dense
        self.deploy = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None


def fuse_conv_and_bn(conv, bn):
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    # fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape).detach())

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    # fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    fusedconv.bias.copy_((torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn).detach())
    return fusedconv


# ---------------------------------------------------#
#   yolo_body
# ---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi, pretrained=False):
        super(YoloBody, self).__init__()
        # -----------------------------------------------#
        #   定义了不同yolov7版本的参数
        # -----------------------------------------------#
        transition_channels = {'l': 32, 'x': 40}[phi]
        block_channels = 32
        panet_channels = {'l': 32, 'x': 64}[phi]
        e = {'l': 2, 'x': 1}[phi]
        n = {'l': 4, 'x': 6}[phi]
        ids = {'l': [-1, -2, -3, -4, -5, -6], 'x': [-1, -3, -5, -7, -8]}[phi]
        conv = {'l': RepConv, 'x': Conv}[phi]
        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        # -----------------------------------------------#

        # ---------------------------------------------------#
        #   生成主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   80, 80, 512
        #   40, 40, 1024
        #   20, 20, 1024
        # ---------------------------------------------------#
        self.backbone = Backbone(transition_channels, block_channels, n, phi, pretrained=pretrained)

        # ------------------------加强特征提取网络------------------------#
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.sppcspc = SPPCSPC(transition_channels * 32, transition_channels * 16)

        self.conv_for_feat1 = Conv(transition_channels * 16, transition_channels * 4)
        self.conv_for_feat2 = Conv(transition_channels * 32, transition_channels * 8)
        self.conv_for_feat3 = Conv(transition_channels * 16, transition_channels * 8)

        self.TB1 = Transition_Block(transition_channels * 4, transition_channels * 4)
        self.MCB1 = Multi_Concat_Block(transition_channels * 16, panet_channels * 4,
                                       transition_channels * 8, e=e, n=n, ids=ids)

        self.TB2 = Transition_Block(transition_channels * 8, transition_channels * 8)
        self.MCB2 = Multi_Concat_Block(transition_channels * 24, panet_channels * 6,
                                       transition_channels * 12, e=e, n=n, ids=ids)

        self.MCB3 = Multi_Concat_Block(transition_channels * 20, panet_channels * 5,
                                       transition_channels * 10, e=e, n=n, ids=ids)
        self.MCB4 = Multi_Concat_Block(transition_channels * 14, panet_channels * 4,
                                       transition_channels * 7, e=e, n=n, ids=ids)

        self.TB5 = Transition_Block(transition_channels * 7, transition_channels * 7)
        self.MCB5 = Multi_Concat_Block(transition_channels * 24, panet_channels * 6,
                                       transition_channels * 12, e=e, n=n, ids=ids)

        self.TB6 = Transition_Block(transition_channels * 12, transition_channels * 12)
        self.MCB6 = Multi_Concat_Block(transition_channels * 36, panet_channels * 9,
                                       transition_channels * 18, e=e, n=n, ids=ids)

        self.rep_conv_1 = conv(transition_channels * 7, transition_channels * 8, 3, 1)
        self.rep_conv_2 = conv(transition_channels * 12, transition_channels * 16, 3, 1)
        self.rep_conv_3 = conv(transition_channels * 18, transition_channels * 32, 3, 1)

        self.yolo_head_P3 = nn.Conv2d(transition_channels * 8, len(anchors_mask[2]) * (5 + num_classes), 1)
        self.yolo_head_P4 = nn.Conv2d(transition_channels * 16, len(anchors_mask[1]) * (5 + num_classes), 1)
        self.yolo_head_P5 = nn.Conv2d(transition_channels * 32, len(anchors_mask[0]) * (5 + num_classes), 1)
    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if isinstance(m, RepConv):
                m.fuse_repvgg_block()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuseforward
        return self

    def forward(self, x):
        #  backbone
        feat1, feat2, feat3 = self.backbone.forward(x)
        # feat1 80, 80, 512
        # feat2 40, 40, 1024
        # feat3 20, 20, 1024
        # ------------------------加强特征提取网络------------------------#
        # 80,80,512 => 80,80,128
        P1 = self.conv_for_feat1(feat1)
        # 40,40,1024 => 40,40,256
        P2 = self.conv_for_feat2(feat2)
        # 20,20,1024 => 20,20,256
        P3 = self.conv_for_feat3(self.sppcspc(feat3))
        # 40,40,256+256 => 40,40,256
        P2 = self.MCB1(torch.cat([P2, self.TB1(P1)], 1))
        # 20,20,256+512 => 20,20,384
        P3 = self.MCB2(torch.cat([P3, self.TB2(P2)], 1))
        # 40,40,384+256 => 40,40,320
        P2 = self.MCB3(torch.cat([P2, self.upsample(P3)], 1))
        # 80,80,320+128 => 80,80,224
        P1 = self.MCB4(torch.cat([P1, self.upsample(P2)], 1))
        # 40,40,320+448 => 40,40,384
        P2 = self.MCB5(torch.cat([P2, self.TB5(P1)], 1))
        # 20,20,384+768 => 20,20,576
        P3 = self.MCB6(torch.cat([P3, self.TB6(P2)], 1))

        P1 = self.rep_conv_1(P1)
        P2 = self.rep_conv_2(P2)
        P3 = self.rep_conv_3(P3)

        out2 = self.yolo_head_P3(P1)
        out1 = self.yolo_head_P4(P2)
        out0 = self.yolo_head_P5(P3)
        # print(P1.shape, P2.shape, P3.shape)

        return [out0, out1, out2]


if __name__ == '__main__':
    yolo_body = YoloBody([[6, 7, 8], [3, 4, 5], [0, 1, 2]], 80, 'l')
    input_tensor = torch.randn(1, 3, 640, 640)
    output = yolo_body(input_tensor)
