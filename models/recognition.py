import torch
import torch.nn as nn


class ResNetBasicStem(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
    ):
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        self._construct_stem(dim_in, dim_out)

    def _construct_stem(self, dim_in, dim_out):
        self.conv = nn.Conv3d(
            dim_in,
            dim_out,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(
            num_features=dim_out, eps=self.eps, momentum=self.bn_mmt
        )
        self.relu = nn.ReLU(self.inplace_relu)
        self.pool_layer = nn.MaxPool3d(
            kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1]
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool_layer(x)
        return x


class VideoModelStem(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
    ):
        super().__init__()
        self.num_pathways = len(dim_in)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        self._construct_stem(dim_in, dim_out)

    def _construct_stem(self, dim_in, dim_out):
        for pathway in range(len(dim_in)):
            stem = ResNetBasicStem(
                dim_in[pathway],
                dim_out[pathway],
                self.kernel[pathway],
                self.stride[pathway],
                self.padding[pathway],
                self.inplace_relu,
                self.eps,
                self.bn_mmt,
            )
            self.add_module("pathway{}_stem".format(pathway), stem)

    def forward(self, x):
        assert (
            len(x) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        for pathway in range(len(x)):
            m = getattr(self, "pathway{}_stem".format(pathway))
            x[pathway] = m(x[pathway])
        return x


###############################################################################
class BottleneckTransform(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        dim_inner,
        num_groups,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
    ):
        super().__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._stride_1x1 = stride_1x1
        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            dilation,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        dilation,
    ):
        (str1x1, str3x3) = (stride, 1) if self._stride_1x1 else (1, stride)

        # Tx1x1, BN, ReLU.
        self.a = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=[self.temp_kernel_size, 1, 1],
            stride=[1, str1x1, str1x1],
            padding=[int(self.temp_kernel_size // 2), 0, 0],
            bias=False,
        )
        self.a_bn = nn.BatchNorm3d(
            num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
        )
        self.a_relu = nn.ReLU(inplace=self._inplace_relu)

        # 1x3x3, BN, ReLU.
        self.b = nn.Conv3d(
            dim_inner,
            dim_inner,
            [1, 3, 3],
            stride=[1, str3x3, str3x3],
            padding=[0, dilation, dilation],
            groups=num_groups,
            bias=False,
            dilation=[1, dilation, dilation],
        )
        self.b_bn = nn.BatchNorm3d(
            num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
        )
        self.b_relu = nn.ReLU(inplace=self._inplace_relu)

        # 1x1x1, BN.
        self.c = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            bias=False,
        )
        self.c_bn = nn.BatchNorm3d(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.c_bn.transform_final_bn = True

    def forward(self, x):
        # Explicitly forward every layer.
        # Branch2a.
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)

        # Branch2b.
        x = self.b(x)
        x = self.b_bn(x)
        x = self.b_relu(x)

        # Branch2c
        x = self.c(x)
        x = self.c_bn(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        dim_inner,
        num_groups=1,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
    ):
        super().__init__()
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._construct(
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            dim_inner,
            num_groups,
            stride_1x1,
            inplace_relu,
            dilation,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        dim_inner,
        num_groups,
        stride_1x1,
        inplace_relu,
        dilation,
    ):
        # Use skip connection with projection if dim or res change.
        if (dim_in != dim_out) or (stride != 1):
            self.branch1 = nn.Conv3d(
                dim_in,
                dim_out,
                kernel_size=1,
                stride=[1, stride, stride],
                padding=0,
                bias=False,
                dilation=1,
            )
            self.branch1_bn = nn.BatchNorm3d(
                num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
            )
        self.branch2 = BottleneckTransform(
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            dim_inner,
            num_groups,
            stride_1x1=stride_1x1,
            inplace_relu=inplace_relu,
            dilation=dilation,
        )
        self.relu = nn.ReLU(self._inplace_relu)

    def forward(self, x):
        if hasattr(self, "branch1"):
            x = self.branch1_bn(self.branch1(x)) + self.branch2(x)
        else:
            x = x + self.branch2(x)
        x = self.relu(x)
        return x


class ResStage(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        stride,
        temp_kernel_sizes,
        num_blocks,
        dim_inner,
        num_groups,
        num_block_temp_kernel,
        dilation,
        stride_1x1=False,
        inplace_relu=True,
    ):
        super().__init__()
        assert all(
            (
                num_block_temp_kernel[i] <= num_blocks[i]
                for i in range(len(temp_kernel_sizes))
            )
        )
        self.num_blocks = num_blocks
        self.temp_kernel_sizes = [
            (temp_kernel_sizes[i] * num_blocks[i])[: num_block_temp_kernel[i]]
            + [1] * (num_blocks[i] - num_block_temp_kernel[i])
            for i in range(len(temp_kernel_sizes))
        ]
        assert (
            len(
                {
                    len(dim_in),
                    len(dim_out),
                    len(temp_kernel_sizes),
                    len(stride),
                    len(num_blocks),
                    len(dim_inner),
                    len(num_groups),
                    len(num_block_temp_kernel),
                }
            )
            == 1
        )
        self.num_pathways = len(self.num_blocks)
        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            stride_1x1,
            inplace_relu,
            dilation,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        stride_1x1,
        inplace_relu,
        dilation,
    ):
        for pathway in range(self.num_pathways):
            for i in range(self.num_blocks[pathway]):
                block = ResBlock(
                    dim_in[pathway] if i == 0 else dim_out[pathway],
                    dim_out[pathway],
                    self.temp_kernel_sizes[pathway][i],
                    stride[pathway] if i == 0 else 1,
                    dim_inner[pathway],
                    num_groups[pathway],
                    stride_1x1=stride_1x1,
                    inplace_relu=inplace_relu,
                    dilation=dilation[pathway],
                )
                self.add_module("pathway{}_res{}".format(pathway, i), block)

    def forward(self, inputs):
        output = []
        for pathway in range(self.num_pathways):
            x = inputs[pathway]
            for i in range(self.num_blocks[pathway]):
                m = getattr(self, "pathway{}_res{}".format(pathway, i))
                x = m(x)
            output.append(x)

        return output


###############################################################################
class ResNetBasicHead(nn.Module):
    def __init__(self, dim_in, num_classes, pool_size, dropout_rate):
        super().__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."

        self.num_pathways = len(pool_size)
        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

    def forward(self, inputs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)

        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))

        x = torch.cat(pool_out, 1)
        x = x.permute((0, 2, 3, 4, 1))

        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = self.projection(x)
        x = x.view(x.shape[0], -1)

        return x


class DeepfakeDetection(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_pathways = 1
        self._construct_network(cfg)

    def _construct_network(self, cfg):
        (d2, d3, d4, d5) = (3, 4, 6, 3)

        num_groups = cfg.num_groups
        width_per_group = cfg.width_per_group
        dim_inner = num_groups * width_per_group

        temp_kernel = [
            [[5]],  # conv1 temporal kernel.
            [[3]],  # res2 temporal kernel.
            [[3, 1]],  # res3 temporal kernel.
            [[3, 1]],  # res4 temporal kernel.
            [[1, 3]],  # res5 temporal kernel.
        ]

        self.s1 = VideoModelStem(
            dim_in=[3],
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
        )

        self.s2 = ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.spatial_strides[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.num_block_temp_kernel[0],
            stride_1x1=cfg.stride_1x1,
            inplace_relu=cfg.inplace_relu,
            dilation=cfg.spatial_dilations[0],
        )

        pool_size = [[2, 1, 1]]
        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.spatial_strides[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.num_block_temp_kernel[1],
            stride_1x1=cfg.stride_1x1,
            inplace_relu=cfg.inplace_relu,
            dilation=cfg.spatial_dilations[1],
        )

        self.s4 = ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.spatial_strides[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.num_block_temp_kernel[2],
            stride_1x1=cfg.stride_1x1,
            inplace_relu=cfg.inplace_relu,
            dilation=cfg.spatial_dilations[2],
        )

        self.s5 = ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.spatial_strides[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.num_block_temp_kernel[3],
            stride_1x1=cfg.stride_1x1,
            inplace_relu=cfg.inplace_relu,
            dilation=cfg.spatial_dilations[3],
        )

        self.head = ResNetBasicHead(
            dim_in=[width_per_group * 32],
            num_classes=1,
            pool_size=[
                [
                    32 // pool_size[0][0],
                    7 // pool_size[0][1],
                    7 // pool_size[0][2],
                ]
            ],
            dropout_rate=0.5,
        )

    def forward(self, xs):
        xs = self.s1(xs)
        xs = self.s2(xs)

        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            xs[pathway] = pool(xs[pathway])

        xs = self.s3(xs)
        xs = self.s4(xs)
        xs = self.s5(xs)
        xs = self.head(xs)

        return xs


class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                self.__dict__[key] = Config(**value)
            else:
                self.__dict__[key] = value


def load_classifier(filepath, device):
    cfg = Config(
        **{
            "inplace_relu": True,
            "num_block_temp_kernel": [[3], [4], [6], [3]],
            "num_groups": 1,
            "spatial_dilations": [[1], [1], [1], [1]],
            "spatial_strides": [[1], [2], [2], [2]],
            "stride_1x1": False,
            "width_per_group": 64,
        }
    )

    model = DeepfakeDetection(cfg)
    model.load_state_dict(torch.load(filepath, map_location="cpu"))
    model.eval()

    torch.backends.cudnn.benchmark = True
    model = model.to(device)

    return model
