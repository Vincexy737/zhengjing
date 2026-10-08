# -*- coding: utf-8 -*-
"""构建期工具：将 RealESRGAN anime 6B 权重转成 ONNX（3通道输入，x4放大）。

**运行应用不需要本脚本** —— `core/sr.py` 直接消费已转换好的
`models/RealESRGAN_x4plus_anime_6B.onnx`。仅当你要自行重建该 onnx 时才用。

依赖 torch（本项目运行期不需要，帧净.spec 里也显式 excludes）：
    pip install torch --index-url https://download.pytorch.org/whl/cpu

前置：把 RealESRGAN_x4plus_anime_6B.pth 放到 models/ 后运行
    python tools/convert_sr.py
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
PTH_PATH = os.path.join(MODEL_DIR, "RealESRGAN_x4plus_anime_6B.pth")
ONNX_PATH = os.path.join(MODEL_DIR, "RealESRGAN_x4plus_anime_6B.onnx")


class ResidualDenseBlock_5(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock_5(nf, gc)
        self.rdb2 = ResidualDenseBlock_5(nf, gc)
        self.rdb3 = ResidualDenseBlock_5(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.conv_body(self.body(fea))
        fea = fea + trunk
        fea = self.lrelu(self.conv_up1(
            F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.conv_up2(
            F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.conv_hr(fea))
        out = self.conv_last(fea)
        return out


def main():
    net = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=6, gc=32, scale=4)
    ckpt = torch.load(PTH_PATH, map_location="cpu", weights_only=True)
    if "params_ema" in ckpt:
        ckpt = ckpt["params_ema"]
    elif "params" in ckpt:
        ckpt = ckpt["params"]
    net.load_state_dict(ckpt, strict=True)
    net.eval()
    print("权重加载成功")

    dummy = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        torch.onnx.export(
            net, dummy, ONNX_PATH,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch", 2: "h", 3: "w"},
                          "output": {0: "batch", 2: "h", 3: "w"}},
            opset_version=14,
            dynamo=False,
        )
    sz = os.path.getsize(ONNX_PATH)
    print(f"ONNX 导出成功: {sz / 1048576:.1f} MB")


if __name__ == "__main__":
    main()
