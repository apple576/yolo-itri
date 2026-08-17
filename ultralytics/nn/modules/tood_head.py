import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import DFL
from ultralytics.nn.modules.conv import Conv
from ultralytics.utils.tal import dist2bbox, make_anchors


class TaskDecomposition(nn.Module):
    """輕量化 TOOD 任務分解模組 (LAM)"""
    def __init__(self, feat_channels=64, stacked_convs=2):
        super().__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs

        self.la_conv1 = nn.Conv2d(feat_channels * stacked_convs, feat_channels, 1)
        self.la_conv2 = nn.Conv2d(feat_channels, stacked_convs, 1)

    def forward(self, feat_concat, avg_feat):
        b, _, h, w = feat_concat.shape
        weight = F.relu(self.la_conv1(avg_feat))
        weight = torch.sigmoid(self.la_conv2(weight))

        feat_reshaped = feat_concat.view(b, self.stacked_convs, self.feat_channels, h, w)
        weight = weight.view(b, self.stacked_convs, 1, 1, 1)
        out_feat = (feat_reshaped * weight).sum(dim=1)
        return out_feat


class TOODHeadYOLO11(nn.Module):
    """特徵層空間增強版 TOOD 檢測頭"""
    dynamic = False
    export = False
    shape = None

    def __init__(self, nc=80, ch=(64, 128, 256), feat_channels=64, stacked_convs=2):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        self.anchors = torch.empty(0)
        self.strides = torch.empty(0)

        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        # 1. 輸入通道對齊層
        self.stem = nn.ModuleList([Conv(c_in, feat_channels, 1) for c_in in ch])

        # 2. 共享卷積塔
        self.inter_convs = nn.ModuleList([
            Conv(feat_channels, feat_channels, 3) for _ in range(stacked_convs)
        ])

        # 3. 任務分解模組
        self.cls_decomp = TaskDecomposition(feat_channels, stacked_convs)
        self.reg_decomp = TaskDecomposition(feat_channels, stacked_convs)

        # 4. TAP 空間機率圖模組 (產生 0~1 的空間注意力圖)
        self.cls_prob_conv1 = nn.Conv2d(feat_channels * stacked_convs, max(feat_channels // 2, 16), 1)
        self.cls_prob_conv2 = nn.Conv2d(max(feat_channels // 2, 16), 1, 3, padding=1)

        # 5. 輸出預測層
        self.tood_cls = nn.Conv2d(feat_channels, self.nc, 3, padding=1)
        self.tood_reg = nn.Conv2d(feat_channels, 4 * self.reg_max, 3, padding=1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        for m in self.modules():
            if isinstance(m, TaskDecomposition):
                nn.init.normal_(m.la_conv1.weight, mean=0.0, std=0.001)
                if m.la_conv1.bias is not None:
                    nn.init.constant_(m.la_conv1.bias, 0.0)
                nn.init.normal_(m.la_conv2.weight, mean=0.0, std=0.001)
                if m.la_conv2.bias is not None:
                    nn.init.constant_(m.la_conv2.bias, 0.0)

        self.bias_init()

    def bias_init(self):
        bias_cls = -math.log((1.0 - 0.01) / 0.01)
        nn.init.constant_(self.tood_cls.bias, bias_cls)
        if hasattr(self, "cls_prob_conv2") and self.cls_prob_conv2.bias is not None:
            nn.init.constant_(self.cls_prob_conv2.bias, 0.0)

    def forward(self, x):
        feats = []
        for i in range(self.nl):
            feat_in = self.stem[i](x[i])
            inter_feats = []
            cur_feat = feat_in
            for conv in self.inter_convs:
                cur_feat = conv(cur_feat)
                inter_feats.append(cur_feat)

            feat_concat = torch.cat(inter_feats, dim=1)
            avg_feat = F.adaptive_avg_pool2d(feat_concat, (1, 1))

            cls_feat = self.cls_decomp(feat_concat, avg_feat)
            reg_feat = self.reg_decomp(feat_concat, avg_feat)

            # 計算空間注意力圖 M (範圍 0~1)
            prob_map = F.relu(self.cls_prob_conv1(feat_concat))
            prob_map = torch.sigmoid(self.cls_prob_conv2(prob_map))

            # 在特徵層進行空間調製 (安全無截斷，且不破壞 Logit 負數定義)
            cls_feat_aligned = cls_feat * (1.0 + prob_map)

            # 純淨 Logits 輸出
            cls_out = self.tood_cls(cls_feat_aligned)
            reg_out = self.tood_reg(reg_feat)

            feats.append(torch.cat([reg_out, cls_out], dim=1))

        if self.training:
            return feats

        y = self._inference(feats, x)
        return y if self.export else (y, feats)

    def _inference(self, feats, x):
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in feats], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        reg_all, cls_all = x_cat.split((self.reg_max * 4, self.nc), 1)
        if self.export:
            return torch.cat([reg_all, cls_all], 1)

        reg_dfl = self.dfl(reg_all)
        box = dist2bbox(reg_dfl, self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        return torch.cat([box, cls_all.sigmoid()], 1)