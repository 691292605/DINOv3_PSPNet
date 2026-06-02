import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

"""
在原有的 DINOv3_PSPNet 的基础上加入了Skip Connection，
从 DINOv3 的第4层（即第3个 Transformer Block）提取特征，
并通过一个 1x1 卷积将其维度从 384 降到 48。这个低级特征图与 PPM 输出的高级特征图（896维）进行拼接，
形成一个包含更多细节信息的特征图（944维），然后送入分类头进行最终的像素级分类预测。
"""

class DINOv3(nn.Module):
    def __init__(self, weight_path=r'D:\\lab\\enter_test\\dinov3_vits16_pretrain.pth', freeze=True):
        super(DINOv3, self).__init__()

        self.model = torch.hub.load("./dinov3-main", 'dinov3_vits16', source='local', pretrained=False)
        checkpoint = torch.load(weight_path, map_location='cpu')
        self.model.load_state_dict(checkpoint, strict=True)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.patch_size = 16
        self.embed_dim = 384
        

    def forward(self, x):
        B, C, H, W = x.shape
        feat_h, feat_w = H // self.patch_size, W // self.patch_size

        tokens_list = self.model.get_intermediate_layers(x, n=[3, 11])
        
        low_tokens = tokens_list[0]  
        high_tokens = tokens_list[1] 
        
        low_feat = low_tokens.reshape(B, feat_h, feat_w, self.embed_dim).permute(0, 3, 1, 2).contiguous()
        high_feat = high_tokens.reshape(B, feat_h, feat_w, self.embed_dim).permute(0, 3, 1, 2).contiguous()
        
        return low_feat, high_feat

class PPM(nn.Module):
    def __init__(self, in_dim=384, reduction_dim=128, bins=(1, 2, 3, 6)):

        super(PPM, self).__init__()
        
        self.features = nn.ModuleList()
        
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            feat = f(x)
            upsampled = F.interpolate(feat, x_size[2:], mode='bilinear', align_corners=True)
            out.append(upsampled)
        return torch.cat(out, 1) #输出维度:384 + 128*4 = 896

class DINO_PSPNet(nn.Module):
    def __init__(self, num_classes=21):
        super(DINO_PSPNet, self).__init__()
        self.backbone = DINOv3()
        self.ppm = PPM(in_dim=384, reduction_dim=128)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(384, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.cls_head = nn.Sequential(
            nn.Conv2d(944, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1), 
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        low_feat, high_feat = self.backbone(x)
        high_feat_ppm = self.ppm(high_feat)  # [B, 896, 64, 64]
        low_feat_reduced = self.low_level_conv(low_feat)  # [B, 48, 64, 64]
        concat_feat = torch.cat([high_feat_ppm, low_feat_reduced], dim=1) # [B, 944, 64, 64]
        out = self.cls_head(concat_feat)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        
        return out


if __name__ == "__main__":
    model = DINO_PSPNet(num_classes=21)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    WEIGHT_PATH = r'./dinov3_vits16_pretrain.pth'

    dummy_input = torch.randn(2, 3, 512, 512)
    dummy_input = dummy_input.to(device)
    prediction = model(dummy_input)
    
    print(f"输入图像形状: {dummy_input.shape}")
    print(f"最终输出形状: {prediction.shape}")
    checkpoint = torch.load(WEIGHT_PATH, map_location='cpu')
    print("Weights keys sample:", list(checkpoint.keys())[:5])



