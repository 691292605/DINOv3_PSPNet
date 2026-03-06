import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))



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
        features = self.model.forward_features(x)
        tokens = features['x_norm_patchtokens']
        feat_h, feat_w = H // self.patch_size, W // self.patch_size
        out = tokens.reshape(B, feat_h, feat_w, self.embed_dim)
        out = out.permute(0, 3, 1, 2).contiguous()
        return out

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
        return torch.cat(out, 1)

class DINO_PSPNet(nn.Module):
    def __init__(self, num_classes=21):
        super(DINO_PSPNet, self).__init__()
        self.backbone = DINOv3()
        self.ppm = PPM(in_dim=384, reduction_dim=128)
        self.cls_head = nn.Sequential(
            nn.Conv2d(896, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1), 
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        feat = self.backbone(x)
        feat = self.ppm(feat)
        out = self.cls_head(feat)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        
        return out


if __name__ == "__main__":
    model = DINO_PSPNet(num_classes=21)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    WEIGHT_PATH = r'D:\lab\enter_test\dinov3_vits16_pretrain.pth'

    dummy_input = torch.randn(2, 3, 512, 512)
    dummy_input = dummy_input.to(device)
    prediction = model(dummy_input)
    
    print(f"输入图像形状: {dummy_input.shape}")
    print(f"最终输出形状: {prediction.shape}")
    checkpoint = torch.load(WEIGHT_PATH, map_location='cpu')
    print("Weights keys sample:", list(checkpoint.keys())[:5])

    new_state_dict = {}
    for k, v in checkpoint.items():
        name = k.replace('backbone.', '') 
        new_state_dict[name] = v
    model.backbone.model.load_state_dict(checkpoint, strict=True)


