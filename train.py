import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import DINO_PSPNet
from data import voc_dataloaders


class CE_DiceLoss(nn.Module):
    def __init__(self, num_classes=21, ignore_index=255):
        super(CE_DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, inputs, target):
        ce = self.ce_loss(inputs, target)
        pred = F.softmax(inputs, dim=1) 
        pred = pred.transpose(1, 2).transpose(2, 3).contiguous().view(-1, self.num_classes)
        target = target.view(-1)
        valid_mask = (target != self.ignore_index)
        pred = pred[valid_mask]
        target = target[valid_mask]  
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        
        smooth = 1e-5
        intersection = torch.sum(pred * target_one_hot, dim=0)
        union = torch.sum(pred, dim=0) + torch.sum(target_one_hot, dim=0)
        
        dice_score = (2.0 * intersection + smooth) / (union + smooth)
        dice_foreground = dice_score[1:]
        dice = 1.0 - torch.mean(dice_foreground)
        
        return ce + dice

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备是: {device}")
    
    Freeze_Epoch = 50       #冻结主干训练
    Freeze_batch_size = 8   
    Freeze_lr = 1e-3 
    
    UnFreeze_Epoch = 10      #解冻主干微调
    Unfreeze_batch_size = 4 
    Unfreeze_lr = 1e-4      
    
    Total_Epoch = Freeze_Epoch + UnFreeze_Epoch
    
    model = DINO_PSPNet(num_classes=21).to(device)
    criterion = CE_DiceLoss(num_classes=21, ignore_index=255)
    scaler = GradScaler() 
    
    best_val_loss = float('inf')
    
    for epoch in range(Total_Epoch):
        if epoch == 0:
            print("[阶段一] 冻结主干，仅训练 PSPNet 头")
            for param in model.backbone.parameters():
                param.requires_grad = False
            
            train_loader, val_loader = voc_dataloaders(batch_size=Freeze_batch_size)
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=Freeze_lr, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=Freeze_Epoch, eta_min=1e-5)
            
        elif epoch == Freeze_Epoch:
            torch.save(model.state_dict(), '50_dino_pspnet.pth')
            print(f"已保存 Epoch {epoch+1} 的权重: 50_dino_pspnet.pth")
            print("[阶段二] 解冻主干最后一层，微调整个模型")

            start_block = 11
            for name, param in model.backbone.named_parameters():
                is_target_block = False
                if "blocks." in name:
                    block_num = int(name.split("blocks.")[1].split(".")[0])
                    if block_num >= start_block:
                        is_target_block = True
                        
                if is_target_block or name.endswith("norm.weight") or name.endswith("norm.bias"):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

                    
            train_loader, val_loader = voc_dataloaders(batch_size=Unfreeze_batch_size)
            
            backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad]
            head_params = [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad]
            
            optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': Unfreeze_lr * 0.001}, 
                {'params': head_params, 'lr': Unfreeze_lr}
            ], weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=UnFreeze_Epoch, eta_min=1e-6)
        
        model.train()
        train_loss = 0.0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Total_Epoch} [Train]")
        
        for images, masks in pbar_train:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar_train.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[-1]['lr']:.6f}"})

        torch.save(model.state_dict(), 'latest_dino_pspnet.pth')
        print(f"已保存 Epoch {epoch+1} 的最新权重: latest_dino_pspnet.pth")
            
        scheduler.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{Total_Epoch}] 验证集 Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_dino_pspnet.pth')
            print(f"保存最佳权重 (Loss: {best_val_loss:.4f})\n")

if __name__ == "__main__":
    main()