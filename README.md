# DINOv3 + PSPNet 语义分割

基于自监督视觉大模型 DINOv3 ViT-S/16 与 PSPNet 的语义分割实现，在 PASCAL VOC 2012 验证集上取得了 **mIoU 85.15%** 的结果。

## 结果

| 方法 | mIoU | PA |
|------|------|-----|
| DINOv3 + PSPNet  | 84.18% | 96.75% |
| DINOv3 + PSPNet + Skip Connection | **85.15%** | **96.85%** |

## 架构概述

```
Input Image (1024×1024)
    │
    ▼
┌────────────────────┐
│  DINOv3 ViT-S/16   │  自监督预训练 Backbone (冻结)
│  → Patch Tokens    │
│  → 低层特征 (Block 3)  │
│  → 高层特征 (Block 11) │
└─────────┬──────────┘
          │
    ┌─────┴─────┐
    │            │
    ▼            ▼
 低层特征     高层特征
 (384-dim)   (384-dim)
    │            │
    ▼            ▼
 1×1 Conv     PPM 模块
 (384→48)    (1,2,3,6 bins)
    │            │
    └─────┬──────┘
          │ Concat (944-dim)
          ▼
    ┌──────────┐
    │ 分类头    │
    │ Conv+BN  │
    │ +Dropout │
    └──────────┘
          │
          ▼
   分割掩码 (1024×1024)
```

### 核心设计

- **DINOv3 Backbone**：利用自监督预训练的 Vision Transformer encoder 提取特征，冻结主干参数保护预训练表征
- **Skip Connection**：从 DINOv3 浅层提取高分辨率边缘特征，与深层语义特征融合，显著改善分割边界质量
- **PPM 模块**：金字塔池化模块 (1×1, 2×2, 3×3, 6×6)，聚合多尺度全局上下文信息

## 训练策略

### 两阶段训练

| 阶段 | Epochs | Batch Size | 学习率 | 说明 |
|------|--------|-----------|--------|------|
| 阶段一：冻结 Backbone | 50 | 8 | 1e-3 | 仅训练 PPM + 分类头 |
| 阶段二：解冻最后一层 | 10 | 4 | 1e-4 (头) / 1e-7 (主干) | 差分学习率微调 |

### 损失函数

CE + Dice 联合损失 (CE_DiceLoss)，同时优化像素级分类准确性与前景形状边界连贯性。

## 项目结构

```
├── model.py        # DINOv3 Backbone + PSPNet 模型定义
├── data.py         # PASCAL VOC 数据集加载
├── train.py        # 两阶段训练入口
├── predict.py      # 推理与可视化
├── PCA.py          # PCA 特征可视化
└── README.md
```

## 依赖

- Python 3.x
- PyTorch
- torchvision
- NumPy
- PIL (Pillow)
- tqdm

## 使用方法

### 数据准备

下载 PASCAL VOC 2012 数据集，放置于 `./data/VOCdevkit/VOC2012/` 目录下。

### 训练

```bash
python train.py
```

### 推理

```bash
python predict.py
```

### PCA 可视化

```bash
python PCA.py
```

## 参考

- [DINOv3](https://github.com/facebookresearch/dinov3)
- [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)
- [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
