import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 注意：请确保能正常导入你的模型
from model import DINO_PSPNet

# PASCAL VOC 标准调色板 (21类 RGB 颜色)
VOC_COLORMAP = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128]
]

def decode_segmap(image_idx, colormap=VOC_COLORMAP):
    r = np.zeros_like(image_idx).astype(np.uint8)
    g = np.zeros_like(image_idx).astype(np.uint8)
    b = np.zeros_like(image_idx).astype(np.uint8)
    for l in range(0, 21):
        idx = image_idx == l
        r[idx] = colormap[l][0]
        g[idx] = colormap[l][1]
        b[idx] = colormap[l][2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def visualize_multiple_predictions(image_paths, model_weight_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("正在加载模型和权重...")
    model = DINO_PSPNet(num_classes=21).to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device, weights_only=True))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    num_imgs = len(image_paths)
    # 动态创建画布：N行 2列，每行高度设为5，宽度设为10
    fig, axes = plt.subplots(nrows=num_imgs, ncols=2, figsize=(10, 5 * num_imgs))
    
    # 如果只有1张图，axes是一维数组，为方便统一处理将其转为二维列表
    if num_imgs == 1:
        axes = [axes]

    print(f"开始生成 {num_imgs} 张图片的分割掩码...")
    
    for i, image_path in enumerate(image_paths):
        original_img = Image.open(image_path).convert('RGB')
        input_tensor = transform(original_img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            
        pred_mask = torch.argmax(output.squeeze(0), dim=0).cpu().numpy()
        color_mask = decode_segmap(pred_mask)

        # 获取文件名用于标题展示
        img_name = os.path.basename(image_path)

        # 绘制左侧：原图
        axes[i][0].imshow(original_img.resize((1024, 1024))) 
        axes[i][0].set_title(f"Original: {img_name}")
        axes[i][0].axis('off')

        # 绘制右侧：预测掩码
        axes[i][1].imshow(color_mask)
        axes[i][1].set_title(f"Prediction: {img_name}")
        axes[i][1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 在这里放入你想要预测的多张图片的路径
    TEST_IMAGE_PATHS = [
        r'.\data\VOCdevkit\VOC2012\JPEGImages\2007_000904.jpg',
        r'.\data\VOCdevkit\VOC2012\JPEGImages\2010_003239.jpg',
        r'.\data\VOCdevkit\VOC2012\JPEGImages\2012_002026.jpg'
    ] 
    
    WEIGHT_PATH = '50_dino_pspnet.pth'
    
    # 确保传入的是列表
    visualize_multiple_predictions(TEST_IMAGE_PATHS, WEIGHT_PATH)