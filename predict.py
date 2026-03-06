import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from model import DINO_PSPNet

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

def visualize_paper_style(image_paths, model_weight_path, save_format='svg'):
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

    fig, axes = plt.subplots(
        nrows=num_imgs, 
        ncols=2, 
        figsize=(6, 3 * num_imgs),
        gridspec_kw={'wspace': 0.04, 'hspace': 0.04}
    )
    
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

        axes[i][0].imshow(original_img.resize((1024, 1024))) 
        axes[i][0].axis('off')
        
        axes[i][1].imshow(color_mask)
        axes[i][1].axis('off')


    axes[-1][0].text(0.5, -0.1, "(a) Image", size=14, ha="center", transform=axes[-1][0].transAxes)
    axes[-1][1].text(0.5, -0.1, "(b) Prediction", size=14, ha="center", transform=axes[-1][1].transAxes)

    save_path = f"paper_figure.{save_format}"
    plt.savefig(save_path, format=save_format, bbox_inches='tight', pad_inches=0.05)
    
    print(f"\n图片已保存为: {save_path}")
    plt.show()

if __name__ == "__main__":
    TEST_IMAGE_PATHS = [
        r'.\data\VOCdevkit\VOC2012\JPEGImages\2007_000904.jpg',
        r'.\data\VOCdevkit\VOC2012\JPEGImages\2010_003239.jpg',
        r'.\data\VOCdevkit\VOC2012\JPEGImages\2008_008093.jpg'  
    ] 
    
    WEIGHT_PATH = 'latest_dino_pspnet.pth'
    
    visualize_paper_style(TEST_IMAGE_PATHS, WEIGHT_PATH, save_format='svg')