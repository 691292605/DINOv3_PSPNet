import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
from torchvision import transforms
from model import DINOv3


def run_dino_diagnostic(img_path, weight_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    original_img = Image.open(img_path).convert('RGB')
    input_tensor = transform(original_img).unsqueeze(0).to(device)

    model = DINOv3(weight_path="D:\lab\enter_test\dinov3_vits16_pretrain.pth").to(device)

    with torch.no_grad():
        features = model(input_tensor)

    B, C, H, W = features.shape
    feat_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
    feat_norm = torch.nn.functional.normalize(feat_flat, p=2, dim=1)

    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(feat_norm.cpu().numpy())

    for i in range(3):
        low, high = np.percentile(pca_results[:, i], [1, 99]) 
        pca_results[:, i] = np.clip(pca_results[:, i], low, high)
        pca_results[:, i] = (pca_results[:, i] - low) / (high - low + 1e-8)

    pca_img = pca_results.reshape(H, W, 3)
    pca_img_resized = cv2.resize(pca_img, (512, 512), interpolation=cv2.INTER_NEAREST)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.resize(np.array(original_img), (1024, 1024)))
    plt.title("Original Image (1024x1024)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(pca_img_resized)
    plt.title("DINOv3 PCA Feature Map")
    plt.axis('off')

    plt.show()


if __name__ == "__main__":
    IMG_PATH = r'D:\lab\enter_test\data\VOCdevkit\VOC2012\JPEGImages\2007_000904.jpg'
    WEIGHT_PATH = r'D:\lab\enter_test\dinov3_vits16_pretrain.pth'

    run_dino_diagnostic(IMG_PATH, WEIGHT_PATH)


