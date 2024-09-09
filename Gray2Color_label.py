import os
from tqdm import tqdm
import numpy as np
from PIL import Image

# 假设这是类别ID到颜色(R, G, B)的映射表
label2color = {
    0: [0, 0, 0], 
    1: [255, 0, 0], 
    2: [0, 255, 0], 
    3: [0, 255, 255], 
    4: [255, 255, 0], 
    5: [0, 0, 255]
}

def label2rgb(label_img, label2color):
    """
    将灰度的语义分割标签图像转换为 RGB 格式
    :param label_img: 输入的灰度图像 (H, W)
    :param label2color: 类别ID到RGB颜色的映射表
    :return: RGB图像 (H, W, 3)
    """
    # 获取图像的高和宽
    h, w = label_img.shape

    # 创建一个空的 RGB 图像
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)

    # 遍历每个类别ID和对应的颜色
    for label, color in label2color.items():
        rgb_img[label_img == label] = color

    return rgb_img

def convert_and_save(label_path, output_path):
    """
    读取灰度标签图像，转换为RGB格式，并保存结果
    :param label_path: 输入的灰度图像路径
    :param output_path: 输出RGB图像路径
    """
    # 读取灰度图像
    label_img = np.array(Image.open(label_path).convert('L'))  # 'L' 表示灰度模式

    # 转换为 RGB 格式
    rgb_img = label2rgb(label_img, label2color)

    # 保存结果
    rgb_image_pil = Image.fromarray(rgb_img)
    rgb_image_pil.save(output_path)

'''
# 使用示例
label_path = 'path_to_grayscale_label.png'  # 输入灰度图像的路径
output_path = 'path_to_output_rgb_image.png'  # 输出RGB图像的路径
convert_and_save(label_path, output_path)
'''
if __name__ == "__main__":
    label_dir = "/data1/gyl/RS_DATASET/FBP/train/gid_labels"
    output_dir = "/data1/gyl/RS_DATASET/FBP/train/rgb_gid_labels"
    label_names = os.listdir(label_dir)
    for label_name in tqdm(label_names):
        label_path = os.path.join(label_dir, label_name)
        output_path = os.path.join(output_dir, label_name)
        convert_and_save(label_path, output_path)
