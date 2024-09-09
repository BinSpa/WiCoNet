import os
from tqdm import tqdm
import numpy as np
from PIL import Image

# 假设这是类别ID到颜色(R, G, B)的映射表
label2color = {
    0: [0, 0, 0], 
    1: [200,0,0], 
    2: [0,200,0], 
    3: [150,250,0], 
    4: [150,200,150], 
    5: [200,0,200], 
    6: [150,0,250], 
    7: [150,150,250], 
    8: [200,150,200], 
    9: [250,200,0], 
    10: [200,200,0],
    11: [0,0,200], 
    12: [250,0,150], 
    13: [0,150,200], 
    14: [0,200,250], 
    15: [150,200,250],
    16: [250,250,250], 
    17: [200,200,200], 
    18: [200,150,150], 
    19: [250,200,150], 
    20: [150,150,0], 
    21: [250,150,150], 
    22: [250,150,0], 
    23: [250,200,250], 
    24: [200,150,0]
}
'''
label2color = {
    0: [0, 0, 0], 
    1: [255, 0, 0], 
    2: [0, 255, 0], 
    3: [0, 255, 255], 
    4: [255, 255, 0], 
    5: [0, 0, 255], 
    6: [255, 0, 255], 
    7: [123, 123, 123]
}
'''
'''
label2color = {
    0: [0, 0, 0], 
    1: [255, 0, 0], 
    2: [0, 255, 0], 
    3: [0, 255, 255], 
    4: [255, 255, 0], 
    5: [0, 0, 255]
}
'''
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
    label_dir = "/data1/gyl/RS_DATASET/FBP/train/fbp_labels"
    output_dir = "/data1/gyl/RS_DATASET/FBP/train/rgb_fbp_labels"
    label_names = os.listdir(label_dir)
    for label_name in tqdm(label_names):
        label_path = os.path.join(label_dir, label_name)
        output_path = os.path.join(output_dir, label_name)
        convert_and_save(label_path, output_path)
