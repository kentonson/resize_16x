# created by kongds at 2024-09-10
# 这是一个主要用于将图像尺寸调整为16的倍数的节点
import numpy as np
import cv2
import torch

class ResizeToMultipleOf16:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "print_to_screen": (["enable", "disable"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    FUNCTION = "resize_image"
    CATEGORY = "Image Processing"

    def resize_image(self, image, print_to_screen="disable"):
        # print(f"--------------image-----------: {type(image)}")   # torch.Tensor
        # 检查输入图像是否为空
        if image is None or image.size == 0:
            raise ValueError("输入图像为空或无效")

        # 转换图像为NumPy数组（如果需要）
        if not isinstance(image, np.ndarray):
            image = np.array(image, dtype=np.float32)
        
        # 确保图像有3个维度（高度、宽度、通道）
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        batch, height, width, channels = image.shape
        # print(f"--------------shape-----------: {batch}, {height}, {width}, {channels}") # 如 (1, 512, 512, 3)
        
        # 计算新的宽度和高度（16的倍数）
        new_width = max(16, ((width + 15) // 16) * 16)
        new_height = max(16, ((height + 15) // 16) * 16)
        # print(f"new_width: {new_width}, new_height: {new_height}")
        
        # 创建一个新的空数组来存储调整大小后的图像
        image_resized = np.zeros((batch, new_height, new_width, channels), dtype=image.dtype)
        
        # 对每个批次中的图像进行调整大小
        for i in range(batch):
            image_resized[i] = cv2.resize(image[i], (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        image_resized = torch.from_numpy(image_resized)
        # batch, height, width, channels = image_resized.shape
        # print(f"--------------shape-----------: {batch}, {height}, {width}, {channels}") # 如 (1, 512, 512, 3)
        
        if print_to_screen == "enable":
            print(f"原始尺寸: {width}x{height}, 新尺寸: {new_width}x{new_height}")
        
        return (image_resized, new_width, new_height)

NODE_CLASS_MAPPINGS = {
    "ResizeToMultipleOf16": ResizeToMultipleOf16
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResizeToMultipleOf16": "Resize Image to Multiple of 16"
}