这是一个 ComfyUI 节点，用于Flux ControlNet节点的图像尺寸自适应调整，由于Flux ControlNet条件输入图像尺寸必须为 16 的倍数，此节点将输入图像就近缩放至 16x，与原图像最多相差 15 个像素宽度。

![](./assets/resize_16x.png)