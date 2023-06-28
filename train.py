from PIL import Image
from torchvision import transforms
import os
import torch
from transformer_v3_1 import Semi_Transformer

# 图像转换操作
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 重新调整图像大小，ResNet50的输入大小通常是224x224
    transforms.ToTensor(),  # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 对应RGB三个通道，将其归一化至[-1,1]
])

device = torch.device("cuda")

# 获取所有的视频文件夹
video_dirs = [os.path.join("/home/zheng/VATN/clip_frame", d) for d in
              os.listdir("/home/zheng/VATN/clip_frame")]
print(f'There are total {len(video_dirs)} files in clip_frame, they are {[d for d in os.listdir("/home/zheng/VATN/clip_frame")]} ')
# 遍历每个视频文件夹
for video_dir in video_dirs:
    # 获取视频中的所有动作文件夹
    action_dirs = [os.path.join(video_dir, d) for d in os.listdir(video_dir)]
    print(f"In this file {video_dir}, there are {len(action_dirs)} actions, they are {[d for d in os.listdir(video_dir)]}")
    # 遍历每个动作文件夹
    for action_dir in action_dirs:
        # 获取动作文件夹中的所有帧
        frame_files = [os.path.join(action_dir, f) for f in os.listdir(action_dir)]

        # 只保留文件名（去掉扩展名）是数字的文件
        frame_files = [f for f in frame_files if os.path.splitext(os.path.basename(f))[0].isdigit()]
        # print(frame_files)
        # 按照文件名（即帧序号）进行排序
        frame_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        # 加载每个帧并进行预处理
        frames = [transform(Image.open(frame_file)) for frame_file in frame_files]

        # 堆叠帧以形成视频剪辑，并增加一个额外的维度以表示批次大小
        clip = torch.stack(frames).unsqueeze(0).to(device)  # 结果的尺寸将是(1, T, C, H, W)，其中T是帧数，C是通道数（即3），H和W是图像的高度和宽度
        print(f"clip.size:{clip.size()}")

        # 这时候你可以将clip输入到你的模型中进行处理
        model = Semi_Transformer(num_classes=len(action_dirs), seq_len=clip.shape[1]).to(device)  # 替换成你的类别数和序列长度
        output = model(clip)