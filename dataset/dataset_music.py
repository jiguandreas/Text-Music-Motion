# import torch
# def process_music_npy_to_seq(music_path, config):
#     music_array = np.load(music_path)  # shape [T, C]
#     ds_rate = config.music_ds_rate if hasattr(config, 'music_ds_rate') else config.ds_rate
#     relative_rate = config.music_relative_rate if hasattr(config, 'music_relative_rate') else config.ds_rate

#     # Step 1: 截取有效长度
#     T_clip = config.structure_generate.n_music // ds_rate
#     music_array = music_array[:T_clip]  # shape [T_clip, C]

#     # Step 2: 转 Tensor，拼接维度
#     music_tensor = torch.from_numpy(music_array).float()  # [T_clip, C]
#     b, t, c = 1, *music_tensor.shape
#     music_tensor = music_tensor.view(b, t // ds_rate, c * ds_rate)  # [1, T_down, C_up]

#     # Step 3: 归一化（可选）
#     if hasattr(config, 'music_normalize') and config.music_normalize:
#         music_tensor = music_tensor / (t // ds_rate * 1.0)

#     # Step 4: 错位（可选）
#     music_tensor = music_tensor[:, ds_rate // relative_rate:]  # [1, T', C']

#     return music_tensor

'''
import os
import glob
import numpy as np
import torch

def process_music_npy_to_seq(music_array, ds_rate=5, relative_rate=5, n_music=200, normalize=False):
    """
    将单个 music_array 处理为 Transformer 可输入的 music_seq。
    """
    T_clip = n_music // ds_rate
    music_array = music_array[:T_clip]  # 裁剪长度

    music_tensor = torch.from_numpy(music_array).float()  # [T_clip, C]
    t, c = music_tensor.shape
    b = 1
    music_tensor = music_tensor.view(b, t // ds_rate, c * ds_rate)  # [1, T', C']

    if normalize:
        music_tensor = music_tensor / (t // ds_rate * 1.0)

    music_tensor = music_tensor[:, ds_rate // relative_rate:]  # 时间错位对齐
    return music_tensor.squeeze(0).numpy()  # [T', C'] → numpy 保存

def batch_process_music_folder(input_dir, output_dir,
                                ds_rate=5, relative_rate=5,
                                n_music=200, normalize=False):
    os.makedirs(output_dir, exist_ok=True)
    music_paths = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    
    for path in music_paths:
        music_array = np.load(path)
        music_seq = process_music_npy_to_seq(music_array,
                                             ds_rate=ds_rate,
                                             relative_rate=relative_rate,
                                             n_music=n_music,
                                             normalize=normalize)
        
        # 保持文件名一致
        filename = os.path.basename(path)
        save_path = os.path.join(output_dir, filename)
        np.save(save_path, music_seq)
        print(f"Processed and saved: {save_path}")

if __name__ == "__main__":
    input_dir = "/disk2/yuyusun/TM2M-GPT/music2motion_data/music"
    output_dir = "/disk2/yuyusun/TM2M-GPT/dataset/HumanML3D/music"
    
    # 设置参数：这些参数应与你训练 Transformer 时保持一致
    batch_process_music_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        ds_rate=5,
        relative_rate=5,
        n_music=200,
        normalize=False
    )



'''
import os
import glob
import numpy as np
import torch
import torch.nn as nn

# 定义一个简单的 MLP 模型来将 360 维度映射到 512
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)

def process_music_npy_to_seq(music_array, mlp_model):
    """
    将单个 music_array 处理为 Transformer 可输入的 music_seq。
    """
    music_tensor = torch.from_numpy(music_array).float()  # [7, 360]
    
    # 使用 MLP 转换为 512 维
    music_tensor = mlp_model(music_tensor)  # [7, 512]
    
    # 取平均作为总 token
    music_mean = music_tensor.mean(dim=0, keepdim=True)  # [1, 512]

    return music_mean.detach().numpy()  # [1, 512] -> numpy 保存

def batch_process_music_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    music_paths = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    
    # 初始化 MLP 模型，将 360 维度映射为 512
    mlp_model = MLP(input_dim=2190, output_dim=512)
    mlp_model.eval()  # 设置为评估模式

    for path in music_paths:
        music_array = np.load(path)  # [7, 360]
        
        # 处理并转换
        music_seq = process_music_npy_to_seq(music_array, mlp_model)
        
        # 保持文件名一致
        filename = os.path.basename(path)
        save_path = os.path.join(output_dir, filename)
        np.save(save_path, music_seq)  # 保存为 [1, 512] 的数据
        print(f"Processed and saved: {save_path}")

if __name__ == "__main__":
    input_dir = "/disk2/yuyusun/TM2M-GPT/music2motion_data/test/musicencoder_music"
    output_dir = "/disk2/yuyusun/TM2M-GPT/music2motion_data/test/final_music"
    
    # 处理并保存
    batch_process_music_folder(input_dir=input_dir, output_dir=output_dir)


