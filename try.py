
# 这是查看motion的size，text框架的。
'''
from dataset.dataset_VQ import DATALoader  # 替换成你的脚本文件名
import torch        

# 设定参数
dataset_name = 't2m'  # 或 'kit'
batch_size = 1  # 设为 1，方便查看单个 motion
window_size = 64

# 获取数据加载器
train_loader = DATALoader(dataset_name, batch_size, window_size=window_size)

# 获取一个 batch
for motion in train_loader:
    print("Motion size:", motion.shape)  # 打印 motion 的 shape
    break  # 只看一个 batch
'''
'''

import json
import sys
from collections import defaultdict

def analyze_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        sys.exit(1)

    # 统计信息
    analysis = {
        'total_size_bytes': len(json.dumps(data).encode('utf-8')),
        'structure': defaultdict(lambda: {'type': None, 'count': 0, 'size': 0, 'meaning': 'TODO'})
    }

    def _analyze(obj, path=''):
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                _analyze(value, new_path)
                # 记录字段信息
                analysis['structure'][new_path]['type'] = type(value).__name__
                analysis['structure'][new_path]['count'] += 1
                analysis['structure'][new_path]['size'] = len(json.dumps(value).encode('utf-8'))
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                _analyze(item, f"{path}[{i}]")
        else:
            # 基本数据类型
            if path not in analysis['structure']:
                analysis['structure'][path] = {
                    'type': type(obj).__name__,
                    'count': 1,
                    'size': len(json.dumps(obj).encode('utf-8')),
                    'meaning': 'TODO'
                }

    _analyze(data)
    return dict(analysis)

if __name__ == '__main__':
    file_path = "/disk2/yuyusun/TM2M-GPT/music2motion_data/aistpp_train_wav/gBR_sBM_cAll_d04_mBR1_ch03.json"
    result = analyze_json(file_path)

    # 打印结果
    print(f"\n{'='*50}")
    print(f"JSON File Analysis: {file_path}")
    print(f"Total Size: {result['total_size_bytes']} bytes")
    print(f"Structure Overview:")
    
    # for field, info in result['structure'].items():
    #     print(f"\nField: {field}")
    #     print(f"  Type: {info['type']}")
    #     print(f"  Count: {info['count']}")
    #     print(f"  Size: {info['size']} bytes")
    #     print(f"  Meaning: {info['meaning']} (需要人工补充说明)")
    # print('='*50)
'''

'''
# 对dance的motion_size进行降采样时间帧至64
import os
import json
import numpy as np

# 定义降采样函数
def downsample_dance(dance_array, target_length=64):
    """
    使用均匀降采样将 dance_array 从 1000 帧降到 64 帧。
    :param dance_array: 原始舞蹈动作数据 (1000, 72)
    :param target_length: 目标时间步长
    :return: 降采样后的舞蹈动作数据 (64, 72)
    """
    original_length = dance_array.shape[0]
    # 计算每个采样步长的间隔
    indices = np.linspace(0, original_length - 1, target_length).astype(int)
    # 使用这些索引从原始数据中采样
    downsampled_dance = dance_array[indices]
    return downsampled_dance

# 读取 JSON 文件并处理
def process_aist_json(json_dir, output_dir, target_length=64):
    """
    处理指定目录下的所有 JSON 文件，对 dance_array 进行降采样。
    :param json_dir: 存放 AIST 数据集 JSON 文件的目录
    :param output_dir: 输出目录，用于保存降采样后的 .npy 文件
    :param target_length: 目标时间步长（默认为 64）
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历 json_dir 中的所有文件
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):  # 只处理 JSON 文件
            json_path = os.path.join(json_dir, filename)
            try:
                # 读取 JSON 文件
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "dance_array" in data:
                        dance_array = np.array(data["dance_array"])  # shape (1000, 72)
                        
                        # 执行降采样
                        downsampled_dance = downsample_dance(dance_array, target_length)
                        
                        # 保存降采样后的数据为 .npy 文件
                        output_filename = os.path.splitext(filename)[0] + ".npy"
                        output_path = os.path.join(output_dir, output_filename)
                        np.save(output_path, downsampled_dance)
                        print(f"已处理 {filename}, 保存为 {output_filename}")
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")

# 设置目录路径
json_dir = "/disk2/yuyusun/TM2M-GPT/music2motion_data/aistpp_train_wav"
output_dir = "/disk2/yuyusun/TM2M-GPT/music2motion_data/new_joint_vecs"

# 执行处理
process_aist_json(json_dir, output_dir, target_length=64)

'''


'''
# 查看单个npy文件的size

import numpy as np
import os

output_dir = "/disk2/yuyusun/TM2M-GPT/aist_dance_npy/gBR_sBM_cAll_d04_mBR1_ch03.npy"


dance_data = np.load(output_dir)
        
        # 打印每个文件的 shape
print(f"npy文件 的 size: {dance_data.shape}")
'''


'''
# 可视化motion为火柴人

import os
import numpy as np
import torch
import imageio
from visualization.plot_3d_global import plot_3d_motion

# 指定 motion 数据路径
motion_data_dir = "/disk2/yuyusun/TM2M-GPT/dataset/HumanML3D/new_joint_vecs"
output_dir = "./motion_visualizations"  # 存放可视化结果的目录
os.makedirs(output_dir, exist_ok=True)

# 获取所有 motion 文件
motion_files = [f for f in os.listdir(motion_data_dir) if f.endswith('.npy')]

# 选择一个或多个 motion 进行可视化
for i, file in enumerate(motion_files[:5]):  # 这里只可视化前 5 个 motion，可调整
    motion_path = os.path.join(motion_data_dir, file)
    
    # 读取 motion 数据
    motion_data = np.load(motion_path)  # shape: (T, J * 3)
    
    # 转换为 PyTorch 张量
    motion_tensor = torch.tensor(motion_data, dtype=torch.float32)
    
    # 设置可视化参数
    output_gif = os.path.join(output_dir, f"{file.replace('.npy', '.gif')}")
    
    # 生成并保存可视化结果
    print(f"Processing {file}...")
    plot_3d_motion([motion_tensor, output_gif, file])
    
print("Motion visualization completed. Check the output GIFs in:", output_dir)
'''

'''
# 用于从AIST数据集的JSON文件中提取dance_array，并以JSON文件中的id作为文件名保存为.npy文件

import os
import json
import numpy as np
from tqdm import tqdm

# 配置路径
input_dir = '/disk2/yuyusun/TM2M-GPT/music2motion_data/aistpp_train_wav'
output_dir = './aist_dance_npy'  # 保存npy文件的目录
os.makedirs(output_dir, exist_ok=True)

def process_json_files(input_dir, output_dir):
    """处理目录下所有JSON文件，提取dance_array保存为npy"""
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        try:
            # 读取JSON文件
            with open(os.path.join(input_dir, json_file), 'r') as f:
                data = json.load(f)
            
            # 提取id和dance_array
            file_id = data.get('id', os.path.splitext(json_file)[0])  # 如果无id字段则用文件名
            dance_array = np.array(data['dance_array'])  # 确保键名正确
            
            # 保存为npy
            output_path = os.path.join(output_dir, f"{file_id}.npy")
            np.save(output_path, dance_array)
            
        except Exception as e:
            print(f"\nError processing {json_file}: {e}")
            continue

if __name__ == '__main__':
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    process_json_files(input_dir, output_dir)
    print("\nAll files processed successfully!")
'''






'''
# 可视化motion的npy文件
import numpy as np
from matplotlib import pyplot as plt
import imageio
from visualization.plot_3d_global import plot_3d_motion,draw_to_batch

# 加载数据
data = np.load('/disk2/yuyusun/TM2M-GPT/dataset/HumanML3D/new_joint_vecs/000000.npy')
print(data.shape)  # 应为 (帧数, 263) 或 (帧数, 关节数, 3)

# 如果是263维特征，需转换为关节坐标
if data.shape[-1] == 263:  # HumanML3D格式
    joints = data[..., :22*3].reshape(len(data), 22, 3)  # 前66维是22个关节的XYZ坐标
else:
    joints = data  # 假设已是 (帧数, 关节数, 3)


# 单文件可视化
output_gif = "000000_motion.gif"
plot_3d_motion(args=(joints, output_gif, "Motion 000000"))

# 或用批量函数（即使只有1个样本）
draw_to_batch(
    smpl_joints_batch=joints[np.newaxis],  # 添加批次维度
    title_batch=["Motion 000000"],
    outname=[output_gif]
)

print(f"动图已保存至: {output_gif}")
'''



# from os.path import join as pjoin

# from common.skeleton import Skeleton
# import numpy as np
# import os
# from common.quaternion import *
# from paramUtil import *

# import torch
# from tqdm import tqdm
# import os



# def uniform_skeleton(positions, target_offset):
#     src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
#     src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
#     src_offset = src_offset.numpy()
#     tgt_offset = target_offset.numpy()
#     # print(src_offset)
#     # print(tgt_offset)
#     '''Calculate Scale Ratio as the ratio of legs'''
#     src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
#     tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

#     scale_rt = tgt_leg_len / src_leg_len
#     # print(scale_rt)
#     src_root_pos = positions[:, 0]
#     tgt_root_pos = src_root_pos * scale_rt

#     '''Inverse Kinematics'''
#     quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
#     # print(quat_params.shape)

#     '''Forward Kinematics'''
#     src_skel.set_offset(target_offset)
#     new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
#     return new_joints


# def process_file(positions, feet_thre):
#     # (seq_len, joints_num, 3)
#     #     '''Down Sample'''
#     #     positions = positions[::ds_num]

#     '''Uniform Skeleton'''
#     positions = uniform_skeleton(positions, tgt_offsets)

#     '''Put on Floor'''
#     floor_height = positions.min(axis=0).min(axis=0)[1]
#     positions[:, :, 1] -= floor_height
#     #     print(floor_height)

#     #     plot_3d_motion("./positions_1.mp4", kinematic_chain, positions, 'title', fps=20)

#     '''XZ at origin'''
#     root_pos_init = positions[0]
#     root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
#     positions = positions - root_pose_init_xz

   

#     '''All initially face Z+'''
#     r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
#     across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
#     across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
#     across = across1 + across2
#     across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

#     # forward (3,), rotate around y-axis
#     forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
#     # forward (3,)
#     forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

#     #     print(forward_init)

#     target = np.array([[0, 0, 1]])
#     root_quat_init = qbetween_np(forward_init, target)
#     root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

#     positions_b = positions.copy()

#     positions = qrot_np(root_quat_init, positions)

#     #     plot_3d_motion("./positions_2.mp4", kinematic_chain, positions, 'title', fps=20)

#     '''New ground truth positions'''
#     global_positions = positions.copy()



#     """ Get Foot Contacts """

#     def foot_detect(positions, thres):
#         velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

#         feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
#         feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
#         feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
#         #     feet_l_h = positions[:-1,fid_l,1]
#         #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
#         feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

#         feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
#         feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
#         feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
#         #     feet_r_h = positions[:-1,fid_r,1]
#         #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
#         feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
#         return feet_l, feet_r
#     #
#     feet_l, feet_r = foot_detect(positions, feet_thre)
#     # feet_l, feet_r = foot_detect(positions, 0.002)

#     '''Quaternion and Cartesian representation'''
#     r_rot = None

#     def get_rifke(positions):
#         '''Local pose'''
#         positions[..., 0] -= positions[:, 0:1, 0]
#         positions[..., 2] -= positions[:, 0:1, 2]
#         '''All pose face Z+'''
#         positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
#         return positions

#     def get_quaternion(positions):
#         skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
#         # (seq_len, joints_num, 4)
#         quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

#         '''Fix Quaternion Discontinuity'''
#         quat_params = qfix(quat_params)
#         # (seq_len, 4)
#         r_rot = quat_params[:, 0].copy()
#         #     print(r_rot[0])
#         '''Root Linear Velocity'''
#         # (seq_len - 1, 3)
#         velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
#         #     print(r_rot.shape, velocity.shape)
#         velocity = qrot_np(r_rot[1:], velocity)
#         '''Root Angular Velocity'''
#         # (seq_len - 1, 4)
#         r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
#         quat_params[1:, 0] = r_velocity
#         # (seq_len, joints_num, 4)
#         return quat_params, r_velocity, velocity, r_rot

#     def get_cont6d_params(positions):
#         skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
#         # (seq_len, joints_num, 4)
#         quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

#         '''Quaternion to continuous 6D'''
#         cont_6d_params = quaternion_to_cont6d_np(quat_params)
#         # (seq_len, 4)
#         r_rot = quat_params[:, 0].copy()
#         #     print(r_rot[0])
#         '''Root Linear Velocity'''
#         # (seq_len - 1, 3)
#         velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
#         #     print(r_rot.shape, velocity.shape)
#         velocity = qrot_np(r_rot[1:], velocity)
#         '''Root Angular Velocity'''
#         # (seq_len - 1, 4)
#         r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
#         # (seq_len, joints_num, 4)
#         return cont_6d_params, r_velocity, velocity, r_rot

#     cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
#     positions = get_rifke(positions)


#     '''Root height'''
#     root_y = positions[:, 0, 1:2]

#     '''Root rotation and linear velocity'''
#     # (seq_len-1, 1) rotation velocity along y-axis
#     # (seq_len-1, 2) linear velovity on xz plane
#     r_velocity = np.arcsin(r_velocity[:, 2:3])
#     l_velocity = velocity[:, [0, 2]]
#     #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
#     root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

#     '''Get Joint Rotation Representation'''
#     # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
#     rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

#     '''Get Joint Rotation Invariant Position Represention'''
#     # (seq_len, (joints_num-1)*3) local joint position
#     ric_data = positions[:, 1:].reshape(len(positions), -1)

#     '''Get Joint Velocity Representation'''
#     # (seq_len-1, joints_num*3)
#     local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
#                         global_positions[1:] - global_positions[:-1])
#     local_vel = local_vel.reshape(len(local_vel), -1)

#     data = root_data
#     data = np.concatenate([data, ric_data[:-1]], axis=-1)
#     data = np.concatenate([data, rot_data[:-1]], axis=-1)
#     #     print(data.shape, local_vel.shape)
#     data = np.concatenate([data, local_vel], axis=-1)
#     data = np.concatenate([data, feet_l, feet_r], axis=-1)

#     return data, global_positions, positions, l_velocity

# def recover_root_rot_pos(data):
#     rot_vel = data[..., 0]
#     r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
#     '''Get Y-axis rotation from rotation velocity'''
#     r_rot_ang[..., 1:] = rot_vel[..., :-1]
#     r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

#     r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
#     r_rot_quat[..., 0] = torch.cos(r_rot_ang)
#     r_rot_quat[..., 2] = torch.sin(r_rot_ang)

#     r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
#     r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
#     '''Add Y-axis rotation to root position'''
#     r_pos = qrot(qinv(r_rot_quat), r_pos)

#     r_pos = torch.cumsum(r_pos, dim=-2)

#     r_pos[..., 1] = data[..., 3]
#     return r_rot_quat, r_pos


# def recover_from_rot(data, joints_num, skeleton):
#     r_rot_quat, r_pos = recover_root_rot_pos(data)

#     r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

#     start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
#     end_indx = start_indx + (joints_num - 1) * 6
#     cont6d_params = data[..., start_indx:end_indx]
#     #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
#     cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
#     cont6d_params = cont6d_params.view(-1, joints_num, 6)

#     positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

#     return positions


# def recover_from_ric(data, joints_num):
#     r_rot_quat, r_pos = recover_root_rot_pos(data)
#     positions = data[..., 4:(joints_num - 1) * 3 + 4]
#     positions = positions.view(positions.shape[:-1] + (-1, 3))

#     '''Add Y-axis rotation to local joints'''
#     positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

#     '''Add root XZ to joints'''
#     positions[..., 0] += r_pos[..., 0:1]
#     positions[..., 2] += r_pos[..., 2:3]

#     '''Concate root and joints'''
#     positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

#     return positions




# if __name__ == "__main__":
#     # 配置参数
#     example_id = "000021"
#     example_path = '/disk2/yuyusun/TM2M-GPT/example/000021.npy'  # 新的示例文件路径
#     l_idx1, l_idx2 = 5, 8
#     fid_r, fid_l = [8, 11], [7, 10]
#     face_joint_indx = [2, 1, 17, 16]
#     r_hip, l_hip = 2, 1
#     joints_num = 22
    
#     # 目录配置
#     data_dir = '/disk2/yuyusun/TM2M-GPT/test/'
#     save_dir1 = '/disk2/yuyusun/TM2M-GPT/test/new_joints/'
#     save_dir2 = '/disk2/yuyusun/TM2M-GPT/test/new_joint_vecs/'
    
#     # 创建输出目录
#     os.makedirs(save_dir1, exist_ok=True)
#     os.makedirs(save_dir2, exist_ok=True)

#     # 检查示例文件是否存在
#     if not os.path.exists(example_path):
#         raise FileNotFoundError(f"示例文件 {example_path} 不存在！请检查路径。")

#     # 初始化骨架参数
#     n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
#     kinematic_chain = t2m_kinematic_chain

#     # 加载示例数据并计算目标骨架偏移量
#     example_data = np.load(example_path)  # 直接使用新路径
#     # print(example_data.shape)
#     example_data = example_data.reshape(len(example_data), -1, 3)
#     example_data = torch.from_numpy(example_data)
#     tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
#     tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])
#     print(tgt_offsets.shape)

#     source_list = os.listdir(data_dir)
#     frame_num = 0
#     for source_file in tqdm(source_list):
#         source_data = np.load(os.path.join(data_dir, source_file))
#         T, _ = source_data.shape
#         source_data = source_data.reshape(T, -1, 3)[:, :joints_num]
#         print(source_data.shape)
#         data, ground_positions, positions, l_velocity = process_file(source_data, 0.002)
#         # print(data.shape, ground_positions.shape, positions.shape, l_velocity.shape)
#         rec_ric_data = recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), joints_num)
#         np.save(pjoin(save_dir1, source_file), rec_ric_data.squeeze().numpy())
#         np.save(pjoin(save_dir2, source_file), data)
#         frame_num += data.shape[0]
#         # except Exception as e:
#         #     print(source_file)
#         #     print(e)
# #         print(source_file)
# #         break

#     print('Total clips: %d, Frames: %d, Duration: %fm' %
#           (len(source_list), frame_num, frame_num / 20 / 60))


'''
import os
import numpy as np
from visualization.plot_3d_global_yuyu import plot_3d_motion

# 输入/输出目录
input_dir = "/disk2/yuyusun/TM2M-GPT/music2motion_data/new_joints"
output_dir = "/disk2/yuyusun/TM2M-GPT/music2motion_data/visualization"
os.makedirs(output_dir, exist_ok=True)

# 遍历所有 .npy 文件
for filename in os.listdir(input_dir):
    if filename.endswith(".npy"):
        input_path = os.path.join(input_dir, filename)
        motion_data = np.load(input_path)  # shape: (T, J, 3)

        # 构建输出 gif 文件名
        name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{name}.gif")

        # 设置标题
        title = f"Motion: {name}"

        # 调用可视化函数
        args = (motion_data, output_path, title)
        print(f"Rendering {filename} -> {output_path}")
        plot_3d_motion(args)

print("全部可视化完成 ✅")
'''


'''
import json
import os

json_dir = "/disk2/yuyusun/TM2M-GPT/music2motion_data/aistpp_music_feat_7.5fps"  # 替换为你的 JSON 文件目录

for filename in os.listdir(json_dir):
    if filename.endswith(".json"):
        json_path = os.path.join(json_dir, filename)
        
        # 加载 JSON 文件
        with open(json_path, 'r') as f:
            data = json.load(f)  # 解析为 Python 对象（dict/list）
        
        # 打印文件名和结构信息
        print(f"\n{filename} 的结构:")
        
        # 情况1：如果是字典，打印键和值的类型/长度
        if isinstance(data, dict):
            print(f"  - 顶层键数量: {len(data)}")
            for key, value in data.items():
                print(f"  - 键 '{key}': 类型={type(value).__name__}", end="")
                if isinstance(value, (list, dict)):
                    print(f", 长度/键数={len(value)}")
                else:
                    print()  # 简单类型（如 str/int/float）
        
        # 情况2：如果是列表，打印长度和首元素类型
        elif isinstance(data, list):
            print(f"  - 列表长度: {len(data)}")
            if len(data) > 0:
                print(f"  - 首元素类型: {type(data[0]).__name__}")
        
        # 其他情况（如单个值）
        else:
            print(f"  - 数据类型: {type(data).__name__}")
'''


'''
import os
import json
import numpy as np

input_dir = "/disk2/yuyusun/TM2M-GPT/music2motion_data/aistpp_train_wav"  # 修改为你的实际路径

for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        json_path = os.path.join(input_dir, filename)
        
        with open(json_path, 'r') as f:
            data = json.load(f)

        music_array = np.array(data['music_array'])   # 转成 numpy 查看 shape
        dance_array = np.array(data['dance_array'])
        id_value = data['id']

        print(f"文件: {filename}")
        print(f" - ID: {id_value} （长度: {len(id_value)}）")
        print(f" - music_array.shape: {music_array.shape}")
        print(f" - dance_array.shape: {dance_array.shape}")
        print("-" * 50)

'''



'''
import os

def count_files(path):
    """统计指定路径下的文件数量"""
    file_count = 0
    
    # 遍历路径下的所有文件和子目录
    for root, dirs, files in os.walk(path):
        file_count += len(files)
    
    return file_count

# 获取用户输入的路径
path = input("请输入要统计的路径: ")

# 检查路径是否存在
if os.path.exists(path):
    total_files = count_files(path)
    print(f"路径 '{path}' 下共有 {total_files} 个文件")
else:
    print("错误: 指定的路径不存在")
'''



'''
# 用于从AIST数据集的JSON文件中提取music_array，并以JSON文件中的id作为文件名保存为.npy文件

import os
import json
import numpy as np
from tqdm import tqdm

# 配置路径
input_dir = '/disk2/yuyusun/TM2M-GPT/music2motion_data/aistpp_train_wav'
output_dir = '/disk2/yuyusun/TM2M-GPT/music2motion_data/test/music'  # 保存npy文件的目录
os.makedirs(output_dir, exist_ok=True)

def process_json_files(input_dir, output_dir):
    """处理目录下所有JSON文件，提取music_array保存为npy"""
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        try:
            # 读取JSON文件
            with open(os.path.join(input_dir, json_file), 'r') as f:
                data = json.load(f)
            
            # 提取id和music_array
            file_id = data.get('id', os.path.splitext(json_file)[0])  # 如果无id字段则用文件名
            music_array = np.array(data['music_array'])  # 确保键名正确
            
            # 保存为npy
            output_path = os.path.join(output_dir, f"{file_id}.npy")
            np.save(output_path, music_array)
            
        except Exception as e:
            print(f"\nError processing {json_file}: {e}")
            continue

if __name__ == '__main__':
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    process_json_files(input_dir, output_dir)
    print("\nAll files processed successfully!")
'''



# 查看npy文件的size

import numpy as np
import os

output_dir = "/disk2/yuyusun/UDE-master/sample_data/a2m"

# 遍历输出目录中的所有 .npy 文件
for filename in os.listdir(output_dir):
    if filename.endswith(".npy"):  # 只处理 .npy 文件
        npy_path = os.path.join(output_dir, filename)
        
        # 加载 .npy 文件
        dance_data = np.load(npy_path)
        
        # 打印每个文件的 shape
        print(f"{filename} 的 size: {dance_data.shape}")


'''
# 把AIST的数据集中的id写入train.txt这个文件里面
import os
import json

# 指定 JSON 文件所在目录
json_dir = "/disk2/yuyusun/TM2M-GPT/music2motion_data/aistpp_test_full_wav"
# 指定输出文件
output_file = "/disk2/yuyusun/TM2M-GPT/music2motion_data/test/test_motion_id.txt"

# 存储所有 ID 的列表
id_list = []

# 遍历 JSON 目录中的所有文件
for filename in os.listdir(json_dir):
    if filename.endswith(".json"):  # 只处理 JSON 文件
        json_path = os.path.join(json_dir, filename)
        
        try:
            # 读取 JSON 文件
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "id" in data:
                    id_list.append(data["id"])
        except Exception as e:
            print(f"读取 {json_path} 时出错: {e}")

# 去重并排序（可选）
id_list = sorted(set(id_list))

# 将 ID 写入目标文件
with open(output_file, "w", encoding="utf-8") as f:
    for motion_id in id_list:
        f.write(motion_id + "\n")

print(f"提取完成，共写入 {len(id_list)} 个 ID 到 {output_file}")
'''


'''
# 对原来的60fps数据进行降采样到20fps
import os
import numpy as np
from scipy.interpolate import interp1d

def resample_motion_linear(motion_array, original_fps=60, target_fps=20):
    length = motion_array.shape[0]
    feature_dim = motion_array.shape[1]
    
    duration = length / original_fps
    new_length = int(duration * target_fps)

    original_times = np.linspace(0, duration, length)
    target_times = np.linspace(0, duration, new_length)

    # 插值每一个特征维度
    interpolated = np.zeros((new_length, feature_dim))
    for dim in range(feature_dim):
        interp_fn = interp1d(original_times, motion_array[:, dim], kind='linear', fill_value="extrapolate")
        interpolated[:, dim] = interp_fn(target_times)

    return interpolated

# 输入输出路径
input_dir = '/disk2/yuyusun/TM2M-GPT/music2motion_data/test/motion'
output_dir = '/disk2/yuyusun/TM2M-GPT/music2motion_data/test/fpssame_dance'

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 遍历所有 .npy 文件进行重采样
for filename in os.listdir(input_dir):
    if filename.endswith('.npy'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 加载 motion 数据
        try:
            motion_data = np.load(input_path)
            if len(motion_data.shape) != 2:
                print(f"跳过非2D数据: {filename}")
                continue

            # 使用线性插值降采样
            motion_resampled = resample_motion_linear(motion_data, original_fps=60, target_fps=20)

            # 保存处理后的数据
            np.save(output_path, motion_resampled)
            print(f"[✔] 处理完成: {filename} -> {motion_resampled.shape}")
        except Exception as e:
            print(f"[✘] 处理失败: {filename}, 错误信息: {e}")
'''
'''
import os

# 路径设置
id_file = "/disk2/yuyusun/TM2M-GPT/music2motion_data/test/music_test_id.txt"
output_dir = "/disk2/yuyusun/TM2M-GPT/music2motion_data/test/music_test_caption"
caption_text = "this is music test data caption"

# 创建输出文件夹（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 读取 ID 列表
with open(id_file, "r") as f:
    ids = [line.strip() for line in f if line.strip()]

# 为每个 ID 创建对应的 caption 文件
for id_str in ids:
    caption_file = os.path.join(output_dir, f"{id_str}.txt")
    with open(caption_file, "w") as f:
        f.write(caption_text)

print(f"完成：已为 {len(ids)} 个 ID 创建 caption 文件。")
'''
