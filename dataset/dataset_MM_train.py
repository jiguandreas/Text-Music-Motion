import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, feat_bias = 5, unit_length = 4, codebook_size = 1024, tokenizer_name=None, music_data_path=None):
        
        self.max_length = 64
        self.pointer = 0
        self.dataset_name = dataset_name

        self.unit_length = unit_length
        self.mot_end_idx = codebook_size
        self.mot_pad_idx = codebook_size + 1
        
        # Set up dataset paths based on the dataset name
        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.joints_num = 22
            self.max_motion_length = 26 if unit_length == 8 else 51
        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.joints_num = 21
            self.max_motion_length = 26 if unit_length == 8 else 51

        split_file = pjoin(self.data_root, 'train.txt')

        # Loading dataset
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        data_dict = {}
        
        for name in tqdm(id_list):
            try:
                m_token_list = np.load(pjoin(self.data_root, tokenizer_name, '%s.npy' % name))

                # Load music sequence (size (1, 512)) for each sample
                music_seq = np.load(pjoin(music_data_path, f'{name}.npy'))

                # Store motion token sequences and corresponding music sequence
                data_dict[name] = {'m_token_list': m_token_list, 'music_seq': music_seq}
                new_name_list.append(name)

            except:
                pass

        self.data_dict = data_dict
        self.name_list = new_name_list

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        m_token_list, music_seq = data['m_token_list'], data['music_seq']
        
        # Select a random motion token sequence
        m_tokens = random.choice(m_token_list)

        # Randomly drop one token at the head or tail of the motion sequence
        coin = np.random.choice([False, False, True])
        if coin:
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]

        m_tokens_len = m_tokens.shape[0]

        # Pad the motion tokens to max length
        if m_tokens_len + 1 < self.max_motion_length:
            m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx, np.ones((self.max_motion_length - 1 - m_tokens_len), dtype=int) * self.mot_pad_idx], axis=0)
        else:
            m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx], axis=0)

        # Return music_seq instead of caption
        return music_seq, m_tokens.reshape(-1), m_tokens_len
'''
Music Sequence shape: torch.Size([32, 1, 512])
Motion Tokens shape: torch.Size([32, 51])
Motion Tokens Length: tensor([35, 42, 38, 35, 42, 39, 42, 38, 43, 47, 47, 43, 43, 43, 36, 36, 43, 47,
        47, 42, 39, 46, 46, 39, 36, 43, 36, 39, 37, 47, 42, 46])
'''
'''
如果 music_seq 的形状是 [32, 1, 512]，那么 music_seq.reshape(-1) 会将其展平（flatten）成一个 1D 向量，
形状为 [16384]（即 32 × 1 × 512 = 16384）。
'''

def DATALoader(dataset_name,
                batch_size, codebook_size, tokenizer_name, music_data_path, unit_length=4,
                num_workers=8): 

    train_loader = torch.utils.data.DataLoader(Text2MotionDataset(dataset_name, codebook_size=codebook_size, tokenizer_name=tokenizer_name, music_data_path=music_data_path, unit_length=unit_length),
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              drop_last=True)
    
    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

# # 假设你已经创建了 DataLoader 实例
# train_loader = DATALoader(dataset_name='t2m', 
#                           batch_size=32, 
#                           codebook_size=1024, 
#                           tokenizer_name='VQVAE', 
#                           music_data_path='/disk2/yuyusun/TM2M-GPT/dataset/HumanML3D/music', 
#                           unit_length=4, 
#                           num_workers=8)

# # 打印每次返回的 music_seq, m_tokens 和 m_tokens_len 的 shape
# for music_seq, m_tokens, m_tokens_len in train_loader:
#     print(f"Music Sequence shape: {music_seq.shape}")
#     print(f"Motion Tokens shape: {m_tokens.shape}")
#     print(f"Motion Tokens Length: {m_tokens_len}")
#     break  # 只查看一个批次，移除 `break` 可以查看更多批次

