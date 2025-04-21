import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import os
import random
import codecs as cs
from tqdm import tqdm

import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate

def collate_fn(batch):
    music_batch = [item for item in batch if item[8] == True]
    text_batch = [item for item in batch if item[8] == False]

    def custom_collate(sub_batch):
        word_embeddings = torch.stack([torch.tensor(item[0]) for item in sub_batch])
        pos_one_hots = torch.stack([torch.tensor(item[1]) for item in sub_batch])
        captions = [item[2] for item in sub_batch]
        sent_lens = torch.tensor([item[3] for item in sub_batch])
        motions = torch.stack([torch.tensor(item[4]) for item in sub_batch])
        motion_lens = torch.tensor([item[5] for item in sub_batch])
        tokens_str = [item[6] for item in sub_batch]
        names = [item[7] for item in sub_batch]
        return {
            'word_embeddings': word_embeddings,
            'pos_one_hots': pos_one_hots,
            'captions': captions,
            'sent_lens': sent_lens,
            'motions': motions,
            'motion_lens': motion_lens,
            'tokens_str': tokens_str,
            'names': names
        }

    mixed_batch = music_batch + text_batch
    return custom_collate(mixed_batch)


class TM2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, is_test, w_vectorizer, feat_bias=5, max_text_len=20, unit_length=4, input_type='mixed'):
        self.max_length = 20
        self.pointer = 0
        self.dataset_name = dataset_name
        self.is_test = is_test
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.w_vectorizer = w_vectorizer
        self.input_type = input_type  # new parameter to control whether to load music, text, or both
        self.music_id_path = "/disk2/yuyusun/TM2M-GPT/dataset/HumanML3D/music_id.txt"
        self.music_id_set = set()
        if os.path.exists(self.music_id_path):
            with open(self.music_id_path, 'r') as f:
                self.music_id_set = set([line.strip() for line in f.readlines()])

        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.music_dir = pjoin(self.data_root, 'music')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 196
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain
            self.meta_dir = '/disk2/yuyusun/TM2M-GPT/meta'
        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.music_dir = pjoin(self.data_root, 'music')
            self.joints_num = 21
            radius = 240 * 8
            fps = 12.5
            dim_pose = 251
            self.max_motion_length = 196
            kinematic_chain = paramUtil.kit_kinematic_chain
            self.meta_dir = '/disk2/yuyusun/TM2M-GPT/meta'

        mean = np.load(pjoin(self.meta_dir, 'Mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'Std.npy'))

        split_file = pjoin(self.data_root, 'test.txt' if is_test else 'val.txt')
        min_motion_len = 40 if self.dataset_name == 't2m' else 24
        joints_num = self.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue

                if name in self.music_id_set:
                    data_dict[name] = {'motion': motion, 'length': len(motion), 'text': None}
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    continue

                text_data = []
                flag = False
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2]) if not np.isnan(float(line_split[2])) else 0.0
                        to_tag = float(line_split[3]) if not np.isnan(float(line_split[3])) else 0.0

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * fps): int(to_tag * fps)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion, 'length': len(n_motion), 'text': [text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                pass
                if flag:
                    data_dict[name] = {'motion': motion, 'length': len(motion), 'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def forward_transform(self, data):
        return (data - self.mean) / self.std

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        motion, m_length, text_list = data['motion'], data['length'], data['text']

        if name in self.music_id_set:
            music_feat = np.load(pjoin(self.music_dir, name + '.npy'))  # shape: [1, 512]
            music_feat = np.repeat(music_feat, self.max_text_len + 2, axis=0)  # [22, 512]
            pos_one_hots = np.zeros((self.max_text_len + 2, 20))  # Dummy pos_one_hots
            tokens = ['music/INPUT'] * (self.max_text_len + 2)
            caption = 'music_input'
            sent_len = len(tokens)
            word_embeddings = music_feat
        else:
            text_data = random.choice(text_list)
            caption, tokens = text_data['caption'], text_data['tokens']
            if len(tokens) < self.max_text_len:
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
            else:
                tokens = tokens[:self.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        motion = (motion - self.mean) / self.std
        if m_length < self.max_motion_length:
            motion = np.concatenate([motion, np.zeros((self.max_motion_length - m_length, motion.shape[1]))], axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), name, name in self.music_id_set

def DATALoader(dataset_name, is_test,
                batch_size, w_vectorizer,
                num_workers = 8, unit_length = 4, input_type='mixed'): 
    
    val_loader = torch.utils.data.DataLoader(TM2MotionDataset(dataset_name, is_test, w_vectorizer, unit_length=unit_length, input_type=input_type),
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              drop_last=True)
    return val_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
