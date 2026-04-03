import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler
import numpy as np
from utils.util import to_tensor
import os
import random
import lmdb
import pickle

class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
    ):
        super(CustomDataset, self).__init__()
        self.db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]
        n_sub_list = [s[3:6] for s in self.keys]
        self.n_sub = len(set(n_sub_list))
        print(f'CustomDataset initialized with {len(self.keys)} samples from {self.n_sub} subjects in {mode} set.')

    def __len__(self):
        return len((self.keys))

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair['sample']
        label = pair['label']
        return data/100, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        # return to_tensor(x_data), to_tensor(y_label).long()
        return to_tensor(x_data), to_tensor(y_label).float()


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir + f'_{params.s_duration}s'

    def get_data_loader(self, contrast=False, n_batch=None):
        train_set = CustomDataset(self.datasets_dir, mode='train')
        val_set = CustomDataset(self.datasets_dir, mode='val')
        test_set = CustomDataset(self.datasets_dir, mode='test')
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set)+len(val_set)+len(test_set))
        if contrast:
            contrast_sampler = ContrastSampler(train_set, n_batch, self.params.batch_size)
        data_loader = {
            'train': DataLoader(
                train_set,
                # batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True if not contrast else None,
                batch_sampler=contrast_sampler if contrast else None 
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
            ),
        }
        print(f'train_loader: {len(data_loader["train"])} batches, val_loader: {len(data_loader["val"])} batches, test_loader: {len(data_loader["test"])} batches')
        return data_loader

class ContrastSampler(BatchSampler):
    def __init__(self, dataset, n_batch, bs):
        self.bs = bs
        self.dataset = dataset
        self.n_batch = n_batch
        self.n_sub = self.dataset.n_sub
        self.n_sample_per_sub = len(self.dataset) // self.n_sub
        self.pairs = []
        self.pairs_epoch = []
        for i in range(self.n_sample_per_sub):
            for sub_i in range(self.n_sub):
                for sub_j in range(sub_i+1, self.n_sub):
                    idx_i = sub_i * self.n_sample_per_sub + i
                    idx_j = sub_j * self.n_sample_per_sub + i
                    self.pairs.append((idx_i, idx_j))
        random.shuffle(self.pairs)
        print(f'ContrastSampler initialized with {len(self.pairs)} pairs of samples from {self.n_sub} subjects.')

    def __iter__(self):
        random.shuffle(self.pairs)
        n_sample = self.n_batch * self.bs
        pairs_to_use = self.pairs[:n_sample // 2]
        all_indices = np.array(pairs_to_use).flatten().tolist()
        
        for i in range(0, len(all_indices), self.bs):
            batch = all_indices[i : i + self.bs]
            batch = np.array(batch).reshape(self.bs //2, 2)
            batch = np.transpose(batch, (1, 0)).reshape(-1)
            yield batch # 每次返回一个包含 bs 个索引的列表

    def __len__(self):
        return self.n_batch