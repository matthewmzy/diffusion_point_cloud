import os
import random
from copy import copy
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from tqdm.auto import tqdm


synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


class ShapeNetCore(Dataset):

    GRAVITATIONAL_AXIS = 1
    
    def __init__(self, path, cates, split, scale_mode, transform=None):
        super().__init__()
        assert isinstance(cates, list), '`cates` must be a list of cate names.'
        assert split in ('train', 'val', 'test')
        assert scale_mode is None or scale_mode in ('global_unit', 'shape_unit', 'shape_bbox', 'shape_half', 'shape_34')
        self.path = path
        if 'all' in cates:
            cates = cate_to_synsetid.keys()
        self.cate_synsetids = [cate_to_synsetid[s] for s in cates]
        self.cate_synsetids.sort()
        self.split = split
        self.scale_mode = scale_mode
        self.transform = transform

        self.pointclouds = []
        self.stats = None

        self.get_statistics()
        self.load()

    def get_statistics(self):

        basename = os.path.basename(self.path)
        dsetname = basename[:basename.rfind('.')]
        stats_dir = os.path.join(os.path.dirname(self.path), dsetname + '_stats')
        os.makedirs(stats_dir, exist_ok=True)

        if len(self.cate_synsetids) == len(cate_to_synsetid):
            stats_save_path = os.path.join(stats_dir, 'stats_all.pt')
        else:
            stats_save_path = os.path.join(stats_dir, 'stats_' + '_'.join(self.cate_synsetids) + '.pt')
        if os.path.exists(stats_save_path):
            self.stats = torch.load(stats_save_path)
            return self.stats

        with h5py.File(self.path, 'r') as f:
            pointclouds = []
            for synsetid in self.cate_synsetids:
                for split in ('train', 'val', 'test'):
                    pointclouds.append(torch.from_numpy(f[synsetid][split][...]))

        all_points = torch.cat(pointclouds, dim=0) # (B, N, 3)
        B, N, _ = all_points.size()
        mean = all_points.view(B*N, -1).mean(dim=0) # (1, 3)
        std = all_points.view(-1).std(dim=0)        # (1, )

        self.stats = {'mean': mean, 'std': std}
        torch.save(self.stats, stats_save_path)
        return self.stats

    def load(self):

        def _enumerate_pointclouds(f):
            for synsetid in self.cate_synsetids:
                cate_name = synsetid_to_cate[synsetid]
                for j, pc in enumerate(f[synsetid][self.split]):
                    yield torch.from_numpy(pc), j, cate_name
        
        with h5py.File(self.path, mode='r') as f:
            for pc, pc_id, cate_name in _enumerate_pointclouds(f):

                if self.scale_mode == 'global_unit':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = self.stats['std'].reshape(1, 1)
                elif self.scale_mode == 'shape_unit':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1)
                elif self.scale_mode == 'shape_half':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1) / (0.5)
                elif self.scale_mode == 'shape_34':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1) / (0.75)
                elif self.scale_mode == 'shape_bbox':
                    pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
                    pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
                    shift = ((pc_min + pc_max) / 2).view(1, 3)
                    scale = (pc_max - pc_min).max().reshape(1, 1) / 2
                else:
                    shift = torch.zeros([1, 3])
                    scale = torch.ones([1, 1])

                pc = (pc - shift) / scale

                self.pointclouds.append({
                    'pointcloud': pc,
                    'cate': cate_name,
                    'id': pc_id,
                    'shift': shift,
                    'scale': scale
                })

        # Deterministically shuffle the dataset
        self.pointclouds.sort(key=lambda data: data['id'], reverse=False)
        random.Random(2020).shuffle(self.pointclouds)

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[idx].items()}
        if self.transform is not None:
            data = self.transform(data)
        return data

class IBSDataset(Dataset):

    def __init__(self, path, split, hand_type='shadow', overfit=False,point_dim=3):
        super().__init__()
        assert split in ('train', 'val', 'test')
        self.path = path
        self.split = split
        self.hand_type = hand_type
        self.point_dim = point_dim
        self.pointclouds = []
        self.overfit = overfit
        
        scene_names = [n[6:11] for n in os.listdir(os.path.join(path, "scene_pc"))]
        self.scene_pcs = [np.load(os.path.join(path, "scene_pc", f"scene_{scene_name}.npy")) for scene_name in scene_names]
        self.ibs_npzs = [np.load(os.path.join(path, hand_type, f"ibs_{scene_name}.npz")) for scene_name in scene_names]
        self.idx2scene = []
        for idx, ibs_npz in enumerate(self.ibs_npzs):
            ibs_nums = 0
            for grasp_code in ibs_npz:
                if len(ibs_npz[grasp_code].shape) == 1:
                    continue
                elif len(ibs_npz[grasp_code].shape) == 2:
                    self.pointclouds.append(torch.from_numpy(ibs_npz[grasp_code]).unsqueeze(0))
                    ibs_nums += 1
                elif len(ibs_npz[grasp_code].shape) == 3:
                    self.pointclouds.append(torch.from_numpy(ibs_npz[grasp_code]))
                    ibs_nums += len(ibs_npz[grasp_code])
                else:
                    raise ValueError
            self.idx2scene += [idx]*ibs_nums
        self.pointclouds = torch.cat(self.pointclouds, dim=0)
        self.pointclouds = self.pointclouds[:, torch.randperm(self.pointclouds.shape[1])[:256], :self.point_dim] # CHANGE
        self.idx2scene = torch.tensor(self.idx2scene)
        # 将self.pointclouds和self.scene_pcs进行相同的shuffle
        torch.random.manual_seed(2020)
        perm = torch.randperm(len(self.pointclouds))
        self.pointclouds = self.pointclouds[perm]
        self.idx2scene = self.idx2scene[perm]

        if self.overfit:
            self.pointclouds = self.pointclouds[0].unsqueeze(0)
        else:
            train_ratio = 0.8
            test_ratio = 0.1
            train_num = int(len(self.pointclouds) * train_ratio)
            test_num = int(len(self.pointclouds) * test_ratio)
            val_num = len(self.pointclouds) - train_num - test_num
            if self.split == 'train':
                self.pointclouds = self.pointclouds[:train_num]
            elif self.split == 'test':
                self.pointclouds = self.pointclouds[train_num:train_num + test_num]
            elif self.split == 'val':
                self.pointclouds = self.pointclouds[train_num + test_num:]
            else:
                raise ValueError
        print(f"Loaded {len(self.pointclouds)} point clouds from {path}.")
        
    def __len__(self):
        return len(self.pointclouds)
    
    def __getitem__(self, idx):
        return {"pointcloud": self.pointclouds[idx],
                "scene_pc":self.scene_pcs[self.idx2scene[idx]],
                "shift":torch.zeros([1,3]),
                "scale":torch.ones([1,1])}
