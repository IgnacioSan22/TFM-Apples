import math
import os.path as osp
from glob import glob

import numpy as np
import scipy.interpolate
import scipy.ndimage
import torch
from torch.utils.data import Dataset

from ..ops import voxelization_idx


class CustomDataset(Dataset):

    CLASSES = None

    def __init__(self,
                 data_root,
                 prefix,
                 suffix,
                 voxel_cfg=None,
                 training=True,
                 repeat=1,
                 logger=None):
        self.data_root = data_root
        self.prefix = prefix
        self.suffix = suffix
        self.voxel_cfg = voxel_cfg
        self.training = training
        self.repeat = repeat
        self.logger = logger
        self.mode = 'train' if training else 'test'
        self.filenames = self.get_filenames()
        self.logger.info(f'Load {self.mode} dataset: {len(self.filenames)} scans')
        self.predict_only = False

    def set_predict_only(self, predict):
        self.predict_only = predict

    def get_filenames(self):
        filenames = glob(osp.join(self.data_root, self.prefix, '*' + self.suffix))
        assert len(filenames) > 0, 'Empty dataset.'
        filenames = sorted(filenames * self.repeat)
        return filenames

    def load(self, filename):
        return torch.load(filename)

    def __len__(self):
        return len(self.filenames)

    def elastic(self, x, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3
        bb = np.abs(x).max(0).astype(np.int32) // gran + 3
        # print('BB in elastic', bb, 'Gran', gran)
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [
            scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0)
            for n in noise
        ]

        def g(x_):
            return np.hstack([i(x_)[:, None] for i in interp])

        return x + g(x) * mag

    def getInstanceInfo(self, xyz, instance_label, semantic_label):
        pt_mean = np.ones((xyz.shape[0], 3), dtype=np.float32) * -100.0
        instance_pointnum = []
        instance_cls = []
        instance_num = int(instance_label.max()) + 1
        for i_ in range(instance_num):
            inst_idx_i = np.where(instance_label == i_)
            xyz_i = xyz[inst_idx_i]
            pt_mean[inst_idx_i] = xyz_i.mean(0)
            instance_pointnum.append(inst_idx_i[0].size)
            cls_idx = inst_idx_i[0][0]
            instance_cls.append(semantic_label[cls_idx])
        pt_offset_label = pt_mean - xyz
        return instance_num, instance_pointnum, instance_cls, pt_offset_label

    def dataAugment(self, xyz, jitter=False, flip=False, rot=False, prob=0.9):
        m = np.eye(3)
        if jitter and np.random.rand() < prob:
            m += np.random.randn(3, 3) * 0.1
        if flip and np.random.rand() < prob:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1
        if rot and np.random.rand() < prob:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0],
                              [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
        else:
            # Empirically, slightly rotate the scene can match the results from checkpoint
            theta = 0.35 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0],
                              [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])

        return np.matmul(xyz, m)

    def crop(self, xyz, step=32):
        # np.savetxt('testCrop.txt', xyz, delimiter=' ')
        xyz_offset = xyz.copy()
        valid_idxs = xyz_offset.min(1) >= 0
        assert valid_idxs.sum() == xyz.shape[0]
        spatial_shape = np.array([self.voxel_cfg.spatial_shape[1]] * 3, dtype=np.float)
        room_range = xyz.max(0) - xyz.min(0)
        while (valid_idxs.sum() > self.voxel_cfg.max_npoint):
            step_temp = step
            # if valid_idxs.sum() > 1e6:
            #     step_temp = step * 2
            # print(spatial_shape, room_range)
            offset = np.clip(spatial_shape - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < spatial_shape).sum(1) == 3)
            spatial_shape[:3] -= step_temp
        return xyz_offset, valid_idxs

    def getCroppedInstLabel(self, instance_label, valid_idxs):
        instance_label = instance_label[valid_idxs]
        j = 0
        while (j < instance_label.max()):
            if (len(np.where(instance_label == j)[0]) == 0):
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label

    def transform_train(self, xyz, rgb, semantic_label, instance_label, aug_prob=0.25):
        xyz_middle = self.dataAugment(xyz, True, True, True, aug_prob)
        xyz = xyz_middle * self.voxel_cfg.scale
        if np.random.rand() < aug_prob:
            xyz = self.elastic(xyz, 6 * self.voxel_cfg.scale // 50, 40 * self.voxel_cfg.scale / 50)
            xyz = self.elastic(xyz, 20 * self.voxel_cfg.scale // 50,
                               160 * self.voxel_cfg.scale / 50)
        # xyz_middle = xyz / self.voxel_cfg.scale
        xyz = xyz - xyz.min(0)
        max_tries = 20
        valid_idxs = np.array([])
        while (max_tries > 0):
            xyz_offset, valid_idxs = self.crop(xyz)
            if valid_idxs.sum() >= self.voxel_cfg.min_npoint:
                xyz = xyz_offset
                break
            max_tries -= 1
        if valid_idxs.sum() < self.voxel_cfg.min_npoint:
            print('Mierdaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
            return None
        xyz = xyz[valid_idxs]
        xyz_middle = xyz_middle[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)
        # print('Transform train:',xyz.shape, xyz_middle.shape, rgb.shape, semantic_label.shape, instance_label.shape)
        return xyz, xyz_middle, rgb, semantic_label, instance_label

    def transform_test(self, xyz, rgb, semantic_label, instance_label):
        xyz_middle = self.dataAugment(xyz, False, False, False)
        xyz = xyz_middle * self.voxel_cfg.scale
        xyz_middle = xyz / self.voxel_cfg.scale
        xyz = xyz - xyz.min(0)
        valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)
        return xyz, xyz_middle, rgb, semantic_label, instance_label

    def __getitem__(self, index):
        # print('Getting item...')
        filename = self.filenames[index]
        # print('Getting item:' , filename)
        scan_id = osp.basename(filename).replace(self.suffix, '')
        # print('scan id:', scan_id)
        data = self.load(filename)
        # print('Loaded initial data:', data)
        data = self.transform_train(*data) if self.training else self.transform_test(*data)
        # print('Transformed initial data:', data)
        if data is None:
            return None
        xyz, xyz_middle, rgb, semantic_label, instance_label = data
        # np.savetxt(filename[:-4] + 'final_apples.txt', xyz[semantic_label == 1], delimiter=' ')
        info = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32), semantic_label)
        # print('Instance info:', info)
        inst_num, inst_pointnum, inst_cls, pt_offset_label = info
        coord = torch.from_numpy(xyz).long()
        coord_float = torch.from_numpy(xyz_middle)
        feat = torch.from_numpy(rgb).float()
        # if self.training:
        #     feat += torch.randn(3) * 0.1
        semantic_label = torch.from_numpy(semantic_label)
        instance_label = torch.from_numpy(instance_label)
        pt_offset_label = torch.from_numpy(pt_offset_label)
        # print('Element gotten...')
        
        # print(f'After transform train apple points:', sum(semantic_label == 1), 'Instances: ', inst_num)
        return (scan_id, coord, coord_float, feat, semantic_label, instance_label, inst_num,
                inst_pointnum, inst_cls, pt_offset_label)

    def collate_fn(self, batch):
        scan_ids = []
        coords = []
        coords_float = []
        feats = []
        semantic_labels = []
        instance_labels = []

        instance_pointnum = []  # (total_nInst), int
        instance_cls = []  # (total_nInst), long
        pt_offset_labels = []

        total_inst_num = 0
        batch_id = 0
        for data in batch:
            if data is None:
                continue
            (scan_id, coord, coord_float, feat, semantic_label, instance_label, inst_num,
             inst_pointnum, inst_cls, pt_offset_label) = data
            instance_label[np.where(instance_label != -100)] += total_inst_num
            total_inst_num += inst_num
            scan_ids.append(scan_id)
            coords.append(torch.cat([coord.new_full((coord.size(0), 1), batch_id), coord], 1))
            coords_float.append(coord_float)
            feats.append(feat)
            semantic_labels.append(semantic_label)
            instance_labels.append(instance_label)
            instance_pointnum.extend(inst_pointnum)
            instance_cls.extend(inst_cls)
            pt_offset_labels.append(pt_offset_label)
            batch_id += 1
        assert batch_id > 0, 'empty batch'
        if batch_id < len(batch):
            self.logger.info(f'batch is truncated from size {len(batch)} to {batch_id}')

        # merge all the scenes in the batch
        coords = torch.cat(coords, 0)  # long (N, 1 + 3), the batch item idx is put in coords[:, 0]
        batch_idxs = coords[:, 0].int()
        coords_float = torch.cat(coords_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)  # float (N, C)
        semantic_labels = torch.cat(semantic_labels, 0).long()  # long (N)
        instance_labels = torch.cat(instance_labels, 0).long()  # long (N)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)
        instance_cls = torch.tensor(instance_cls, dtype=torch.long)  # long (total_nInst)
        pt_offset_labels = torch.cat(pt_offset_labels).float()

        spatial_shape = np.clip(
            coords.max(0)[0][1:].numpy() + 1, self.voxel_cfg.spatial_shape[0], None)
        voxel_coords, v2p_map, p2v_map = voxelization_idx(coords, batch_id)
        # print('Instance info', torch.unique(instance_labels))
        return {
            'scan_ids': scan_ids,
            'coords': coords,
            'batch_idxs': batch_idxs,
            'voxel_coords': voxel_coords,
            'p2v_map': p2v_map,
            'v2p_map': v2p_map,
            'coords_float': coords_float,
            'feats': feats,
            'semantic_labels': semantic_labels,
            'instance_labels': instance_labels,
            'instance_pointnum': instance_pointnum,
            'instance_cls': instance_cls,
            'pt_offset_labels': pt_offset_labels,
            'spatial_shape': spatial_shape,
            'batch_size': batch_id,
        }
