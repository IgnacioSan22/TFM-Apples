import csv
import os.path as osp
import os
from glob import glob

import numpy as np
import torch

from ..ops import voxelization_idx
from .custom import CustomDataset
import pylas


class ApplesSizeDataSet(CustomDataset):

    CLASSES = ('tree', 'apple')
    
    def __init__(self, data_root, prefix, suffix, voxel_cfg=None, training=True, repeat=1, logger=None, size_path=''):
        super().__init__(data_root, prefix, suffix, voxel_cfg, training, repeat, logger)
        self.size_data = {}
        if os.path.exists(size_path):
            file = open(size_path)
            csvreader = csv.reader(file,delimiter=';')
            header = next(csvreader)
            # print(header)
            self.size_data[0] = -1
            for elem in csvreader:
                self.size_data[int(elem[5])] = float(elem[2])
                
    def getCroppedInstLabel(self, instance_label, valid_idxs, instance_sizes):
        instance_label = instance_label[valid_idxs]
        j = 0
        new_instance_sizes = (instance_label.copy())
        while (j < instance_label.max()):
            if (len(np.where(instance_label == j)[0]) == 0):
                new_instance_sizes[new_instance_sizes[:,0] == instance_label.max(),0] = j
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label, new_instance_sizes

    
    def get_filenames(self):
        filenames = [osp.join(self.data_root, f) for f in os.listdir(self.data_root)]
        assert len(filenames) > 0, 'Empty dataset.'
        filenames = sorted(filenames * self.repeat)
        return filenames

    def load(self, filename):
        # global_shift = np.array([-322599., -4609540., -255.])
        with pylas.open(filename) as fh:
            las = fh.read()
            x = las.x
            y = las.y
            z = las.z
            xyz = (np.column_stack((x,y,z)))
            # xyz = xyz + global_shift
            #Normalization step to avoid too large numbers
            xyz = xyz - (np.mean(xyz, axis=0))
            # furthest_distance = np.max(np.sqrt(np.sum(abs(xyz)**2,axis=-1)))
            # xyz = xyz / furthest_distance
            # xyz = xyz * 100.0
            xyz = np.ascontiguousarray(xyz, dtype=np.float32)

            R = las.points["red"]
            G = las.points["green"]
            B = las.points["blue"]
            rgb =   (np.column_stack((R,G,B))>>8).astype(np.uint8)
            rgb = (rgb / 127.5) - 1.0
            rgb = np.ascontiguousarray(rgb, dtype=np.float32)
            

            if not self.predict_only:
                # semantic_label = (las.points['semantic_label'].copy())
                # instance_label = (las.points['instance_label'].astype(np.int32).copy())
                instance_label = (las.points['Scalar_field'].astype(np.int32).copy())
                semantic_label = np.zeros(instance_label.shape, dtype=np.int)
                semantic_label[instance_label > 0] = 1
                instance_sizes = np.array([[key,self.size_data[key]] for key in np.unique(instance_label)])
                # instance_label[np.isnan(instance_label)] = 0
            else:
                semantic_label = np.zeros(R.shape)
                instance_label = np.zeros(R.shape)
                instance_sizes = np.array([[0, -1.]])
        
        # subsample data
        if self.training:
            N = xyz.shape[0]
            treeInds = np.where(semantic_label == 0)[0]
            treeInds = np.random.choice(treeInds, int(len(treeInds) * 0.3), replace=False)
            inds = np.concatenate((np.where(semantic_label == 1)[0], treeInds), axis=0)
            xyz = xyz[inds]
            rgb = rgb[inds]
            semantic_label = semantic_label[inds]
            
            instance_label, instance_sizes = self.getCroppedInstLabel(instance_label, inds, instance_sizes)
            
        return xyz, rgb, semantic_label, instance_label, instance_sizes

    def crop(self, xyz, step=0.5):
        return super().crop(xyz, step=step)

    def transform_test(self, xyz, rgb, semantic_label, instance_label, instace_sizes):
        # devide into 4 piecies
        inds = np.arange(xyz.shape[0])
        piece_1 = inds[::4]
        piece_2 = inds[1::4]
        piece_3 = inds[2::4]
        piece_4 = inds[3::4]
        xyz_aug = self.dataAugment(xyz, False, False, False)

        xyz_list = []
        xyz_middle_list = []
        rgb_list = []
        semantic_label_list = []
        instance_label_list = []
        for batch, piece in enumerate([piece_1, piece_2, piece_3, piece_4]):
            xyz_middle = xyz_aug[piece]
            xyz = xyz_middle * self.voxel_cfg.scale
            xyz -= xyz.min(0)
            xyz_list.append(np.concatenate([np.full((xyz.shape[0], 1), batch), xyz], 1))
            xyz_middle_list.append(xyz_middle)
            rgb_list.append(rgb[piece])
            semantic_label_list.append(semantic_label[piece])
            instance_label_list.append(instance_label[piece])
        xyz = np.concatenate(xyz_list, 0)
        xyz_middle = np.concatenate(xyz_middle_list, 0)
        rgb = np.concatenate(rgb_list, 0)
        semantic_label = np.concatenate(semantic_label_list, 0)
        instance_label = np.concatenate(instance_label_list, 0)
        valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)  # TODO remove this
        return xyz, xyz_middle, rgb, semantic_label, instance_label, instace_sizes

    def transform_train(self, xyz, rgb, semantic_label, instance_label, instance_sizes, aug_prob=0.25):
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
        instance_label = self.getCroppedInstLabel(instance_label, valid_idxs, instance_sizes)
        # print('Transform train:',xyz.shape, xyz_middle.shape, rgb.shape, semantic_label.shape, instance_label.shape)
        return xyz, xyz_middle, rgb, semantic_label, instance_label, instance_sizes


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
        xyz, xyz_middle, rgb, semantic_label, instance_label, instance_sizes = data
        # np.savetxt(filename[:-4] + 'final_apples.txt', xyz[semantic_label == 1], delimiter=' ')
        info = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32), semantic_label)
        # print('Instance info:', info)
        inst_num, inst_pointnum, inst_cls, pt_offset_label = info
        coord = torch.from_numpy(xyz).long()
        coord_float = torch.from_numpy(xyz_middle)
        feat = torch.from_numpy(rgb).float()
        if self.training:
            feat += torch.randn(3) * 0.1
        semantic_label = torch.from_numpy(semantic_label)
        instance_label = torch.from_numpy(instance_label)
        pt_offset_label = torch.from_numpy(pt_offset_label)
        instance_sizes = torch.from_numpy(instance_sizes)
        # print('Element gotten...')
        
        # print(f'After transform train apple points:', sum(semantic_label == 1), 'Instances: ', inst_num)
        return (scan_id, coord, coord_float, feat, semantic_label, instance_label, inst_num,
                inst_pointnum, inst_cls, pt_offset_label, instance_sizes)

    def collate_fn(self, batch):
        scan_ids = []
        coords = []
        coords_float = []
        feats = []
        semantic_labels = []
        instance_labels = []
        instance_sizes = []
        instance_pointnum = []  # (total_nInst), int
        instance_cls = []  # (total_nInst), long
        pt_offset_labels = []

        total_inst_num = 0
        batch_id = 0
        if self.training:
            
            for data in batch:
                if data is None:
                    continue
                (scan_id, coord, coord_float, feat, semantic_label, instance_label, inst_num,
                inst_pointnum, inst_cls, pt_offset_label, instance_size) = data
                instance_label[np.where(instance_label != -100)] += total_inst_num
                instance_size[:,0] += total_inst_num
                total_inst_num += inst_num
                scan_ids.append(scan_id)
                coords.append(torch.cat([coord.new_full((coord.size(0), 1), batch_id), coord], 1))
                coords_float.append(coord_float)
                feats.append(feat)
                instance_sizes.append(instance_size)
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
            instance_sizes = torch.cat(instance_sizes, 0) 
            semantic_labels = torch.cat(semantic_labels, 0).long()  # long (N)
            instance_labels = torch.cat(instance_labels, 0).long()  # long (N)
            instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)
            instance_cls = torch.tensor(instance_cls, dtype=torch.long)  # long (total_nInst)
            pt_offset_labels = torch.cat(pt_offset_labels).float()

            spatial_shape = np.clip(
                coords.max(0)[0][1:].numpy() + 1, self.voxel_cfg.spatial_shape[0], None)
            voxel_coords, v2p_map, p2v_map = voxelization_idx(coords, batch_id)
        else:
            batch_id = 1
            (scan_id, coord, coord_float, feat, semantic_label, instance_label, inst_num, inst_pointnum,
            inst_cls, pt_offset_label, instance_sizes) = batch[0]
            scan_ids = [scan_id]
            coords = coord.long()
            batch_idxs = torch.zeros_like(coord[:, 0].int())
            coords_float = coord_float.float()
            feats = feat.float()
            semantic_labels = semantic_label.long()
            instance_labels = instance_label.long()
            instance_pointnum = torch.tensor([inst_pointnum], dtype=torch.int)
            instance_cls = torch.tensor([inst_cls], dtype=torch.long)
            pt_offset_labels = pt_offset_label.float()
            spatial_shape = np.clip((coords.max(0)[0][1:] + 1).numpy(), self.voxel_cfg.spatial_shape[0],
                                    None)
            voxel_coords, v2p_map, p2v_map = voxelization_idx(coords, 1)
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
            'instance_sizes': instance_sizes,
            'batch_size': batch_id,
        }