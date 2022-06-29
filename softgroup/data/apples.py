import os.path as osp
import os
from glob import glob
import sys

import numpy as np
import torch

from ..ops import voxelization_idx
from .custom import CustomDataset
import pylas


class ApplesDataSet(CustomDataset):

    CLASSES = ('tree', 'apple')

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
                # instance_label[np.isnan(instance_label)] = 0
            else:
                semantic_label = np.zeros(R.shape)
                instance_label = np.zeros(R.shape)
        
        # subsample data
        if self.training:
            N = xyz.shape[0]
            treeInds = np.where(semantic_label == 0)[0]
            treeInds = np.random.choice(treeInds, int(len(treeInds) * 0.3), replace=False)
            inds = np.concatenate((np.where(semantic_label == 1)[0], treeInds), axis=0)
            xyz = xyz[inds]
            rgb = rgb[inds]
            semantic_label = semantic_label[inds]
            # print('pre',np.unique(instance_label[inds]))
            instance_label = self.getCroppedInstLabel(instance_label, inds)
            # print('post',np.unique(instance_label))
        return xyz, rgb, semantic_label, instance_label

    def crop(self, xyz, step=0.5):
        return super().crop(xyz, step=step)

    def transform_test(self, xyz, rgb, semantic_label, instance_label):
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
        return xyz, xyz_middle, rgb, semantic_label, instance_label

    def collate_fn(self, batch):
        if self.training:
            return super().collate_fn(batch)

        # assume 1 scan only
        (scan_id, coord, coord_float, feat, semantic_label, instance_label, inst_num, inst_pointnum,
         inst_cls, pt_offset_label) = batch[0]
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
        return {
            'scan_ids': scan_ids,
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
            'batch_size': 1
        }
