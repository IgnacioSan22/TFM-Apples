import functools
import numpy as np
import spconv.pytorch as spconv
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

    @force_fp32(apply_to=('semantic_scores, pt_offsets'))
    def forward_grouping(self,
                         semantic_scores,
                         pt_offsets,
                         batch_idxs,
                         coords_float,
                         grouping_cfg=None):
        proposals_idx_list = []
        proposals_offset_list = []
        batch_size = batch_idxs.max() + 1
        semantic_scores = semantic_scores.softmax(dim=-1)
        proposals_idx, proposals_offset = None, None

        radius = self.grouping_cfg.radius
        mean_active = self.grouping_cfg.mean_active
        npoint_thr = self.grouping_cfg.npoint_thr
        class_numpoint_mean = torch.tensor(
            self.grouping_cfg.class_numpoint_mean, dtype=torch.float32)
        for class_id in range(self.semantic_classes):
            if class_id in self.grouping_cfg.ignore_classes:
                continue
            scores = semantic_scores[:, class_id].contiguous()
            object_idxs = (scores > self.grouping_cfg.score_thr).nonzero().view(-1)
            if object_idxs.size(0) < self.test_cfg.min_npoint:
                continue
            batch_idxs_ = batch_idxs[object_idxs]
            batch_offsets_ = self.get_batch_offsets(batch_idxs_, batch_size)
            coords_ = coords_float[object_idxs]
            pt_offsets_ = pt_offsets[object_idxs]
            idx, start_len = ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_, batch_offsets_,
                                            radius, mean_active)
            proposals_idx, proposals_offset = bfs_cluster(class_numpoint_mean, idx.cpu(),
                                                        start_len.cpu(), npoint_thr, class_id)
            proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
            # print('Class id', class_id,torch.unique(proposals_idx[:,0], return_counts=True))
            # merge proposals
            if len(proposals_offset_list) > 0:
                proposals_idx[:, 0] += sum([x.size(0) for x in proposals_offset_list]) - 1
                proposals_offset += proposals_offset_list[-1][-1]
                proposals_offset = proposals_offset[1:]
            if proposals_idx.size(0) > 0:
                proposals_idx_list.append(proposals_idx)
                proposals_offset_list.append(proposals_offset)
        if len(proposals_idx_list) > 0:
            proposals_idx = torch.cat(proposals_idx_list, dim=0)
            proposals_offset = torch.cat(proposals_offset_list)
        return proposals_idx, proposals_offset
    
    
    @force_fp32(apply_to=('semantic_scores, pt_offsets'))
    def forward_grouping(self,
                         semantic_scores,
                         pt_offsets,
                         batch_idxs,
                         coords_float,
                         grouping_cfg=None):
        for rad in [0.005,0.01,0.05,0.1,0.5]:
            proposals_idx_list = []
            proposals_offset_list = []
            batch_size = batch_idxs.max() + 1
            semantic_scores = semantic_scores.softmax(dim=-1)
            proposals_idx, proposals_offset = None, None

            # radius = self.grouping_cfg.radius
            radius = rad
            mean_active = self.grouping_cfg.mean_active
            npoint_thr = self.grouping_cfg.npoint_thr
            class_numpoint_mean = torch.tensor(
                self.grouping_cfg.class_numpoint_mean, dtype=torch.float32)
            for class_id in range(self.semantic_classes):
                if class_id in self.grouping_cfg.ignore_classes:
                    continue
                scores = semantic_scores[:, class_id].contiguous()
                object_idxs = (scores > self.grouping_cfg.score_thr).nonzero().view(-1)
                if object_idxs.size(0) < self.test_cfg.min_npoint:
                    continue
                batch_idxs_ = batch_idxs[object_idxs]
                batch_offsets_ = self.get_batch_offsets(batch_idxs_, batch_size)
                coords_ = coords_float[object_idxs]
                pt_offsets_ = pt_offsets[object_idxs]
                idx, start_len = ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_, batch_offsets_,
                                                radius, mean_active)
                proposals_idx, proposals_offset = bfs_cluster(class_numpoint_mean, idx.cpu(),
                                                            start_len.cpu(), npoint_thr, class_id)
                proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
                # print('Class id', class_id,torch.unique(proposals_idx[:,0], return_counts=True))
                # merge proposals
                if len(proposals_offset_list) > 0:
                    proposals_idx[:, 0] += sum([x.size(0) for x in proposals_offset_list]) - 1
                    proposals_offset += proposals_offset_list[-1][-1]
                    proposals_offset = proposals_offset[1:]
                if proposals_idx.size(0) > 0:
                    proposals_idx_list.append(proposals_idx)
                    proposals_offset_list.append(proposals_offset)
            if len(proposals_idx_list) > 0:
                proposals_idx = torch.cat(proposals_idx_list, dim=0)
                proposals_offset = torch.cat(proposals_offset_list)
            if proposals_idx.size(0) > 0:
                best_prop_idx, off = self.best_proposals(proposals_idx, proposals_offset)
                best_prop_idx = best_prop_idx.cpu().numpy()
                props = []
                for i in range(1,len(off)):
                    props.append(coords_float[best_prop_idx[off[i-1]:off[i],1]].cpu().numpy())
                    
                props = np.array(props, dtype=object)
                # print(props)
                label = [[ind]*len(props[ind]) for ind in range(len(props))]
                label = np.concatenate(label).ravel()
                concat = np.concatenate(props,axis=0)
                concat = np.column_stack((concat,label))
                np.savetxt(f'conacta_props_{rad}.txt', concat)
                print(f'Proposals saved with radius {radius}!!')
            else:
                print(f'Empty proposals for radius {rad}')
        print(f'Proposals saved and finisheeed!!!!!!!!!!!')
        return proposals_idx, proposals_offset
    
    
from functions import (ballquery_batch_p, bfs_cluster, get_mask_iou_on_cluster, get_mask_iou_on_pred,
                   get_mask_label, global_avg_pool, sec_max, sec_min, voxelization,
                   voxelization_idx)

def get_batch_offsets(self, batch_idxs, bs):
    batch_offsets = torch.zeros(bs + 1).int().cuda()
    for i in range(bs):
        batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
    assert batch_offsets[-1] == batch_idxs.shape[0]
    return batch_offsets

def main():
    batch_idxs = torch.load('batch_idxs.pt')
    semantic_scores = torch.load('semantic_scores.pt')
    coords_float = torch.load('coords_float.pt')
    pt_offsets = torch.load('pt_offsets.pt')

    proposals_idx_list = []
    proposals_offset_list = []
    batch_size = batch_idxs.max() + 1
    semantic_scores = semantic_scores.softmax(dim=-1)
    proposals_idx, proposals_offset = None, None

    radius = 0.01
    mean_active = 200
    npoint_thr = 0.1
    class_numpoint_mean = torch.tensor(
        [-1., 4022.], dtype=torch.float32)
    for class_id in range(2):

        scores = semantic_scores[:, class_id].contiguous()
        object_idxs = (scores > 0.1).nonzero().view(-1)
        if object_idxs.size(0) < 500:
            continue
        batch_idxs_ = batch_idxs[object_idxs]
        batch_offsets_ = get_batch_offsets(batch_idxs_, batch_size)
        coords_ = coords_float[object_idxs]
        pt_offsets_ = pt_offsets[object_idxs]
        idx, start_len = ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_, batch_offsets_,
                                            radius, mean_active)
        proposals_idx, proposals_offset = bfs_cluster(class_numpoint_mean, idx.cpu(),
                                                        start_len.cpu(), npoint_thr, class_id)
        proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
        # print('Class id', class_id,torch.unique(proposals_idx[:,0], return_counts=True))
        # merge proposals
        if len(proposals_offset_list) > 0:
            proposals_idx[:, 0] += sum([x.size(0) for x in proposals_offset_list]) - 1
            proposals_offset += proposals_offset_list[-1][-1]
            proposals_offset = proposals_offset[1:]
        if proposals_idx.size(0) > 0:
            proposals_idx_list.append(proposals_idx)
            proposals_offset_list.append(proposals_offset)
    if len(proposals_idx_list) > 0:
        proposals_idx = torch.cat(proposals_idx_list, dim=0)
        proposals_offset = torch.cat(proposals_offset_list)
        
    proposals_idx, proposals_offset

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(coords_float[proposals_idx[:,1],0], coords_float[proposals_idx[:,1],1], coords_float[proposals_idx[:,1],2], cmap='Greens')

if __name__ == '__main__':
    main()