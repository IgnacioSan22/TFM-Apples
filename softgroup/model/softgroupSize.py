import functools
import numpy as np
import spconv.pytorch as spconv
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..ops import (ballquery_batch_p, bfs_cluster, get_mask_iou_on_cluster, get_mask_iou_on_pred,
                   get_mask_label, global_avg_pool, sec_max, sec_min, voxelization,
                   voxelization_idx)
from ..util import cuda_cast, force_fp32, rle_encode
from .blocks import MLP, ResidualBlock, SizeNet, UBlock, SizeMLP


class SoftGroupSize(nn.Module):

    def __init__(self,
                 channels=32,
                 num_blocks=7,
                 semantic_only=False,
                 semantic_classes=2,
                 instance_classes=2,
                 sem2ins_classes=[],
                 ignore_label=-100,
                 grouping_cfg=None,
                 instance_voxel_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 fixed_modules=[]):
        super().__init__()
        self.channels = channels
        self.num_blocks = num_blocks
        self.semantic_only = semantic_only
        self.semantic_classes = semantic_classes
        self.instance_classes = instance_classes
        self.sem2ins_classes = sem2ins_classes
        self.ignore_label = ignore_label
        self.grouping_cfg = grouping_cfg
        self.instance_voxel_cfg = instance_voxel_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fixed_modules = fixed_modules

        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        # backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                6, channels, kernel_size=3, padding=1, bias=False, indice_key='subm1'))
        block_channels = [channels * (i + 1) for i in range(num_blocks)]
        self.unet = UBlock(block_channels, norm_fn, 2, block, indice_key_id=1)
        self.output_layer = spconv.SparseSequential(norm_fn(channels), nn.ReLU())

        # point-wise prediction
        self.semantic_linear = MLP(channels, semantic_classes, norm_fn=norm_fn, num_layers=2)
        self.offset_linear = MLP(channels, 3, norm_fn=norm_fn, num_layers=2)

        # topdown refinement path
        if not semantic_only:
            self.tiny_unet = UBlock([channels, 2 * channels], norm_fn, 2, block, indice_key_id=11)
            self.tiny_unet_outputlayer = spconv.SparseSequential(norm_fn(channels), nn.ReLU())
            self.cls_linear = nn.Linear(channels, instance_classes) # + 1
            self.mask_linear = MLP(channels, instance_classes, norm_fn=None, num_layers=2) # + 1
            self.iou_score_linear = nn.Linear(channels, instance_classes) # + 1
            if self.train_cfg.full_feats_size:
                self.size_regression = SizeNet(self.train_cfg.size_feats)
            else:    
                self.size_regression = MLP(self.train_cfg.size_feats, 1, norm_fn=None, num_layers=2) # Features from tiny U-net + [mean, median, min, max]
            # inputSize = channels + 4
            # self.size_regression = SizeMLP([inputSize, inputSize*2, 1], norm_fn=None, num_layers=2) # Features from tiny U-net + [mean, median, min, max]

        self.init_weights()

        for mod in fixed_modules:
            mod = getattr(self, mod)
            for param in mod.parameters():
                param.requires_grad = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MLP):
                m.init_weights()
        if not self.semantic_only:
            for m in [self.cls_linear, self.iou_score_linear]:
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super().train(mode)
        for mod in self.fixed_modules:
            mod = getattr(self, mod)
            for m in mod.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def forward(self, batch, return_loss=False):
        if return_loss:
            return self.forward_train(**batch)
        else:
            return self.forward_test(**batch)

    @cuda_cast
    def forward_train(self, batch_idxs, voxel_coords, p2v_map, v2p_map, coords_float, feats,
                      semantic_labels, instance_labels, instance_pointnum, instance_cls,
                      pt_offset_labels, spatial_shape, instance_sizes, batch_size, **kwargs):
        losses = {}
        feats = torch.cat((feats, coords_float), 1)
        voxel_feats = voxelization(feats, p2v_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        semantic_scores, pt_offsets, output_feats = self.forward_backbone(input, v2p_map)
        # print('Output features big Unet:', output_feats.shape, semantic_scores.shape)
        point_wise_loss = self.point_wise_loss(semantic_scores, pt_offsets, semantic_labels,
                                               instance_labels, pt_offset_labels)
        losses.update(point_wise_loss)

        # instance losses
        if not self.semantic_only:
            proposals_idx, proposals_offset = self.forward_grouping(semantic_scores, pt_offsets,
                                                                    batch_idxs, coords_float,
                                                                    self.grouping_cfg)

            proposals_idx, proposals_offset = self.best_proposals(proposals_idx, proposals_offset)
            # print('proposal idx shape', proposals_idx.shape)
            if proposals_offset.shape[0] > self.train_cfg.max_proposal_num:
                proposals_offset = proposals_offset[:self.train_cfg.max_proposal_num + 1]
                proposals_idx = proposals_idx[:proposals_offset[-1]]
                assert proposals_idx.shape[0] == proposals_offset[-1]
            inst_feats, inst_map = self.clusters_voxelization(
                proposals_idx,
                proposals_offset,
                output_feats,
                coords_float,
                rand_quantize=True,
                **self.instance_voxel_cfg)
            
            if self.train_cfg.full_feats_size:
                off_feats = self.get_offset_full_feats(semantic_scores, proposals_offset, coords_float, proposals_idx, pt_offsets)
            else:
                off_feats = self.get_offset_4feats(semantic_scores, proposals_offset, proposals_idx, pt_offsets)
                
            instance_batch_idxs, cls_scores, iou_scores, mask_scores, sizes = self.forward_instance(
                inst_feats, inst_map, off_feats, proposals_offset)
            
            instance_loss = self.instance_loss(cls_scores, mask_scores, iou_scores, proposals_idx,
                                               proposals_offset, instance_labels, instance_pointnum,
                                               instance_cls, instance_batch_idxs, instance_sizes, sizes)
            #Custom loss weight to force the network to predict less instances
            # instance_loss['num_instances'] = torch.tensor(((cls_scores.size(0) * self.train_cfg.loos_per_instance)), requires_grad=True)
            losses.update(instance_loss)
        return self.parse_losses(losses)
    
    def get_offset_4feats(self, semantic_scores, proposals_offset, proposals_idx, pt_offsets):
        '''Return 4 features obtained agregating the radius of each point: [Mean, Median, Min, Max]'''
        off_feats = semantic_scores.new_full((len(proposals_offset) - 1, 4), 0, dtype=torch.float)
        for i in range(len(off_feats)):
            prop_off = proposals_offset[i]
            prop_ind = proposals_idx[prop_off:(prop_off+self.test_cfg.min_npoint), 1].long()
            radius = torch.linalg.norm(pt_offsets[prop_ind],dim=1)
            off_feats[i, 0] = torch.mean(radius)
            off_feats[i, 1] = torch.median(radius)
            off_feats[i, 2] = torch.min(radius)
            off_feats[i, 3] = torch.max(radius)
        return off_feats
    
    
    def get_offset_full_feats(self, semantic_scores, proposals_offset, coords_float, proposals_idx, pt_offsets):
        off_feats = semantic_scores.new_full((len(proposals_offset) - 1, self.test_cfg.min_npoint, 3), 0, dtype=torch.float)
        for i in range(len(off_feats)):
            prop_off = proposals_offset[i]
            prop_ind = proposals_idx[prop_off:(prop_off+self.test_cfg.min_npoint), 1].long()
            # radius = torch.linalg.norm(pt_offsets[prop_ind],dim=1)
            # off_feats[i, :, 0] = radius
            off_feats[i, :, 0:3] = coords_float[prop_ind]
        return off_feats

    def point_wise_loss(self, semantic_scores, pt_offsets, semantic_labels, instance_labels,
                        pt_offset_labels):
        losses = {}
        # weight=torch.from_numpy(np.array([0.7, 0.3], dtype=np.float32)).cuda(), 
        semantic_loss = F.cross_entropy(
            semantic_scores, semantic_labels, ignore_index=self.ignore_label)
        losses['semantic_loss'] = semantic_loss

        pos_inds = instance_labels != self.ignore_label
        if pos_inds.sum() == 0:
            offset_loss = 0 * pt_offsets.sum()
        else:
            offset_loss = F.l1_loss(
                pt_offsets[pos_inds], pt_offset_labels[pos_inds], reduction='sum') / pos_inds.sum()
        losses['offset_loss'] = offset_loss
        return losses


    def size_error(self, proposals_idx, proposals_offset,
                      instance_labels, instance_pointnum, instance_cls, instance_sizes, sizes):

        proposals_idx = proposals_idx[:, 1].cuda()
        proposals_offset = proposals_offset.cuda()

        # cal iou of clustered instance
        ious_on_cluster = get_mask_iou_on_cluster(proposals_idx, proposals_offset, instance_labels,
                                                  instance_pointnum)

        # filter out background instances
        fg_inds = (instance_cls != self.ignore_label)
        fg_ious_on_cluster = ious_on_cluster[:, fg_inds]

        # assign proposal to gt idx. -1: negative, 0 -> num_gts - 1: positive
        num_proposals = fg_ious_on_cluster.size(0)
        num_gts = fg_ious_on_cluster.size(1)
        assigned_gt_inds = fg_ious_on_cluster.new_full((num_proposals, ), -1, dtype=torch.long)

        # overlap > thr on fg instances are positive samples
        max_iou, argmax_iou = fg_ious_on_cluster.max(1)
        pos_inds = max_iou >= self.train_cfg.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_iou[pos_inds]
        
        # allow low-quality proposals with best iou to be as positive sample
        # in case pos_iou_thr is too high to achieve
        match_low_quality = getattr(self.train_cfg, 'match_low_quality', False)
        min_pos_thr = getattr(self.train_cfg, 'min_pos_thr', 0)
        if match_low_quality:
            gt_max_iou, gt_argmax_iou = fg_ious_on_cluster.max(0)
            for i in range(num_gts):
                if gt_max_iou[i] >= min_pos_thr:
                    assigned_gt_inds[gt_argmax_iou[i]] = i
        
        #compute size loss
        valid_objs = assigned_gt_inds[assigned_gt_inds >= 0]
        size_loss = F.huber_loss(torch.ravel(sizes[assigned_gt_inds >= 0]), instance_sizes[valid_objs,1], reduction='sum')
        
        return size_loss

    @force_fp32(apply_to=('cls_scores', 'mask_scores', 'iou_scores'))
    def instance_loss(self, cls_scores, mask_scores, iou_scores, proposals_idx, proposals_offset,
                      instance_labels, instance_pointnum, instance_cls, instance_batch_idxs, instance_sizes, sizes):
        losses = {}
        proposals = proposals_idx[:, 0]
        proposals_idx = proposals_idx[:, 1].cuda()
        proposals_offset = proposals_offset.cuda()

        # cal iou of clustered instance
        ious_on_cluster = get_mask_iou_on_cluster(proposals_idx, proposals_offset, instance_labels,
                                                  instance_pointnum)

        # filter out background instances
        fg_inds = (instance_cls != self.ignore_label)
        fg_instance_cls = instance_cls[fg_inds]
        fg_ious_on_cluster = ious_on_cluster[:, fg_inds]

        # assign proposal to gt idx. -1: negative, 0 -> num_gts - 1: positive
        num_proposals = fg_ious_on_cluster.size(0)
        num_gts = fg_ious_on_cluster.size(1)
        assigned_gt_inds = fg_ious_on_cluster.new_full((num_proposals, ), -1, dtype=torch.long)

        # overlap > thr on fg instances are positive samples
        max_iou, argmax_iou = fg_ious_on_cluster.max(1)
        pos_inds = max_iou >= self.train_cfg.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_iou[pos_inds]
        
        # allow low-quality proposals with best iou to be as positive sample
        # in case pos_iou_thr is too high to achieve
        match_low_quality = getattr(self.train_cfg, 'match_low_quality', False)
        min_pos_thr = getattr(self.train_cfg, 'min_pos_thr', 0)
        if match_low_quality:
            gt_max_iou, gt_argmax_iou = fg_ious_on_cluster.max(0)
            for i in range(num_gts):
                if gt_max_iou[i] >= min_pos_thr:
                    assigned_gt_inds[gt_argmax_iou[i]] = i

        # compute cls loss. follow detection convention: 0 -> K - 1 are fg, K is bg
        labels = fg_instance_cls.new_full((num_proposals, ), self.instance_classes - 1)
        pos_inds = assigned_gt_inds >= 0
        labels[pos_inds] = fg_instance_cls[assigned_gt_inds[pos_inds]]
        cls_loss = F.cross_entropy(cls_scores, labels)
        losses['cls_loss'] = cls_loss

        # compute mask loss
        mask_cls_label = labels[instance_batch_idxs.long()]
        slice_inds = torch.arange(
            0, mask_cls_label.size(0), dtype=torch.long, device=mask_cls_label.device)
        mask_scores_sigmoid_slice = mask_scores.sigmoid()[slice_inds, mask_cls_label]
        mask_label = get_mask_label(proposals_idx, proposals_offset, instance_labels, instance_cls,
                                    instance_pointnum, ious_on_cluster, self.train_cfg.pos_iou_thr)
        mask_label_weight = (mask_label != -1).float()
        mask_label[mask_label == -1.] = 0.5  # any value is ok
        mask_loss = F.binary_cross_entropy(
            mask_scores_sigmoid_slice, mask_label, weight=mask_label_weight, reduction='sum')
        mask_loss /= (mask_label_weight.sum() + 1)
        losses['mask_loss'] = mask_loss

        # compute iou score loss
        ious = get_mask_iou_on_pred(proposals_idx, proposals_offset, instance_labels,
                                    instance_pointnum, mask_scores_sigmoid_slice.detach())
        fg_ious = ious[:, fg_inds]
        gt_ious, _ = fg_ious.max(1)
        slice_inds = torch.arange(0, labels.size(0), dtype=torch.long, device=labels.device)
        iou_score_weight = (labels < self.instance_classes).float()
        iou_score_slice = iou_scores[slice_inds, labels]
        iou_score_loss = F.mse_loss(iou_score_slice, gt_ious, reduction='none')
        iou_score_loss = (iou_score_loss * iou_score_weight).sum() / (iou_score_weight.sum() + 1)
        losses['iou_score_loss'] = iou_score_loss
        
        #compute size loss
        valid_objs = assigned_gt_inds[assigned_gt_inds >= 0]
        size_loss = F.mse_loss(torch.ravel(sizes[assigned_gt_inds >= 0]).float(), instance_sizes[valid_objs,1].float(), reduction='sum')
        losses['size_loss'] = size_loss
        
        return losses

    def parse_losses(self, losses):
        loss = sum(v for v in losses.values())
        losses['loss'] = loss
        for loss_name, loss_value in losses.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            losses[loss_name] = loss_value.item()
        return loss, losses

    @cuda_cast
    def forward_test(self, batch_idxs, voxel_coords, p2v_map, v2p_map, coords_float, feats,
                     semantic_labels, instance_labels, pt_offset_labels, spatial_shape, batch_size,
                     scan_ids, **kwargs):
        feats = torch.cat((feats, coords_float), 1)
        voxel_feats = voxelization(feats, p2v_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        semantic_scores, pt_offsets, output_feats = self.forward_backbone(
            input, v2p_map, x4_split=self.test_cfg.x4_split)
        if self.test_cfg.x4_split:
            coords_float = self.merge_4_parts(coords_float)
            semantic_labels = self.merge_4_parts(semantic_labels)
            instance_labels = self.merge_4_parts(instance_labels)
            pt_offset_labels = self.merge_4_parts(pt_offset_labels)
        semantic_preds = semantic_scores.max(1)[1]
        print('Semantic scores shape:', semantic_scores.shape, semantic_preds.sum())
        ret = dict(
            scan_id=scan_ids[0],
            coords_float=coords_float.cpu().numpy(),
            semantic_preds=semantic_preds.cpu().numpy(),
            semantic_labels=semantic_labels.cpu().numpy(),
            offset_preds=pt_offsets.cpu().numpy(),
            offset_labels=pt_offset_labels.cpu().numpy(),
            instance_labels=instance_labels.cpu().numpy())
        if not self.semantic_only:
            proposals_idx, proposals_offset = self.forward_grouping(semantic_scores, pt_offsets,
                                                                    batch_idxs, coords_float,
                                                                    self.grouping_cfg)
            proposals_idx, proposals_offset = self.best_proposals(proposals_idx, proposals_offset)
            inst_feats, inst_map = self.clusters_voxelization(proposals_idx, proposals_offset,
                                                              output_feats, coords_float,
                                                              **self.instance_voxel_cfg)
            # print(inst_feats, inst_map)
            if self.train_cfg.full_feats_size:
                off_feats = self.get_offset_full_feats(semantic_scores, proposals_offset, coords_float, proposals_idx, pt_offsets)
            else:
                off_feats = self.get_offset_4feats(semantic_scores, proposals_offset, proposals_idx, pt_offsets)
                
            _, cls_scores, iou_scores, mask_scores, sizes = self.forward_instance(
                inst_feats, inst_map, off_feats, proposals_offset)
            
            pred_instances = self.get_instances(scan_ids[0], proposals_idx, semantic_scores,
                                                cls_scores, iou_scores, mask_scores, sizes)
            gt_instances = self.get_gt_instances(semantic_labels, instance_labels)
            # size_regression_err = self.size_error(proposals_idx, proposals_offset,
            #           instance_labels, instance_pointnum, instance_cls, instance_sizes, sizes)
            ret.update(dict(pred_instances=pred_instances, gt_instances=gt_instances))
        return ret

    def forward_backbone(self, input, input_map, x4_split=False):
        if x4_split:
            output_feats = self.forward_4_parts(input, input_map)
            output_feats = self.merge_4_parts(output_feats)
        else:
            output = self.input_conv(input)
            output = self.unet(output)
            output = self.output_layer(output)
            output_feats = output.features[input_map.long()]

        semantic_scores = self.semantic_linear(output_feats)
        pt_offsets = self.offset_linear(output_feats)
        return semantic_scores, pt_offsets, output_feats

    def forward_4_parts(self, x, input_map):
        """Helper function for s3dis: devide and forward 4 parts of a scene."""
        outs = []
        for i in range(4):
            inds = x.indices[:, 0] == i
            feats = x.features[inds]
            coords = x.indices[inds]
            coords[:, 0] = 0
            x_new = spconv.SparseConvTensor(
                indices=coords, features=feats, spatial_shape=x.spatial_shape, batch_size=1)
            out = self.input_conv(x_new)
            out = self.unet(out)
            out = self.output_layer(out)
            outs.append(out.features)
        outs = torch.cat(outs, dim=0)
        return outs[input_map.long()]

    def merge_4_parts(self, x):
        """Helper function for s3dis: take output of 4 parts and merge them."""
        inds = torch.arange(x.size(0), device=x.device)
        p1 = inds[::4]
        p2 = inds[1::4]
        p3 = inds[2::4]
        p4 = inds[3::4]
        ps = [p1, p2, p3, p4]
        x_split = torch.split(x, [p.size(0) for p in ps])
        x_new = torch.zeros_like(x)
        for i, p in enumerate(ps):
            x_new[p] = x_split[i]
        return x_new

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

    def forward_instance(self, inst_feats, inst_map, pt_offsets, proposals_off):
        feats = self.tiny_unet(inst_feats)
        feats = self.tiny_unet_outputlayer(feats)

        # predict mask scores
        mask_scores = self.mask_linear(feats.features)
        mask_scores = mask_scores[inst_map.long()]
        instance_batch_idxs = feats.indices[:, 0][inst_map.long()]

        featsPooling = self.global_pool(feats)
        cls_scores = self.cls_linear(featsPooling)
        iou_scores = self.iou_score_linear(featsPooling)
        if self.train_cfg.full_feats_size:
            if self.train_cfg.size_feats > 32:
                coords_and_feat = proposals_off.new_full((len(proposals_off) - 1, self.test_cfg.min_npoint, self.train_cfg.size_feats), 0, dtype=torch.float)
                mapped_feats = feats.features[inst_map.long()]
                for i in range(len(proposals_off) - 1):
                    coords_and_feat[i,:,0:32] = mapped_feats[proposals_off[i].long():(proposals_off[i]+self.test_cfg.min_npoint).long(),:]
                    coords_and_feat[i,:,32:self.train_cfg.size_feats] = pt_offsets[i]
            else:
                coords_and_feat = pt_offsets
            coords_and_feat = coords_and_feat.permute(0,2,1).cuda()
            # print('shape de cords and feat', coords_and_feat.shape)
            # coords_and_feat = torch.hstack((aux.cuda(),pt_offsets))
        else:
            coords_and_feat = torch.hstack((featsPooling,pt_offsets))
        regression_size = self.size_regression(coords_and_feat)

        return instance_batch_idxs, cls_scores, iou_scores, mask_scores, regression_size

    @force_fp32(apply_to=('semantic_scores', 'cls_scores', 'iou_scores', 'mask_scores'))
    def get_instances(self, scan_id, proposals_idx, semantic_scores, cls_scores, iou_scores,
                      mask_scores, sizes):
        num_instances = cls_scores.size(0)
        num_points = semantic_scores.size(0)
        cls_scores = cls_scores.softmax(1)
        semantic_pred = semantic_scores.max(1)[1]
        cls_pred_list, score_pred_list, mask_pred_list, sizes_list = [], [], [], []
        print('Numero de instancias:', num_instances)
        for i in range(self.instance_classes):
            if i in self.sem2ins_classes:
                cls_pred = cls_scores.new_tensor([i + 1], dtype=torch.long)
                score_pred = cls_scores.new_tensor([1.], dtype=torch.float32)
                mask_pred = (semantic_pred == i)[None, :].int()
            else:
                cls_pred = cls_scores.new_full((num_instances, ), i + 1, dtype=torch.long)
                cur_cls_scores = cls_scores[:, i]
                cur_iou_scores = iou_scores[:, i]
                cur_mask_scores = mask_scores[:, i]
                score_pred = cur_cls_scores * cur_iou_scores.clamp(0, 1)
                
                mask_inds = cur_mask_scores > self.test_cfg.mask_score_thr
                cur_proposals_idx = proposals_idx[mask_inds].long()
                
                uniques, count = torch.unique(cur_proposals_idx[:,0], return_counts=True)
                preserve = count[count >= self.test_cfg.min_npoint]
                # print('valid instances: ', preserve)
                remove_instances = torch.any(torch.stack([torch.eq(cur_proposals_idx, elem).logical_or_(torch.eq(cur_proposals_idx, elem)) for elem in uniques[count >= self.test_cfg.min_npoint]], dim=0), dim = 0)
                cur_proposals_idx = cur_proposals_idx[remove_instances[:,0]]
                new_labels = np.concatenate([[ind]*preserve[ind] for ind in range(len(preserve))]).ravel()
                cur_proposals_idx[:,0] = torch.from_numpy(new_labels).long()
                    
                inds_valid = count >= self.test_cfg.min_npoint
                sizes = sizes[inds_valid]
                cls_pred = cls_pred[inds_valid]
                score_pred = score_pred[inds_valid]
                cur_cls_scores = cur_cls_scores[inds_valid]
                print('Numero de instancias post-process:', cls_pred.size(0))
                mask_pred = torch.zeros((cls_pred.size(0), num_points), dtype=torch.int, device='cuda')
                mask_pred[cur_proposals_idx[:, 0], cur_proposals_idx[:, 1]] = 1             
                
                # filter low score instance
                inds = cur_cls_scores > self.test_cfg.cls_score_thr
                sizes = sizes[inds]
                cls_pred = cls_pred[inds]
                score_pred = score_pred[inds]
                mask_pred = mask_pred[inds]

            sizes_list.append(sizes)
            cls_pred_list.append(cls_pred)
            score_pred_list.append(score_pred)
            mask_pred_list.append(mask_pred)
        cls_pred = torch.cat(cls_pred_list).cpu().numpy()
        sizes = torch.cat(sizes_list).cpu().numpy()
        score_pred = torch.cat(score_pred_list).cpu().numpy()
        mask_pred = torch.cat(mask_pred_list).cpu().numpy()

        instances = []
        for i in range(cls_pred.shape[0]):
            pred = {}
            pred['scan_id'] = scan_id
            pred['size'] = sizes[i][0]
            pred['label_id'] = cls_pred[i]
            pred['conf'] = score_pred[i]
            # rle encode mask to save memory
            pred['pred_mask'] = rle_encode(mask_pred[i])
            instances.append(pred)
        return instances

    def get_gt_instances(self, semantic_labels, instance_labels):
        """Get gt instances for evaluation."""
        # convert to evaluation format 0: ignore, 1->N: valid
        label_shift = self.semantic_classes - self.instance_classes
        semantic_labels = semantic_labels - label_shift + 1
        semantic_labels[semantic_labels < 0] = 0
        instance_labels += 1
        ignore_inds = instance_labels < 0
        # scannet encoding rule
        gt_ins = semantic_labels * 1000 + instance_labels
        gt_ins[ignore_inds] = 0
        gt_ins = gt_ins.cpu().numpy()
        return gt_ins

    @force_fp32(apply_to='feats')
    def clusters_voxelization(self,
                              clusters_idx,
                              clusters_offset,
                              feats,
                              coords,
                              scale,
                              spatial_shape,
                              rand_quantize=False):
        batch_idx = clusters_idx[:, 0].cuda().long()
        c_idxs = clusters_idx[:, 1].cuda()
        feats = feats[c_idxs.long()]
        coords = coords[c_idxs.long()]

        coords_min = sec_min(coords, clusters_offset.cuda())
        coords_max = sec_max(coords, clusters_offset.cuda())

        # 0.01 to ensure voxel_coords < spatial_shape
        # print(f'Scale: {scale}, Spatial shape: {spatial_shape}, Coord range scaled: {((coords_max - coords_min) / spatial_shape).max(1)[0]}')
        clusters_scale = 1 / ((coords_max - coords_min) / spatial_shape).max(1)[0] - 0.01
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        coords_min = coords_min * clusters_scale[:, None]
        coords_max = coords_max * clusters_scale[:, None]
        clusters_scale = clusters_scale[batch_idx]
        coords = coords * clusters_scale[:, None]

        if rand_quantize:
            # after this, coords.long() will have some randomness
            range = coords_max - coords_min
            coords_min -= torch.clamp(spatial_shape - range - 0.001, min=0) * torch.rand(3).cuda()
            coords_min -= torch.clamp(spatial_shape - range + 0.001, max=0) * torch.rand(3).cuda()
        coords_min = coords_min[batch_idx]
        coords -= coords_min
        # assert coords.shape.numel() == ((coords >= 0) * (coords < spatial_shape)).sum()
        coords = coords.long()
        coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), coords.cpu()], 1)

        out_coords, inp_map, out_map = voxelization_idx(coords, int(clusters_idx[-1, 0]) + 1)
        out_feats = voxelization(feats, out_map.cuda())
        spatial_shape = [spatial_shape] * 3
        voxelization_feats = spconv.SparseConvTensor(out_feats,
                                                     out_coords.int().cuda(), spatial_shape,
                                                     int(clusters_idx[-1, 0]) + 1)
        return voxelization_feats, inp_map

    def get_batch_offsets(self, batch_idxs, bs):
        batch_offsets = torch.zeros(bs + 1).int().cuda()
        for i in range(bs):
            batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
        assert batch_offsets[-1] == batch_idxs.shape[0]
        return batch_offsets

    @force_fp32(apply_to=('x'))
    def global_pool(self, x, expand=False):
        indices = x.indices[:, 0]
        batch_counts = torch.bincount(indices)
        batch_offset = torch.cumsum(batch_counts, dim=0)
        pad = batch_offset.new_full((1, ), 0)
        batch_offset = torch.cat([pad, batch_offset]).int()
        x_pool = global_avg_pool(x.features, batch_offset)
        if not expand:
            return x_pool

        x_pool_expand = x_pool[indices.long()]
        x.features = torch.cat((x.features, x_pool_expand), dim=1)
        return x
    
    def best_proposals(self,prop_idx, prop_off):
        ids, count = torch.unique(prop_idx[:,0], return_counts=True)
        preserve = count[count >= self.test_cfg.min_npoint]
        remove_instances = torch.any(torch.stack([torch.eq(prop_idx, elem).logical_or_(torch.eq(prop_idx, elem)) for elem in ids[count >= self.test_cfg.min_npoint]], dim=0), dim = 0)
        prop_idx = prop_idx[remove_instances[:,0]]
        new_labels = np.concatenate([[ind]*preserve[ind] for ind in range(len(preserve))]).ravel()
        prop_idx[:,0] = torch.from_numpy(new_labels)
        offset = torch.cumsum(preserve, dim=0)
        offset = torch.cat([torch.tensor([0]), offset], dim=0).type(torch.int32)
        assert prop_idx.shape[0] == offset[-1]
        return prop_idx, offset
