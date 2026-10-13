import torch
import torch.nn as nn
from mmdet.models import HEADS
from mmdet.models.utils.transformer import inverse_sigmoid

from projects.mmdet3d_plugin.models.dense_heads.streampetr_head import StreamPETRHead
from projects.mmdet3d_plugin.models.utils.deform_cross_attn import BEVDeformCrossAttention
from projects.mmdet3d_plugin.models.utils.misc import topk_gather
from projects.mmdet3d_plugin.models.utils.positional_encoding import (
    nerf_positional_encoding, pos2posemb1d, pos2posemb3d)


@HEADS.register_module()
class StreamPETRLidarHead(StreamPETRHead):
    """在每一层图像交叉注意力之后，再对 PointPillars BEV 做可变形交叉注意力。

    图像交叉注意力由配置决定：PETRMultiheadAttention 或 ImageDeformCrossAttention。
    时序打开时，第一层只看当前帧；这一层 refine 之后，用历史框换掉低分 query。
    每一层的预测中心写回 reference，下一层围着新框采样。
    """

    def __init__(self, lidar_in_channels=384, lidar_num_points=4, **kwargs):
        super().__init__(**kwargs)
        self.defer_temporal = True
        self.lidar_embed = nn.Sequential(
            nn.Linear(lidar_in_channels, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        n_layers = len(self.transformer.decoder.layers)
        self.lidar_cross_attn = nn.ModuleList([
            BEVDeformCrossAttention(
                embed_dims=self.embed_dims, num_heads=8,
                num_points=lidar_num_points, dropout=0.1)
            for _ in range(n_layers)
        ])
        self.lidar_norm = nn.ModuleList([
            nn.LayerNorm(self.embed_dims) for _ in range(n_layers)
        ])

    def _lidar_tokens(self, lidar_bev):
        _, _, h, w = lidar_bev.shape
        memory = self.lidar_embed(lidar_bev.flatten(2).transpose(1, 2))
        return memory, h, w

    def _decode_layer_with_lidar(
            self, layer, lidar_attn, lidar_ln, query, key, value,
            query_pos, key_pos, temp_memory, temp_pos, attn_masks,
            lidar_key, lidar_hw, reference_points, lidar2img, img_hw, img_pad):
        if attn_masks is None:
            attn_masks = [None for _ in range(layer.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [attn_masks for _ in range(layer.num_attn)]
        norm_index = attn_index = ffn_index = 0
        identity = query
        after_image = False
        for op in layer.operation_order:
            if op == 'self_attn':
                if temp_memory is not None:
                    temp_key = temp_value = torch.cat([query, temp_memory], dim=0)
                    tpos = torch.cat([query_pos, temp_pos], dim=0)
                else:
                    temp_key = temp_value = query
                    tpos = query_pos
                query = layer.attentions[attn_index](
                    query, temp_key, temp_value,
                    identity if layer.pre_norm else None,
                    query_pos=query_pos, key_pos=tpos,
                    attn_mask=attn_masks[attn_index])
                attn_index += 1
                identity = query
            elif op == 'norm':
                query = layer.norms[norm_index](query)
                norm_index += 1
                if after_image:
                    if lidar_key is not None:
                        query = lidar_attn(
                            query, value=lidar_key,
                            identity=identity if layer.pre_norm else None,
                            query_pos=query_pos,
                            reference_points=reference_points[..., :2],
                            spatial_hw=lidar_hw)
                        query = lidar_ln(query)
                    identity = query
                    after_image = False
            elif op == 'cross_attn':
                query = layer.attentions[attn_index](
                    query, key, value,
                    identity if layer.pre_norm else None,
                    query_pos=query_pos, key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    reference_points=reference_points,
                    lidar2img=lidar2img, img_hw=img_hw, img_pad=img_pad)
                attn_index += 1
                identity = query
                after_image = True
            elif op == 'ffn':
                query = layer.ffns[ffn_index](
                    query, identity if layer.pre_norm else None)
                ffn_index += 1
        return query

    def _identity_ego_motion(self, reference):
        if not self.with_ego_pos:
            return None
        eye = torch.eye(4, device=reference.device, dtype=reference.dtype)
        eye = eye.view(1, 1, 4, 4).expand(
            reference.shape[0], reference.shape[1], 4, 4).contiguous()
        motion = torch.cat([
            torch.zeros_like(reference), eye[..., :3, :].flatten(-2)], dim=-1)
        return nerf_positional_encoding(motion)

    def _refresh_query_pos(self, reference, slot_time, slot_motion):
        pos = self.query_embedding(pos2posemb3d(reference))
        if slot_motion is not None:
            pos = self.ego_pose_pe(pos, slot_motion)
        pos = pos + self.time_embedding(pos2posemb1d(slot_time))
        return pos.transpose(0, 1).contiguous()

    def _layer_prediction(self, level, query, reference_points):
        hidden = query.transpose(0, 1)
        cls_score = self.cls_branches[level](hidden)
        delta = self.reg_branches[level](hidden)
        base = inverse_sigmoid(reference_points.clamp(1e-4, 1 - 1e-4))
        centers = (delta[..., 0:3] + base).sigmoid()
        coords = torch.cat([centers, delta[..., 3:]], dim=-1)
        return cls_score, coords

    def _splice_temporal(
            self, query, query_pos, reference, cls_score, pad, prev_exists,
            slot_time, slot_motion, temp_memory, temp_pos):
        """用历史框换掉当前帧里置信度较低的 query。总数不变，去噪 query 留在最前面。"""
        k = self.num_propagated
        prev = prev_exists.reshape(-1).to(device=query.device, dtype=query.dtype)
        temp_memory = temp_memory.to(dtype=query.dtype)
        temp_pos = temp_pos.to(dtype=query_pos.dtype)
        if pad > 0:
            dn_query, cur_query = query[:pad], query[pad:]
            dn_pos, cur_pos = query_pos[:pad], query_pos[pad:]
            dn_ref, cur_ref = reference[:, :pad], reference[:, pad:]
            dn_time, cur_time = slot_time[:, :pad], slot_time[:, pad:]
            dn_motion = None if slot_motion is None else slot_motion[:, :pad]
            cur_motion = None if slot_motion is None else slot_motion[:, pad:]
            cur_cls = cls_score[:, pad:]
        else:
            dn_query = dn_pos = dn_ref = dn_time = dn_motion = None
            cur_query, cur_pos, cur_ref = query, query_pos, reference
            cur_time, cur_motion, cur_cls = slot_time, slot_motion, cls_score
        n_keep = cur_query.shape[0] - k
        if n_keep <= 0:
            return query, query_pos, reference, slot_time, slot_motion
        index = torch.topk(cur_cls.detach().max(dim=-1).values, n_keep, dim=1).indices

        def gather_batch(feat):
            pick = index.unsqueeze(-1).expand(-1, -1, feat.shape[-1])
            return torch.gather(feat, 1, pick)

        kept_query = gather_batch(cur_query.transpose(0, 1)).transpose(0, 1)
        kept_pos = gather_batch(cur_pos.transpose(0, 1)).transpose(0, 1)
        kept_ref = gather_batch(cur_ref)
        kept_time = gather_batch(cur_time)
        kept_motion = None if cur_motion is None else gather_batch(cur_motion)
        temp_ref = self._deferred_reference[:, :k].to(dtype=cur_ref.dtype).clamp(1e-4, 1 - 1e-4)
        temp_time = self.memory_timestamp[:, :k].to(dtype=cur_time.dtype)
        merged_query = torch.cat([temp_memory[:, :k].transpose(0, 1), kept_query], dim=0)
        merged_pos = torch.cat([temp_pos[:, :k].transpose(0, 1), kept_pos], dim=0)
        merged_ref = torch.cat([temp_ref, kept_ref], dim=1)
        merged_time = torch.cat([temp_time, kept_time], dim=1)
        merged_motion = None
        if kept_motion is not None and self._deferred_ego_motion is not None:
            temp_motion = self._deferred_ego_motion[:, :k].to(dtype=kept_motion.dtype)
            merged_motion = torch.cat([temp_motion, kept_motion], dim=1)
        alive_q = prev.view(1, -1, 1)
        alive_b = prev.view(-1, 1, 1)
        merged_query = merged_query * alive_q + cur_query * (1 - alive_q)
        merged_pos = merged_pos * alive_q + cur_pos * (1 - alive_q)
        merged_ref = merged_ref * alive_b + cur_ref * (1 - alive_b)
        merged_time = merged_time * alive_b + cur_time * (1 - alive_b)
        if merged_motion is not None:
            merged_motion = merged_motion * alive_b + cur_motion * (1 - alive_b)
        if pad > 0:
            query = torch.cat([dn_query, merged_query], dim=0)
            query_pos = torch.cat([dn_pos, merged_pos], dim=0)
            reference = torch.cat([dn_ref, merged_ref], dim=1)
            slot_time = torch.cat([dn_time, merged_time], dim=1)
            if dn_motion is not None and merged_motion is not None:
                slot_motion = torch.cat([dn_motion, merged_motion], dim=1)
        else:
            query, query_pos, reference = merged_query, merged_pos, merged_ref
            slot_time = merged_time
            if merged_motion is not None:
                slot_motion = merged_motion
        return (
            query.contiguous(), query_pos.contiguous(), reference.contiguous(),
            slot_time, slot_motion)

    def _run_temporal_decoder(
            self, memory, tgt, query_pos, pos_embed, attn_mask,
            temp_memory, temp_pos, lidar_bev, reference_points,
            lidar2img, img_hw, img_pad, prev_exists, pad):
        memory = memory.transpose(0, 1).contiguous()
        query_pos = query_pos.transpose(0, 1).contiguous()
        pos_embed = pos_embed.transpose(0, 1).contiguous()
        query = tgt.transpose(0, 1).contiguous()
        rest = 0
        temp_key = temp_key_pos = None
        if temp_memory is not None and self.num_propagated > 0:
            rest = temp_memory.shape[1] - self.num_propagated
            if rest > 0:
                temp_key = temp_memory[:, self.num_propagated:].transpose(0, 1).contiguous()
                temp_key_pos = temp_pos[:, self.num_propagated:].transpose(0, 1).contiguous()
        later_mask = attn_mask
        if attn_mask is not None and rest > 0:
            query_n = attn_mask.shape[0]
            later_mask = attn_mask.new_zeros(query_n, query_n + rest)
            later_mask[:, :query_n] = attn_mask
        lidar_mem, lidar_hw = None, None
        if lidar_bev is not None:
            lidar_mem, lidar_h, lidar_w = self._lidar_tokens(lidar_bev)
            lidar_mem = lidar_mem.transpose(0, 1).contiguous()
            lidar_hw = (lidar_h, lidar_w)
        slot_time = reference_points.new_zeros(reference_points.shape[:2] + (1,))
        slot_motion = self._identity_ego_motion(reference_points)
        decoder = self.transformer.decoder
        intermediate, classes, coords = [], [], []
        n_layers = len(decoder.layers)
        for i, layer in enumerate(decoder.layers):
            use_memory = temp_key is not None and i > 0
            query = self._decode_layer_with_lidar(
                layer, self.lidar_cross_attn[i], self.lidar_norm[i],
                query, memory, memory, query_pos, pos_embed,
                temp_key if use_memory else None,
                temp_key_pos if use_memory else None,
                [later_mask if use_memory else attn_mask, None],
                lidar_mem, lidar_hw, reference_points, lidar2img, img_hw, img_pad)
            if decoder.post_norm is not None:
                query = decoder.post_norm(query)
            intermediate.append(query)
            cls_score, coord = self._layer_prediction(i, query, reference_points)
            classes.append(cls_score)
            coords.append(coord)
            reference_points = coord[..., 0:3].clamp(1e-4, 1 - 1e-4)
            if (
                    i == 0 and i + 1 < n_layers and self.use_temporal
                    and temp_memory is not None and self.num_propagated > 0):
                query_pos = self._refresh_query_pos(reference_points, slot_time, slot_motion)
                query, query_pos, reference_points, slot_time, slot_motion = self._splice_temporal(
                    query, query_pos, reference_points, cls_score, pad, prev_exists,
                    slot_time, slot_motion, temp_memory, temp_pos)
            elif i + 1 < n_layers:
                query_pos = self._refresh_query_pos(reference_points, slot_time, slot_motion)
        outs = torch.stack(intermediate).transpose(1, 2).contiguous()
        return outs, torch.stack(classes), torch.stack(coords)

    def forward(self, memory_center, img_metas, topk_indexes=None, **data):
        self.pre_update_memory(data)
        x = data['img_feats']
        b, n, c, h, w = x.shape
        memory = x.permute(0, 1, 3, 4, 2).reshape(b, n * h * w, c)
        memory = topk_gather(memory, topk_indexes)
        pos_embed, cone = self.position_embeding(data, memory_center, topk_indexes, img_metas)
        memory = self.memory_embed(memory)
        memory = self.spatial_alignment(memory, cone)
        pos_embed = self.featurized_pe(pos_embed, memory)
        reference_points = self.reference_points.weight
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(
            b, reference_points, img_metas)
        query_pos = self.query_embedding(pos2posemb3d(reference_points))
        tgt = torch.zeros_like(query_pos)
        tgt, query_pos, reference_points, temp_memory, temp_pos, _ = (
            self.temporal_alignment(query_pos, tgt, reference_points))
        lidar2img = data['lidar2img']
        if lidar2img.dim() == 3:
            lidar2img = lidar2img.reshape(b, n, 4, 4)
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        pad = mask_dict['pad_size'] if mask_dict is not None else 0
        outs_dec, all_cls_scores, all_bbox_preds = self._run_temporal_decoder(
            memory, tgt, query_pos, pos_embed, attn_mask,
            temp_memory, temp_pos, data.get('lidar_bev'), reference_points,
            lidar2img, (n, h, w), (pad_h, pad_w), data['prev_exists'], pad)
        outs_dec = torch.nan_to_num(outs_dec)
        all_cls_scores = torch.nan_to_num(all_cls_scores)
        all_bbox_preds = torch.nan_to_num(all_bbox_preds)
        rec_ego_pose = torch.eye(
            4, device=outs_dec.device).view(1, 1, 4, 4).repeat(b, self.num_query, 1, 1)
        centers = (
            all_bbox_preds[..., 0:3] * (self.pc_range[3:6] - self.pc_range[0:3])
            + self.pc_range[0:3])
        all_bbox_preds = torch.cat([centers, all_bbox_preds[..., 3:]], dim=-1)
        self.post_update_memory(
            data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict)
        if mask_dict and mask_dict['pad_size'] > 0:
            pad = mask_dict['pad_size']
            mask_dict['output_known_lbs_bboxes'] = (
                all_cls_scores[:, :, :pad, :], all_bbox_preds[:, :, :pad, :])
            return {
                'all_cls_scores': all_cls_scores[:, :, pad:, :],
                'all_bbox_preds': all_bbox_preds[:, :, pad:, :],
                'dn_mask_dict': mask_dict,
            }
        return {
            'all_cls_scores': all_cls_scores,
            'all_bbox_preds': all_bbox_preds,
            'dn_mask_dict': None,
        }
