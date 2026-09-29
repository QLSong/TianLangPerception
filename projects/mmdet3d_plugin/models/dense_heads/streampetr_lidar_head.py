import torch
import torch.nn as nn
from mmdet.models import HEADS
from mmdet.models.utils.transformer import inverse_sigmoid

from projects.mmdet3d_plugin.models.dense_heads.streampetr_head import StreamPETRHead
from projects.mmdet3d_plugin.models.utils.deform_cross_attn import BEVDeformCrossAttention
from projects.mmdet3d_plugin.models.utils.misc import topk_gather
from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb3d


@HEADS.register_module()
class StreamPETRLidarHead(StreamPETRHead):
    """在每一层图像交叉注意力之后，再对 PointPillars BEV 做可变形交叉注意力。

    图像交叉注意力由配置决定：PETRMultiheadAttention 或 ImageDeformCrossAttention。
    """

    def __init__(self, lidar_in_channels=384, lidar_num_points=4, **kwargs):
        super().__init__(**kwargs)
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

    def _run_temporal_decoder(
            self, memory, tgt, query_pos, pos_embed, attn_mask,
            temp_memory, temp_pos, lidar_bev, reference_points,
            lidar2img, img_hw, img_pad):
        memory = memory.transpose(0, 1).contiguous()
        query_pos = query_pos.transpose(0, 1).contiguous()
        pos_embed = pos_embed.transpose(0, 1).contiguous()
        tgt = tgt.transpose(0, 1).contiguous()
        if temp_memory is not None:
            temp_memory = temp_memory.transpose(0, 1).contiguous()
            temp_pos = temp_pos.transpose(0, 1).contiguous()
        lidar_mem, lidar_hw = None, None
        if lidar_bev is not None:
            lidar_mem, lidar_h, lidar_w = self._lidar_tokens(lidar_bev)
            lidar_mem = lidar_mem.transpose(0, 1).contiguous()
            lidar_hw = (lidar_h, lidar_w)
        decoder = self.transformer.decoder
        query = tgt
        intermediate = []
        attn_masks = [attn_mask, None]
        for i, layer in enumerate(decoder.layers):
            query = self._decode_layer_with_lidar(
                layer, self.lidar_cross_attn[i], self.lidar_norm[i],
                query, memory, memory, query_pos, pos_embed,
                temp_memory, temp_pos, attn_masks, lidar_mem, lidar_hw,
                reference_points, lidar2img, img_hw, img_pad)
            if decoder.post_norm is not None:
                intermediate.append(decoder.post_norm(query))
            else:
                intermediate.append(query)
        return torch.stack(intermediate).transpose(1, 2).contiguous()

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
        tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose = (
            self.temporal_alignment(query_pos, tgt, reference_points))
        lidar2img = data['lidar2img']
        if lidar2img.dim() == 3:
            lidar2img = lidar2img.reshape(b, n, 4, 4)
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        outs_dec = self._run_temporal_decoder(
            memory, tgt, query_pos, pos_embed, attn_mask,
            temp_memory, temp_pos, data.get('lidar_bev'), reference_points,
            lidar2img, (n, h, w), (pad_h, pad_w))
        outs_dec = torch.nan_to_num(outs_dec)
        outputs_classes, outputs_coords = [], []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            tmp = self.reg_branches[lvl](outs_dec[lvl])
            centers = (tmp[..., 0:3] + reference[..., 0:3]).sigmoid()
            tmp = torch.cat([centers, tmp[..., 3:]], dim=-1)
            outputs_classes.append(outputs_class)
            outputs_coords.append(tmp)
        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
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
