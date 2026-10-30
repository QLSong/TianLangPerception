import copy

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
    """相机 + 激光雷达的时序检测头。

    每一层的顺序是：自注意力、相机交叉注意力、激光雷达 BEV 可变形交叉注意力、FFN。
    相机交叉注意力默认看原图 token，由配置决定是 PETRMultiheadAttention
    或 ImageDeformCrossAttention。use_img_bev 为真时不看原图，改为对
    PlaneViewTransformer 输出的相机 BEV 做可变形交叉注意力。
    激光雷达 BEV 来自 PointPillars，reference 的 xy 按 pc_range 归一化后直接对应画布。

    时序打开时（defer_temporal），第 0 层只看当前帧。这一层 refine 之后，
    用上一帧 memory 里最靠前的 num_propagated 个历史框换掉当前低分 query。
    去噪 query 始终留在序列最前面，不参与替换。
    每一层预测出的中心会 detach 后写回 reference，下一层围着新框采样。
    分类和回归头按层分开，因为后面的层预测的是相对上一层中心的小位移。

    张量排布：
    - 注意力里的 query / query_pos 是 (query 数, batch, 通道)，batch 在中间。
    - reference、分类分数、框是 (batch, query 数, …)，batch 在最前。
    - reference 的 xyz 在 pc_range 内归一化到 (0, 1)。损失和 memory 用的是米制中心。
    """

    def __init__(self, lidar_in_channels=384, lidar_num_points=4,
                 use_img_bev=False, img_bev_channels=64, img_bev_num_points=4, **kwargs):
        super().__init__(**kwargs)
        self.use_img_bev = bool(use_img_bev)
        # False 时检测只看雷达 BEV，跳过图像交叉注意力。
        self.attend_image = True
        # 历史框等到第 0 层 refine 之后再拼进来，temporal_alignment 里先不拼接。
        self.defer_temporal = True
        # PointPillars 输出 384 通道，压到和图像 query 相同的 embed_dims。
        self.lidar_embed = nn.Sequential(
            nn.Linear(lidar_in_channels, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        n_layers = len(self.transformer.decoder.layers)
        # 每层一块 BEV 可变形注意力，插在图像交叉注意力后面的 LayerNorm 之后。
        self.lidar_cross_attn = nn.ModuleList([
            BEVDeformCrossAttention(
                embed_dims=self.embed_dims, num_heads=8,
                num_points=lidar_num_points, dropout=0.1)
            for _ in range(n_layers)
        ])
        self.lidar_norm = nn.ModuleList([
            nn.LayerNorm(self.embed_dims) for _ in range(n_layers)
        ])
        # 相机 BEV 走和激光雷达同一套可变形注意力，通道先压到 embed_dims。
        self.img_bev_embed = None
        self.img_bev_cross_attn = None
        if self.use_img_bev:
            self.img_bev_embed = nn.Sequential(
                nn.Linear(img_bev_channels, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims),
            )
            self.img_bev_cross_attn = nn.ModuleList([
                BEVDeformCrossAttention(
                    embed_dims=self.embed_dims, num_heads=8,
                    num_points=img_bev_num_points, dropout=0.1)
                for _ in range(n_layers)
            ])
            for layer in self.transformer.decoder.layers:
                layer.attentions[1] = nn.Identity()
        # 父类把同一个预测头重复放进 ModuleList。这里按层复制成独立权重。
        self.num_pred = n_layers
        self.cls_branches = nn.ModuleList(
            [copy.deepcopy(self.cls_branches[0]) for _ in range(n_layers)])
        self.reg_branches = nn.ModuleList(
            [copy.deepcopy(self.reg_branches[0]) for _ in range(n_layers)])

    def init_weights(self):
        super().init_weights()
        # 最后一层线性置零，训练开始时位移为 0，框先落在当前 reference 上。
        for branch in self.reg_branches:
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)

    def _bev_tokens(self, bev, embed):
        """(B, C, H, W) 的 BEV 铺成 (B, H*W, embed) 的 token。H 是 y，W 是 x。"""
        _, _, h, w = bev.shape
        memory = embed(bev.flatten(2).transpose(1, 2))
        return memory, h, w

    def _lidar_tokens(self, lidar_bev):
        return self._bev_tokens(lidar_bev, self.lidar_embed)

    def _decode_layer_with_lidar(
            self, layer, lidar_attn, lidar_ln, query, key, value,
            query_pos, key_pos, temp_memory, temp_pos, attn_masks,
            lidar_key, lidar_hw, reference_points, lidar2img, img_hw, img_pad,
            img_bev_key=None, img_bev_hw=None, img_bev_attn=None, attend_image=True):
        """跑一层 decoder，并在图像交叉注意力的 LayerNorm 之后加上 BEV 注意力。

        配置里的 operation_order 是 self_attn、norm、cross_attn、norm、ffn、norm。
        after_image 在图像交叉注意力结束时置位，下一次遇到 norm 时先做完这个
        LayerNorm，再对 BEV 采样。BEV 注意力自己再过一层 LayerNorm，然后才进 FFN。
        pre_norm 为假时，各注意力模块内部用当前 query 做残差，这里传入的 identity 不用。

        temp_memory 只在第 1 层及之后传入。它是 memory 里去掉已拼进 query 的那
        num_propagated 个之后的剩余条目，作为自注意力的额外 key。
        """
        if attn_masks is None:
            attn_masks = [None for _ in range(layer.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [attn_masks for _ in range(layer.num_attn)]
        norm_index = attn_index = ffn_index = 0
        identity = query
        after_image = False
        for op in layer.operation_order:
            if op == 'self_attn':
                # 有历史 memory 时，key 是当前 query 再接上剩余 memory。
                # 用 tpos 承接位置编码，避免覆盖循环外还要继续用的 temp_pos。
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
                    # xy 已按检测 pc_range 归一化，和柱面图画布是同一块范围。
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
                if attend_image and img_bev_key is not None:
                    # xy 按检测 pc_range 归一化，和相机 BEV 是同一块范围。
                    query = img_bev_attn(
                        query, value=img_bev_key,
                        identity=identity if layer.pre_norm else None,
                        query_pos=query_pos,
                        reference_points=reference_points[..., :2],
                        spatial_hw=img_bev_hw)
                elif attend_image:
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
        """当前帧 query 的自车运动编码：平移为 0，旋转是单位阵。

        返回值已经过 NeRF 位置编码，通道数和 memory 里的历史运动编码一致。
        历史框用的是 memory 中保存的速度、时间差和位姿，不走这个函数。
        """
        if not self.with_ego_pos:
            return None
        eye = torch.eye(4, device=reference.device, dtype=reference.dtype)
        eye = eye.view(1, 1, 4, 4).expand(
            reference.shape[0], reference.shape[1], 4, 4).contiguous()
        # 前 3 维是速度占位，后面 12 维是位姿矩阵的前三行。
        motion = torch.cat([
            torch.zeros_like(reference), eye[..., :3, :].flatten(-2)], dim=-1)
        return nerf_positional_encoding(motion)

    def _refresh_query_pos(self, reference, slot_time, slot_motion):
        """用新的归一化中心重算 query 位置编码，返回 (query 数, batch, 通道)。

        slot_time 对当前帧是 0，历史槽位是 memory 里的时间差。
        slot_motion 已经是 NeRF 编码，这里只再过一次 ego_pose_pe。
        """
        pos = self.query_embedding(pos2posemb3d(reference))
        if slot_motion is not None:
            pos = self.ego_pose_pe(pos, slot_motion)
        pos = pos + self.time_embedding(pos2posemb1d(slot_time))
        return pos.transpose(0, 1).contiguous()

    def _layer_prediction(self, level, query, reference_points):
        """一层的分类和框。中心是在 reference 上加位移再 sigmoid。

        query 进来是 (query 数, batch, 通道)。
        中心仍是 pc_range 内的归一化坐标。尺寸、朝向、速度是回归头的原始输出，
        和损失里 normalize_bbox 的 log 尺寸、sin/cos 朝向对齐。
        reference 在本层当作常数，层与层之间的截断在调用方完成。
        """
        hidden = query.transpose(0, 1)
        cls_score = self.cls_branches[level](hidden)
        delta = self.reg_branches[level](hidden)
        # cls_score = self.cls_branches(hidden)
        # delta = self.reg_branches(hidden)
        base = inverse_sigmoid(reference_points.clamp(1e-4, 1 - 1e-4))
        centers = (delta[..., 0:3] + base).sigmoid()
        coords = torch.cat([centers, delta[..., 3:]], dim=-1)
        return cls_score, coords

    def _splice_temporal(
            self, query, query_pos, reference, cls_score, pad, prev_exists,
            slot_time, slot_motion, temp_memory, temp_pos):
        """用历史框换掉当前帧里置信度较低的 query。总数不变，去噪 query 留在最前面。

        当前 query 按各类 logit 的最大值取前 num_query - num_propagated 个，
        前面接上 memory 最前的 num_propagated 个。这 num_propagated 个是上一帧
        写回 memory 时排在最前的 topk，内容、位置编码、中心、时间差和自车运动
        都换成历史的。被换掉的当前 query 从这一层之后不再出现。

        prev_exists 为 0 的样本是场景第一帧，按 batch 维退回替换前的当前 query，
        避免把别的场景或全零 memory 当成历史框。topk 的下标不回传梯度。
        """
        k = self.num_propagated
        prev = prev_exists.reshape(-1).to(device=query.device, dtype=query.dtype)
        temp_memory = temp_memory.to(dtype=query.dtype)
        temp_pos = temp_pos.to(dtype=query_pos.dtype)
        # 去噪段在序列前部，只在后面的匹配 query 里做筛选。
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
            # feat 是 (batch, query, 通道)。每个样本保留的下标可以不同。
            pick = index.unsqueeze(-1).expand(-1, -1, feat.shape[-1])
            return torch.gather(feat, 1, pick)

        kept_query = gather_batch(cur_query.transpose(0, 1)).transpose(0, 1)
        kept_pos = gather_batch(cur_pos.transpose(0, 1)).transpose(0, 1)
        kept_ref = gather_batch(cur_ref)
        kept_time = gather_batch(cur_time)
        kept_motion = None if cur_motion is None else gather_batch(cur_motion)
        # 历史中心在 temporal_alignment 里已变到当前自车并做了归一化。
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
        # query 是 (query 数, batch, 通道)，所以 prev 的 batch 维放在中间。
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
            lidar2img, img_hw, img_pad, prev_exists, pad, img_bev=None,
            attend_image=True):
        """逐层解码。第 0 层结束时插入历史框，之后每层用新中心刷新位置编码。

        进来的 memory、tgt、query_pos、pos_embed 都是 batch 在前，这里转成
        (长度, batch, 通道) 供 mmcv 注意力使用。
        返回的 outs 是 (层数, batch, query 数, 通道)，分类和框同样是层数在前。
        第 0 层的框按替换前的 query 顺序保存，后面的层按替换后的顺序保存。
        损失对每一层单独做匈牙利匹配，两套顺序各自对应自己的预测。
        """
        memory = memory.transpose(0, 1).contiguous()
        query_pos = query_pos.transpose(0, 1).contiguous()
        pos_embed = pos_embed.transpose(0, 1).contiguous()
        query = tgt.transpose(0, 1).contiguous()
        rest = 0
        temp_key = temp_key_pos = None
        # 最前 num_propagated 个会在第 0 层之后拼进 query，自注意力只额外看后面的条目。
        if temp_memory is not None and self.num_propagated > 0:
            rest = temp_memory.shape[1] - self.num_propagated
            if rest > 0:
                temp_key = temp_memory[:, self.num_propagated:].transpose(0, 1).contiguous()
                temp_key_pos = temp_pos[:, self.num_propagated:].transpose(0, 1).contiguous()
        later_mask = attn_mask
        if attn_mask is not None and rest > 0:
            # 去噪掩码仍盖住 query 之间的部分。多出来的 memory key 对所有 query 可见。
            query_n = attn_mask.shape[0]
            later_mask = attn_mask.new_zeros(query_n, query_n + rest)
            later_mask[:, :query_n] = attn_mask
        lidar_mem, lidar_hw = None, None
        if lidar_bev is not None:
            lidar_mem, lidar_h, lidar_w = self._lidar_tokens(lidar_bev)
            lidar_mem = lidar_mem.transpose(0, 1).contiguous()
            lidar_hw = (lidar_h, lidar_w)
        img_bev_key, img_bev_hw = None, None
        if img_bev is not None:
            img_bev_key, img_h, img_w = self._bev_tokens(img_bev, self.img_bev_embed)
            img_bev_key = img_bev_key.transpose(0, 1).contiguous()
            img_bev_hw = (img_h, img_w)
        # 当前帧槽位的时间差是 0，自车运动是单位阵。历史槽位在拼接时换掉。
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
                lidar_mem, lidar_hw, reference_points, lidar2img, img_hw, img_pad,
                img_bev_key, img_bev_hw,
                None if img_bev_key is None else self.img_bev_cross_attn[i],
                attend_image)
            # post_norm 只进入这一层的输出和预测头。下一层仍从原始 query 接着算。
            hidden = decoder.post_norm(query) if decoder.post_norm is not None else query
            intermediate.append(hidden)
            cls_score, coord = self._layer_prediction(i, hidden, reference_points)
            classes.append(cls_score)
            coords.append(coord)
            # 截断梯度，下一层把这个中心当作新的 anchor，不再回传到这一层的采样坐标。
            reference_points = coord[..., 0:3].detach().clamp(1e-4, 1 - 1e-4)
            if (
                    i == 0 and i + 1 < n_layers and self.use_temporal
                    and temp_memory is not None and self.num_propagated > 0):
                query_pos = self._refresh_query_pos(reference_points, slot_time, slot_motion)
                query, query_pos, reference_points, slot_time, slot_motion = self._splice_temporal(
                    query, query_pos, reference_points, cls_score, pad, prev_exists,
                    slot_time, slot_motion, temp_memory, temp_pos)
            elif i + 1 < n_layers:
                # 拼接之后，历史槽位的 slot_time / slot_motion 已经换成 memory 里的值。
                query_pos = self._refresh_query_pos(reference_points, slot_time, slot_motion)
        outs = torch.stack(intermediate).transpose(1, 2).contiguous()
        return outs, torch.stack(classes), torch.stack(coords)

    def forward(self, memory_center, img_metas, topk_indexes=None, **data):
        """单帧前向。训练时由 obtain_history_memory 按时间维逐帧调用。

        use_img_bev 为假时，img_feats 是 (B, 相机数, C, H, W)。
        为真时 img_bev 是 PlaneViewTransformer 的 (B, C, Y, X)，原图 token 不进检测头。
        lidar_bev 是 (B, C, H, W)，没有时间维。
        返回的分类和框都不含去噪 query。框的前三维是米制中心，其余维是回归原值。
        """
        # 把 memory 从上一帧的全局坐标变回当前自车。场景切换时 prev_exists 为 0，会清掉。
        self.pre_update_memory(data)
        attend_image = self.attend_image
        img_bev = data.get('img_bev') if self.use_img_bev and attend_image else None
        if attend_image and self.use_img_bev:
            if img_bev is None:
                raise RuntimeError('检测头需要 img_view_transformer 的 BEV，batch 里没有 img_bev')
            if img_bev.dim() == 5:
                img_bev = img_bev[:, -1]
            b = img_bev.shape[0]
            memory = img_bev.new_zeros(b, 1, self.embed_dims)
            pos_embed = memory
        elif not attend_image:
            lidar_bev = data.get('lidar_bev')
            if lidar_bev is None:
                raise RuntimeError('单雷达检测没有 lidar_bev')
            if lidar_bev.dim() == 5:
                lidar_bev = lidar_bev[:, -1]
                data['lidar_bev'] = lidar_bev
            b = lidar_bev.shape[0]
            memory = lidar_bev.new_zeros(b, 1, self.embed_dims)
            pos_embed = memory
        else:
            x = data['img_feats']
            b, n, c, h, w = x.shape
            # (B, 相机, H, W, C) 再摊成 token。相机顺序和 lidar2img 的相机维一致。
            memory = x.permute(0, 1, 3, 4, 2).reshape(b, n * h * w, c)
            memory = topk_gather(memory, topk_indexes)
            pos_embed, cone = self.position_embeding(data, memory_center, topk_indexes, img_metas)
            memory = self.memory_embed(memory)
            memory = self.spatial_alignment(memory, cone)
            pos_embed = self.featurized_pe(pos_embed, memory)
        reference_points = self.reference_points.weight
        # 训练时在 anchor 前面垫上去噪 query，并生成阻止匹配 query 看见去噪段的掩码。
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(
            b, reference_points, img_metas)
        query_pos = self.query_embedding(pos2posemb3d(reference_points))
        tgt = torch.zeros_like(query_pos)
        # defer_temporal 为真时，这里只准备历史 memory，不把它拼到当前 query 上。
        tgt, query_pos, reference_points, temp_memory, temp_pos, _ = (
            self.temporal_alignment(query_pos, tgt, reference_points))
        if self.use_img_bev or not attend_image:
            lidar2img, img_hw, img_pad = None, None, None
        else:
            lidar2img = data['lidar2img']
            if lidar2img.dim() == 3:
                lidar2img = lidar2img.reshape(b, n, 4, 4)
            pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
            img_hw, img_pad = (n, h, w), (pad_h, pad_w)
        pad = mask_dict['pad_size'] if mask_dict is not None else 0
        outs_dec, all_cls_scores, all_bbox_preds = self._run_temporal_decoder(
            memory, tgt, query_pos, pos_embed, attn_mask,
            temp_memory, temp_pos, data.get('lidar_bev'), reference_points,
            lidar2img, img_hw, img_pad, data['prev_exists'], pad, img_bev,
            attend_image)
        outs_dec = torch.nan_to_num(outs_dec)
        all_cls_scores = torch.nan_to_num(all_cls_scores)
        all_bbox_preds = torch.nan_to_num(all_bbox_preds)
        # 当前帧 query 在本车坐标系里，位姿先记成单位阵。post_update 再乘上 ego_pose 变到全局。
        # 长度是 num_query：post_update 会先剥掉去噪段，再按分数取 topk。
        rec_ego_pose = torch.eye(
            4, device=outs_dec.device).view(1, 1, 4, 4).repeat(b, self.num_query, 1, 1)
        # 归一化中心还原成米，尺寸和朝向保持回归头的输出，供损失和 memory 使用。
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
