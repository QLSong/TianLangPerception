"""把 mydata 配置的单帧前向导出成 TensorRT，测 GPU 耗时。

图里包含骨干、IPM、占用头和检测解码。不含 get_bboxes 的 CPU 后处理。
可变形注意力走 mmcv 的 CUDA MultiScaleDeformableAttnFunction，
导出时接到 TensorRT 自带的 MultiscaleDeformableAttnPlugin_TRT。
外参逆和 bda 逆不进图：bda 是单位阵，自车到相机的外参在图外求逆，输入是 sensor2ego。
平方改成乘法，Slice 的 int64 上界收成 int32，FP32 引擎才和 PyTorch 对齐。
这份 TensorRT 8.6 的 FP16 引擎分类会变成全 0，默认编 FP32。

示例：
export LD_LIBRARY_PATH=/opt/streampetr/lib/python3.8/site-packages/tensorrt:\
/opt/streampetr/lib/python3.8/site-packages/nvidia/cublas/lib:\
/opt/streampetr/lib/python3.8/site-packages/nvidia/cuda_runtime/lib:\
/opt/streampetr/lib/python3.8/site-packages/nvidia/cudnn/lib:\
/opt/streampetr/lib/python3.8/site-packages/nvidia/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
/opt/streampetr/bin/python tools/bench_trt_mydata.py
/opt/streampetr/bin/python tools/bench_trt_mydata.py --fp16
/opt/streampetr/bin/python tools/bench_trt_mydata.py --int8
"""
import argparse
import os
import os.path as osp
import sys
import time

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
os.chdir(osp.dirname(osp.dirname(osp.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from mmcv import Config
from mmdet3d.models import build_model

import projects.mmdet3d_plugin  # noqa: F401
import projects.mmdet3d_plugin.models.necks.plane_view_transformer as plane_view
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
from projects.mmdet3d_plugin.core.checkpoint import load_pretrained
from projects.mmdet3d_plugin.models.detectors.streampetr_fusion import StreamPETRFusion

CONFIG = 'projects/configs/TianLangEyes/streampetr_convnext_pp_det10m_mydata.py'
_ONNX_INT32 = 6


def _reshape_heads(graph, sampled):
    """插件给出 (B, Q, heads, dim)，CUDA 函数返回 (B, Q, heads * dim)。"""
    shape = graph.op('Shape', sampled)
    starts = graph.op('Constant', value_t=torch.tensor([0], dtype=torch.int64))
    ends = graph.op('Constant', value_t=torch.tensor([2], dtype=torch.int64))
    axes = graph.op('Constant', value_t=torch.tensor([0], dtype=torch.int64))
    head = graph.op('Slice', shape, starts, ends, axes)
    tail = graph.op('Constant', value_t=torch.tensor([-1], dtype=torch.int64))
    return graph.op('Reshape', sampled, graph.op('Concat', head, tail, axis_i=0))


def _symbolic_ms_deform(graph, value, spatial_shapes, level_start_index,
                        sampling_locations, attention_weights, im2col_step):
    """导出成 TensorRT 自带的 CUDA plugin。索引是 int32。"""
    del im2col_step
    spatial_shapes = graph.op('Cast', spatial_shapes, to_i=_ONNX_INT32)
    level_start_index = graph.op('Cast', level_start_index, to_i=_ONNX_INT32)
    sampled = graph.op(
        'MultiscaleDeformableAttnPlugin_TRT',
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights)
    return _reshape_heads(graph, sampled)


MultiScaleDeformableAttnFunction.symbolic = staticmethod(_symbolic_ms_deform)


def _rational_distort(x, y, dist):
    """和训练时的 rational 模型一致，但把无效点的多项式限制在有限范围。

    相机背后的点深度被抬到 1e-5 后，归一化坐标会到 1e6，r^6 直接溢出。
    有效像素的归一化坐标大约在 [-2, 2]，钳到 [-8, 8] 不会碰到它们。
    TensorRT 的 INT8 校准会统计这些中间张量，inf 会让动态范围 min > max。
    """
    x = x.clamp(-8, 8)
    y = y.clamp(-8, 8)
    k1, k2, p1, p2, k3, k4, k5, k6 = [item[..., None] for item in dist.unbind(-1)]
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    denom = (1 + k4 * r2 + k5 * r4 + k6 * r6).clamp(min=1e-4)
    scale = ((1 + k1 * r2 + k2 * r4 + k3 * r6) / denom).clamp(-1e3, 1e3)
    xd = x * scale + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * scale + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return xd, yd


def _ego_to_image(ego, sensor2ego, cam2imgs, post_rots, post_trans, bda, distortion=None):
    """bda 固定为单位阵，不再在图里求逆。"""
    del bda
    xyz = ego
    rotation = sensor2ego[:, :, :3, :3]
    translation = sensor2ego[:, :, :3, 3]
    xyz = xyz - translation[:, :, None, :]
    cam = rotation.transpose(-1, -2)[:, :, None].matmul(xyz.unsqueeze(-1)).squeeze(-1)
    if distortion is not None:
        depth = cam[..., 2]
        safe = depth.clamp(min=1e-5)
        xn, yn = _rational_distort(cam[..., 0] / safe, cam[..., 1] / safe, distortion)
        cam = torch.stack((xn * safe, yn * safe, depth), dim=-1)
    lifted = post_rots[:, :, None].matmul(cam2imgs[:, :, None].matmul(cam.unsqueeze(-1))).squeeze(-1)
    depth = lifted[..., 2]
    safe = depth.clamp(min=1e-5)
    view = torch.stack((
        (lifted[..., 0] / safe).clamp(-1e4, 1e4),
        (lifted[..., 1] / safe).clamp(-1e4, 1e4),
        depth,
    ), dim=-1)
    pix = post_rots[:, :, None].matmul(view.unsqueeze(-1)).squeeze(-1) + post_trans[:, :, None, :]
    return pix[..., 0].clamp(-1e4, 1e4), pix[..., 1].clamp(-1e4, 1e4), cam[..., 2]


def _lss_inputs_ready(self, img_feats, sensor2ego, intrinsics, distortion=None):
    """sensor2ego 是图外求好的相机到自车。这里不再调用 linalg.inv。"""
    b, n = img_feats.shape[:2]
    device = img_feats.device
    if sensor2ego.dim() == 5:
        sensor2ego = sensor2ego[:, -1]
        intrinsics = intrinsics[:, -1]
        if distortion is not None and distortion.dim() == 4:
            distortion = distortion[:, -1]
    ego2global = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(b, n, 1, 1)
    cam2imgs = intrinsics[..., :3, :3]
    post_rots = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(b, n, 1, 1)
    post_trans = torch.zeros(b, n, 3, device=device)
    bda = torch.eye(3, device=device).unsqueeze(0).repeat(b, 1, 1)
    packed = [img_feats, sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda]
    if distortion is not None:
        packed.append(distortion.to(device=device, dtype=img_feats.dtype))
    return packed


plane_view.ego_to_image = _ego_to_image
StreamPETRFusion._lss_inputs = _lss_inputs_ready


class InferenceGraph(nn.Module):
    """和 simple_test 同一条计算，只留张量。检测不走时序，占用复用同一张 BEV。"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img, sensor2ego, intrinsics, distortion):
        model = self.model
        batch, views, channels, height, width = img.shape
        flat = img.reshape(batch * views, channels, height, width)
        feats = model.img_backbone(flat)
        if isinstance(feats, dict):
            feats = list(feats.values())
        feats = model.img_neck(feats)
        level = feats[model.position_level]
        _, channels, feat_h, feat_w = level.shape
        level = level.view(batch, views, channels, feat_h, feat_w)
        bev = model._camera_bev(level, sensor2ego, intrinsics, distortion)
        data = dict(
            img=img,
            img_feats=level,
            extrinsics=sensor2ego,
            intrinsics=intrinsics,
            distortion=distortion,
            img_bev=bev,
            prev_exists=img.new_zeros(batch),
        )
        outs = model.pts_bbox_head(level.new_zeros(1), [None], None, **data)
        occ = model.forward_occ(level, data, camera_bev=bev)
        return outs['all_cls_scores'], outs['all_bbox_preds'], occ


def build(config):
    cfg = Config.fromfile(config)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_pretrained(model, cfg.load_from)
    model.eval().cuda()
    final_dim = tuple(cfg.ida_aug_conf['final_dim'])
    model.img_view_transformer.input_size = final_dim
    graph = InferenceGraph(model).cuda().eval()
    height, width = final_dim
    views = len(cfg.demo_cameras)
    img = torch.randn(1, views, 3, height, width, device='cuda')
    # 真实相机传入标定里的 T_base_sensor。这里的单位阵逆还是单位阵。
    ego2cam = torch.eye(4, device='cuda').view(1, 1, 4, 4).repeat(1, views, 1, 1)
    sensor2ego = torch.linalg.inv(ego2cam)
    intrinsics = torch.eye(4, device='cuda').view(1, 1, 4, 4).repeat(1, views, 1, 1)
    intrinsics[:, :, 0, 0] = 300
    intrinsics[:, :, 1, 1] = 300
    intrinsics[:, :, 0, 2] = width / 2
    intrinsics[:, :, 1, 2] = height / 2
    distortion = torch.zeros(1, views, 8, device='cuda')
    return graph, (img, sensor2ego, intrinsics, distortion)


def elapsed_ms(launch, warmup, repeat):
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeat):
        start.record()
        launch()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return sum(samples) / len(samples), samples[len(samples) // 2]


def export_onnx(graph, inputs, path):
    # 自定义 plugin 不在 onnx schema 里，检查会误报。TensorRT 解析器认识这个算子。
    torch.onnx.utils._C._check_onnx_proto = lambda proto: None
    with torch.no_grad():
        torch.onnx.export(
            graph, inputs, path,
            input_names=['img', 'sensor2ego', 'intrinsics', 'distortion'],
            output_names=['cls', 'box', 'occ'],
            opset_version=16,
            do_constant_folding=False,
        )


def patch_onnx(path):
    """平方用乘法，Slice 上界用 int32 最大值。TensorRT 8.6 直接吃原图会算错。"""
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(path)
    nodes = []
    squares = 0
    for node in model.graph.node:
        if node.op_type == 'Pow':
            nodes.append(helper.make_node(
                'Mul', [node.input[0], node.input[0]], list(node.output), name=node.name))
            squares += 1
        else:
            nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    clipped = 0
    int32_max = np.array(np.iinfo(np.int32).max, dtype=np.int64)
    for node in model.graph.node:
        if node.op_type != 'Constant':
            continue
        for attr in node.attribute:
            if attr.name != 'value':
                continue
            values = numpy_helper.to_array(attr.t)
            if values.dtype == np.int64 and values.size and int(values.reshape(-1)[0]) == np.iinfo(np.int64).max:
                attr.t.CopyFrom(numpy_helper.from_array(int32_max))
                clipped += 1
    # 全零的线性层没有 INT8 scale，bias 量化会直接失败。对角线上放一个可忽略的值。
    nudged = 0
    for init in model.graph.initializer:
        if init.data_type != onnx.TensorProto.FLOAT:
            continue
        weight = np.array(numpy_helper.to_array(init), copy=True)
        if weight.ndim < 2 or weight.size == 0:
            continue
        channels = weight.reshape(weight.shape[0], -1)
        dead = np.max(np.abs(channels), axis=1) == 0
        if not np.any(dead):
            continue
        width = channels.shape[1]
        for index in np.where(dead)[0]:
            channels[index, index % width] = 1e-5
        init.CopyFrom(numpy_helper.from_array(channels.reshape(weight.shape), init.name))
        nudged += int(dead.sum())
    onnx.save(model, path)
    print('ONNX 修正：平方 {} 处，Slice 上界 {} 处，空权重通道 {} 个'.format(
        squares, clipped, nudged), flush=True)


def make_int8_calibrator(batches, cache_path):
    """熵校准。校准器类要继承 TensorRT 的接口，所以等 tensorrt 导入后再定义。"""
    import tensorrt as trt

    class Int8Calibrator(trt.IInt8EntropyCalibrator2):

        def __init__(self):
            trt.IInt8EntropyCalibrator2.__init__(self)
            self.batches = batches
            self.cache_path = cache_path
            self.index = 0
            self.bound = []

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.index >= len(self.batches):
                return None
            batch = self.batches[self.index]
            self.index += 1
            self.bound = [batch[name].contiguous() for name in names]
            print('INT8 校准 {}/{}'.format(self.index, len(self.batches)), flush=True)
            return [int(tensor.data_ptr()) for tensor in self.bound]

        def read_calibration_cache(self):
            if not osp.isfile(self.cache_path):
                return None
            with open(self.cache_path, 'rb') as handle:
                return handle.read()

        def write_calibration_cache(self, cache):
            with open(self.cache_path, 'wb') as handle:
                handle.write(bytes(cache))

    return Int8Calibrator()


def calibration_batches(cfg, count):
    """按 demo 的缩放和归一化取帧。外参在这里求逆，和图的 sensor2ego 输入一致。"""
    tools_dir = osp.dirname(osp.abspath(__file__))
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import demo_mydata

    camera_names = list(cfg.demo_cameras)
    cameras, _ = demo_mydata.load_sensors(cfg.demo_calib, camera_names)
    frames = demo_mydata.index_frames(cfg.demo_data_root, camera_names)
    if count < 1:
        raise ValueError('校准帧数要大于 0')
    if len(frames) > count:
        indexes = np.linspace(0, len(frames) - 1, count, dtype=int)
        frames = [frames[int(i)] for i in indexes]
    mean = np.array(cfg.img_norm_cfg['mean'], dtype=np.float32)
    std = np.array(cfg.img_norm_cfg['std'], dtype=np.float32)
    final_dim = tuple(cfg.ida_aug_conf['final_dim'])
    batches = []
    for frame in frames:
        images, intrinsics, extrinsics, _, distortion = demo_mydata.prepare_images(
            frame, cameras, final_dim, mean, std)
        ego2cam = torch.tensor(np.stack(extrinsics), dtype=torch.float32, device='cuda').unsqueeze(0)
        batches.append({
            'img': torch.tensor(np.stack(images), dtype=torch.float32, device='cuda').unsqueeze(0),
            'sensor2ego': torch.linalg.inv(ego2cam),
            'intrinsics': torch.tensor(np.stack(intrinsics), dtype=torch.float32, device='cuda').unsqueeze(0),
            'distortion': torch.tensor(np.stack(distortion), dtype=torch.float32, device='cuda').unsqueeze(0),
        })
    print('INT8 校准用 {} 帧真实图像'.format(len(batches)), flush=True)
    return batches


def sanitize_calibration_cache(path):
    """0 和 inf 的 scale 会让 bias 量化断言失败。换成幅度 1 对应的 scale。"""
    import struct
    lines = open(path).read().splitlines()
    fallback = struct.pack('!f', 1.0 / 127.0).hex()
    changed = 0
    output = [lines[0]]
    for line in lines[1:]:
        if ': ' not in line:
            output.append(line)
            continue
        name, raw = line.rsplit(': ', 1)
        invalid = raw == '0'
        if not invalid and len(raw) == 8 and all(char in '0123456789abcdef' for char in raw):
            value = struct.unpack('!f', bytes.fromhex(raw))[0]
            invalid = not (0 < value < 1e4)
        if invalid:
            line = '{}: {}'.format(name, fallback)
            changed += 1
        output.append(line)
    with open(path, 'w') as handle:
        handle.write('\n'.join(output) + '\n')
    return changed


def build_engine(onnx_path, engine_path, fp16, int8=False, calibrator=None):
    import tensorrt as trt

    def once(current):
        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, '')
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        with open(onnx_path, 'rb') as handle:
            if not parser.parse(handle.read()):
                errors = [parser.get_error(i) for i in range(parser.num_errors)]
                raise RuntimeError('\n'.join(str(item) for item in errors))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
        if int8:
            if current is None:
                raise RuntimeError('INT8 需要校准器')
            # 可变形注意力插件只有 FP16/FP32，其余层走 INT8。
            config.set_flag(trt.BuilderFlag.INT8)
            config.set_flag(trt.BuilderFlag.FP16)
            config.int8_calibrator = current
        elif fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        print('开始构建 TensorRT 引擎，fp16={}，int8={}'.format(fp16 or int8, int8), flush=True)
        return builder.build_serialized_network(network, config)

    started = time.time()
    if int8 and osp.isfile(calibrator.cache_path):
        os.remove(calibrator.cache_path)
    serialized = once(calibrator)
    if serialized is None and int8:
        changed = sanitize_calibration_cache(calibrator.cache_path)
        print('校准表有 {} 个无效 scale，换成有限值后重新构建'.format(changed), flush=True)
        reader = make_int8_calibrator(calibrator.batches, calibrator.cache_path)
        serialized = once(reader)
    if serialized is None:
        raise RuntimeError('TensorRT 构建失败')
    blob = bytes(serialized)
    with open(engine_path, 'wb') as handle:
        handle.write(blob)
    print('引擎构建用了 {:.1f} 秒，大小 {:.1f} MB'.format(
        time.time() - started, len(blob) / 1024 / 1024), flush=True)
    return blob


def trt_ms(blob, inputs, warmup, repeat):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(blob)
    context = engine.create_execution_context()
    feeds = dict(zip(['img', 'sensor2ego', 'intrinsics', 'distortion'], inputs))
    dtype_of = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
    }
    stream = torch.cuda.current_stream()
    use_v3 = hasattr(engine, 'num_io_tensors') and not hasattr(engine, 'num_bindings')
    if use_v3:
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                tensor = feeds[name].contiguous()
                context.set_input_shape(name, tuple(tensor.shape))
            else:
                tensor = torch.empty(
                    tuple(engine.get_tensor_shape(name)), device='cuda',
                    dtype=dtype_of[engine.get_tensor_dtype(name)])
            context.set_tensor_address(name, tensor.data_ptr())

        def launch():
            context.execute_async_v3(stream.cuda_stream)
    else:
        bindings = [0] * engine.num_bindings
        for index in range(engine.num_bindings):
            name = engine.get_binding_name(index)
            if engine.binding_is_input(index):
                tensor = feeds[name].contiguous()
                context.set_binding_shape(index, tuple(tensor.shape))
            else:
                tensor = torch.empty(
                    tuple(context.get_binding_shape(index)), device='cuda',
                    dtype=dtype_of[engine.get_binding_dtype(index)])
            bindings[index] = tensor.data_ptr()

        def launch():
            context.execute_async_v2(bindings, stream.cuda_stream)

    return elapsed_ms(launch, warmup, repeat)


def main():
    parser = argparse.ArgumentParser(description='导出 mydata 模型并测 TensorRT 耗时')
    parser.add_argument('config', nargs='?', default=CONFIG)
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument('--fp16', action='store_true', help='编 FP16。TensorRT 8.6 上这张图的分类输出是错的')
    precision.add_argument('--int8', action='store_true', help='用真实图像做熵校准，编 INT8。插件层保持 FP16')
    parser.add_argument('--calib-batches', type=int, default=32, help='INT8 校准帧数，从数据里均匀抽取')
    parser.add_argument('--out-dir', default='work_dirs/bench_trt_mydata')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--repeat', type=int, default=50)
    args = parser.parse_args()

    graph, inputs = build(args.config)
    with torch.no_grad():
        outputs = graph(*inputs)
    print('输出', [tuple(item.shape) for item in outputs], flush=True)

    def run_pytorch():
        with torch.no_grad():
            graph(*inputs)

    mean, median = elapsed_ms(run_pytorch, args.warmup, args.repeat)
    print('PyTorch 平均 {:.2f} ms，中位数 {:.2f} ms'.format(mean, median), flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.int8:
        tag = 'int8'
    elif args.fp16:
        tag = 'fp16'
    else:
        tag = 'fp32'
    onnx_path = osp.join(args.out_dir, 'streampetr_mydata.onnx')
    engine_path = osp.join(args.out_dir, 'streampetr_mydata_{}.engine'.format(tag))
    export_onnx(graph, inputs, onnx_path)
    patch_onnx(onnx_path)
    print('ONNX {:.1f} MB'.format(os.path.getsize(onnx_path) / 1024 / 1024), flush=True)
    calibrator = None
    if args.int8:
        cfg = Config.fromfile(args.config)
        batches = calibration_batches(cfg, args.calib_batches)
        calibrator = make_int8_calibrator(
            batches, osp.join(args.out_dir, 'streampetr_mydata_int8.cache'))
    blob = build_engine(onnx_path, engine_path, fp16=args.fp16, int8=args.int8, calibrator=calibrator)
    mean, median = trt_ms(blob, inputs, args.warmup, args.repeat)
    print('TensorRT {} 平均 {:.2f} ms，中位数 {:.2f} ms'.format(tag.upper(), mean, median), flush=True)
    print('引擎 {}'.format(engine_path), flush=True)


if __name__ == '__main__':
    main()
