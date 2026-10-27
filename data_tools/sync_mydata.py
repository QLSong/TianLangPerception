"""把 test_my_data 的 rosbag 对齐成一帧。

两路雷达、四路环视都是 10 Hz，定位 /lio/robo/odom 大约 200 Hz。
以 front_up 的时间戳为基准，其余传感器取最近一帧。点云布局和
tools/demo_mydata.py 的 27 字节记录一致：xyz、intensity 各一个 float32。

示例：
/opt/streampetr/bin/python data_tools/sync_mydata.py \
  /mnt/data/nas/datasets/test_my_data/2026-10-26-09-21-26 \
  --out /mnt/data/nas/datasets/test_my_data/2026-10-26-09-21-26

--out 下按传感器各建一个文件夹：cam0 到 cam3 是 jpg，front_up 和 back_down 是 binary pcd。
同一帧的文件名都是 000000.jpg / 000000.pcd。定位写在 poses.json。
"""
import argparse
import json
import os
import os.path as osp
import struct
import ctypes

import cv2
import numpy as np

LIDARS = {
    'front_up': '/front_up/rslidar_points',
    'back_down': '/back_down/rslidar_points',
}
CAMERAS = {
    'cam0': '/cam0/image/compressed',
    'cam1': '/cam1/image/compressed',
    'cam2': '/cam2/image/compressed',
    'cam3': '/cam3/image/compressed',
}
ODOM = {
    'robo_odom': '/lio/robo/odom',
    'imu_odom': '/lio/imu/odom',
}
MASTER = 'front_up'
SENSOR_TOPICS = {}
SENSOR_TOPICS.update(LIDARS)
SENSOR_TOPICS.update(CAMERAS)
SENSOR_TOPICS.update(ODOM)
POINT_STRIDE = 27
# 10 Hz 传感器半个周期。定位更密，同样用这个上限，避免偶发间隔把整帧丢掉。
MAX_DT = 0.05


def _fields(buf):
    index = 0
    fields = {}
    while index < len(buf):
        size = struct.unpack_from('<I', buf, index)[0]
        index += 4
        item = buf[index:index + size]
        index += size
        key, value = item.split(b'=', 1)
        fields[key.decode()] = value
    return fields


def _u32(buf, index):
    return struct.unpack_from('<I', buf, index)[0]


class _Cursor(object):
    def __init__(self, buf):
        self.buf = buf
        self.index = 0

    def u8(self):
        value = self.buf[self.index]
        self.index += 1
        return value

    def u32(self):
        value = _u32(self.buf, self.index)
        self.index += 4
        return value

    def f64(self):
        value = struct.unpack_from('<d', self.buf, self.index)[0]
        self.index += 8
        return value

    def time(self):
        sec, nsec = struct.unpack_from('<II', self.buf, self.index)
        self.index += 8
        return sec + nsec * 1e-9

    def string(self):
        size = self.u32()
        raw = self.buf[self.index:self.index + size]
        self.index += size
        return raw.split(b'\x00', 1)[0].decode()


def _header(cursor):
    cursor.u32()
    stamp = cursor.time()
    frame_id = cursor.string()
    return stamp, frame_id


class _Lz4(object):
    def __init__(self):
        lib = ctypes.CDLL('liblz4.so.1')
        lib.LZ4F_createDecompressionContext.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        lib.LZ4F_createDecompressionContext.restype = ctypes.c_size_t
        lib.LZ4F_freeDecompressionContext.argtypes = [ctypes.c_void_p]
        lib.LZ4F_freeDecompressionContext.restype = ctypes.c_size_t
        lib.LZ4F_decompress.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p]
        lib.LZ4F_decompress.restype = ctypes.c_size_t
        lib.LZ4F_isError.argtypes = [ctypes.c_size_t]
        lib.LZ4F_isError.restype = ctypes.c_uint
        lib.LZ4F_getErrorName.argtypes = [ctypes.c_size_t]
        lib.LZ4F_getErrorName.restype = ctypes.c_char_p
        self.lib = lib

    def decompress(self, src, size):
        ctx = ctypes.c_void_p()
        err = self.lib.LZ4F_createDecompressionContext(ctypes.byref(ctx), 100)
        if self.lib.LZ4F_isError(err):
            raise RuntimeError(self.lib.LZ4F_getErrorName(err).decode())
        dst = ctypes.create_string_buffer(size)
        dst_size = ctypes.c_size_t(size)
        src_buf = ctypes.create_string_buffer(src)
        src_size = ctypes.c_size_t(len(src))
        ret = self.lib.LZ4F_decompress(ctx, dst, ctypes.byref(dst_size), src_buf, ctypes.byref(src_size), None)
        self.lib.LZ4F_freeDecompressionContext(ctx)
        if self.lib.LZ4F_isError(ret):
            raise RuntimeError(self.lib.LZ4F_getErrorName(ret).decode())
        if dst_size.value != size:
            raise RuntimeError('lz4 解出 {} 字节，块头写的是 {}'.format(dst_size.value, size))
        return dst.raw


def _stamp_of(sec, nsec):
    return sec + nsec * 1e-9


def _bag_header(handle):
    magic = handle.read(13)
    if magic != b'#ROSBAG V2.0\n':
        raise RuntimeError('不是 ROS1 bag')
    hdr_len = _u32(handle.read(4), 0)
    header = _fields(handle.read(hdr_len))
    data_len = _u32(handle.read(4), 0)
    handle.seek(data_len, os.SEEK_CUR)
    return struct.unpack('<Q', header['index_pos'])[0]


def _topic_ids(handle, index_pos):
    """连接记录在文件尾的索引里，chunk 前面读不到话题名。"""
    handle.seek(index_pos)
    found = {}
    while True:
        raw = handle.read(4)
        if len(raw) < 4:
            break
        hdr_len = _u32(raw, 0)
        header = _fields(handle.read(hdr_len))
        data_len = _u32(handle.read(4), 0)
        op = header['op'][0]
        if op != 0x07:
            handle.seek(data_len, os.SEEK_CUR)
            continue
        handle.seek(data_len, os.SEEK_CUR)
        topic = header['topic'].decode()
        conn = _u32(header['conn'], 0)
        for name, wanted in SENSOR_TOPICS.items():
            if topic == wanted:
                found[conn] = name
    missing = [name for name in SENSOR_TOPICS if name not in found.values()]
    if missing:
        raise RuntimeError('bag 里没有 {}'.format(', '.join(missing)))
    return found


def index_bag(path):
    """只读每个 chunk 后面的索引，不展开点云和图像。"""
    messages = {name: [] for name in SENSOR_TOPICS}
    chunks = []
    with open(path, 'rb') as handle:
        index_pos = _bag_header(handle)
        body = handle.tell()
        topic_of = _topic_ids(handle, index_pos)
        handle.seek(body)
        chunk_index = 0
        while handle.tell() < index_pos:
            chunk_pos = handle.tell()
            raw = handle.read(4)
            if len(raw) < 4:
                break
            hdr_len = _u32(raw, 0)
            header = _fields(handle.read(hdr_len))
            data_len = _u32(handle.read(4), 0)
            op = header['op'][0]
            if op == 0x05:
                size = _u32(header['size'], 0)
                handle.seek(data_len, os.SEEK_CUR)
                chunks.append([chunk_pos, size, data_len])
                chunk_index += 1
                if chunk_index % 2000 == 0:
                    print('索引 chunk {}'.format(chunk_index), flush=True)
            elif op == 0x04:
                conn = _u32(header['conn'], 0)
                count = _u32(header['count'], 0)
                blob = handle.read(data_len)
                name = topic_of.get(conn)
                if name is None:
                    continue
                base = chunk_index - 1
                for item in range(count):
                    sec, nsec, offset = struct.unpack_from('<III', blob, item * 12)
                    messages[name].append([_stamp_of(sec, nsec), base, int(offset)])
            else:
                raise RuntimeError('bag 里出现未处理的记录 op={}'.format(op))
    for name, rows in messages.items():
        rows.sort(key=lambda row: row[0])
        if not rows:
            raise RuntimeError('bag 里没有 {}'.format(SENSOR_TOPICS[name]))
    return {'chunks': chunks, 'messages': messages}


def _nearest(rows, stamp):
    stamps = rows[:, 0]
    pos = int(np.searchsorted(stamps, stamp))
    choices = []
    if pos < len(stamps):
        choices.append(pos)
    if pos > 0:
        choices.append(pos - 1)
    best = min(choices, key=lambda item: abs(float(stamps[item]) - stamp))
    delta = float(stamps[best]) - stamp
    return best, delta


def synchronize(messages, max_dt):
    """以 front_up 为基准。任一 10 Hz 传感器或定位超出 max_dt 就丢掉这一帧。"""
    master = messages[MASTER]
    arrays = {name: np.asarray(rows, dtype=np.float64) for name, rows in messages.items()}
    frames = []
    dropped = {name: 0 for name in SENSOR_TOPICS if name != MASTER}
    for index in range(len(master)):
        stamp = float(master[index][0])
        chosen = {MASTER: index}
        deltas = {MASTER: 0.0}
        keep = True
        for name in SENSOR_TOPICS:
            if name == MASTER:
                continue
            match, delta = _nearest(arrays[name], stamp)
            deltas[name] = delta
            chosen[name] = match
            if abs(delta) > max_dt:
                dropped[name] += 1
                keep = False
        if keep:
            frames.append({'t': stamp, 'id': chosen, 'dt': deltas})
    return frames, dropped


def parse_cloud(buf):
    cursor = _Cursor(buf)
    stamp, frame_id = _header(cursor)
    height, width = cursor.u32(), cursor.u32()
    nfields = cursor.u32()
    fields = []
    for _ in range(nfields):
        name = cursor.string()
        offset, datatype, count = cursor.u32(), cursor.u8(), cursor.u32()
        fields.append((name, offset, datatype, count))
    bigendian = cursor.u8()
    point_step, row_step = cursor.u32(), cursor.u32()
    size = cursor.u32()
    if bigendian:
        raise RuntimeError('点云是大端，当前只读小端')
    if point_step != POINT_STRIDE or fields[0][0] != 'x':
        raise RuntimeError('点云字段 {} step {}'.format(fields, point_step))
    raw = np.frombuffer(buf, dtype=np.uint8, offset=cursor.index, count=size)
    count = height * width
    if raw.size != count * point_step:
        raise RuntimeError('点数 {} 和数据长度 {} 不一致'.format(count, raw.size))
    rec = raw.reshape(count, point_step)
    xyz = rec[:, :12].copy().view(np.float32).reshape(count, 3)
    intensity = rec[:, 12:16].copy().view(np.float32).reshape(count)
    return {
        'stamp': stamp,
        'frame_id': frame_id,
        'xyz': xyz,
        'intensity': intensity,
        'points': count,
    }


def read_jpeg(buf):
    """取出 CompressedImage 里的 jpeg 字节。"""
    cursor = _Cursor(buf)
    stamp, frame_id = _header(cursor)
    fmt = cursor.string()
    size = cursor.u32()
    raw = buf[cursor.index:cursor.index + size]
    if not fmt.startswith('jpeg') or raw[:2] != b'\xff\xd8':
        raise RuntimeError('{} 不是 jpeg，format={}'.format(frame_id, fmt))
    return {'stamp': stamp, 'frame_id': frame_id, 'jpeg': np.frombuffer(raw, dtype=np.uint8)}


def parse_image(buf):
    payload = read_jpeg(buf)
    image = cv2.imdecode(payload['jpeg'], cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError('解不开 {} 图像'.format(payload['frame_id']))
    payload['image'] = image
    return payload


def parse_odom(buf):
    cursor = _Cursor(buf)
    stamp, frame_id = _header(cursor)
    child = cursor.string()
    position = [cursor.f64() for _ in range(3)]
    quaternion = [cursor.f64() for _ in range(4)]
    return {
        'stamp': stamp,
        'frame_id': frame_id,
        'child_frame_id': child,
        'position': position,
        'quaternion_xyzw': quaternion,
    }


class SyncReader(object):
    """按对齐结果从 bag 里取出一帧的图像、点云和定位。"""

    def __init__(self, bag_path, index, frames, calib_path=None):
        self.bag_path = bag_path
        self.calib_path = calib_path
        self.index = index
        self.frames = frames
        self._handle = open(bag_path, 'rb')
        self._lz4 = _Lz4()
        self._chunk_id = None
        self._chunk = None

    def close(self):
        self._handle.close()

    def __len__(self):
        return len(self.frames)

    def _chunk_bytes(self, chunk_id):
        if chunk_id == self._chunk_id:
            return self._chunk
        pos, size, data_len = self.index['chunks'][chunk_id]
        self._handle.seek(pos)
        hdr_len = _u32(self._handle.read(4), 0)
        self._handle.seek(hdr_len, os.SEEK_CUR)
        got = _u32(self._handle.read(4), 0)
        if got != data_len:
            raise RuntimeError('chunk {} 长度和索引不一致'.format(chunk_id))
        self._chunk = self._lz4.decompress(self._handle.read(data_len), size)
        self._chunk_id = chunk_id
        return self._chunk

    def _message(self, name, msg_index):
        stamp, chunk_id, offset = self.index['messages'][name][msg_index]
        plain = self._chunk_bytes(int(chunk_id))
        offset = int(offset)
        hdr_len = _u32(plain, offset)
        header = _fields(plain[offset + 4:offset + 4 + hdr_len])
        if header['op'][0] != 0x02:
            raise RuntimeError('{} 的偏移 {} 不是消息'.format(name, offset))
        data_len = _u32(plain, offset + 4 + hdr_len)
        start = offset + 8 + hdr_len
        return stamp, plain[start:start + data_len]

    def frame(self, index):
        meta = self.frames[index]
        images = {}
        clouds = {}
        for name in CAMERAS:
            _, buf = self._message(name, meta['id'][name])
            images[name] = parse_image(buf)
        for name in LIDARS:
            _, buf = self._message(name, meta['id'][name])
            clouds[name] = parse_cloud(buf)
        odom = {}
        for name in ODOM:
            _, buf = self._message(name, meta['id'][name])
            odom[name] = parse_odom(buf)
        return {
            'stamp': meta['t'],
            'dt': meta['dt'],
            'images': images,
            'clouds': clouds,
            'odom': odom,
        }


def find_inputs(root):
    bags = [name for name in os.listdir(root) if name.endswith('.bag')]
    if len(bags) != 1:
        raise RuntimeError('{} 里 bag 有 {} 个'.format(root, len(bags)))
    calib = osp.join(root, 'calibration.yaml')
    if not osp.isfile(calib):
        calib = None
    return osp.join(root, bags[0]), calib


def build(root, out_dir, max_dt=MAX_DT):
    bag_path, calib_path = find_inputs(root)
    os.makedirs(out_dir, exist_ok=True)
    cache_path = osp.join(out_dir, 'bag_index.json')
    bag_size = os.path.getsize(bag_path)
    cached = None
    if osp.isfile(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as handle:
            cached = json.load(handle)
        if cached.get('bag_size') != bag_size or cached.get('bag') != bag_path:
            cached = None
    if cached is None:
        print('扫描 {}'.format(bag_path), flush=True)
        indexed = index_bag(bag_path)
        cached = {
            'bag': bag_path,
            'bag_size': bag_size,
            'calib': calib_path,
            'chunks': indexed['chunks'],
            'messages': indexed['messages'],
        }
        with open(cache_path, 'w', encoding='utf-8') as handle:
            json.dump(cached, handle)
        print('索引写到 {}'.format(cache_path), flush=True)
    frames, dropped = synchronize(cached['messages'], max_dt)
    sync_path = osp.join(out_dir, 'sync.json')
    summary = {
        'bag': bag_path,
        'calib': calib_path,
        'master': MASTER,
        'max_dt': max_dt,
        'topics': SENSOR_TOPICS,
        'counts': {name: len(rows) for name, rows in cached['messages'].items()},
        'dropped': dropped,
        'frames': len(frames),
    }
    with open(sync_path, 'w', encoding='utf-8') as handle:
        json.dump({'summary': summary, 'frames': frames}, handle)
    print('对齐 {} 帧，上限 {:.0f} ms，写到 {}'.format(len(frames), max_dt * 1000, sync_path))
    for name, count in summary['counts'].items():
        print('  {} {} 条'.format(name, count))
    if frames:
        worst = {name: 0.0 for name in SENSOR_TOPICS}
        for item in frames:
            for name, delta in item['dt'].items():
                worst[name] = max(worst[name], abs(delta))
        for name, delta in worst.items():
            print('  {} 最大偏差 {:.1f} ms，超出上限 {} 帧'.format(
                name, delta * 1000, dropped.get(name, 0)))
    return SyncReader(bag_path, cached, frames, calib_path)


def write_pcd(path, xyz, intensity):
    """binary PCD，字段 x y z intensity，float32，小端。"""
    count = int(xyz.shape[0])
    header = (
        '# .PCD v0.7 - Point Cloud Data file format\n'
        'VERSION 0.7\n'
        'FIELDS x y z intensity\n'
        'SIZE 4 4 4 4\n'
        'TYPE F F F F\n'
        'COUNT 1 1 1 1\n'
        'WIDTH {n}\n'
        'HEIGHT 1\n'
        'VIEWPOINT 0 0 0 1 0 0 0\n'
        'POINTS {n}\n'
        'DATA binary\n'
    ).format(n=count).encode('ascii')
    packed = np.empty((count, 4), np.float32)
    if count:
        packed[:, :3] = xyz
        packed[:, 3] = intensity
    with open(path, 'wb') as handle:
        handle.write(header)
        handle.write(packed.tobytes())


def export_folders(reader, out_dir):
    """四路图像写成 jpg，两路点云写成 pcd，每一路一个文件夹。"""
    cameras = list(CAMERAS)
    lidars = list(LIDARS)
    for name in cameras + lidars:
        os.makedirs(osp.join(out_dir, name), exist_ok=True)
    total = len(reader)
    poses = []
    for index in range(total):
        meta = reader.frames[index]
        stem = '{:06d}'.format(index)
        _, odom_buf = reader._message('robo_odom', meta['id']['robo_odom'])
        pose = parse_odom(odom_buf)
        files = {}
        for name in cameras:
            _, image_buf = reader._message(name, meta['id'][name])
            jpeg = read_jpeg(image_buf)['jpeg']
            rel = osp.join(name, stem + '.jpg')
            with open(osp.join(out_dir, rel), 'wb') as handle:
                handle.write(jpeg.tobytes())
            files[name] = rel
        for name in lidars:
            _, cloud_buf = reader._message(name, meta['id'][name])
            cloud = parse_cloud(cloud_buf)
            xyz = np.ascontiguousarray(cloud['xyz'])
            intensity = np.ascontiguousarray(cloud['intensity'])
            keep = np.isfinite(xyz).all(axis=1) & np.isfinite(intensity)
            rel = osp.join(name, stem + '.pcd')
            write_pcd(osp.join(out_dir, rel), xyz[keep], intensity[keep])
            files[name] = rel
        poses.append({
            'index': index,
            'stamp': meta['t'],
            'odom_stamp': pose['stamp'],
            'frame_id': pose['frame_id'],
            'position': pose['position'],
            'quaternion_xyzw': pose['quaternion_xyzw'],
            'files': files,
        })
        if index % 200 == 0:
            print('写出 {}/{}'.format(index, total), flush=True)
    pose_path = osp.join(out_dir, 'poses.json')
    with open(pose_path, 'w', encoding='utf-8') as handle:
        json.dump({'cameras': cameras, 'lidars': lidars, 'frames': poses}, handle)
    print('{} 帧已按路写到 {}，定位在 {}'.format(total, out_dir, pose_path), flush=True)


def export_synced(reader, path):
    """每帧的定位、四路 jpeg、两路点云写进一个 npz。

    图像按帧、再按 camera_names 的顺序首尾相接，image_jpeg_count 是每张 jpeg 的字节数。
    点云同样首尾相接，front_up_count / back_down_count 是每帧点数。
    """
    total = len(reader)
    cameras = list(CAMERAS)
    position = np.zeros((total, 3), np.float64)
    quaternion = np.zeros((total, 4), np.float64)
    stamp = np.zeros(total, np.float64)
    odom_stamp = np.zeros(total, np.float64)
    pose_frame = np.empty(total, dtype='U16')
    image_names = np.empty((total, len(cameras)), dtype='U32')
    image_jpeg = []
    image_jpeg_count = np.zeros((total, len(cameras)), np.int32)
    front_xyz, front_intensity = [], []
    back_xyz, back_intensity = [], []
    front_count = np.zeros(total, np.int32)
    back_count = np.zeros(total, np.int32)
    piles = {
        'front_up': (front_xyz, front_intensity, front_count),
        'back_down': (back_xyz, back_intensity, back_count),
    }
    for index in range(total):
        meta = reader.frames[index]
        _, odom_buf = reader._message('robo_odom', meta['id']['robo_odom'])
        pose = parse_odom(odom_buf)
        position[index] = pose['position']
        quaternion[index] = pose['quaternion_xyzw']
        stamp[index] = meta['t']
        odom_stamp[index] = pose['stamp']
        pose_frame[index] = pose['frame_id']
        for column, name in enumerate(cameras):
            _, image_buf = reader._message(name, meta['id'][name])
            jpeg = read_jpeg(image_buf)['jpeg']
            image_names[index, column] = '{:06d}_{}.jpg'.format(index, name)
            image_jpeg.append(jpeg)
            image_jpeg_count[index, column] = jpeg.size
        for name, (xyz_pile, intensity_pile, counts) in piles.items():
            _, cloud_buf = reader._message(name, meta['id'][name])
            cloud = parse_cloud(cloud_buf)
            xyz = np.ascontiguousarray(cloud['xyz'])
            intensity = np.ascontiguousarray(cloud['intensity'])
            keep = np.isfinite(xyz).all(axis=1) & np.isfinite(intensity)
            xyz_pile.append(xyz[keep])
            intensity_pile.append(intensity[keep])
            counts[index] = int(keep.sum())
        if index % 200 == 0:
            print('写出 {}/{}'.format(index, total), flush=True)
    np.savez(
        path,
        stamp=stamp,
        odom_stamp=odom_stamp,
        position=position,
        quaternion_xyzw=quaternion,
        pose_frame=pose_frame,
        camera_names=np.array(cameras),
        image_names=image_names,
        image_jpeg=np.concatenate(image_jpeg),
        image_jpeg_count=image_jpeg_count,
        lidar_names=np.array(['front_up', 'back_down']),
        front_up_xyz=np.concatenate(front_xyz),
        front_up_intensity=np.concatenate(front_intensity),
        front_up_count=front_count,
        back_down_xyz=np.concatenate(back_xyz),
        back_down_intensity=np.concatenate(back_intensity),
        back_down_count=back_count,
    )
    print('对齐结果 {} 帧写到 {}，图像 {} MB，前雷达 {} 点，后雷达 {} 点'.format(
        total, path,
        int(image_jpeg_count.sum() / (1024 * 1024)),
        int(front_count.sum()), int(back_count.sum())), flush=True)


def packed_slice(count, index):
    """第 index 段在拼接数组里的 [start, end)。count 可以是一维，或按行展开。"""
    flat = np.reshape(count, -1)
    end = int(np.cumsum(flat)[index])
    return end - int(flat[index]), end


def lidar_slice(count, index):
    """第 index 帧在拼接数组里的 [start, end)。"""
    return packed_slice(count, index)


def image_slice(count, frame, camera):
    """第 frame 帧、第 camera 路 jpeg 在 image_jpeg 里的 [start, end)。"""
    cameras = count.shape[1] if getattr(count, 'ndim', 1) == 2 else 1
    return packed_slice(count, frame * cameras + camera)


def parse_args():
    parser = argparse.ArgumentParser(description='对齐两路雷达、四路图像和定位')
    parser.add_argument('root', help='含 bag 和 calibration.yaml 的目录')
    parser.add_argument('--out', default='work_dirs/sync_mydata')
    parser.add_argument('--max-dt', type=float, default=MAX_DT, help='同步允许的最大时间差，秒')
    parser.add_argument('--preview', type=int, default=0, help='抽出前几帧图像，检查能不能解码')
    parser.add_argument('--save', default='', help='另外把定位、jpeg 和点云写进这个 npz')
    return parser.parse_args()


def main():
    args = parse_args()
    reader = build(args.root, args.out, args.max_dt)
    export_folders(reader, args.out)
    if args.save:
        export_synced(reader, args.save)
    if args.preview <= 0 or len(reader) == 0:
        reader.close()
        return
    preview = osp.join(args.out, 'preview')
    os.makedirs(preview, exist_ok=True)
    step = max(1, len(reader) // args.preview)
    picked = list(range(0, len(reader), step))[:args.preview]
    for index in picked:
        sample = reader.frame(index)
        for name, image in sample['images'].items():
            cv2.imwrite(osp.join(preview, '{:04d}_{}.jpg'.format(index, name)), image['image'])
        pose = sample['odom']['robo_odom']
        print('帧 {} stamp {:.3f} 点 {} / {} 定位 {} {}'.format(
            index, sample['stamp'],
            sample['clouds']['front_up']['points'],
            sample['clouds']['back_down']['points'],
            [round(v, 3) for v in pose['position']],
            pose['frame_id']))
    reader.close()


if __name__ == '__main__':
    main()
