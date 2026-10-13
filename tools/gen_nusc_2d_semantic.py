"""用 GroundingDINO + SAM 给 nuScenes 训练集生成 2D 语义伪标签。

输出和 GaussianOcc 的读取方式一致：
data/nuscenes/nuscenes_semantic/samples/CAM_*/xxx_mask.bin
int8，形状 900×1600。类别编号随后由训练代码映射到 Occ3D。

依赖：
  pip install segment-anything
  pip install git+https://github.com/IDEA-Research/GroundingDINO.git
权重放在 ckpts/sam_vit_h_4b8939.pth 和 ckpts/groundingdino_swinb_cogcoor.pth。
"""
import argparse
import os
import pickle

import numpy as np
import torch

BOX_THRESHOLD = 0.20
TEXT_THRESHOLD = 0.20
# 第一组按词分开检测，其余组整句检测，和 GroundedSAM_OccNeRF 一致。
TEXT_PROMPTS = [
    ['Pedestrian', 'Bus', 'Truck'],
    ['Bicycle, Bicyclist'],
    ['Motorcycle, Motorcyclist'],
    ['Barrier, Barricade'],
    ['Sedan, Highway, Terrain, Tree, Sidewalk, Building, Bridge, Pole, Billboard, Light, Ashbin, Crane, Trailer, Cone, Sky'],
]
CAMERAS = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT',
    'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT',
]
NAME_TO_ID = {
    'sedan': 1,
    'highway': 2,
    'bus': 3,
    'truck': 4,
    'terrain': 5,
    'tree': 6,
    'sidewalk': 7,
    'bicycle': 8,
    'bicyclist': 8,
    'barrier': 9,
    'barricade': 9,
    'person': 10,
    'pedestrian': 10,
    'building': 11,
    'bridge': 11,
    'pole': 11,
    'billboard': 11,
    'light': 11,
    'ashbin': 11,
    'motorcycle': 12,
    'motorcyclist': 12,
    'crane': 13,
    'trailer': 14,
    'cone': 15,
    'sky': 16,
}


def mask_path(dataroot, image_path):
    rel = os.path.relpath(os.path.abspath(image_path), os.path.abspath(dataroot))
    stem, _ = os.path.splitext(rel)
    return os.path.join(dataroot, 'nuscenes_semantic', stem + '_mask.bin')


def load_models(device, sam_ckpt, dino_ckpt, bert_dir):
    import groundingdino
    from groundingdino.models import build_model
    from groundingdino.util.misc import clean_state_dict
    from groundingdino.util.slconfig import SLConfig
    from segment_anything import SamPredictor, sam_model_registry

    config = os.path.join(
        os.path.dirname(groundingdino.__file__), 'config', 'GroundingDINO_SwinB_cfg.py')
    if not os.path.isfile(config):
        raise FileNotFoundError(config)
    args = SLConfig.fromfile(config)
    args.device = device
    if os.path.isdir(bert_dir):
        args.text_encoder_type = os.path.abspath(bert_dir)
    dino = build_model(args)
    checkpoint = torch.load(dino_ckpt, map_location='cpu')
    dino.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    dino.eval()
    sam = sam_model_registry['vit_h'](checkpoint=sam_ckpt)
    sam.to(device=device)
    return dino, SamPredictor(sam)


def boxes_for_prompt(model, image, prompt, device):
    from groundingdino.util.inference import predict
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=device)
    keep = []
    names = []
    for index, phrase in enumerate(phrases):
        name = phrase.lower().strip().rstrip('.')
        if name in NAME_TO_ID:
            keep.append(index)
            names.append(name)
    if not keep:
        empty = boxes.new_zeros((0, 4))
        return empty, boxes.new_zeros((0,)), []
    keep = torch.tensor(keep, device=boxes.device, dtype=torch.long)
    return boxes[keep], logits[keep], names


def generate_mask(dino, predictor, image_path, device):
    from groundingdino.util import box_ops
    from groundingdino.util.inference import load_image

    image_source, image = load_image(image_path)
    image = image.to(device)
    boxes, logits, phrases = [], [], []
    for group in TEXT_PROMPTS:
        for prompt in group:
            one_boxes, one_logits, one_phrases = boxes_for_prompt(dino, image, prompt, device)
            boxes.append(one_boxes)
            logits.append(one_logits)
            phrases.extend(one_phrases)
    boxes = torch.cat(boxes, dim=0)
    logits = torch.cat(logits, dim=0)
    height, width = image_source.shape[:2]
    if boxes.shape[0] == 0:
        return np.full((height, width), -1, dtype=np.int8)

    predictor.set_image(image_source)
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor(
        [width, height, width, height], device=boxes.device)
    transformed = predictor.transform.apply_boxes_torch(
        boxes_xyxy, image_source.shape[:2]).to(device)
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed,
        multimask_output=False,
        return_logits=True)
    masks = masks.cpu()
    logits = logits.cpu()
    score = torch.zeros(height, width, 1)
    layers = [score]
    for index in range(masks.shape[0]):
        layers.append(masks[index].reshape(height, width, 1) * logits[index].reshape(1, 1, 1))
    chosen = torch.cat(layers, dim=-1).argmax(dim=-1).cpu().numpy()
    names = ['uncertain'] + phrases
    ids = np.array([NAME_TO_ID.get(names[index], -1) for index in range(len(names))], dtype=np.int8)
    return ids[chosen]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='data/nuscenes')
    parser.add_argument('--infos', default='data/nuscenes/nuscenes2d_temporal_infos_train.pkl')
    parser.add_argument('--sam-ckpt', default='ckpts/sam_vit_h_4b8939.pth')
    parser.add_argument('--dino-ckpt', default='ckpts/groundingdino_swinb_cogcoor.pth')
    parser.add_argument('--bert-dir', default='ckpts/bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int, default=0, help='只处理前 N 帧，0 表示全部')
    args = parser.parse_args()

    for path in (args.sam_ckpt, args.dino_ckpt, args.infos):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    with open(args.infos, 'rb') as handle:
        infos = pickle.load(handle)['infos']
    if args.limit > 0:
        infos = infos[:args.limit]

    device = args.device
    dino, predictor = load_models(device, args.sam_ckpt, args.dino_ckpt, args.bert_dir)
    done = 0
    skipped = 0
    with torch.no_grad():
        for info in infos:
            for camera in CAMERAS:
                image_path = info['cams'][camera]['data_path']
                output = mask_path(args.dataroot, image_path)
                if os.path.isfile(output):
                    skipped += 1
                    continue
                mask = generate_mask(dino, predictor, image_path, device)
                os.makedirs(os.path.dirname(output), exist_ok=True)
                mask.astype(np.int8).tofile(output)
                done += 1
                if done % 20 == 0:
                    print(f'wrote {done}, skipped {skipped}, last {output}', flush=True)
    print(f'finished, wrote {done}, skipped {skipped}')


if __name__ == '__main__':
    main()
