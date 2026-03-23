#!/usr/bin/env python3
"""
VGGT vs DA3 多场景批量测试 — 对比 abs_distance 精度
从 V8 结果中选取多个退化严重的场景，对比 DA3 和 VGGT 的距离估算精度
"""

import os
import sys
import json
import time
import logging
import gc
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

PROJECT_ROOT = Path(__file__).parent.parent
VGGT_ROOT = Path('/home/tione/notebook/tianjungu/projects/Spatial-MLLM/src')
DA3_ROOT = Path('/home/tione/notebook/tianjungu/projects/Depth-Anything-3')

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(VGGT_ROOT))
sys.path.insert(0, str(DA3_ROOT / 'src'))

VIDEO_DIRS = [
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/arkitscenes',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannet',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannetpp',
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

NUM_FRAMES = 8


def find_video_path(scene_name: str) -> Optional[str]:
    for dir_path in VIDEO_DIRS:
        video_path = os.path.join(dir_path, f"{scene_name}.mp4")
        if os.path.exists(video_path):
            return video_path
    return None


def sample_video_frames(video_path: str, num_frames: int = 8):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return [], []
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, indices.tolist()


# ================================================================
# DA3
# ================================================================
class DA3Engine:
    def __init__(self, device='cuda'):
        from core.perception_da3_full import DA3FullEstimator
        self.da3 = DA3FullEstimator(model_name="da3nested-giant-large", device=device, process_res=504)
        self.device = device
    
    def infer(self, frames):
        pred = self.da3.estimate_multiview(frames, ref_view_strategy="saddle_balanced")
        return {
            'depth_maps': pred.depth_maps,
            'extrinsics': pred.extrinsics,
            'intrinsics': pred.intrinsics,
            'proc_shape': pred.depth_maps.shape[1:3],
        }
    
    def unload(self):
        del self.da3.model
        del self.da3
        gc.collect(); torch.cuda.empty_cache()


# ================================================================
# VGGT
# ================================================================
class VGGTEngine:
    def __init__(self, device='cuda'):
        from models.vggt.models.vggt import VGGT
        self.model = VGGT.from_pretrained("facebook/VGGT-1B")
        self.model = self.model.to(device).eval()
        self.device = device
    
    def preprocess(self, frames, target_size=518):
        from torchvision import transforms as TF
        to_tensor = TF.ToTensor()
        images = []
        for frame in frames:
            img = Image.fromarray(frame)
            w, h = img.size
            new_w = target_size
            new_h = round(h * (new_w / w) / 14) * 14
            img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
            t = to_tensor(img)
            if new_h > target_size:
                start_y = (new_h - target_size) // 2
                t = t[:, start_y:start_y + target_size, :]
            images.append(t)
        shapes = set((t.shape[1], t.shape[2]) for t in images)
        if len(shapes) > 1:
            max_h = max(s[0] for s in shapes)
            max_w = max(s[1] for s in shapes)
            padded = []
            for t in images:
                hp = max_h - t.shape[1]; wp = max_w - t.shape[2]
                if hp > 0 or wp > 0:
                    t = F.pad(t, (wp//2, wp-wp//2, hp//2, hp-hp//2), value=1.0)
                padded.append(t)
            images = padded
        return torch.stack(images)
    
    def infer(self, frames):
        from models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
        
        images = self.preprocess(frames).to(self.device)
        N, C, H, W = images.shape
        dtype = self.model.dtype
        if dtype != torch.float32:
            images = images.to(dtype)
        
        with torch.no_grad():
            predictions = self.model(images)
        
        world_points = predictions['world_points'].cpu().float()[0].numpy()
        world_points_conf = predictions['world_points_conf'].cpu().float()[0].numpy()
        depth = predictions['depth'].cpu().float()[0, :, :, :, 0].numpy()
        pose_enc = predictions['pose_enc'].cpu().float()
        
        extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, image_size_hw=(H, W))
        extrinsics_np = extrinsics[0].numpy()
        intrinsics_np = intrinsics[0].numpy()
        
        return {
            'depth_maps': depth,
            'extrinsics': extrinsics_np,
            'intrinsics': intrinsics_np,
            'world_points': world_points,
            'world_points_conf': world_points_conf,
            'proc_shape': (H, W),
        }
    
    def unload(self):
        del self.model
        gc.collect(); torch.cuda.empty_cache()


# ================================================================
# GroundingDINO (一次加载，多次检测)
# ================================================================
class GDINOEngine:
    def __init__(self, device='cuda'):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)
        self.model.eval()
        self.device = device
    
    def detect(self, frames, objects):
        prompt = " . ".join(objects) + " ."
        all_dets = []
        for frame in frames:
            img = Image.fromarray(frame)
            inputs = self.processor(images=img, text=prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                threshold=0.25, text_threshold=0.25,
                target_sizes=[img.size[::-1]]
            )[0]
            frame_dets = []
            labels_list = results.get('text_labels', results.get('labels', []))
            for box, score, label in zip(results['boxes'], results['scores'], labels_list):
                box = box.cpu().numpy()
                frame_dets.append({
                    'label': label.strip().lower(),
                    'bbox_xyxy': box,
                    'confidence': float(score.cpu()),
                })
            all_dets.append(frame_dets)
        return all_dets
    
    def unload(self):
        del self.model, self.processor
        gc.collect(); torch.cuda.empty_cache()


# ================================================================
# 3D 坐标计算
# ================================================================

def compute_3d_unproject(result, frame_idx, bbox_xyxy, orig_shape):
    """传统 bbox中心+中值深度 反投影"""
    depth_maps = result['depth_maps']
    intrinsics = result['intrinsics']
    extrinsics = result['extrinsics']
    proc_H, proc_W = result['proc_shape']
    orig_H, orig_W = orig_shape
    
    scale_x = proc_W / orig_W
    scale_y = proc_H / orig_H
    x1, y1, x2, y2 = bbox_xyxy
    x1s, y1s = x1 * scale_x, y1 * scale_y
    x2s, y2s = x2 * scale_x, y2 * scale_y
    
    x1i = int(np.clip(x1s, 0, proc_W - 1))
    y1i = int(np.clip(y1s, 0, proc_H - 1))
    x2i = int(np.clip(x2s, 0, proc_W - 1))
    y2i = int(np.clip(y2s, 0, proc_H - 1))
    
    depth_region = depth_maps[frame_idx, y1i:y2i+1, x1i:x2i+1]
    depth = float(np.median(depth_region)) if depth_region.size > 0 else float(depth_maps[frame_idx, (y1i+y2i)//2, (x1i+x2i)//2])
    
    cu = (x1s + x2s) / 2
    cv = (y1s + y2s) / 2
    K = intrinsics[frame_idx]
    fx, fy = K[0, 0], K[1, 1]
    cx_k, cy_k = K[0, 2], K[1, 2]
    
    x_cam = (cu - cx_k) / fx * depth
    y_cam = (cv - cy_k) / fy * depth
    cam_point = np.array([x_cam, y_cam, depth])
    
    R = extrinsics[frame_idx, :3, :3]
    t_vec = extrinsics[frame_idx, :3, 3]
    world_point = R.T @ cam_point - R.T @ t_vec
    
    w_px = x2s - x1s
    h_px = y2s - y1s
    max_size = max(w_px * depth / fx, h_px * depth / fy)
    
    return world_point, depth, float(max_size)


def compute_3d_vggt_worldpts(result, frame_idx, bbox_xyxy, orig_shape):
    """VGGT world_points 直接法"""
    world_points = result['world_points']
    world_points_conf = result['world_points_conf']
    depth_maps = result['depth_maps']
    proc_H, proc_W = result['proc_shape']
    orig_H, orig_W = orig_shape
    
    scale_x = proc_W / orig_W
    scale_y = proc_H / orig_H
    x1, y1, x2, y2 = bbox_xyxy
    x1s, y1s = x1 * scale_x, y1 * scale_y
    x2s, y2s = x2 * scale_x, y2 * scale_y
    
    x1i = int(np.clip(x1s, 0, proc_W - 1))
    y1i = int(np.clip(y1s, 0, proc_H - 1))
    x2i = int(np.clip(x2s, 0, proc_W - 1))
    y2i = int(np.clip(y2s, 0, proc_H - 1))
    
    region_pts = world_points[frame_idx, y1i:y2i+1, x1i:x2i+1, :]
    region_conf = world_points_conf[frame_idx, y1i:y2i+1, x1i:x2i+1]
    region_depth = depth_maps[frame_idx, y1i:y2i+1, x1i:x2i+1]
    
    valid_mask = (region_conf > 0.5) & (region_depth > 0.01)
    
    if valid_mask.sum() > 0:
        pts = region_pts[valid_mask]
        centroid = np.median(pts, axis=0)
        if valid_mask.sum() > 10:
            pct = np.percentile(pts, [5, 95], axis=0)
            size_3d = pct[1] - pct[0]
            max_size = float(np.linalg.norm(size_3d))
        else:
            max_size = 0.0
    else:
        cy_i = int((y1i + y2i) / 2)
        cx_i = int((x1i + x2i) / 2)
        centroid = world_points[frame_idx, cy_i, cx_i, :]
        max_size = 0.0
    
    return centroid, float(np.median(region_depth)), max_size


# ================================================================
# 主测试逻辑
# ================================================================

def select_test_samples(n=20):
    """从 V8 结果中选取 abs_distance 测试样本"""
    results_path = PROJECT_ROOT / 'outputs/agentic_pipeline_v8_full/merged/detailed_results.json'
    with open(results_path) as f:
        data = json.load(f)
    
    samples = []
    for item in data:
        if item.get('question_type') != 'object_abs_distance':
            continue
        scene = item.get('scene_name', '')
        vp = find_video_path(scene)
        if not vp:
            continue
        
        # 从 question 提取物体名
        q = item.get('question', '')
        # "what is the distance between the X and the Y"
        import re
        m = re.search(r'between the (.+?) and the (.+?)[\s\(]', q)
        if not m:
            continue
        obj1, obj2 = m.group(1).strip(), m.group(2).strip()
        
        gt_val = item.get('ground_truth', '')
        try:
            gt_float = float(gt_val)
        except:
            continue
        
        samples.append({
            'scene': scene,
            'video': vp,
            'obj1': obj1,
            'obj2': obj2,
            'gt_distance': gt_float,
            'v8_pred': item.get('prediction', ''),
            'v8_score': item.get('score', 0),
        })
    
    # 按 score 排序，取最差的
    samples.sort(key=lambda x: x['v8_score'])
    
    # 去重场景 - 每个场景最多取2个
    seen_scenes = defaultdict(int)
    selected = []
    for s in samples:
        if seen_scenes[s['scene']] < 2 and len(selected) < n:
            selected.append(s)
            seen_scenes[s['scene']] += 1
    
    return selected


def run_batch_test():
    """批量测试"""
    test_samples = select_test_samples(n=20)
    
    print(f"\n{'='*100}")
    print(f"VGGT vs DA3 Batch Comparison — {len(test_samples)} abs_distance samples")
    print(f"{'='*100}\n")
    
    # 按场景分组
    scene_samples = defaultdict(list)
    for s in test_samples:
        scene_samples[s['scene']].append(s)
    
    print(f"Scenes: {list(scene_samples.keys())}")
    
    # 初始化引擎
    logger.info("Loading DA3...")
    da3_engine = DA3Engine()
    
    logger.info("Loading VGGT...")
    vggt_engine = VGGTEngine()
    
    logger.info("Loading GroundingDINO...")
    gdino_engine = GDINOEngine()
    
    # 结果收集
    all_results = []
    
    for scene_name, samples in scene_samples.items():
        video_path = samples[0]['video']
        logger.info(f"\n{'='*60}")
        logger.info(f"Scene: {scene_name} ({len(samples)} questions)")
        
        # 采样帧
        frames, _ = sample_video_frames(video_path, NUM_FRAMES)
        if not frames:
            logger.warning(f"No frames for {scene_name}")
            continue
        orig_shape = (frames[0].shape[0], frames[0].shape[1])
        
        # 收集所有物体
        all_objects = set()
        for s in samples:
            all_objects.add(s['obj1'])
            all_objects.add(s['obj2'])
        objects_list = list(all_objects)
        
        # 运行推理
        logger.info(f"DA3 inference...")
        da3_result = da3_engine.infer(frames)
        
        logger.info(f"VGGT inference...")
        vggt_result = vggt_engine.infer(frames)
        
        logger.info(f"GroundingDINO detection for: {objects_list}")
        all_dets = gdino_engine.detect(frames, objects_list)
        
        # 对每个物体聚合3D位置
        obj_positions = {}  # {label: {da3: [...], vggt_wp: [...], vggt_up: [...]}}
        obj_sizes = {}
        
        for fi in range(len(frames)):
            for det in all_dets[fi]:
                label = det['label']
                bbox = det['bbox_xyxy']
                
                if label not in obj_positions:
                    obj_positions[label] = {'da3': [], 'vggt_wp': [], 'vggt_up': []}
                    obj_sizes[label] = {'da3': [], 'vggt_wp': []}
                
                p_da3, d_da3, s_da3 = compute_3d_unproject(da3_result, fi, bbox, orig_shape)
                obj_positions[label]['da3'].append(p_da3)
                obj_sizes[label]['da3'].append(s_da3)
                
                p_vggt_wp, d_vggt, s_vggt = compute_3d_vggt_worldpts(vggt_result, fi, bbox, orig_shape)
                obj_positions[label]['vggt_wp'].append(p_vggt_wp)
                obj_sizes[label]['vggt_wp'].append(s_vggt)
                
                p_vggt_up, _, _ = compute_3d_unproject(vggt_result, fi, bbox, orig_shape)
                obj_positions[label]['vggt_up'].append(p_vggt_up)
        
        # 计算 scale ratio from depth
        da3_mean_depth = da3_result['depth_maps'].mean()
        vggt_mean_depth = vggt_result['depth_maps'].mean()
        depth_scale_ratio = da3_mean_depth / (vggt_mean_depth + 1e-8)
        
        # 对每个问题计算距离
        for sample in samples:
            obj1 = sample['obj1'].lower()
            obj2 = sample['obj2'].lower()
            gt_dist = sample['gt_distance']
            
            # 找匹配的检测标签
            def find_label(target, positions):
                if target in positions:
                    return target
                for k in positions:
                    if target in k or k in target:
                        return k
                return None
            
            l1 = find_label(obj1, obj_positions)
            l2 = find_label(obj2, obj_positions)
            
            result_entry = {
                'scene': scene_name,
                'obj1': obj1, 'obj2': obj2,
                'gt_distance': gt_dist,
                'v8_pred': sample['v8_pred'],
                'v8_score': sample['v8_score'],
                'depth_scale_ratio': float(depth_scale_ratio),
            }
            
            if l1 and l2 and obj_positions[l1]['da3'] and obj_positions[l2]['da3']:
                # DA3 距离
                p1_da3 = np.median(obj_positions[l1]['da3'], axis=0)
                p2_da3 = np.median(obj_positions[l2]['da3'], axis=0)
                dist_da3 = float(np.linalg.norm(p1_da3 - p2_da3))
                
                # VGGT world_points 距离
                p1_vggt_wp = np.median(obj_positions[l1]['vggt_wp'], axis=0)
                p2_vggt_wp = np.median(obj_positions[l2]['vggt_wp'], axis=0)
                dist_vggt_wp = float(np.linalg.norm(p1_vggt_wp - p2_vggt_wp))
                
                # VGGT unproject 距离
                p1_vggt_up = np.median(obj_positions[l1]['vggt_up'], axis=0)
                p2_vggt_up = np.median(obj_positions[l2]['vggt_up'], axis=0)
                dist_vggt_up = float(np.linalg.norm(p1_vggt_up - p2_vggt_up))
                
                # VGGT 按 depth_scale_ratio 缩放
                dist_vggt_wp_scaled = dist_vggt_wp * depth_scale_ratio
                dist_vggt_up_scaled = dist_vggt_up * depth_scale_ratio
                
                # MRA 计算
                def mra(pred, gt):
                    return max(0, 1.0 - abs(pred - gt) / gt)
                
                result_entry.update({
                    'da3_dist': dist_da3,
                    'vggt_wp_dist': dist_vggt_wp,
                    'vggt_up_dist': dist_vggt_up,
                    'vggt_wp_scaled': dist_vggt_wp_scaled,
                    'vggt_up_scaled': dist_vggt_up_scaled,
                    'da3_mra': mra(dist_da3, gt_dist),
                    'vggt_wp_mra': mra(dist_vggt_wp, gt_dist),
                    'vggt_up_mra': mra(dist_vggt_up, gt_dist),
                    'vggt_wp_scaled_mra': mra(dist_vggt_wp_scaled, gt_dist),
                    'detected': True,
                    'l1': l1, 'l2': l2,
                    'l1_n_dets': len(obj_positions[l1]['da3']),
                    'l2_n_dets': len(obj_positions[l2]['da3']),
                })
            else:
                result_entry.update({
                    'detected': False,
                    'l1': l1, 'l2': l2,
                    'missing': f"{'obj1' if not l1 else ''} {'obj2' if not l2 else ''}".strip(),
                })
            
            all_results.append(result_entry)
    
    # 卸载模型
    da3_engine.unload()
    vggt_engine.unload()
    gdino_engine.unload()
    
    # ================================================================
    # 汇总输出
    # ================================================================
    print(f"\n{'='*120}")
    print(f"{'VGGT vs DA3 Distance Comparison Results':^120}")
    print(f"{'='*120}")
    
    detected_results = [r for r in all_results if r.get('detected', False)]
    undetected = [r for r in all_results if not r.get('detected', False)]
    
    print(f"\nDetected: {len(detected_results)}/{len(all_results)} samples")
    if undetected:
        print(f"Undetected ({len(undetected)}):")
        for r in undetected:
            print(f"  scene={r['scene']}, {r['obj1']}↔{r['obj2']}, missing={r.get('missing','')}")
    
    # 逐样本对比表
    print(f"\n{'Scene':<15} {'Objects':<30} {'GT':>6} {'V8':>7} {'DA3':>7} {'VGGT_WP':>8} {'VGGT_UP':>8} {'VWP_Scl':>8} | {'DA3_MRA':>8} {'VWP_MRA':>8} {'VUP_MRA':>8} {'VWP_S_MRA':>10}")
    print("-" * 150)
    
    for r in detected_results:
        v8p = r.get('v8_pred', '?')
        try:
            v8p_f = f"{float(v8p):.2f}"
        except:
            v8p_f = str(v8p)[:6]
        
        print(f"{r['scene']:<15} {r['obj1'][:12]}↔{r['obj2'][:12]:<15} "
              f"{r['gt_distance']:>6.1f} {v8p_f:>7} "
              f"{r['da3_dist']:>7.2f} {r['vggt_wp_dist']:>8.3f} {r['vggt_up_dist']:>8.3f} {r['vggt_wp_scaled']:>8.2f} | "
              f"{r['da3_mra']:>8.3f} {r['vggt_wp_mra']:>8.3f} {r['vggt_up_mra']:>8.3f} {r['vggt_wp_scaled_mra']:>10.3f}")
    
    # 汇总统计
    if detected_results:
        print(f"\n{'='*80}")
        print(f"AGGREGATE STATISTICS ({len(detected_results)} samples)")
        print(f"{'='*80}")
        
        da3_mras = [r['da3_mra'] for r in detected_results]
        vggt_wp_mras = [r['vggt_wp_mra'] for r in detected_results]
        vggt_up_mras = [r['vggt_up_mra'] for r in detected_results]
        vggt_wp_s_mras = [r['vggt_wp_scaled_mra'] for r in detected_results]
        
        print(f"  DA3 unproject:        mean MRA = {np.mean(da3_mras):.4f}, median = {np.median(da3_mras):.4f}")
        print(f"  VGGT world_points:    mean MRA = {np.mean(vggt_wp_mras):.4f}, median = {np.median(vggt_wp_mras):.4f}")
        print(f"  VGGT depth+unproject: mean MRA = {np.mean(vggt_up_mras):.4f}, median = {np.median(vggt_up_mras):.4f}")
        print(f"  VGGT WP scaled:       mean MRA = {np.mean(vggt_wp_s_mras):.4f}, median = {np.median(vggt_wp_s_mras):.4f}")
        
        # 深度scale ratio统计
        scale_ratios = [r['depth_scale_ratio'] for r in detected_results]
        print(f"\n  Depth scale ratio (DA3/VGGT): mean={np.mean(scale_ratios):.2f}, "
              f"std={np.std(scale_ratios):.2f}, range=[{min(scale_ratios):.2f}, {max(scale_ratios):.2f}]")
        
        # 距离误差统计
        da3_errors = [abs(r['da3_dist'] - r['gt_distance']) for r in detected_results]
        vggt_wp_errors = [abs(r['vggt_wp_dist'] - r['gt_distance']) for r in detected_results]
        vggt_up_errors = [abs(r['vggt_up_dist'] - r['gt_distance']) for r in detected_results]
        
        print(f"\n  DA3 abs error:     mean={np.mean(da3_errors):.3f}m, median={np.median(da3_errors):.3f}m")
        print(f"  VGGT WP abs error: mean={np.mean(vggt_wp_errors):.3f}m, median={np.median(vggt_wp_errors):.3f}m")
        print(f"  VGGT UP abs error: mean={np.mean(vggt_up_errors):.3f}m, median={np.median(vggt_up_errors):.3f}m")
        
        # DA3 vs VGGT head-to-head
        da3_wins = sum(1 for d, v in zip(da3_mras, vggt_wp_mras) if d > v)
        vggt_wins = sum(1 for d, v in zip(da3_mras, vggt_wp_mras) if v > d)
        ties = len(da3_mras) - da3_wins - vggt_wins
        print(f"\n  Head-to-head (DA3 vs VGGT WP): DA3 wins={da3_wins}, VGGT wins={vggt_wins}, ties={ties}")
        
        da3_wins_s = sum(1 for d, v in zip(da3_mras, vggt_wp_s_mras) if d > v)
        vggt_wins_s = sum(1 for d, v in zip(da3_mras, vggt_wp_s_mras) if v > d)
        print(f"  Head-to-head (DA3 vs VGGT WP Scaled): DA3 wins={da3_wins_s}, VGGT wins={vggt_wins_s}")
    
    # 保存结果
    out_dir = PROJECT_ROOT / 'outputs' / 'vggt_vs_da3_comparison'
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / 'batch_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    run_batch_test()
