#!/usr/bin/env python3
"""
VGGT vs DA3 3D Reconstruction Comparison Test

对同一组视频帧，分别使用 VGGT 和 DA3 进行 3D 重建，比较：
1. 深度图精度
2. 相机位姿一致性
3. 物体3D坐标精度（GroundingDINO bbox → 3D点）
4. SAM mask → 3D centroid vs bbox中心 → 3D点 的精度差异

Usage:
    python scripts/test_vggt_vs_da3.py --scene 42444950 --num_frames 8
    python scripts/test_vggt_vs_da3.py --scene scene0050_00 --num_frames 16
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import numpy as np
import cv2
import torch
import torch.nn.functional as F

# === 环境设置 ===
os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
VGGT_ROOT = Path('/home/tione/notebook/tianjungu/projects/Spatial-MLLM/src')
SAM3_ROOT = Path('/home/tione/notebook/tianjungu/projects/sam3')
DA3_ROOT = Path('/home/tione/notebook/tianjungu/projects/Depth-Anything-3')

# 添加必要路径
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(VGGT_ROOT))
sys.path.insert(0, str(DA3_ROOT / 'src'))
sys.path.insert(0, str(SAM3_ROOT))

VIDEO_DIRS = [
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/arkitscenes',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannet',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannetpp',
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# 工具函数
# ============================================================

def find_video_path(scene_name: str) -> Optional[str]:
    for dir_path in VIDEO_DIRS:
        video_path = os.path.join(dir_path, f"{scene_name}.mp4")
        if os.path.exists(video_path):
            return video_path
    return None


def sample_video_frames(video_path: str, num_frames: int = 8) -> Tuple[List[np.ndarray], List[int]]:
    """从视频均匀采样帧，返回 (frames_rgb_list, frame_indices)"""
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


def save_frames_as_temp(frames: List[np.ndarray], tmp_dir: str) -> List[str]:
    """保存帧为临时图片文件（VGGT 需要路径列表输入）"""
    os.makedirs(tmp_dir, exist_ok=True)
    paths = []
    for i, frame in enumerate(frames):
        path = os.path.join(tmp_dir, f"frame_{i:04d}.png")
        cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        paths.append(path)
    return paths


# ============================================================
# DA3 Pipeline
# ============================================================

def run_da3(frames: List[np.ndarray], device='cuda') -> Dict:
    """运行 DA3 多视图推理"""
    from core.perception_da3_full import DA3FullEstimator
    
    logger.info(f"[DA3] Loading model...")
    t0 = time.time()
    da3 = DA3FullEstimator(model_name="da3nested-giant-large", device=device, process_res=504)
    t_load = time.time() - t0
    logger.info(f"[DA3] Model loaded in {t_load:.1f}s")
    
    logger.info(f"[DA3] Running multiview inference on {len(frames)} frames...")
    t0 = time.time()
    pred = da3.estimate_multiview(frames, ref_view_strategy="saddle_balanced")
    t_infer = time.time() - t0
    logger.info(f"[DA3] Inference done in {t_infer:.1f}s")
    logger.info(f"[DA3] depth_maps: {pred.depth_maps.shape}, extrinsics: {pred.extrinsics.shape}, intrinsics: {pred.intrinsics.shape}")
    
    # 提取相机中心
    camera_centers = []
    for i in range(pred.extrinsics.shape[0]):
        R = pred.extrinsics[i, :3, :3]
        t = pred.extrinsics[i, :3, 3]
        cam_pos = -R.T @ t
        camera_centers.append(cam_pos)
    
    result = {
        'depth_maps': pred.depth_maps,           # (N, H, W)
        'extrinsics': pred.extrinsics,            # (N, 3, 4)
        'intrinsics': pred.intrinsics,            # (N, 3, 3)
        'camera_centers': np.array(camera_centers),  # (N, 3)
        'proc_shape': pred.depth_maps.shape[1:3],
        'load_time': t_load,
        'infer_time': t_infer,
    }
    
    # 释放
    del da3.model
    del da3
    import gc; gc.collect(); torch.cuda.empty_cache()
    
    return result


# ============================================================
# VGGT Pipeline
# ============================================================

def preprocess_frames_for_vggt(frames: List[np.ndarray], target_size=518) -> torch.Tensor:
    """
    将 numpy RGB 帧预处理为 VGGT 输入格式
    模拟 load_and_preprocess_images 的 crop 模式，但接受 numpy 数组而非文件路径
    
    Returns: (N, 3, H, W) tensor in [0, 1]
    """
    from torchvision import transforms as TF
    from PIL import Image
    
    to_tensor = TF.ToTensor()
    images = []
    
    for frame in frames:
        img = Image.fromarray(frame)
        width, height = img.size
        
        # crop 模式: width=518, height 按比例（可被14整除）
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
        
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        t = to_tensor(img)  # (3, H, W) in [0, 1]
        
        # 如果高度 > 518，居中裁剪
        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            t = t[:, start_y:start_y + target_size, :]
        
        images.append(t)
    
    # 检查是否有不同尺寸
    shapes = set((t.shape[1], t.shape[2]) for t in images)
    if len(shapes) > 1:
        max_h = max(s[0] for s in shapes)
        max_w = max(s[1] for s in shapes)
        padded = []
        for t in images:
            h_pad = max_h - t.shape[1]
            w_pad = max_w - t.shape[2]
            if h_pad > 0 or w_pad > 0:
                t = F.pad(t, (w_pad // 2, w_pad - w_pad // 2, h_pad // 2, h_pad - h_pad // 2), value=1.0)
            padded.append(t)
        images = padded
    
    return torch.stack(images)  # (N, 3, H, W)


def run_vggt(frames: List[np.ndarray], device='cuda') -> Dict:
    """运行 VGGT 推理"""
    from models.vggt.models.vggt import VGGT
    from models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
    
    logger.info(f"[VGGT] Loading model from facebook/VGGT-1B...")
    t0 = time.time()
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model = model.to(device).eval()
    t_load = time.time() - t0
    logger.info(f"[VGGT] Model loaded in {t_load:.1f}s")
    
    # 预处理
    logger.info(f"[VGGT] Preprocessing {len(frames)} frames...")
    images = preprocess_frames_for_vggt(frames)  # (N, 3, H, W)
    N, C, H, W = images.shape
    logger.info(f"[VGGT] Input shape: {images.shape}")
    
    # 推理
    logger.info(f"[VGGT] Running inference...")
    t0 = time.time()
    with torch.no_grad():
        # VGGT 期望 [B, S, 3, H, W] 或 [S, 3, H, W]
        images_gpu = images.to(device)
        dtype = model.dtype
        if dtype != torch.float32:
            images_gpu = images_gpu.to(dtype)
        
        predictions = model(images_gpu)  # images shape: [S, 3, H, W] → auto add batch dim
    t_infer = time.time() - t0
    logger.info(f"[VGGT] Inference done in {t_infer:.1f}s")
    
    # 提取结果
    world_points = predictions['world_points'].cpu().float()       # [B, S, H, W, 3]
    world_points_conf = predictions['world_points_conf'].cpu().float()  # [B, S, H, W]
    depth = predictions['depth'].cpu().float()                      # [B, S, H, W, 1]
    depth_conf = predictions['depth_conf'].cpu().float()            # [B, S, H, W]
    pose_enc = predictions['pose_enc'].cpu().float()                # [B, S, 9]
    
    # Decode pose to extrinsics and intrinsics
    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        pose_enc, image_size_hw=(H, W)
    )
    # extrinsics: [B, S, 3, 4], intrinsics: [B, S, 3, 3]
    
    extrinsics_np = extrinsics[0].numpy()  # (S, 3, 4)
    intrinsics_np = intrinsics[0].numpy()  # (S, 3, 3)
    world_points_np = world_points[0].numpy()  # (S, H, W, 3)
    world_points_conf_np = world_points_conf[0].numpy()  # (S, H, W)
    depth_np = depth[0, :, :, :, 0].numpy()  # (S, H, W)
    depth_conf_np = depth_conf[0].numpy()  # (S, H, W)
    
    # 提取相机中心 (same convention: w2c [R|t], cam_center = -R^T @ t)
    camera_centers = []
    for i in range(extrinsics_np.shape[0]):
        R = extrinsics_np[i, :3, :3]
        t = extrinsics_np[i, :3, 3]
        cam_pos = -R.T @ t
        camera_centers.append(cam_pos)
    
    result = {
        'depth_maps': depth_np,                      # (S, H, W)
        'extrinsics': extrinsics_np,                  # (S, 3, 4)
        'intrinsics': intrinsics_np,                  # (S, 3, 3)
        'world_points': world_points_np,              # (S, H, W, 3)
        'world_points_conf': world_points_conf_np,    # (S, H, W)
        'depth_conf': depth_conf_np,                  # (S, H, W)
        'camera_centers': np.array(camera_centers),   # (S, 3)
        'proc_shape': (H, W),
        'load_time': t_load,
        'infer_time': t_infer,
    }
    
    # 释放
    del model, predictions
    import gc; gc.collect(); torch.cuda.empty_cache()
    
    return result


# ============================================================
# GroundingDINO 物体检测
# ============================================================

def run_grounding_dino(frames: List[np.ndarray], objects: List[str], device='cuda') -> List[List[Dict]]:
    """
    对每一帧运行 GroundingDINO 检测指定物体
    
    Returns: detections[frame_idx] = [{label, bbox_xyxy, conf}, ...]
    """
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    
    logger.info(f"[GDINO] Loading model...")
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)
    model.eval()
    
    prompt = " . ".join(objects) + " ."
    
    all_dets = []
    for i, frame in enumerate(frames):
        from PIL import Image
        img = Image.fromarray(frame)
        inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        results = processor.post_process_grounded_object_detection(
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
                'bbox_xyxy': box,  # x1, y1, x2, y2 in original image coords
                'confidence': float(score.cpu()),
            })
        all_dets.append(frame_dets)
    
    del model, processor
    import gc; gc.collect(); torch.cuda.empty_cache()
    
    return all_dets


# ============================================================
# SAM3 Segmentation (如可用)
# ============================================================

def run_sam3_segmentation(frame: np.ndarray, bbox_xyxy: np.ndarray, device='cuda') -> Optional[np.ndarray]:
    """
    用 SAM3 对单帧图像中的 bbox 区域生成 mask
    
    Args:
        frame: RGB (H, W, 3)
        bbox_xyxy: (x1, y1, x2, y2)
    
    Returns:
        mask: (H, W) bool array, or None if SAM3 not available
    """
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        from PIL import Image
        
        model = build_sam3_image_model()
        processor = Sam3Processor(model)
        
        img = Image.fromarray(frame)
        state = processor.set_image(img)
        
        # SAM3 box format: [cx, cy, w, h] normalized
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy
        cx = ((x1 + x2) / 2) / W
        cy = ((y1 + y2) / 2) / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H
        
        output = processor.add_geometric_prompt(
            box=[cx, cy, bw, bh],
            label=1,
            state=state,
        )
        
        masks = output.get('masks', None)
        if masks is not None and len(masks) > 0:
            # 取第一个 mask
            mask = masks[0]
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            return mask.astype(bool)
        
        return None
    except Exception as e:
        logger.warning(f"[SAM3] Failed: {e}")
        return None


# ============================================================
# 3D 坐标计算
# ============================================================

def compute_3d_from_da3(da3_result: Dict, frame_idx: int, bbox_xyxy: np.ndarray,
                        orig_shape: Tuple[int, int]) -> Dict:
    """
    使用 DA3 结果计算物体 3D 坐标（bbox 中心 + 中值深度反投影）
    与 grid64_real_test.py 的 build_grid 方法一致
    """
    depth_maps = da3_result['depth_maps']
    intrinsics = da3_result['intrinsics']
    extrinsics = da3_result['extrinsics']
    proc_H, proc_W = da3_result['proc_shape']
    orig_H, orig_W = orig_shape
    
    # 坐标缩放到处理分辨率
    scale_x = proc_W / orig_W
    scale_y = proc_H / orig_H
    x1, y1, x2, y2 = bbox_xyxy
    x1s, y1s = x1 * scale_x, y1 * scale_y
    x2s, y2s = x2 * scale_x, y2 * scale_y
    
    # bbox 中值深度
    x1i = int(np.clip(x1s, 0, proc_W - 1))
    y1i = int(np.clip(y1s, 0, proc_H - 1))
    x2i = int(np.clip(x2s, 0, proc_W - 1))
    y2i = int(np.clip(y2s, 0, proc_H - 1))
    
    depth_region = depth_maps[frame_idx, y1i:y2i+1, x1i:x2i+1]
    if depth_region.size > 0:
        depth = float(np.median(depth_region))
    else:
        cx_i = int((x1i + x2i) / 2)
        cy_i = int((y1i + y2i) / 2)
        depth = float(depth_maps[frame_idx, cy_i, cx_i])
    
    # 像素 → 相机坐标
    cu = (x1s + x2s) / 2
    cv = (y1s + y2s) / 2
    K = intrinsics[frame_idx]
    fx, fy = K[0, 0], K[1, 1]
    cx_k, cy_k = K[0, 2], K[1, 2]
    
    x_cam = (cu - cx_k) / fx * depth
    y_cam = (cv - cy_k) / fy * depth
    z_cam = depth
    cam_point = np.array([x_cam, y_cam, z_cam])
    
    # 相机 → 世界 (w2c → world = R^T @ cam - R^T @ t)
    R = extrinsics[frame_idx, :3, :3]
    t_vec = extrinsics[frame_idx, :3, 3]
    world_point = R.T @ cam_point - R.T @ t_vec
    
    # 3D尺寸
    w_px = x2s - x1s
    h_px = y2s - y1s
    w_3d = w_px * depth / fx
    h_3d = h_px * depth / fy
    
    return {
        'position_3d': world_point,
        'depth': depth,
        'size_3d': (float(w_3d), float(h_3d)),
        'method': 'DA3_bbox_center',
    }


def compute_3d_from_vggt_bbox(vggt_result: Dict, frame_idx: int, bbox_xyxy: np.ndarray,
                               orig_shape: Tuple[int, int]) -> Dict:
    """
    使用 VGGT 的 world_points 直接获取 bbox 区域内点的 3D 坐标
    - bbox 区域内的 world_points 取中位数作为 centroid
    - 用 world_points_conf 做加权或过滤
    """
    world_points = vggt_result['world_points']  # (S, H, W, 3)
    world_points_conf = vggt_result['world_points_conf']  # (S, H, W)
    depth_maps = vggt_result['depth_maps']  # (S, H, W)
    proc_H, proc_W = vggt_result['proc_shape']
    orig_H, orig_W = orig_shape
    
    # 坐标缩放
    scale_x = proc_W / orig_W
    scale_y = proc_H / orig_H
    x1, y1, x2, y2 = bbox_xyxy
    x1s, y1s = x1 * scale_x, y1 * scale_y
    x2s, y2s = x2 * scale_x, y2 * scale_y
    
    x1i = int(np.clip(x1s, 0, proc_W - 1))
    y1i = int(np.clip(y1s, 0, proc_H - 1))
    x2i = int(np.clip(x2s, 0, proc_W - 1))
    y2i = int(np.clip(y2s, 0, proc_H - 1))
    
    # 取 bbox 区域的 world_points
    region_pts = world_points[frame_idx, y1i:y2i+1, x1i:x2i+1, :]  # (h, w, 3)
    region_conf = world_points_conf[frame_idx, y1i:y2i+1, x1i:x2i+1]  # (h, w)
    region_depth = depth_maps[frame_idx, y1i:y2i+1, x1i:x2i+1]  # (h, w)
    
    # 过滤低置信度点
    valid_mask = (region_conf > 0.5) & (region_depth > 0.01)
    
    if valid_mask.sum() > 0:
        pts = region_pts[valid_mask]  # (K, 3)
        confs = region_conf[valid_mask]  # (K,)
        
        # 方法1: 直接取 world_points 中位数
        centroid_median = np.median(pts, axis=0)
        
        # 方法2: 置信度加权平均
        weights = confs / confs.sum()
        centroid_weighted = np.sum(pts * weights[:, None], axis=0)
        
        median_depth = float(np.median(region_depth[valid_mask]))
    else:
        # fallback: 取中心点
        cy_i = int((y1i + y2i) / 2)
        cx_i = int((x1i + x2i) / 2)
        centroid_median = world_points[frame_idx, cy_i, cx_i, :]
        centroid_weighted = centroid_median.copy()
        median_depth = float(depth_maps[frame_idx, cy_i, cx_i])
    
    # 估算 3D 尺寸（从 world_points 范围）
    if valid_mask.sum() > 10:
        pts_range = np.percentile(pts, [5, 95], axis=0)  # (2, 3)
        size_3d = pts_range[1] - pts_range[0]  # (3,) 各轴范围
        w_3d = float(np.max(size_3d[:2]))  # XY 平面上的最大维度（近似）
        h_3d = float(size_3d[2]) if len(size_3d) > 2 else w_3d
    else:
        w_3d, h_3d = 0.0, 0.0
    
    return {
        'position_3d_median': centroid_median,
        'position_3d_weighted': centroid_weighted,
        'depth': median_depth,
        'size_3d': (w_3d, h_3d),
        'num_valid_points': int(valid_mask.sum()),
        'total_points': int(region_pts.shape[0] * region_pts.shape[1]),
        'method': 'VGGT_world_points_bbox',
    }


def compute_3d_from_vggt_extrinsics(vggt_result: Dict, frame_idx: int, bbox_xyxy: np.ndarray,
                                      orig_shape: Tuple[int, int]) -> Dict:
    """
    使用 VGGT 的 depth + extrinsics + intrinsics 做传统反投影（与 DA3 同方法）
    这样可以对比：VGGT depth/pose vs DA3 depth/pose 在相同计算方法下的差异
    """
    depth_maps = vggt_result['depth_maps']
    intrinsics = vggt_result['intrinsics']
    extrinsics = vggt_result['extrinsics']
    proc_H, proc_W = vggt_result['proc_shape']
    orig_H, orig_W = orig_shape
    
    # 同 DA3 流程
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
    if depth_region.size > 0:
        depth = float(np.median(depth_region))
    else:
        depth = float(depth_maps[frame_idx, int((y1i+y2i)/2), int((x1i+x2i)/2)])
    
    cu = (x1s + x2s) / 2
    cv = (y1s + y2s) / 2
    K = intrinsics[frame_idx]
    fx, fy = K[0, 0], K[1, 1]
    cx_k, cy_k = K[0, 2], K[1, 2]
    
    x_cam = (cu - cx_k) / fx * depth
    y_cam = (cv - cy_k) / fy * depth
    z_cam = depth
    cam_point = np.array([x_cam, y_cam, z_cam])
    
    R = extrinsics[frame_idx, :3, :3]
    t_vec = extrinsics[frame_idx, :3, 3]
    world_point = R.T @ cam_point - R.T @ t_vec
    
    w_px = x2s - x1s
    h_px = y2s - y1s
    w_3d = w_px * depth / fx
    h_3d = h_px * depth / fy
    
    return {
        'position_3d': world_point,
        'depth': depth,
        'size_3d': (float(w_3d), float(h_3d)),
        'method': 'VGGT_depth_unproject',
    }


# ============================================================
# 主比较逻辑
# ============================================================

def compare_reconstructions(scene_name: str, objects: List[str], num_frames: int = 8, 
                            device: str = 'cuda', skip_sam: bool = False):
    """主比较函数"""
    
    video_path = find_video_path(scene_name)
    if not video_path:
        logger.error(f"Video not found for scene: {scene_name}")
        return
    
    logger.info(f"=" * 80)
    logger.info(f"Scene: {scene_name}")
    logger.info(f"Video: {video_path}")
    logger.info(f"Objects: {objects}")
    logger.info(f"Frames: {num_frames}")
    logger.info(f"=" * 80)
    
    # 1. 采样视频帧
    frames, frame_indices = sample_video_frames(video_path, num_frames)
    logger.info(f"Sampled {len(frames)} frames, shape: {frames[0].shape}")
    orig_shape = (frames[0].shape[0], frames[0].shape[1])
    
    # 2. 运行 DA3
    logger.info(f"\n{'='*40} DA3 {'='*40}")
    da3_result = run_da3(frames, device)
    
    # 3. 运行 VGGT
    logger.info(f"\n{'='*40} VGGT {'='*40}")
    vggt_result = run_vggt(frames, device)
    
    # 4. 运行 GroundingDINO 检测
    logger.info(f"\n{'='*40} GroundingDINO {'='*40}")
    all_dets = run_grounding_dino(frames, objects, device)
    
    # 5. 对每个物体、每帧计算 3D 坐标
    logger.info(f"\n{'='*40} 3D Comparison {'='*40}")
    
    results_per_object = defaultdict(lambda: {
        'da3': [], 'vggt_wp': [], 'vggt_unproj': [],
    })
    
    for fi in range(len(frames)):
        dets = all_dets[fi]
        for det in dets:
            label = det['label']
            if label not in [o.lower() for o in objects]:
                continue
            
            bbox = det['bbox_xyxy']
            
            # DA3 方法
            da3_3d = compute_3d_from_da3(da3_result, fi, bbox, orig_shape)
            results_per_object[label]['da3'].append(da3_3d)
            
            # VGGT world_points 直接方法
            vggt_wp = compute_3d_from_vggt_bbox(vggt_result, fi, bbox, orig_shape)
            results_per_object[label]['vggt_wp'].append(vggt_wp)
            
            # VGGT depth+unproject 方法（同DA3流程，但用VGGT的depth/pose）
            vggt_up = compute_3d_from_vggt_extrinsics(vggt_result, fi, bbox, orig_shape)
            results_per_object[label]['vggt_unproj'].append(vggt_up)
    
    # 6. 汇总对比
    logger.info(f"\n{'='*80}")
    logger.info(f"RESULTS SUMMARY")
    logger.info(f"{'='*80}")
    
    # 6a. 性能对比
    print(f"\n[Performance]")
    print(f"  DA3:  load={da3_result['load_time']:.1f}s, infer={da3_result['infer_time']:.1f}s, shape={da3_result['proc_shape']}")
    print(f"  VGGT: load={vggt_result['load_time']:.1f}s, infer={vggt_result['infer_time']:.1f}s, shape={vggt_result['proc_shape']}")
    
    # 6b. 相机位姿对比
    print(f"\n[Camera Centers (world coords)]")
    da3_cams = da3_result['camera_centers']
    vggt_cams = vggt_result['camera_centers']
    print(f"  DA3  cam trajectory range: {da3_cams.min(axis=0)} → {da3_cams.max(axis=0)}")
    print(f"  VGGT cam trajectory range: {vggt_cams.min(axis=0)} → {vggt_cams.max(axis=0)}")
    
    # 归一化后对比（两者的世界坐标系可能不同，但相对距离应一致）
    da3_cam_dists = np.linalg.norm(da3_cams - da3_cams[0:1], axis=1)
    vggt_cam_dists = np.linalg.norm(vggt_cams - vggt_cams[0:1], axis=1)
    print(f"  DA3  cam distances from frame0: {np.round(da3_cam_dists, 3)}")
    print(f"  VGGT cam distances from frame0: {np.round(vggt_cam_dists, 3)}")
    
    if da3_cam_dists.max() > 0 and vggt_cam_dists.max() > 0:
        scale_ratio = np.median(da3_cam_dists[1:] / (vggt_cam_dists[1:] + 1e-8))
        print(f"  Scale ratio (DA3/VGGT): {scale_ratio:.4f}")
    
    # 6c. 深度图对比
    print(f"\n[Depth Statistics]")
    da3_depths = da3_result['depth_maps']
    vggt_depths = vggt_result['depth_maps']
    print(f"  DA3  depth: shape={da3_depths.shape}, range=[{da3_depths.min():.3f}, {da3_depths.max():.3f}], mean={da3_depths.mean():.3f}")
    print(f"  VGGT depth: shape={vggt_depths.shape}, range=[{vggt_depths.min():.3f}, {vggt_depths.max():.3f}], mean={vggt_depths.mean():.3f}")
    
    # 6d. 物体3D坐标对比
    print(f"\n[Object 3D Positions]")
    
    for obj_name, methods in results_per_object.items():
        print(f"\n  Object: '{obj_name}' (detected in {len(methods['da3'])} frames)")
        
        if not methods['da3']:
            print(f"    No detections!")
            continue
        
        # 聚合: 取各帧结果的中位数
        da3_positions = np.array([r['position_3d'] for r in methods['da3']])
        vggt_wp_positions = np.array([r['position_3d_median'] for r in methods['vggt_wp']])
        vggt_up_positions = np.array([r['position_3d'] for r in methods['vggt_unproj']])
        
        da3_agg = np.median(da3_positions, axis=0)
        vggt_wp_agg = np.median(vggt_wp_positions, axis=0)
        vggt_up_agg = np.median(vggt_up_positions, axis=0)
        
        da3_depths_obj = [r['depth'] for r in methods['da3']]
        vggt_depths_obj = [r['depth'] for r in methods['vggt_wp']]
        
        print(f"    DA3 aggregated position:         {np.round(da3_agg, 4)}")
        print(f"    VGGT world_points aggregated:    {np.round(vggt_wp_agg, 4)}")
        print(f"    VGGT unproject aggregated:       {np.round(vggt_up_agg, 4)}")
        print(f"    DA3 median depth:   {np.median(da3_depths_obj):.3f}m")
        print(f"    VGGT median depth:  {np.median(vggt_depths_obj):.3f}m")
        
        # 帧间一致性（标准差 — 越小越稳定）
        da3_std = np.std(da3_positions, axis=0)
        vggt_wp_std = np.std(vggt_wp_positions, axis=0)
        print(f"    DA3 position std:   {np.round(da3_std, 4)} (total={np.linalg.norm(da3_std):.4f})")
        print(f"    VGGT wp pos std:    {np.round(vggt_wp_std, 4)} (total={np.linalg.norm(vggt_wp_std):.4f})")
        
        # 3D尺寸
        da3_sizes = [r['size_3d'] for r in methods['da3']]
        vggt_sizes = [r['size_3d'] for r in methods['vggt_wp']]
        da3_max_size = np.median([max(s[0], s[1]) for s in da3_sizes])
        vggt_max_size = np.median([max(s[0], s[1]) for s in vggt_sizes])
        print(f"    DA3 median max_size: {da3_max_size:.3f}m")
        print(f"    VGGT median max_size: {vggt_max_size:.3f}m")
    
    # 6e. 物体间距离对比
    obj_names = list(results_per_object.keys())
    if len(obj_names) >= 2:
        print(f"\n[Pairwise Distances]")
        for i in range(len(obj_names)):
            for j in range(i+1, len(obj_names)):
                name_i, name_j = obj_names[i], obj_names[j]
                mi, mj = results_per_object[name_i], results_per_object[name_j]
                
                if not mi['da3'] or not mj['da3']:
                    continue
                
                da3_pi = np.median([r['position_3d'] for r in mi['da3']], axis=0)
                da3_pj = np.median([r['position_3d'] for r in mj['da3']], axis=0)
                
                vggt_pi = np.median([r['position_3d_median'] for r in mi['vggt_wp']], axis=0)
                vggt_pj = np.median([r['position_3d_median'] for r in mj['vggt_wp']], axis=0)
                
                vggt_up_pi = np.median([r['position_3d'] for r in mi['vggt_unproj']], axis=0)
                vggt_up_pj = np.median([r['position_3d'] for r in mj['vggt_unproj']], axis=0)
                
                dist_da3 = float(np.linalg.norm(da3_pi - da3_pj))
                dist_vggt_wp = float(np.linalg.norm(vggt_pi - vggt_pj))
                dist_vggt_up = float(np.linalg.norm(vggt_up_pi - vggt_up_pj))
                
                print(f"  {name_i} ↔ {name_j}:")
                print(f"    DA3 distance:          {dist_da3:.3f}m")
                print(f"    VGGT world_points:     {dist_vggt_wp:.3f}m")
                print(f"    VGGT depth+unproject:  {dist_vggt_up:.3f}m")
    
    # 6f. VGGT world_points 置信度统计
    print(f"\n[VGGT World Points Confidence]")
    wp_conf = vggt_result['world_points_conf']  # (S, H, W)
    print(f"  Overall: mean={wp_conf.mean():.4f}, median={np.median(wp_conf):.4f}, "
          f"min={wp_conf.min():.4f}, max={wp_conf.max():.4f}")
    print(f"  Per-frame means: {[f'{wp_conf[i].mean():.4f}' for i in range(min(5, wp_conf.shape[0]))]}")
    
    # 清理临时文件
    return {
        'da3': da3_result,
        'vggt': vggt_result,
        'detections': all_dets,
        'objects_3d': dict(results_per_object),
    }


# ============================================================
# Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="VGGT vs DA3 3D Reconstruction Comparison")
    parser.add_argument('--scene', type=str, default='42444950',
                        help='Scene name (e.g., 42444950, scene0050_00)')
    parser.add_argument('--objects', type=str, nargs='+', default=['chair', 'sofa'],
                        help='Objects to detect and compare')
    parser.add_argument('--num_frames', type=int, default=8,
                        help='Number of frames to sample')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--skip_sam', action='store_true',
                        help='Skip SAM3 segmentation')
    
    args = parser.parse_args()
    
    compare_reconstructions(
        scene_name=args.scene,
        objects=args.objects,
        num_frames=args.num_frames,
        device=args.device,
        skip_sam=args.skip_sam,
    )


if __name__ == '__main__':
    main()
