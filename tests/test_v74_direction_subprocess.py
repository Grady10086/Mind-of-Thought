#!/usr/bin/env python3
"""
V7.4 - 方向问题修复版 (使用subprocess实现真正的GPU隔离)

修复内容：
1. 使用subprocess而非multiprocessing确保CUDA设备隔离
2. 正则表达式支持 hard 格式
3. 正确的方向计算算法
4. 保存完整的 position_3d 坐标
"""

import os
import sys
import json
import re
import gc
import argparse
import logging
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from datetime import datetime

import numpy as np
from tqdm import tqdm

# 设置环境
os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# 方向计算算法
# ============================================================================

def compute_direction_hard(standing_pos, facing_pos, target_pos):
    """
    计算三物体方向 (hard格式)
    
    建立局部坐标系：
    - 原点: standing position
    - 前方: 指向 facing position
    - 右侧: 垂直于前方（右手定则）
    
    返回: front-left, front-right, back-left, back-right
    """
    standing = np.array(standing_pos)
    facing = np.array(facing_pos)
    target = np.array(target_pos)
    
    # 使用X-Z平面（水平面），忽略Y（高度）
    face_dir = np.array([facing[0] - standing[0], facing[2] - standing[2]])
    face_norm = np.linalg.norm(face_dir)
    if face_norm < 1e-6:
        return "unknown", {}
    face_dir = face_dir / face_norm
    
    # 右方向（顺时针90度）
    right_dir = np.array([face_dir[1], -face_dir[0]])
    
    # 目标相对位置
    target_rel = np.array([target[0] - standing[0], target[2] - standing[2]])
    
    # 投影
    front = np.dot(target_rel, face_dir)
    right = np.dot(target_rel, right_dir)
    
    # 方向判断
    fb = "front" if front > 0 else "back"
    lr = "right" if right > 0 else "left"
    
    debug = {
        'standing': standing.tolist(),
        'facing': facing.tolist(),
        'target': target.tolist(),
        'face_dir': face_dir.tolist(),
        'right_dir': right_dir.tolist(),
        'front_proj': float(front),
        'right_proj': float(right),
    }
    
    return f"{fb}-{lr}", debug


def compute_direction_simple(ref_pos, target_pos):
    """
    计算两物体方向 (easy/medium格式)
    假设默认朝向为+Z方向
    """
    ref = np.array(ref_pos)
    target = np.array(target_pos)
    
    dx = target[0] - ref[0]
    dz = target[2] - ref[2] if len(target) > 2 else 0
    
    # 简单的左右判断（基于X轴）
    lr = "right" if dx > 0 else "left"
    fb = "front" if dz > 0 else "behind"
    
    return lr, fb


# ============================================================================
# Worker 脚本内容
# ============================================================================

WORKER_SCRIPT = '''
#!/usr/bin/env python3
"""GPU Worker for V7.4 Direction Test"""

import os
import sys
import json
import re
import gc
import logging
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
import torch

# 设置环境
os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ============================================================================
# 基础数据结构
# ============================================================================

@dataclass
class MindMapEntity:
    label: str
    instances: List[Dict] = field(default_factory=list)
    position_3d: np.ndarray = None
    size_3d: np.ndarray = None
    confidence: float = 0.0

# ============================================================================
# 扩展词汇表
# ============================================================================

EXTENDED_VOCABULARY = [
    "chair", "table", "sofa", "couch", "stove", "tv", "television",
    "bed", "cabinet", "shelf", "desk", "lamp", "refrigerator",
    "sink", "toilet", "bathtub", "door", "window", "picture",
    "pillow", "cushion", "monitor", "backpack", "bag",
    "trash can", "trash bin", "mirror", "towel", "plant",
    "nightstand", "closet", "microwave", "printer", "washer",
    "fireplace", "piano", "guitar", "clock", "fan",
]

# ============================================================================
# 心智地图构建器
# ============================================================================

class MindMapBuilder:
    """心智地图构建器"""
    
    def __init__(self, device: str = 'cuda', num_frames: int = 32, box_threshold: float = 0.25):
        self.device = device
        self.num_frames = num_frames
        self.box_threshold = box_threshold
        self._labeler = None
        self._depth_estimator = None
        self.focal_length = 500
    
    def load_models(self):
        if self._labeler is None:
            from core.semantic_labeler import GroundingDINOLabeler
            self._labeler = GroundingDINOLabeler(
                model_id="IDEA-Research/grounding-dino-base",
                device=self.device,
                box_threshold=self.box_threshold,
                text_threshold=0.25,
            )
            self._labeler.load_model()
        
        if self._depth_estimator is None:
            from core.perception import DepthEstimator
            self._depth_estimator = DepthEstimator(
                model_name="depth-anything/DA3-Large",
                device=self.device,
                half_precision=True,
            )
    
    def unload(self):
        if self._labeler is not None:
            try:
                del self._labeler.model
                del self._labeler.processor
            except:
                pass
            self._labeler = None
        if self._depth_estimator is not None:
            self._depth_estimator = None
        gc.collect()
        torch.cuda.empty_cache()
    
    def build_from_video(self, video_path: str, target_objects: List[str] = None) -> Tuple[Dict[str, MindMapEntity], int]:
        """构建心智地图"""
        self.load_models()
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return {}, 0
        
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        vocab = list(set((target_objects or []) + EXTENDED_VOCABULARY))
        prompt = " . ".join(vocab) + " ."
        
        all_detections = defaultdict(list)
        
        for fidx, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            
            # 检测
            detections = self._labeler.detect(frame_rgb, prompt)
            
            # 深度估计
            depth_tensor, _ = self._depth_estimator.infer_single(frame_rgb)
            depth = depth_tensor.squeeze().cpu().numpy()  # (H, W)
            
            for det in detections:
                label = det.label.lower()
                box = det.bbox_pixels
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                box_w = box[2] - box[0]
                box_h = box[3] - box[1]
                
                # 获取深度
                x_int = int(np.clip(cx, 0, w-1))
                y_int = int(np.clip(cy, 0, h-1))
                z = float(depth[y_int, x_int])
                
                # 3D位置估算
                x_3d = (cx - w/2) * z / self.focal_length
                y_3d = (cy - h/2) * z / self.focal_length
                
                all_detections[label].append({
                    'frame_idx': int(frame_idx),
                    'bbox': [float(b) for b in box],
                    'confidence': float(det.confidence),
                    'depth': z,
                    'x_3d': x_3d,
                    'y_3d': y_3d,
                    'z_3d': z,
                })
        
        cap.release()
        
        # 聚合为心智地图
        mind_map = {}
        for label, instances in all_detections.items():
            if not instances:
                continue
            
            # 平均位置
            x_avg = np.mean([d['x_3d'] for d in instances])
            y_avg = np.mean([d['y_3d'] for d in instances])
            z_avg = np.mean([d['z_3d'] for d in instances])
            conf_avg = np.mean([d['confidence'] for d in instances])
            
            mind_map[label] = MindMapEntity(
                label=label,
                instances=instances,
                position_3d=np.array([x_avg, y_avg, z_avg]),
                confidence=conf_avg,
            )
        
        return mind_map, total_frames


# ============================================================================
# 尺度校准器
# ============================================================================

CALIBRATION_OBJECTS = {
    'door': 2.0,
    'chair': 0.85,
    'table': 0.75,
    'desk': 0.75,
    'bed': 0.5,
    'sofa': 0.85,
    'refrigerator': 1.7,
    'toilet': 0.4,
    'tv': 0.5,
}

class ScaleCalibrator:
    def calibrate(self, mind_map: Dict[str, MindMapEntity]):
        @dataclass
        class Result:
            scale_factor: float = 1.0
            ref_object: str = None
        
        for ref_label, expected_size in CALIBRATION_OBJECTS.items():
            if ref_label in mind_map:
                entity = mind_map[ref_label]
                if entity.position_3d is not None:
                    # 简单的尺度校准
                    return Result(scale_factor=1.0, ref_object=ref_label)
        return Result()


# ============================================================================
# 方向计算
# ============================================================================

def compute_direction_hard(standing_pos, facing_pos, target_pos):
    standing = np.array(standing_pos)
    facing = np.array(facing_pos)
    target = np.array(target_pos)
    
    face_dir = np.array([facing[0] - standing[0], facing[2] - standing[2]])
    face_norm = np.linalg.norm(face_dir)
    if face_norm < 1e-6:
        return "unknown", {}
    face_dir = face_dir / face_norm
    
    right_dir = np.array([face_dir[1], -face_dir[0]])
    target_rel = np.array([target[0] - standing[0], target[2] - standing[2]])
    
    front = np.dot(target_rel, face_dir)
    right = np.dot(target_rel, right_dir)
    
    fb = "front" if front > 0 else "back"
    lr = "right" if right > 0 else "left"
    
    return f"{fb}-{lr}", {
        'front_proj': float(front),
        'right_proj': float(right),
    }


def compute_direction_simple(ref_pos, target_pos):
    ref = np.array(ref_pos)
    target = np.array(target_pos)
    dx = target[0] - ref[0]
    return "right" if dx > 0 else "left"


# ============================================================================
# 问题解析
# ============================================================================

def parse_direction_question(question: str) -> Dict:
    q = question.lower()
    
    # hard格式: standing by X, facing Y, is Z to my...
    m = re.search(r"standing by (?:the )?(\w+).*?facing (?:the )?(\w+).*?is (?:the )?(\w+) to my", q)
    if m:
        return {'format': 'hard', 'standing': m.group(1), 'facing': m.group(2), 'target': m.group(3)}
    
    # medium格式
    m = re.search(r"direction of (?:the )?(\w+) from (?:the )?(\w+)", q)
    if m:
        return {'format': 'medium', 'target': m.group(1), 'ref': m.group(2)}
    
    # easy格式
    m = re.search(r"is (?:the )?(\w+) to the.*?of (?:the )?(\w+)", q)
    if m:
        return {'format': 'easy', 'target': m.group(1), 'ref': m.group(2)}
    
    # 另一种hard格式
    m = re.search(r"standing by (?:the )?(\w+).*?facing (?:the )?(\w+).*?is (?:the )?(\w+) to the", q)
    if m:
        return {'format': 'hard_lr', 'standing': m.group(1), 'facing': m.group(2), 'target': m.group(3)}
    
    return None


# ============================================================================
# 物体匹配
# ============================================================================

def find_object_in_mindmap(query: str, mind_map: Dict[str, MindMapEntity]) -> Optional[MindMapEntity]:
    query = query.lower().strip()
    
    # 精确匹配
    if query in mind_map:
        return mind_map[query]
    
    # 模糊匹配
    for label, entity in mind_map.items():
        if query in label or label in query:
            return entity
    
    # 同义词
    synonyms = {
        'couch': 'sofa',
        'television': 'tv',
        'fridge': 'refrigerator',
    }
    if query in synonyms and synonyms[query] in mind_map:
        return mind_map[synonyms[query]]
    
    return None


# ============================================================================
# 规则推理
# ============================================================================

def rule_answer_direction(mind_map, question, options):
    extra_info = {'parsed': None, 'positions': {}, 'computed_direction': None, 'debug': {}}
    
    parsed = parse_direction_question(question)
    extra_info['parsed'] = parsed
    
    if not parsed:
        return options[0][0] if options else "A", "Could not parse", extra_info
    
    if parsed['format'] in ['hard', 'hard_lr']:
        standing_e = find_object_in_mindmap(parsed['standing'], mind_map)
        facing_e = find_object_in_mindmap(parsed['facing'], mind_map)
        target_e = find_object_in_mindmap(parsed['target'], mind_map)
        
        extra_info['positions'] = {
            'standing': {'query': parsed['standing'], 'found': standing_e.label if standing_e else None,
                        'pos': standing_e.position_3d.tolist() if standing_e and standing_e.position_3d is not None else None},
            'facing': {'query': parsed['facing'], 'found': facing_e.label if facing_e else None,
                      'pos': facing_e.position_3d.tolist() if facing_e and facing_e.position_3d is not None else None},
            'target': {'query': parsed['target'], 'found': target_e.label if target_e else None,
                      'pos': target_e.position_3d.tolist() if target_e and target_e.position_3d is not None else None},
        }
        
        if not all([standing_e, facing_e, target_e]):
            return options[0][0] if options else "A", "Objects not found", extra_info
        
        if any(e.position_3d is None for e in [standing_e, facing_e, target_e]):
            return options[0][0] if options else "A", "Missing position_3d", extra_info
        
        computed, debug = compute_direction_hard(
            standing_e.position_3d, facing_e.position_3d, target_e.position_3d
        )
        extra_info['computed_direction'] = computed
        extra_info['debug'] = debug
        
        for i, opt in enumerate(options):
            if computed in opt.lower():
                return chr(65 + i), f"Computed {computed}", extra_info
        
        return options[0][0] if options else "A", f"Computed {computed} (no match)", extra_info
    
    elif parsed['format'] in ['medium', 'easy']:
        ref_e = find_object_in_mindmap(parsed['ref'], mind_map)
        target_e = find_object_in_mindmap(parsed['target'], mind_map)
        
        extra_info['positions'] = {
            'ref': {'query': parsed['ref'], 'found': ref_e.label if ref_e else None,
                   'pos': ref_e.position_3d.tolist() if ref_e and ref_e.position_3d is not None else None},
            'target': {'query': parsed['target'], 'found': target_e.label if target_e else None,
                      'pos': target_e.position_3d.tolist() if target_e and target_e.position_3d is not None else None},
        }
        
        if not all([ref_e, target_e]):
            return options[0][0] if options else "A", "Objects not found", extra_info
        
        if any(e.position_3d is None for e in [ref_e, target_e]):
            return options[0][0] if options else "A", "Missing position_3d", extra_info
        
        computed = compute_direction_simple(ref_e.position_3d, target_e.position_3d)
        extra_info['computed_direction'] = computed
        
        for i, opt in enumerate(options):
            if computed in opt.lower():
                return chr(65 + i), f"Computed {computed}", extra_info
        
        return options[0][0] if options else "A", f"Computed {computed} (no match)", extra_info
    
    return options[0][0] if options else "A", "Unknown format", extra_info


# ============================================================================
# Main Worker
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    with open(args.input) as f:
        samples = json.load(f)
    
    device = 'cuda'
    builder = MindMapBuilder(device=device, num_frames=32)
    calibrator = ScaleCalibrator()
    
    results = []
    scene_cache = {}
    
    for sample in samples:
        scene_name = sample['scene_name']
        question = sample['question']
        gt = sample['ground_truth']
        options = sample.get('options', [])
        
        try:
            if scene_name not in scene_cache:
                video_path = sample['video_path']
                mind_map, total_frames = builder.build_from_video(video_path)
                
                calibration = calibrator.calibrate(mind_map)
                scene_cache[scene_name] = {'mind_map': mind_map, 'calibration': calibration}
            
            cached = scene_cache[scene_name]
            mind_map = cached['mind_map']
            
            rule_pred, rule_reasoning, extra_info = rule_answer_direction(mind_map, question, options)
            rule_correct = rule_pred == gt
            
            mind_map_full = {}
            for label, entity in mind_map.items():
                mind_map_full[label] = {
                    'label': label,
                    'position_3d': entity.position_3d.tolist() if entity.position_3d is not None else None,
                    'confidence': entity.confidence,
                }
            
            results.append({
                'scene_name': scene_name,
                'question': question,
                'question_type': sample['question_type'],
                'options': options,
                'ground_truth': gt,
                'rule_prediction': rule_pred,
                'rule_reasoning': rule_reasoning,
                'correct': rule_correct,
                'extra_info': extra_info,
                'mind_map': mind_map_full,
                'error': None,
            })
            
        except Exception as e:
            results.append({
                'scene_name': scene_name,
                'question': question,
                'question_type': sample['question_type'],
                'options': options,
                'ground_truth': gt,
                'rule_prediction': None,
                'correct': None,
                'error': str(e),
            })
    
    builder.unload()
    
    with open(args.output, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
'''


# ============================================================================
# 数据加载
# ============================================================================

VIDEO_DIRS = [
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/arkitscenes',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannet',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannetpp',
]

def find_video_path(scene_name: str) -> Optional[str]:
    """找到视频路径"""
    for dir_path in VIDEO_DIRS:
        video_path = os.path.join(dir_path, f"{scene_name}.mp4")
        if os.path.exists(video_path):
            return video_path
    return None


def load_direction_data() -> List[Dict]:
    """加载方向问题数据"""
    from datasets import load_dataset
    
    logger.info("加载 VSI-Bench 数据集...")
    ds = load_dataset(
        'nyu-visionx/VSI-Bench',
        split='test',
        cache_dir='/home/tione/notebook/tianjungu/hf_cache/vsibench'
    )
    
    samples = []
    for item in ds:
        if 'direction' not in item['question_type']:
            continue
        
        scene_name = item['scene_name']
        video_path = find_video_path(scene_name)
        
        if not video_path:
            continue
        
        samples.append({
            'scene_name': scene_name,
            'video_path': video_path,
            'question': item['question'],
            'question_type': item['question_type'],
            'options': item.get('options', []),
            'ground_truth': item['ground_truth'],
        })
    
    logger.info(f"筛选方向问题: {len(samples)} 条")
    return samples


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='V7.4 方向问题测试 (subprocess隔离)')
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default='outputs')
    args = parser.parse_args()
    
    # 加载数据
    samples = load_direction_data()
    
    if args.max_samples:
        samples = samples[:args.max_samples]
    
    num_gpus = min(args.num_gpus, len(samples))
    logger.info(f"使用 {num_gpus} 个 GPU 处理 {len(samples)} 样本")
    
    # 按场景分组
    scene_samples = defaultdict(list)
    for s in samples:
        scene_samples[s['scene_name']].append(s)
    
    # 分配到各GPU
    scenes = list(scene_samples.keys())
    gpu_assignments = [[] for _ in range(num_gpus)]
    for i, scene in enumerate(scenes):
        gpu_id = i % num_gpus
        gpu_assignments[gpu_id].extend(scene_samples[scene])
    
    # 写入worker脚本
    worker_script_path = PROJECT_ROOT / 'tests' / '_worker_v74.py'
    with open(worker_script_path, 'w') as f:
        f.write(WORKER_SCRIPT)
    
    # 启动所有GPU进程
    processes = []
    temp_files = []
    
    for gpu_id in range(num_gpus):
        if not gpu_assignments[gpu_id]:
            continue
        
        # 写入输入文件
        input_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(gpu_assignments[gpu_id], input_file)
        input_file.close()
        
        # 输出文件
        output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        output_file.close()
        
        temp_files.append((input_file.name, output_file.name))
        
        # 启动进程
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        env['HIP_VISIBLE_DEVICES'] = str(gpu_id)
        
        cmd = [
            sys.executable,
            str(worker_script_path),
            '--input', input_file.name,
            '--output', output_file.name,
        ]
        
        logger.info(f"启动 GPU {gpu_id} 进程: {len(gpu_assignments[gpu_id])} 样本")
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append((gpu_id, p))
    
    # 等待所有进程完成
    all_results = []
    for gpu_id, p in tqdm(processes, desc="等待GPU进程"):
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            logger.error(f"GPU {gpu_id} 进程失败: {stderr.decode()[:500]}")
            continue
        
        # 读取结果
        _, output_file = temp_files[gpu_id]
        try:
            with open(output_file) as f:
                results = json.load(f)
            all_results.extend(results)
            logger.info(f"GPU {gpu_id}: {len(results)} 结果")
        except Exception as e:
            logger.error(f"GPU {gpu_id} 结果读取失败: {e}")
    
    # 清理临时文件
    for inp, out in temp_files:
        try:
            os.unlink(inp)
            os.unlink(out)
        except:
            pass
    
    try:
        os.unlink(worker_script_path)
    except:
        pass
    
    # 分析结果
    logger.info(f"\n{'='*80}")
    logger.info("V7.4 方向问题测试结果")
    logger.info(f"{'='*80}")
    
    stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'error': 0, 'parsed': 0, 'computed': 0})
    
    for r in all_results:
        qtype = r['question_type']
        stats[qtype]['total'] += 1
        
        if r.get('error'):
            stats[qtype]['error'] += 1
        elif r.get('correct'):
            stats[qtype]['correct'] += 1
        
        extra = r.get('extra_info', {})
        if extra.get('parsed'):
            stats[qtype]['parsed'] += 1
        if extra.get('computed_direction'):
            stats[qtype]['computed'] += 1
    
    total_correct = 0
    total_samples = 0
    
    for qtype, s in sorted(stats.items()):
        total = s['total']
        correct = s['correct']
        error = s['error']
        parsed = s['parsed']
        computed = s['computed']
        
        total_samples += total
        total_correct += correct
        
        logger.info(f"\n{qtype}: {total} 样本")
        logger.info(f"  错误: {error} ({100*error/total:.1f}%)")
        logger.info(f"  成功解析: {parsed} ({100*parsed/total:.1f}%)")
        logger.info(f"  成功计算: {computed} ({100*computed/total:.1f}%)")
        logger.info(f"  规则正确: {correct} ({100*correct/total:.1f}%)")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"总计: {total_samples} 样本")
    logger.info(f"正确: {total_correct} ({100*total_correct/total_samples:.1f}%)")
    logger.info(f"{'='*80}")
    
    # 保存结果
    output_dir = Path(args.output_dir) / f"v74_direction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'detailed_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    summary = {
        'total': total_samples,
        'correct': total_correct,
        'accuracy': total_correct / total_samples if total_samples > 0 else 0,
        'by_type': {k: dict(v) for k, v in stats.items()},
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n结果保存到: {output_dir}")
    
    # 打印几个典型案例
    logger.info(f"\n{'='*80}")
    logger.info("典型案例分析")
    logger.info(f"{'='*80}")
    
    for qtype in stats.keys():
        # 找一个成功计算的案例
        for r in all_results:
            if r['question_type'] == qtype and not r.get('error') and r.get('extra_info', {}).get('computed_direction'):
                logger.info(f"\n[{qtype}] 案例:")
                logger.info(f"  问题: {r['question'][:100]}...")
                logger.info(f"  GT: {r['ground_truth']}")
                logger.info(f"  预测: {r['rule_prediction']}")
                logger.info(f"  计算方向: {r['extra_info']['computed_direction']}")
                logger.info(f"  正确: {r['correct']}")
                if r.get('mind_map'):
                    logger.info(f"  心智地图物体: {list(r['mind_map'].keys())[:5]}...")
                break


if __name__ == '__main__':
    main()
