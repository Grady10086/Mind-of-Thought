#!/usr/bin/env python3
"""
V7.5 - 使用DA3多帧融合 + 扩展词汇表的方向问题测试

修复内容：
1. 使用 infer_video 而不是 infer_single (多帧时序平滑)
2. 扩展词汇表覆盖缺失物体
3. 统一尺度校准
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
from dataclasses import dataclass, field
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
# 扩展词汇表 - 添加缺失物体
# ============================================================================

EXTENDED_VOCABULARY_V75 = [
    # V7原有
    "chair", "table", "sofa", "couch", "stove", "tv", "television",
    "bed", "cabinet", "shelf", "desk", "lamp", "refrigerator", "fridge",
    "sink", "toilet", "bathtub", "door", "window", "picture", "painting",
    "pillow", "cushion", "monitor", "backpack", "bag", "heater",
    "trash can", "trash bin", "mirror", "towel", "plant", "cup",
    "nightstand", "closet", "microwave", "printer", "washer", "dryer",
    "oven", "counter", "drawer", "curtain", "rug", "carpet", "clock",
    "fan", "air conditioner", "bookshelf", "armchair", "stool",
    # V7.5新增 - 基于缺失分析
    "telephone", "phone", "keyboard", "laptop", "computer",
    "ceiling", "floor", "wall", "ceiling light", "ceiling fan",
    "radiator", "whiteboard", "blackboard", "dishwasher",
    "fireplace", "piano", "guitar", "vase", "bottle",
    "box", "basket", "pot", "pan", "kettle",
]

# ============================================================================
# Worker 脚本内容 - 使用 infer_video
# ============================================================================

WORKER_SCRIPT = '''
#!/usr/bin/env python3
"""GPU Worker for V7.5 Direction Test - 使用DA3多帧融合"""

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

@dataclass
class MindMapEntity:
    label: str
    instances: List[Dict] = field(default_factory=list)
    position_3d: np.ndarray = None
    size_3d: np.ndarray = None
    confidence: float = 0.0

EXTENDED_VOCABULARY = [
    "chair", "table", "sofa", "couch", "stove", "tv", "television",
    "bed", "cabinet", "shelf", "desk", "lamp", "refrigerator", "fridge",
    "sink", "toilet", "bathtub", "door", "window", "picture", "painting",
    "pillow", "cushion", "monitor", "backpack", "bag", "heater",
    "trash can", "trash bin", "mirror", "towel", "plant", "cup",
    "nightstand", "closet", "microwave", "printer", "washer", "dryer",
    "oven", "counter", "drawer", "curtain", "rug", "carpet", "clock",
    "fan", "air conditioner", "bookshelf", "armchair", "stool",
    "telephone", "phone", "keyboard", "laptop", "computer",
    "ceiling", "floor", "wall", "ceiling light", "ceiling fan",
    "radiator", "whiteboard", "blackboard", "dishwasher",
    "fireplace", "piano", "guitar", "vase", "bottle",
    "box", "basket", "pot", "pan", "kettle",
]

class MindMapBuilderV75:
    """心智地图构建器 V7.5 - 使用DA3多帧融合"""
    
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
        """构建心智地图 - 使用多帧融合"""
        self.load_models()
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return {}, 0
        
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        vocab = list(set((target_objects or []) + EXTENDED_VOCABULARY))
        prompt = " . ".join(vocab) + " ."
        
        # 收集所有帧
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        
        if not frames:
            return {}, 0
        
        H, W = frames[0].shape[:2]
        
        # ========================================
        # 关键改变: 使用 infer_video 多帧融合
        # ========================================
        try:
            depth_pred = self._depth_estimator.infer_video(
                frames, 
                normalize=False,  # 不归一化，保持相对深度
                apply_temporal_smoothing=True  # 启用时序平滑
            )
            depth_maps = depth_pred.depth_raw.cpu().numpy()  # (T, H, W)
            
            # 统一尺度校准：假设中值深度为2.5米
            median_depth = np.median(depth_maps)
            if median_depth > 0:
                scale = 2.5 / median_depth
                depth_maps = depth_maps * scale
        except Exception as e:
            logger.warning(f"infer_video 失败: {e}, 回退到逐帧")
            depth_maps = []
            for frame in frames:
                depth_tensor, _ = self._depth_estimator.infer_single(frame, normalize=False)
                depth_map = depth_tensor.squeeze().cpu().numpy()
                median = np.median(depth_map)
                if median > 0:
                    depth_map = depth_map * (2.5 / median)
                depth_maps.append(depth_map)
            depth_maps = np.stack(depth_maps)
        
        # 物体检测 + 3D位置估算
        all_detections = defaultdict(list)
        
        for fidx, frame in enumerate(frames):
            depth_map = depth_maps[fidx]
            if depth_map.shape != (H, W):
                depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)
            
            detections = self._labeler.detect(frame, prompt)
            
            for det in detections:
                label = det.label.lower().strip()
                if label.startswith('##'):
                    continue
                
                box = det.bbox_pixels
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                
                x_int = int(np.clip(cx, 0, W-1))
                y_int = int(np.clip(cy, 0, H-1))
                z = float(depth_map[y_int, x_int])
                
                x_3d = (cx - W/2) * z / self.focal_length
                y_3d = (cy - H/2) * z / self.focal_length
                
                all_detections[label].append({
                    'frame_idx': fidx,
                    'bbox': [float(b) for b in box],
                    'confidence': float(det.confidence),
                    'depth': z,
                    'x_3d': x_3d,
                    'y_3d': y_3d,
                    'z_3d': z,
                })
        
        # 聚合
        mind_map = {}
        for label, instances in all_detections.items():
            if not instances:
                continue
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


# 其余代码与V7.4相同...
CALIBRATION_OBJECTS = {
    'door': 2.0, 'chair': 0.85, 'table': 0.75, 'desk': 0.75,
    'bed': 0.5, 'sofa': 0.85, 'refrigerator': 1.7, 'toilet': 0.4, 'tv': 0.5,
}

class ScaleCalibrator:
    def calibrate(self, mind_map):
        @dataclass
        class Result:
            scale_factor: float = 1.0
            ref_object: str = None
        return Result()

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
    
    return f"{fb}-{lr}", {'front_proj': float(front), 'right_proj': float(right)}

def compute_direction_simple(ref_pos, target_pos):
    dx = target_pos[0] - ref_pos[0]
    return "right" if dx > 0 else "left"

def parse_direction_question(question: str):
    q = question.lower()
    
    m = re.search(r"standing by (?:the )?(\w+).*?facing (?:the )?(\w+).*?is (?:the )?(\w+) to my", q)
    if m:
        return {'format': 'hard', 'standing': m.group(1), 'facing': m.group(2), 'target': m.group(3)}
    
    m = re.search(r"direction of (?:the )?(\w+) from (?:the )?(\w+)", q)
    if m:
        return {'format': 'medium', 'target': m.group(1), 'ref': m.group(2)}
    
    m = re.search(r"is (?:the )?(\w+) to the.*?of (?:the )?(\w+)", q)
    if m:
        return {'format': 'easy', 'target': m.group(1), 'ref': m.group(2)}
    
    m = re.search(r"standing by (?:the )?(\w+).*?facing (?:the )?(\w+).*?is (?:the )?(\w+) to the", q)
    if m:
        return {'format': 'hard_lr', 'standing': m.group(1), 'facing': m.group(2), 'target': m.group(3)}
    
    return None

def find_object_in_mindmap(query, mind_map):
    query = query.lower().strip()
    if query in mind_map:
        return mind_map[query]
    for label, entity in mind_map.items():
        if query in label or label in query:
            return entity
    synonyms = {'couch': 'sofa', 'television': 'tv', 'fridge': 'refrigerator', 'phone': 'telephone'}
    if query in synonyms and synonyms[query] in mind_map:
        return mind_map[synonyms[query]]
    return None

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
        
        computed, debug = compute_direction_hard(standing_e.position_3d, facing_e.position_3d, target_e.position_3d)
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

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    with open(args.input) as f:
        samples = json.load(f)
    
    builder = MindMapBuilderV75(device='cuda', num_frames=32)
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
                mind_map, total_frames = builder.build_from_video(sample['video_path'])
                scene_cache[scene_name] = {'mind_map': mind_map}
            
            mind_map = scene_cache[scene_name]['mind_map']
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
    for dir_path in VIDEO_DIRS:
        video_path = os.path.join(dir_path, f"{scene_name}.mp4")
        if os.path.exists(video_path):
            return video_path
    return None


def load_direction_data() -> List[Dict]:
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
    parser = argparse.ArgumentParser(description='V7.5 方向问题测试 (DA3多帧融合 + 扩展词汇表)')
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default='outputs')
    args = parser.parse_args()
    
    samples = load_direction_data()
    
    if args.max_samples:
        samples = samples[:args.max_samples]
    
    num_gpus = min(args.num_gpus, len(samples))
    logger.info(f"使用 {num_gpus} 个 GPU 处理 {len(samples)} 样本")
    
    scene_samples = defaultdict(list)
    for s in samples:
        scene_samples[s['scene_name']].append(s)
    
    scenes = list(scene_samples.keys())
    gpu_assignments = [[] for _ in range(num_gpus)]
    for i, scene in enumerate(scenes):
        gpu_id = i % num_gpus
        gpu_assignments[gpu_id].extend(scene_samples[scene])
    
    worker_script_path = PROJECT_ROOT / 'tests' / '_worker_v75.py'
    with open(worker_script_path, 'w') as f:
        f.write(WORKER_SCRIPT)
    
    processes = []
    temp_files = []
    
    for gpu_id in range(num_gpus):
        if not gpu_assignments[gpu_id]:
            continue
        
        input_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(gpu_assignments[gpu_id], input_file)
        input_file.close()
        
        output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        output_file.close()
        
        temp_files.append((input_file.name, output_file.name))
        
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        env['HIP_VISIBLE_DEVICES'] = str(gpu_id)
        
        cmd = [sys.executable, str(worker_script_path), '--input', input_file.name, '--output', output_file.name]
        
        logger.info(f"启动 GPU {gpu_id} 进程: {len(gpu_assignments[gpu_id])} 样本")
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append((gpu_id, p))
    
    all_results = []
    for gpu_id, p in tqdm(processes, desc="等待GPU进程"):
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            logger.error(f"GPU {gpu_id} 进程失败: {stderr.decode()[:500]}")
            continue
        
        _, output_file = temp_files[gpu_id]
        try:
            with open(output_file) as f:
                results = json.load(f)
            all_results.extend(results)
            logger.info(f"GPU {gpu_id}: {len(results)} 结果")
        except Exception as e:
            logger.error(f"GPU {gpu_id} 结果读取失败: {e}")
    
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
    logger.info("V7.5 方向问题测试结果 (DA3多帧融合 + 扩展词汇表)")
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
        if extra.get('computed_direction') and extra['computed_direction'] != 'unknown':
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
    
    if total_samples > 0:
        logger.info(f"\n{'='*80}")
        logger.info(f"总计: {total_samples} 样本")
        logger.info(f"正确: {total_correct} ({100*total_correct/total_samples:.1f}%)")
        logger.info(f"{'='*80}")
    
    # 保存结果
    output_dir = Path(args.output_dir) / f"v75_direction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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


if __name__ == '__main__':
    main()
