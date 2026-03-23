# Spatial Intelligence MindCube

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **端到端空间智能推理系统**：结合 3D 几何重建 (DA3) 与视觉语言模型 (VLM)，构建可进化的 3D 心智地图 (Mind Map) 实现复杂空间推理任务。

## 📊 核心能力

| 任务类型 | 能力描述 | V21 表现 |
|---------|---------|---------|
| **Object Counting** | 多物体计数 | 94.1% |
| **Object Size Estimation** | 3D 尺寸估计 | 95.5% |
| **Room Size Estimation** | 房间面积/体积估计 | 97.6% |
| **Absolute Distance** | 物体到相机距离 | 80.3% |
| **Relative Distance** | 物体间距离比较 | 50.4% |
| **Direction (Easy/Hard)** | 相对方位判断 | 57.6% / 42.1% |
| **Appearance Order** | 首次出现顺序 | 61.8% |
| **Route Planning** | 路径规划 | 32.0% |
| **Overall** | VSI-Bench 综合 | **70.4%** |

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Spatial Intelligence MindCube           │
├─────────────────────────────────────────────────────────────┤
│  Input: 室内全景视频 (VSI-Bench / Cambrian-W / 自定义)       │
├─────────────────────────────────────────────────────────────┤
│  Perception Layer          │  Reasoning Layer              │
│  ─────────────────         │  ───────────────              │
│  DA3 Depth Anything 3      │  Manager (VLM Decision)        │
│  ├─ 深度估计               │  ├─ 工具选择                   │
│  ├─ 3D 点云重建            │  ├─ 多轮推理                   │
│  └─ 相机位姿估计           │  └─ 闭环验证                   │
│                            │                                │
│  Grid Mind Map (256³)      │  Tools                         │
│  ├─ 稀疏体素表示           │  ├─ CODER (几何计算)           │
│  ├─ 实体符号化             │  ├─ CRITIC (质检)              │
│  └─ 时空关联               │  ├─ EVOLUTOR (自修复)          │
│                            │  └─ GRID_SLICE (可视化)        │
│  Self-Evolution            │                                │
│  ├─ Auto-ADD               │  Self-Verification             │
│  ├─ Self-Verify            │  └─ VL Pairwise Verification   │
│  └─ Confidence-Driven      │                                │
├─────────────────────────────────────────────────────────────┤
│  Output: 空间问答答案 (选择/数值)                            │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 环境配置

```bash
# 1. 克隆仓库
git clone https://github.com/Grady10086/Spatial-Intelligence-MindCube.git
cd Spatial-Intelligence-MindCube

# 2. 创建环境
conda create -n mindcube python=3.10
conda activate mindcube

# 3. 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate deepspeed
pip install -r requirements.txt

# 4. 配置环境变量
export HF_HOME=/path/to/hf_cache
export http_proxy=http://your-proxy:port  # 如需代理
```

### 数据准备

```bash
# 下载 VSI-Bench 数据集
huggingface-cli download --repo-type dataset nyu-visionx/VSI-Bench --local-dir ./data/vsibench

# 或准备自定义视频数据
# 格式: video.mp4 + questions.json
```

### 运行评测

```bash
# 完整评测 (8 GPU)
python scripts/run_mindcube_vsibench.py \
    --model_path /path/to/Qwen3-VL-8B-Instruct \
    --data_root ./data/vsibench \
    --num_gpus 8 \
    --fps 2 \
    --grid_resolution 256

# 快速测试 (单 GPU, 10 样本)
python scripts/run_mindcube_vsibench.py \
    --model_path /path/to/Qwen3-VL-8B-Instruct \
    --data_root ./data/vsibench \
    --num_gpus 1 \
    --max_samples 10 \
    --fps 2
```

## 📁 代码结构

```
Spatial-Intelligence-MindCube/
├── core/                          # 核心模块
│   ├── grid256_mind_map.py       # Grid Mind Map (256³体素)
│   ├── da3_perception.py         # DA3 3D重建封装
│   ├── voxel_map.py              # 稀疏体素哈希表
│   ├── visibility.py             # 射线投射可见性
│   └── semantic_labeler.py       # 语义标注 (CLIP/GroundingDINO)
│
├── scripts/                       # 脚本
│   ├── grid64_agentic_pipeline_v21.py  # V21 主Pipeline
│   ├── run_mindcube_vsibench.py  # VSI-Bench 评测入口
│   ├── merge_results.py          # 多GPU结果合并
│   └── calculate_metrics.py      # 指标计算
│
├── tests/                         # 单元测试
│   ├── test_grid_map.py
│   ├── test_coder.py
│   └── test_evolution.py
│
├── docs/                          # 文档
│   ├── DA3_Usage_Guide.md
│   ├── Architecture_Evolution.md
│   └── Ablation_Study.md
│
├── data/                          # 数据 (gitignore)
├── outputs/                       # 输出结果 (gitignore)
└── requirements.txt
```

## 🧠 核心概念

### 1. Grid Mind Map (3D心智地图)

基于 DA3 重建的 3D 场景，构建稀疏体素表示：

```python
from core.grid256_mind_map import Grid256MindMap

# 从 DA3 输出构建
mind_map = Grid256MindMap.from_da3_output(
    glb_path="scene.glb",
    resolution=256,  # 256³ 体素
    voxel_size=0.1   # 10cm 分辨率
)

# 查询实体
entities = mind_map.query_entities(
    position=(x, y, z),
    radius=2.0
)

# 生成 VL Prompt
prompt = mind_map.to_prompt(format='yaml')
```

### 2. Agentic Pipeline (V21)

```python
# Pipeline 流程
question → Manager(选择工具) → 执行 → Self-Verify → Decide
                ↓
        ┌───────┴───────┐
        ↓               ↓
    CODER计算      CRITIC质检
    (几何推理)      (一致性检查)
        ↓               ↓
        └───────┬───────┘
                ↓
           EVOLUTOR修复
           (ADD/DELETE/FILTER)
                ↓
           闭环验证 (最多3轮)
```

### 3. Self-Evolution 机制

- **Auto-ADD**: CODER 返回 "not found" 时自动补全缺失实体
- **Self-Verify**: CODER + VL 结果矛盾时触发修复
- **Confidence-Driven**: Manager 自评 confidence 低时请求更多工具

## 📈 实验结果

### VSI-Bench (5130 样本)

| 版本 | 基座模型 | Grid | FPS | Overall | MCA | NA |
|------|---------|------|-----|---------|-----|-----|
| V5 | Qwen3-VL-8B | 64³ | 16 | 64.1% | - | - |
| V6 | Qwen3-VL-8B | 64³ | 32 | 65.5% | - | - |
| V10 | Qwen3-VL-8B | 256³ | 32 | 67.6% | - | - |
| **V21** | **Qwen3-VL-8B** | **256³** | **2** | **70.4%** | **51.1%** | **88.5%** |

### 关键改进

| 改进点 | 效果 |
|--------|------|
| Grid 64³ → 256³ | direction_hard +8.9%, rel_distance +4.8% |
| Self-Verification | 减少 VL 推翻 CODER 正确答案 |
| FPS=2 对齐官方 | 信息量提升，与官方结果可比 |

## 💡 使用示例

### 自定义视频推理

```python
from scripts.grid64_agentic_pipeline_v21 import MindCubePipeline

# 初始化
pipeline = MindCubePipeline(
    vl_model_path="Qwen/Qwen3-VL-8B-Instruct",
    grid_resolution=256,
    fps=2
)

# 处理视频
result = pipeline.process_video(
    video_path="path/to/video.mp4",
    question="How many chairs are in the room?",
    options=["A. 1", "B. 2", "C. 3", "D. 4"]  # 可选
)

print(f"Answer: {result['answer']}")
print(f"Reasoning: {result['reasoning']}")
print(f"Confidence: {result['confidence']}")
```

### 批量评测

```bash
# 多 GPU 并行
for gpu_id in {0..7}; do
    python scripts/run_mindcube_vsibench.py \
        --shard_id $gpu_id \
        --num_shards 8 \
        --gpu_id $gpu_id &
done
wait

# 合并结果
python scripts/merge_results.py \
    --input_dir outputs/vsibench_gpu* \
    --output outputs/vsibench_merged.json
```

## 🔧 高级配置

### ROCm (AMD GPU) 支持

```bash
# 环境变量
export MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK=1
export MIOPEN_LOG_LEVEL=1

# 替换 Flash Attention
pip install aiter
# 修改 transformers/modeling_flash_attention_utils.py
# from aiter import flash_attn_varlen_func as flash_varlen_fn
```

### DeepSpeed 训练

```bash
deepspeed --num_gpus=8 scripts/train_qwen3vl_vsi590k.py \
    --deepspeed scripts/ds_config_zero2.json \
    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
    --dataset_path data/vsi590k_subset
```

## 📚 相关资源

- **论文**: [VSI-Bench: Benchmarking Video Spatial Intelligence](https://arxiv.org/abs/2406.08691)
- **DA3**: [Depth Anything V3](https://github.com/DepthAnything/Depth-Anything-V3)
- **VLM**: [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)

## 🤝 贡献

欢迎提交 Issue 和 PR！

## 📄 许可证

MIT License

## 🙏 致谢

- VSI-Bench 数据集: NYU VisionX
- DA3: Depth Anything Team
- Qwen3-VL: Alibaba Cloud
