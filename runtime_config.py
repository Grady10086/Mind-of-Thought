from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_ROOT / 'config'
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'

_VSIBENCH_SPLITS = ('arkitscenes', 'scannet', 'scannetpp')
_GDINO_DEFAULT_MODEL = 'IDEA-Research/grounding-dino-base'
_LOADED_ENV = False


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[len('export '):]
        key, sep, value = line.partition('=')
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_local_env() -> None:
    global _LOADED_ENV
    if _LOADED_ENV:
        return
    _LOADED_ENV = True
    _load_env_file(CONFIG_DIR / 'local.env')


def ensure_runtime_env() -> None:
    load_local_env()
    hf_home = os.environ.get('HF_HOME') or os.environ.get('MOT_HF_HOME')
    if not hf_home:
        hf_home = str(PROJECT_ROOT / '.cache' / 'huggingface')
    os.environ.setdefault('HF_HOME', hf_home)

    hf_endpoint = os.environ.get('MOT_HF_ENDPOINT')
    if hf_endpoint:
        os.environ.setdefault('HF_ENDPOINT', hf_endpoint)

    if os.environ.get('MOT_ENABLE_ROCM_WORKAROUND', '1') == '1':
        os.environ.setdefault('MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK', '1')


def _split_paths(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    parts: List[str] = []
    for chunk in raw.replace('\n', os.pathsep).replace(',', os.pathsep).split(os.pathsep):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return parts


def _expand_video_candidate(candidate: Path) -> List[str]:
    if not candidate.exists():
        return []
    if candidate.is_dir():
        split_dirs = [candidate / split for split in _VSIBENCH_SPLITS if (candidate / split).is_dir()]
        if split_dirs:
            return [str(path.resolve()) for path in split_dirs]
        if any(candidate.glob('*.mp4')):
            return [str(candidate.resolve())]
    return []


def get_video_dirs(extra_dirs: Optional[Iterable[str]] = None) -> List[str]:
    ensure_runtime_env()
    candidates: List[str] = []
    if extra_dirs:
        candidates.extend(str(Path(entry).expanduser()) for entry in extra_dirs if entry)
    candidates.extend(_split_paths(os.environ.get('MOT_VIDEO_DIRS')))
    candidates.extend(_split_paths(os.environ.get('VIDEO_DIRS')))
    for local_root in (
        DATA_DIR / 'videos',
        DATA_DIR / 'vsibench',
        PROJECT_ROOT / 'videos',
    ):
        candidates.extend(_expand_video_candidate(local_root))

    resolved: List[str] = []
    seen = set()
    for entry in candidates:
        for expanded in _expand_video_candidate(Path(entry).expanduser()):
            if expanded not in seen:
                seen.add(expanded)
                resolved.append(expanded)
    return resolved


def resolve_eval_manifest(path: Optional[str] = None) -> Path:
    ensure_runtime_env()
    candidates: List[Path] = []
    if path:
        candidates.append(Path(path).expanduser())
    env_path = os.environ.get('MOT_INPUT_RESULTS')
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(DATA_DIR / 'eval_samples.json')
    candidates.extend(sorted(DATA_DIR.glob('eval_samples*.json')))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        'Could not find an evaluation manifest. Pass --input_results, set MOT_INPUT_RESULTS, '
        'or place data/eval_samples.json in the repository.'
    )


def resolve_vl_model_path(model_path: Optional[str] = None) -> str:
    ensure_runtime_env()
    candidates: List[str] = []
    if model_path:
        candidates.append(model_path)
    env_model = os.environ.get('MOT_VL_MODEL')
    if env_model:
        candidates.append(env_model)

    hf_home = Path(os.environ['HF_HOME']).expanduser()
    candidates.append(str(hf_home / 'Qwen' / 'Qwen3-VL-8B-Instruct'))
    candidates.append(str(PROJECT_ROOT / 'models' / 'Qwen' / 'Qwen3-VL-8B-Instruct'))

    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path.resolve())
        if '/' in candidate and not candidate.startswith('/') and len(candidate.split('/')) == 2:
            return candidate

    raise FileNotFoundError(
        'Could not resolve a Qwen3-VL model path. Set MOT_VL_MODEL, pass --vl-model, '
        'or place the model under HF_HOME/Qwen/Qwen3-VL-8B-Instruct.'
    )


def _looks_like_repo_id(candidate: str) -> bool:
    if not candidate or candidate.startswith(('/', './', '../', '~')):
        return False
    parts = [part for part in candidate.split('/') if part]
    return len(parts) == 2


def _has_model_weights(path: Path) -> bool:
    return any(
        (path / filename).exists()
        for filename in (
            'model.safetensors',
            'model.safetensors.index.json',
            'pytorch_model.bin',
            'pytorch_model.bin.index.json',
        )
    )


def _iter_snapshot_candidates(root: Path) -> List[Path]:
    candidates: List[Path] = []
    if not root.exists():
        return candidates
    if root.is_dir() and (root / 'config.json').exists() and _has_model_weights(root):
        candidates.append(root.resolve())
    snapshots_dir = root / 'snapshots'
    if snapshots_dir.is_dir():
        for snapshot in sorted(snapshots_dir.iterdir()):
            if snapshot.is_dir() and (snapshot / 'config.json').exists() and _has_model_weights(snapshot):
                candidates.append(snapshot.resolve())
    return candidates


def resolve_grounding_dino_model(model_ref: Optional[str] = None) -> str:
    ensure_runtime_env()
    explicit_candidates: List[str] = []
    if model_ref:
        explicit_candidates.append(model_ref)
    for env_key in ('MOT_GDINO_MODEL', 'MOT_GROUNDING_DINO_MODEL', 'GROUNDING_DINO_MODEL'):
        value = os.environ.get(env_key)
        if value:
            explicit_candidates.append(value)

    repo_fallback: Optional[str] = None
    for candidate in explicit_candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path.resolve())
        if _looks_like_repo_id(candidate):
            repo_fallback = candidate

    repo_name = repo_fallback or _GDINO_DEFAULT_MODEL
    hf_home = Path(os.environ['HF_HOME']).expanduser()
    model_key = f"models--{repo_name.replace('/', '--')}"
    local_roots = [
        hf_home / model_key,
        hf_home / 'hub' / model_key,
        PROJECT_ROOT / 'models' / Path(repo_name).name,
        PROJECT_ROOT / 'models' / repo_name,
    ]
    for root in local_roots:
        snapshots = _iter_snapshot_candidates(root)
        if snapshots:
            return str(snapshots[0])

    return repo_name


def resolve_da3_src_path() -> Optional[Path]:
    ensure_runtime_env()
    candidates: List[Path] = []
    env_src = os.environ.get('MOT_DA3_SRC')
    if env_src:
        candidates.append(Path(env_src).expanduser())
    env_root = os.environ.get('MOT_DA3_ROOT')
    if env_root:
        candidates.append(Path(env_root).expanduser() / 'src')
    candidates.extend([
        PROJECT_ROOT / 'third_party' / 'Depth-Anything-3' / 'src',
        PROJECT_ROOT.parent / 'Depth-Anything-3' / 'src',
        PROJECT_ROOT.parent / 'projects' / 'Depth-Anything-3' / 'src',
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None
