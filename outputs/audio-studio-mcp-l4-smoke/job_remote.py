"""Remote Colab L4 environment bootstrapper for kanaworksai-hue/audio-studio-mcp."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path('/content/colab_job')
EXPECTED_GPU = 'L4'
HF_REPO_ID = 'stabilityai/stable-audio-3-medium'
HF_REVISION = 'main'
HF_ALLOW_PATTERNS = ['model_config.json', 'model.safetensors', 't5gemma-b-b-ul2/*']
UPSTREAM_COMMIT = '77e0398c26f7d22d801ec1bb036307e8b231f741'

EXTRA_PIP_PACKAGES = [
    'einops>=0.8.2',
    'einops-exts>=0.0.4',
    'packaging>=24',
    'safetensors>=0.7.0',
    'tqdm>=4.67',
    'huggingface-hub>=1.7.1',
    'transformers>=5.8.0',
    'soundfile>=0.12',
    'numpy>=1.24',
    'pydantic>=2',
    'mcp>=2.3,<3',
    'editables>=0.5',
    'sentencepiece>=0.2',
]

ROOT.mkdir(parents=True, exist_ok=True)
OUT = ROOT / 'results'
OUT.mkdir(parents=True, exist_ok=True)

env = dict(
    os.environ,
    PYTHONUNBUFFERED='1',
    HF_HUB_DISABLE_PROGRESS_BARS='1',
    HF_XET_HIGH_PERFORMANCE='1',
    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
    HF_HOME=str(ROOT / 'hf-cache'),
    TOKENIZERS_PARALLELISM='false',
)

# Read and immediately shred temporary Hugging Face token from disk
secret = ROOT / '.hf_token'
if secret.exists():
    env['HF_TOKEN'] = secret.read_text(encoding='utf-8').strip()
    secret.unlink()


def run(args, timeout=900, cwd=None):
    print('RUN ' + ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), env=env, cwd=cwd, check=True, timeout=timeout)


def status(state, **kw):
    (ROOT / 'status.json').write_text(
        json.dumps(dict(state=state, time=time.time(), **kw)), encoding='utf-8'
    )
    print('STATUS ' + state, flush=True)


try:
    timings = {}
    gpu = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], text=True
    ).strip()
    print('GPU ' + gpu, flush=True)
    if EXPECTED_GPU and EXPECTED_GPU not in gpu:
        raise RuntimeError(f'Requested {EXPECTED_GPU}; got {gpu}')
    (OUT / 'hardware.txt').write_text(gpu + '\n' + Path('/proc/meminfo').read_text(encoding='utf-8'), encoding='utf-8')

    status('cloning_repos')
    t0 = time.monotonic()
    repo_dir = ROOT / 'audio-studio-mcp'
    upstream_dir = repo_dir / 'vendor' / 'stable-audio-3'
    if not repo_dir.exists():
        run(['git', 'clone', '--depth', '1', 'https://github.com/kanaworksai-hue/audio-studio-mcp.git', repo_dir], 180)
    if not upstream_dir.exists():
        upstream_dir.parent.mkdir(parents=True, exist_ok=True)
        run(['git', 'clone', '--filter=blob:none', '--no-checkout', 'https://github.com/Stability-AI/stable-audio-3.git', upstream_dir], 180)
        run(['git', '-C', upstream_dir, 'fetch', '--quiet', 'origin', UPSTREAM_COMMIT], 180)
        run(['git', '-C', upstream_dir, '-c', 'advice.detachedHead=false', 'checkout', '--quiet', '--force', UPSTREAM_COMMIT], 180)
    timings['git_clone_seconds'] = round(time.monotonic() - t0, 2)

    status('installing_environment')
    t0 = time.monotonic()
    run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], 180)
    run(['uv', 'venv', '--python', '3.11', ROOT / 'model-venv'], 180)
    venv_py = ROOT / 'model-venv/bin/python'
    run([
        'uv', 'pip', 'install', '--python', venv_py,
        'torch==2.7.1', 'torchaudio==2.7.1',
        '--index-url', 'https://download.pytorch.org/whl/cu126',
    ], 700)
    run(['uv', 'pip', 'install', '--python', venv_py, *EXTRA_PIP_PACKAGES], 700)
    run(['uv', 'pip', 'install', '--python', venv_py, '--no-deps', '-e', repo_dir], 180)
    timings['uv_venv_install_seconds'] = round(time.monotonic() - t0, 2)

    status('downloading_weights')
    t0 = time.monotonic()
    download_cmd = (
        "from huggingface_hub import snapshot_download; "
        f"p = snapshot_download({HF_REPO_ID!r}, revision={HF_REVISION!r}, "
        f"max_workers=8, allow_patterns={HF_ALLOW_PATTERNS!r}); "
        "print('SNAPSHOT_PATH', p)"
    )
    run([venv_py, '-c', download_cmd], 1200)
    timings['weight_download_seconds'] = round(time.monotonic() - t0, 2)

    (ROOT / 'setup_timings.json').write_text(json.dumps(timings, indent=2), encoding='utf-8')
    print('TIMINGS ' + json.dumps(timings), flush=True)

    status('generating')
    run([venv_py, ROOT / 'generate_remote.py'], 3600)
    status('complete')
except Exception as e:
    (OUT / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
    traceback.print_exc()
    status('failed', error=str(e))
finally:
    env.pop('HF_TOKEN', None)
    if secret.exists():
        secret.unlink()
