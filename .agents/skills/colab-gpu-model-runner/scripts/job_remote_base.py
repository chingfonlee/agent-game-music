"""General-purpose remote Colab GPU environment bootstrapper and model downloader."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

# Configure these per project/task
ROOT = Path('/content/colab_job')
EXPECTED_GPU = 'L4'  # e.g., 'L4', 'A100', or 'T4'
HF_REPO_ID = 'ORG/MODEL_NAME'
HF_REVISION = 'main'
HF_ALLOW_PATTERNS = ['*.json', '*.safetensors', '*.model', '*.txt', 'LICENSE', 'README.md']
EXTRA_PIP_PACKAGES = [
    'transformers>=4.48.0',
    'diffusers>=0.32.0',
    'accelerate>=1.2.0',
    'safetensors>=0.4.5',
    'soundfile>=0.13.0',
    'sentencepiece>=0.2.0',
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
    gpu = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], text=True
    ).strip()
    print('GPU ' + gpu, flush=True)
    if EXPECTED_GPU and EXPECTED_GPU not in gpu:
        raise RuntimeError(f'Requested {EXPECTED_GPU}; got {gpu}')
    (OUT / 'hardware.txt').write_text(gpu + '\n' + Path('/proc/meminfo').read_text(encoding='utf-8'), encoding='utf-8')

    status('installing_environment')
    run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], 180)

    # Always use an isolated Python 3.11 uv venv with matched PyTorch + TorchAudio + TorchVision CUDA wheels
    run(['uv', 'venv', '--python', '3.11', ROOT / 'model-venv'], 180)
    venv_py = ROOT / 'model-venv/bin/python'
    run([
        'uv', 'pip', 'install', '--python', venv_py,
        'torch==2.7.1', 'torchaudio==2.7.1', 'torchvision==0.22.1',
        '--index-url', 'https://download.pytorch.org/whl/cu126',
    ], 700)
    run(['uv', 'pip', 'install', '--python', venv_py, *EXTRA_PIP_PACKAGES], 700)

    if HF_REPO_ID and HF_REPO_ID != 'ORG/MODEL_NAME':
        status('downloading_weights')
        download_cmd = (
            "from huggingface_hub import snapshot_download; "
            f"snapshot_download({HF_REPO_ID!r}, revision={HF_REVISION!r}, "
            f"max_workers=8, allow_patterns={HF_ALLOW_PATTERNS!r})"
        )
        run([venv_py, '-c', download_cmd], 1200)

    status('generating')
    run([venv_py, ROOT / 'generate_remote.py'], 7200)
    status('complete')
except Exception as e:
    (OUT / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
    traceback.print_exc()
    status('failed', error=str(e))
finally:
    env.pop('HF_TOKEN', None)
    if secret.exists():
        secret.unlink()
