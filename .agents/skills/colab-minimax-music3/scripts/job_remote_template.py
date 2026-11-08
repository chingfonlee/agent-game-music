"""Remote Colab L4 setup and execution for MiniMax-Music3 MP3 generation (v2.1 Verified)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path('/content/funky_house_mp3')
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
    if 'L4' not in gpu:
        raise RuntimeError(f'Requested L4; got {gpu}')
    (OUT / 'hardware.txt').write_text(gpu + '\n' + Path('/proc/meminfo').read_text(encoding='utf-8'), encoding='utf-8')

    status('installing_minimax')
    run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], 180)

    # Always use an isolated Python 3.11 uv venv with matched torch==2.7.1+cu126 and torchaudio==2.7.1+cu126
    # (takes ~15s on Colab 10Gbps and avoids system CUDA 13.0/13.2 mismatch between torch and torchaudio).
    run(['uv', 'venv', '--python', '3.11', ROOT / 'music-venv'], 180)
    music_py = ROOT / 'music-venv/bin/python'
    run([
        'uv', 'pip', 'install', '--python', music_py,
        'torch==2.7.1', 'torchaudio==2.7.1',
        '--index-url', 'https://download.pytorch.org/whl/cu126',
    ], 700)
    run([
        'uv', 'pip', 'install', '--python', music_py,
        'git+https://github.com/huggingface/diffusers@dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d',
        'transformers==5.22.2',
        'accelerate==1.15.0',
        'soundfile==0.14.0',
        'numpy==2.4.6',
        'sentencepiece==0.2.3',
        'hf-transfer>=0.1.8',
    ], 700)

    # Fast parallel download with hf_transfer + max_workers=8
    download_cmd = (
        "from huggingface_hub import snapshot_download; "
        "snapshot_download('MiniMaxAI/MiniMax-Music3', "
        "revision='1b525ccb685ae809f33027c51c351eeb5833cf72', "
        "max_workers=8, "
        "allow_patterns=['modular_model_index.json','condition_encoder/*','language_model/*',"
        "'rvq_depth_decoder/*','scheduler/*','tokenizer/*','transformer/*','vocoder/*','LICENSE','README.md'])"
    )
    run([music_py, '-c', download_cmd], 1200)

    status('generating_minimax')
    run([music_py, ROOT / 'generate_remote.py'], 7200)
    status('complete')
except Exception as e:
    (OUT / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
    traceback.print_exc()
    status('failed', error=str(e))
finally:
    env.pop('HF_TOKEN', None)
    if secret.exists():
        secret.unlink()

