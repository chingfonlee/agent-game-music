"""General-purpose remote inference template with atomic ZIP checkpointing and VRAM cleanup."""
import gc
import hashlib
import json
from pathlib import Path
import time
import zipfile

import torch

# Enable Ada/Ampere Tensor Core TF32 acceleration
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ROOT = Path('/content/colab_job')
OUT = ROOT / 'results'
PACKS = ROOT / 'packs'
OUT.mkdir(parents=True, exist_ok=True)
PACKS.mkdir(parents=True, exist_ok=True)

ITEMS = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))


def update_progress(current: str):
    entries = [{'name': f.name, 'size': f.stat().st_size} for f in sorted(PACKS.glob('*.zip'))]
    tmp = ROOT / 'progress.tmp.json'
    tmp.write_text(json.dumps({
        'current': current,
        'packs': entries,
        'time': time.time(),
    }), encoding='utf-8')
    tmp.replace(ROOT / 'progress.json')


def save_artifact_pack(item: dict, output_file: Path, wall_seconds: float, extra_meta: dict | None = None):
    """Package output file + SHA-256 metadata sidecar into an atomic ZIP archive."""
    pack_name = f"{item['id']}.zip"
    file_sha256 = hashlib.sha256(output_file.read_bytes()).hexdigest()
    meta = {
        'item': item,
        'pack_name': pack_name,
        'file_size_bytes': output_file.stat().st_size,
        'generation_wall_seconds': round(wall_seconds, 2),
        'gpu': torch.cuda.get_device_name(),
        'peak_gpu_allocated_bytes': int(torch.cuda.max_memory_allocated()),
        'peak_gpu_reserved_bytes': int(torch.cuda.max_memory_reserved()),
        'files': {
            output_file.name: file_sha256,
        },
        **(extra_meta or {}),
    }
    sidecar = output_file.with_suffix(output_file.suffix + '.json')
    sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    temp_zip = PACKS / (pack_name + '.tmp')
    with zipfile.ZipFile(temp_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for f in (output_file, sidecar):
            z.write(f, f.relative_to(OUT))
    temp_zip.replace(PACKS / pack_name)

    update_progress(f"saved {item['id']}")
    print(
        f"SAVED {item['id']} size={meta['file_size_bytes']/1024/1024:.2f}MiB "
        f"generation={wall_seconds:.1f}s peak={meta['peak_gpu_allocated_bytes']/1024**3:.2f}GiB",
        flush=True,
    )


def main():
    # TODO: Load your Hugging Face pipeline / model here (e.g. in dtype=torch.bfloat16 with CPU offload if >18GiB)
    pipe = None

    for item in ITEMS:
        if (ROOT / 'STOP').exists():
            return
        pack_name = f"{item['id']}.zip"
        if (PACKS / pack_name).exists():
            continue

        update_progress(f"generating {item['id']}")
        print(f"GENERATING {item['id']}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()

        # TODO: Run model inference and write output file to OUT / f"{item['id']}.ext"
        output_file = OUT / f"{item['id']}.bin"
        elapsed = time.monotonic() - t0

        save_artifact_pack(item, output_file, elapsed)

        # Prevent VRAM fragmentation across batch items
        gc.collect()
        torch.cuda.empty_cache()

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    print('MODEL_DONE', flush=True)


if __name__ == '__main__':
    main()
