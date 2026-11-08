---
name: colab-gpu-model-runner
description: >-
  General-purpose, fault-tolerant workflow for loading and running any Hugging
  Face or PyTorch generative model (audio, music, image, video, TTS, LLM, 3D) on
  Google Colab GPUs (L4, A100, T4) via the colab CLI. Use this skill whenever a
  task involves provisioning a Colab GPU, installing PyTorch/Diffusers/Transformers
  without CUDA version conflicts, fast model weight downloading, incremental
  artifact recovery, local SHA-256 verification, Compute Unit budget enforcement,
  and guaranteed automatic VM shutdown.
---

# Colab GPU Hugging Face Model Runner Skill

This skill provides a general-purpose, reusable architecture for deploying and running **any Hugging Face / PyTorch generative model** on Google Colab GPUs (`L4`, `A100`, or `T4`) via the `colab` CLI with maximum speed, zero CUDA version conflicts, strict Compute Unit budget protection, and guaranteed VM shutdown.

---

## 1. Six Golden Rules for Colab GPU Model Loading

### Rule 1: Isolated `uv venv` with Matched PyTorch + TorchAudio/TorchVision Wheels
- **Why**: Colab's system Python image pre-installs `torchaudio` / `torchvision` compiled against a specific minor CUDA version (e.g., `cu130`). Using `--system-site-packages` and installing `diffusers` or `transformers` often upgrades `torch` from PyPI (`cu132`), triggering:
  `RuntimeError: Detected that PyTorch and TorchAudio were compiled with different CUDA versions.`
- **Solution**: Always create a clean, isolated `uv venv --python 3.11` and install matched wheels from the official PyTorch index (`--index-url https://download.pytorch.org/whl/cu126`). On Colab's 10 Gbps network, `uv` resolves and installs the entire PyTorch + CUDA stack in **~15–21 seconds**.

### Rule 2: High-Speed Weight Download (`HF_XET_HIGH_PERFORMANCE=1` + `max_workers=8` + `allow_patterns`)
- In `huggingface-hub >= 1.30`, `HF_HUB_ENABLE_HF_TRANSFER` is deprecated in favor of **`HF_XET_HIGH_PERFORMANCE=1`** (powered by `hf-xet`, installed automatically with `huggingface-hub`).
- Always pass `max_workers=8` and explicit `allow_patterns=[...]` (or `ignore_patterns=['*.gguf', '*.bin']`) to `snapshot_download()` so you never waste time or disk space downloading redundant checkpoint formats.

### Rule 3: Zero-Leak Hugging Face Token Handling (`.hf_token` Auto-Shredding)
- Never hardcode tokens in scripts, prompts, or CLI command lines.
- Upload `~/.cache/huggingface/token` to a temporary remote file (`.hf_token`), set `chmod 0o600`, read it into the remote Python process's `env['HF_TOKEN']`, and **immediately `unlink()` the file from disk before running any setup or model code** (and again in `finally:`).

### Rule 4: GPU Hardware & VRAM Optimization (L4 / A100 / T4)
- **Verify Accelerator Early**: Run `nvidia-smi --query-gpu=name,memory.total` before installing packages to confirm the requested GPU (e.g., `L4` 23,034 MiB) was assigned.
- **Prevent Allocator Fragmentation**: Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the environment before importing `torch`.
- **Precision & Tensor Cores**:
  - On **L4 (Ada, CC 8.9)** and **A100 (Ampere, CC 8.0)**: Enable `torch.backends.cuda.matmul.allow_tf32 = True`, `torch.backends.cudnn.allow_tf32 = True`, and load weights in `torch.bfloat16`.
  - On **T4 (Turing, CC 7.5)**: Use `torch.float16` (T4 lacks native fast BF16).
- **Attention Kernels**: Rely on PyTorch 2.7's built-in `sdpa` (FlashAttention-2 / Memory-Efficient Attention). **Never** compile `flash-attn` from source on Colab (wastes 15–25 minutes of GPU Compute Units).
- **Multi-Component Pipelines**: If total model weights + activations approach ~80% of VRAM (e.g., >18 GiB on L4), enable stage-level/model-level CPU offload (`enable_model_cpu_offload()` or `ComponentsManager.enable_auto_cpu_offload(device='cuda')`).
- **Inter-Item Cleanup**: In batch generation loops, explicitly `del` outputs and run `gc.collect(); torch.cuda.empty_cache()` after saving each item.

### Rule 5: Resilient `colab` CLI Proxy Handling & Fail-Fast Polling
- **Port 8080 Warmup Delay & Retries**: Immediately after `colab new -s <SESSION> --gpu <GPU>` returns `Session READY`, the remote Jupyter proxy on Port 8080 may still be warming up for 5–10 seconds. Sleep 6 seconds after `colab new` and wrap all `colab exec` / `upload` / `download` calls with exponential backoff retries (`retries=4`).
- **Fail-Fast on Remote Error**: Poll `status.json` every 15 seconds. If `status.get('state') == 'failed'`, break out of the monitoring loop **immediately on the first check** to trigger VM shutdown without wasting idle Compute Units.

### Rule 6: Atomic Checkpointing, Multi-Layer Local Verification & Guaranteed Shutdown
- **Atomic Remote Packaging**: Write each generated output + its `.json` metadata sidecar (containing the remote **SHA-256** digest) into `packs/<name>.zip.tmp`, then atomically rename to `packs/<name>.zip`.
- **Multi-Layer Local Verification (`verify_pack`)**:
  1. **ZIP CRC Check**: `zipfile.ZipFile.testzip()` must pass.
  2. **SHA-256 Match**: Local file SHA-256 must match the remote SHA-256 recorded in the sidecar `.json`.
  3. **Format/Content Decode Check**: Decode/inspect the output locally (e.g., `soundfile.info` for audio, `PIL.Image.open().verify()` for images, `ffprobe` for video) to ensure it is non-empty and uncorrupted.
- **Guaranteed `colab stop` (`finally:` Block)**:
  - Only exit the normal loop once all expected items pass local verification (or upon fatal error / budget cap).
  - In `finally:`, terminate any local foreground client, run `colab stop -s <SESSION>` (with up to 3 retries), verify `No active sessions found on server.` via `colab sessions`, and write `shutdown.json` and `usage-after.txt` (`Usage rate: 0.00/hr`, `Active assignments: 0`).

---

## 2. Reusable Script Templates

Copy and adapt these three modular templates into your task's local output directory:

1. **[scripts/job_remote_base.py](./scripts/job_remote_base.py)**:
   Remote environment bootstrapper. Verifies GPU hardware, shreds `.hf_token` into memory, creates the isolated Python 3.11 `uv venv`, installs matched `cu126` PyTorch wheels + model packages, downloads Hugging Face checkpoints with `HF_XET_HIGH_PERFORMANCE=1` and `max_workers=8`, and runs `generate_remote.py`.
2. **[scripts/generate_remote_base.py](./scripts/generate_remote_base.py)**:
   Remote inference loop template. Configures `allow_tf32`, loads the Hugging Face model/pipeline in `bfloat16`, loops over `manifest.json`, performs atomic `.zip` checkpointing with SHA-256 sidecars, and runs `gc.collect() + torch.cuda.empty_cache()` between items.
3. **[scripts/supervise_base.py](./scripts/supervise_base.py)**:
   Local supervisor. Checks initial Compute Unit balance, provisions the Colab GPU with warmup retries, runs the remote job in the foreground, incrementally downloads and verifies each `.zip` pack, enforces `MAX_BATCH_SPEND`, and guarantees `colab stop` in `finally:`.
