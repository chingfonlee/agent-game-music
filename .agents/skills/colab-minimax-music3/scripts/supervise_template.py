"""Local L4 supervisor for 2 Funky House MP3 tracks: strict 10-unit budget, 4-layer MP3 verification, guaranteed Colab stop."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import zipfile

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
SESSION = 'funky-house-b-mp3'
REMOTE = '/content/funky_house_mp3/'
MAX_BATCH_SPEND = 8.5  # Strict cap under user's 10-unit budget (expected spend: ~1.3 units)
EXPECTED_TRACKS = 2

env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
started = False
foreground_client = None
downloaded = set()
verified_details = {}


def call(args, timeout=90, check=True, retries=4):
    last_err = ''
    for attempt in range(retries):
        try:
            p = subprocess.run(
                ['colab', *args],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=timeout,
                env=env,
            )
            if not p.returncode:
                return p.returncode, p.stdout + p.stderr
            last_err = (p.stdout + p.stderr)[-1600:]
            if not check:
                return p.returncode, p.stdout + p.stderr
        except subprocess.TimeoutExpired as e:
            last_err = f'TimeoutExpired: {e}'
        if attempt < retries - 1:
            time.sleep(5 * (attempt + 1))
    if check:
        raise RuntimeError(last_err)
    return 1, last_err


def usage():
    _, s = call(['usage'])
    m = re.search(r'Current balance:\s*([\d,.]+)', s)
    if not m:
        raise RuntimeError('Cannot read account balance: ' + s)
    return float(m.group(1).replace(',', '')), s


def get(remote, local, timeout=90, optional=False):
    code, _ = call(
        ['download', '-s', SESSION, REMOTE + remote, str(ROOT / local)],
        timeout,
        check=not optional,
        retries=1 if optional else 3,
    )
    return code == 0


def verify_pack(name: str):
    """4-layer verification of downloaded MP3 pack before marking track as complete."""
    archive = ROOT / 'packs' / name
    if not archive.exists() or archive.stat().st_size < 1_000_000:
        raise ValueError(f'Archive missing or too small: {archive}')

    # Layer 1: ZIP CRC integrity & safe path extraction
    with zipfile.ZipFile(archive) as z:
        bad_crc = z.testzip()
        if bad_crc is not None:
            raise ValueError(f'Candidate ZIP CRC failed on {bad_crc}')
        for entry in z.namelist():
            p = Path(entry)
            if p.is_absolute() or '..' in p.parts or p.parts[0] != 'minimax-music-3':
                raise ValueError(f'Unexpected archive entry path: {entry}')
            if p.suffix.lower() == '.wav':
                raise ValueError(f'WAV file forbidden in archive: {entry}')
        z.extractall(ROOT / 'results')

    # Layer 2: Sidecar metadata & SHA-256 cryptographic hash match
    metas = list((ROOT / 'results').glob('minimax-music-3/*.mp3.json'))
    selected = [p for p in metas if json.loads(p.read_text(encoding='utf-8')).get('pack_name') == name]
    if len(selected) != 1:
        raise ValueError(f'Candidate metadata not uniquely identified for {name}')
    meta_path = selected[0]
    meta = json.loads(meta_path.read_text(encoding='utf-8'))

    for filename, expected_sha256 in meta['files'].items():
        f = meta_path.parent / filename
        if not f.exists():
            raise ValueError(f'Extracted MP3 file missing: {f}')
        actual_sha256 = hashlib.sha256(f.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(f'SHA-256 mismatch for {filename}: {actual_sha256} != {expected_sha256}')

    # Layer 3: Full MP3 stream decode, stereo/duration/finite/RMS energy verification
    mp3_path = meta_path.with_suffix('')  # strips .json -> leaves .mp3
    if mp3_path.suffix.lower() != '.mp3' or mp3_path.stat().st_size < 1_500_000:
        raise ValueError(f'Invalid MP3 file or size < 1.5MB: {mp3_path}')

    info = sf.info(str(mp3_path))
    if info.format != 'MP3' or info.channels != 2 or info.samplerate != 44100:
        raise ValueError(f'Unexpected MP3 stream header: {info}')
    if not (120.0 <= info.duration <= 220.0):
        raise ValueError(f'MP3 duration out of bounds ({info.duration:.2f}s)')

    decoded, _ = sf.read(str(mp3_path), dtype='float32', always_2d=True)
    if not np.isfinite(decoded).all():
        raise ValueError('Decoded MP3 contains non-finite samples')
    rms = float(np.sqrt((decoded ** 2).mean()))
    peak = float(np.abs(decoded).max())
    if rms < 0.01 or peak < 0.1:
        raise ValueError(f'Decoded MP3 energy too low (rms={rms:.4f}, peak={peak:.4f})')

    # Layer 4: Record verified state locally
    downloaded.add(name)
    verified_details[name] = {
        'mp3_path': str(mp3_path),
        'file_size_bytes': mp3_path.stat().st_size,
        'sha256': hashlib.sha256(mp3_path.read_bytes()).hexdigest(),
        'format': info.format,
        'subtype': info.subtype,
        'samplerate': info.samplerate,
        'channels': info.channels,
        'duration_seconds': round(float(info.duration), 2),
        'decoded_rms': round(rms, 4),
        'decoded_peak': round(peak, 4),
        'bpm': meta['item']['bpm'],
        'label': meta['item']['label'],
        'seed': meta['seed'],
        'generation_wall_seconds': meta['generation_wall_seconds'],
    }
    (ROOT / 'downloaded.json').write_text(json.dumps(sorted(downloaded), indent=2), encoding='utf-8')
    (ROOT / 'verification.json').write_text(
        json.dumps(verified_details, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(
        f"LOCAL_MP3_VERIFIED {len(downloaded)}/{EXPECTED_TRACKS} {mp3_path.name} "
        f"duration={info.duration:.1f}s size={mp3_path.stat().st_size/1024/1024:.2f}MiB "
        f"sha256={verified_details[name]['sha256'][:12]}... rms={rms:.4f}",
        flush=True,
    )


def recover_packs():
    if not get('progress.json', 'progress.json', 60, optional=True):
        return
    progress = json.loads((ROOT / 'progress.json').read_text(encoding='utf-8'))
    for item in progress.get('packs', []):
        name = item['name']
        if name in downloaded:
            continue
        if '/' in name or '\\' in name or not name.endswith('.zip'):
            raise ValueError(f'Unexpected pack name: {name}')
        if get('packs/' + name, 'packs/' + name, 180):
            verify_pack(name)


try:
    (ROOT / 'packs').mkdir(exist_ok=True)
    (ROOT / 'results').mkdir(exist_ok=True)
    if (ROOT / 'downloaded.json').exists():
        downloaded = set(json.loads((ROOT / 'downloaded.json').read_text(encoding='utf-8')))

    initial_balance, before_text = usage()
    (ROOT / 'usage-before.txt').write_text(before_text, encoding='utf-8')
    print(f'INITIAL_BALANCE {initial_balance:.2f} compute units (budget cap: {MAX_BATCH_SPEND:.1f} units)', flush=True)

    started = True
    print(call(['new', '-s', SESSION, '--gpu', 'L4'], 180)[1], flush=True)
    time.sleep(6)

    init_script = ROOT / 'initialize_remote.py'
    init_script.write_text(
        "from pathlib import Path\nimport subprocess\n"
        "r=Path('/content/funky_house_mp3'); r.mkdir(parents=True, exist_ok=True)\n"
        "s=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True)\n"
        "print(s)\nassert 'L4' in s\n",
        encoding='utf-8',
    )
    print(call(['exec', '-s', SESSION, '-f', str(init_script), '--timeout', '40'], 70)[1], flush=True)

    for name in ('music_manifest.json', 'generate_remote.py', 'job_remote.py'):
        call(['upload', '-s', SESSION, str(ROOT / name), REMOTE + name], 120)

    token_path = Path.home() / '.cache' / 'huggingface' / 'token'
    token = (os.environ.get('HF_TOKEN') or (token_path.read_text(encoding='utf-8').strip() if token_path.exists() else '')).strip()
    if not token:
        raise RuntimeError('Hugging Face login token unavailable on local machine')
    fd, secret = tempfile.mkstemp(prefix='funky-hf-', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(token)
        call(['upload', '-s', SESSION, secret, REMOTE + '.hf_token'], 120)
    finally:
        if os.path.exists(secret):
            os.remove(secret)
        token = None

    foreground = ROOT / 'run_foreground.py'
    foreground.write_text(
        "from pathlib import Path\nimport os,subprocess,sys\n"
        "r=Path('/content/funky_house_mp3');os.chmod(r/'.hf_token',0o600)\n"
        "f=open(r/'job.log','ab',buffering=0)\n"
        "p=subprocess.Popen([sys.executable,'-u',str(r/'job_remote.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)\n"
        "(r/'job.pid').write_text(str(p.pid));print('OWNED_JOB_PID',p.pid,flush=True)\n"
        "try:code=p.wait(timeout=7200)\n"
        "except subprocess.TimeoutExpired:p.terminate();raise\n"
        "print('JOB_RETURN',code,flush=True)\n",
        encoding='utf-8',
    )
    clientlog = open(ROOT / 'foreground-client.log', 'ab', buffering=0)
    foreground_client = subprocess.Popen(
        ['colab', 'exec', '-s', SESSION, '-f', str(foreground), '--timeout', '7300'],
        stdout=clientlog,
        stderr=subprocess.STDOUT,
        env=env,
    )
    (ROOT / 'foreground-client.pid').write_text(str(foreground_client.pid), encoding='utf-8')
    print(f'Foreground Colab job client PID {foreground_client.pid}', flush=True)

    t0 = time.monotonic()
    last_usage = 0.0
    last_brief = ''
    failures = 0
    failed_at = None

    while time.monotonic() - t0 < 7200:
        if (ROOT / 'STOP_REQUEST').exists():
            print('USER_STOP_REQUEST', flush=True)
            break
        try:
            recover_packs()
            if len(downloaded) >= EXPECTED_TRACKS:
                print('ALL_2_MP3_TRACKS_DOWNLOADED_AND_VERIFIED', flush=True)
                break

            if get('status.json', 'remote-status.json', 60, optional=True):
                status = json.loads((ROOT / 'remote-status.json').read_text(encoding='utf-8'))
                if get('job.log', 'live-job.log', 75, optional=True):
                    s = (ROOT / 'live-job.log').read_text(encoding='utf-8', errors='replace')
                    brief = '\n'.join(
                        x for x in s.splitlines()
                        if x.startswith(('STATUS', 'GENERATING', 'SAVED', 'MODEL_DONE', 'RuntimeError', 'Traceback'))
                    )
                    brief = '\n'.join(brief.splitlines()[-4:])
                    if brief != last_brief:
                        print(json.dumps(status) + '\n' + brief, flush=True)
                        last_brief = brief

                if status.get('state') == 'complete':
                    recover_packs()
                    if len(downloaded) >= EXPECTED_TRACKS:
                        print('ALL_2_MP3_TRACKS_DOWNLOADED_AND_VERIFIED', flush=True)
                        break
                    raise RuntimeError(f"Remote job completed but only {len(downloaded)}/{EXPECTED_TRACKS} MP3s verified")

                if status.get('state') == 'failed':
                    print(f"REMOTE_JOB_FAILED: {status.get('error')}", flush=True)
                    break
            failures = 0
        except Exception as e:
            failures += 1
            print('MONITOR_ERROR ' + str(e)[-600:], flush=True)
            if failures >= 3:
                break

        if time.monotonic() - last_usage > 45:
            cur_balance, us = usage()
            last_usage = time.monotonic()
            spent = initial_balance - cur_balance
            (ROOT / 'usage-current.txt').write_text(us, encoding='utf-8')
            print(f'BUDGET spent={spent:.2f}/{MAX_BATCH_SPEND:.1f} (balance={cur_balance:.2f})', flush=True)
            if 'Active assignments: 0' in us:
                print('ASSIGNMENT_LOST', flush=True)
                break
            if spent >= MAX_BATCH_SPEND:
                print('BUDGET_STOP', flush=True)
                break

        time.sleep(15)

except Exception as e:
    print('SUPERVISOR_ERROR ' + str(e), flush=True)
finally:
    if started:
        try:
            print('Final pack recovery check before stopping Colab VM...', flush=True)
            recover_packs()
            for name in ('job.log', 'status.json'):
                get(name, 'colab-job.log' if name == 'job.log' else name, 90, optional=True)
        except Exception as e:
            print('FINAL_RECOVERY_WARNING ' + str(e)[-300:], flush=True)
        finally:
            if foreground_client is not None and foreground_client.poll() is None:
                try:
                    foreground_client.terminate()
                except Exception:
                    pass
            stopped = False
            for attempt in range(3):
                try:
                    out = call(['stop', '-s', SESSION], 120)[1]
                    print(out, flush=True)
                    stopped = True
                    break
                except Exception as e:
                    print('STOP_ERROR ' + str(e)[-300:], flush=True)
            try:
                final_balance, us = usage()
                (ROOT / 'usage-after.txt').write_text(us, encoding='utf-8')
                sessions_out = call(['sessions'])[1]
                (ROOT / 'shutdown.json').write_text(
                    json.dumps(
                        dict(
                            session=SESSION,
                            stop_command_succeeded=stopped,
                            downloaded_and_verified=len(downloaded),
                            expected_tracks=EXPECTED_TRACKS,
                            total_spent_units=round(initial_balance - final_balance, 2),
                            final_balance_units=final_balance,
                            active_sessions_output=sessions_out.strip(),
                        ),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding='utf-8',
                )
                print(f'FINAL_BATCH_SPEND {initial_balance - final_balance:.2f} units', flush=True)
                print(sessions_out, flush=True)
            except Exception as e:
                print('FINAL_CHECK_WARNING ' + str(e), flush=True)
