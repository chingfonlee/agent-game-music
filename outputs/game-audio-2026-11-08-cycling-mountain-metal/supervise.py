"""Local Colab L4 supervisor template for colab-audio-studio-mcp skill."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import wave
import zipfile

ROOT = Path(__file__).resolve().parent
SESSION = 'audio-studio-l4'
GPU_TYPE = 'L4'
REMOTE = '/content/colab_job/'
MAX_BATCH_SPEND = 4.0

# Auto-detect expected item count from manifest.json
MANIFEST_PATH = ROOT / 'manifest.json'
EXPECTED_ITEMS = len(json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))) if MANIFEST_PATH.exists() else 1

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
    """Verify ZIP CRC integrity, safe paths, non-zero size, local-vs-remote SHA-256 match, and audio decode."""
    archive = ROOT / 'packs' / name
    if not archive.exists() or archive.stat().st_size == 0:
        raise ValueError(f'Archive missing or empty: {archive}')

    with zipfile.ZipFile(archive) as z:
        bad_crc = z.testzip()
        if bad_crc is not None:
            raise ValueError(f'ZIP CRC failed on {bad_crc}')
        for entry in z.namelist():
            p = Path(entry)
            if p.is_absolute() or '..' in p.parts:
                raise ValueError(f'Unsafe archive path: {entry}')
        z.extractall(ROOT / 'results')

    metas = list((ROOT / 'results').rglob('*.json'))
    selected = [p for p in metas if json.loads(p.read_text(encoding='utf-8')).get('pack_name') == name]
    if len(selected) != 1:
        raise ValueError(f'Sidecar metadata not uniquely identified for {name}')
    meta_path = selected[0]
    meta = json.loads(meta_path.read_text(encoding='utf-8'))

    for filename, expected_sha256 in meta['files'].items():
        f = meta_path.parent / filename
        if not f.exists() or f.stat().st_size == 0:
            raise ValueError(f'Extracted output file missing or empty: {f}')
        actual_sha256 = hashlib.sha256(f.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(f'SHA-256 mismatch for {filename}: {actual_sha256} != {expected_sha256}')
        if f.suffix.lower() == '.wav':
            with wave.open(str(f), 'rb') as wf:
                if wf.getnframes() <= 0 or wf.getframerate() <= 0:
                    raise ValueError(f'Invalid WAV header in {f}')

    qa = meta.get('qa_report') or {}
    if not qa.get('passed', False):
        raise ValueError(f'Remote QA gate did not pass for {name}: {qa}')
    fdiag = qa.get('final_diagnostics') or {}
    repairs = qa.get('repairs_applied') or []

    downloaded.add(name)
    verified_details[name] = meta
    (ROOT / 'downloaded.json').write_text(json.dumps(sorted(downloaded), indent=2), encoding='utf-8')
    (ROOT / 'verification.json').write_text(
        json.dumps(verified_details, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(
        f'LOCAL_VERIFIED {len(downloaded)}/{EXPECTED_ITEMS} {name} | '
        f'QA=PASSED repairs={len(repairs)} intro_drop={fdiag.get("intro_drop_db")}dB '
        f'sfx_zone={fdiag.get("sfx_zone_3to6k_pct")}% seam={fdiag.get("seam_jump")}',
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
    print(call(['new', '-s', SESSION, '--gpu', GPU_TYPE], 180)[1], flush=True)
    time.sleep(6)  # Allow Port 8080 Jupyter proxy warmup

    init_script = ROOT / 'initialize_remote.py'
    init_script.write_text(
        f"from pathlib import Path\nimport subprocess\n"
        f"r=Path({REMOTE!r}); r.mkdir(parents=True, exist_ok=True)\n"
        f"s=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True)\n"
        f"print(s)\nassert {GPU_TYPE!r} in s\n",
        encoding='utf-8',
    )
    print(call(['exec', '-s', SESSION, '-f', str(init_script), '--timeout', '40'], 70)[1], flush=True)

    for name in ('manifest.json', 'generate_remote.py', 'job_remote.py'):
        call(['upload', '-s', SESSION, str(ROOT / name), REMOTE + name], 120)

    token_path = Path.home() / '.cache' / 'huggingface' / 'token'
    token = (os.environ.get('HF_TOKEN') or (token_path.read_text(encoding='utf-8').strip() if token_path.exists() else '')).strip()
    if token:
        fd, secret = tempfile.mkstemp(prefix='colab-hf-', suffix='.tmp')
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
        f"from pathlib import Path\nimport os,subprocess,sys\n"
        f"r=Path({REMOTE!r})\n"
        f"if (r/'.hf_token').exists(): os.chmod(r/'.hf_token',0o600)\n"
        f"f=open(r/'job.log','ab',buffering=0)\n"
        f"p=subprocess.Popen([sys.executable,'-u',str(r/'job_remote.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)\n"
        f"(r/'job.pid').write_text(str(p.pid));print('OWNED_JOB_PID',p.pid,flush=True)\n"
        f"try:code=p.wait(timeout=3600)\n"
        f"except subprocess.TimeoutExpired:p.terminate();raise\n"
        f"print('JOB_RETURN',code,flush=True)\n",
        encoding='utf-8',
    )
    clientlog = open(ROOT / 'foreground-client.log', 'ab', buffering=0)
    foreground_client = subprocess.Popen(
        ['colab', 'exec', '-s', SESSION, '-f', str(foreground), '--timeout', '3700'],
        stdout=clientlog,
        stderr=subprocess.STDOUT,
        env=env,
    )
    (ROOT / 'foreground-client.pid').write_text(str(foreground_client.pid), encoding='utf-8')

    t0 = time.monotonic()
    last_usage = 0.0
    last_brief = ''
    failures = 0

    while time.monotonic() - t0 < 3600:
        if (ROOT / 'STOP_REQUEST').exists():
            print('USER_STOP_REQUEST', flush=True)
            break
        try:
            recover_packs()
            if len(downloaded) >= EXPECTED_ITEMS:
                print('ALL_ITEMS_DOWNLOADED_AND_VERIFIED', flush=True)
                break

            if get('status.json', 'remote-status.json', 60, optional=True):
                status = json.loads((ROOT / 'remote-status.json').read_text(encoding='utf-8'))
                if get('job.log', 'live-job.log', 75, optional=True):
                    s = (ROOT / 'live-job.log').read_text(encoding='utf-8', errors='replace')
                    brief = '\n'.join(
                        x for x in s.splitlines()
                        if x.startswith(('STATUS', 'TIMINGS', 'MODEL_LOADED', 'GENERATING', 'QA_INITIAL', 'QA_REPAIR', 'QA_PASSED', 'SAVED', 'MODEL_DONE', 'RuntimeError', 'Traceback'))
                    )
                    brief = '\n'.join(brief.splitlines()[-6:])
                    if brief != last_brief:
                        print(json.dumps(status) + '\n' + brief, flush=True)
                        last_brief = brief

                if status.get('state') == 'complete':
                    recover_packs()
                    if len(downloaded) >= EXPECTED_ITEMS:
                        print('ALL_ITEMS_DOWNLOADED_AND_VERIFIED', flush=True)
                        break
                    raise RuntimeError(f"Remote job completed but only {len(downloaded)}/{EXPECTED_ITEMS} items verified")

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
            recover_packs()
            for name in ('job.log', 'status.json', 'setup_timings.json'):
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
                            expected_items=EXPECTED_ITEMS,
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
