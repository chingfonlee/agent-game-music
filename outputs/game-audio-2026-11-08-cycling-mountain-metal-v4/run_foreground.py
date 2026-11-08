from pathlib import Path
import os,subprocess,sys
r=Path('/content/colab_job/')
if (r/'.hf_token').exists(): os.chmod(r/'.hf_token',0o600)
f=open(r/'job.log','ab',buffering=0)
p=subprocess.Popen([sys.executable,'-u',str(r/'job_remote.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(r/'job.pid').write_text(str(p.pid));print('OWNED_JOB_PID',p.pid,flush=True)
try:code=p.wait(timeout=3600)
except subprocess.TimeoutExpired:p.terminate();raise
print('JOB_RETURN',code,flush=True)
