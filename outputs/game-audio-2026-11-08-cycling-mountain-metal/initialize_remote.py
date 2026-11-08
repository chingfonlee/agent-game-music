from pathlib import Path
import subprocess
r=Path('/content/colab_job/'); r.mkdir(parents=True, exist_ok=True)
s=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True)
print(s)
assert 'L4' in s
