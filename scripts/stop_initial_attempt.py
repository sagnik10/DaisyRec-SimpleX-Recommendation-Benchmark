import os, signal
from pathlib import Path
root='/mnt/c/Users/Administrator/Downloads/DaisyRec-v2.0_Updated/DaisyRec-v2.0'
targets=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():
  continue
 try:
  args=(proc/'cmdline').read_bytes().decode().split('\0')
 except (OSError,UnicodeDecodeError):
  continue
 if root not in ' '.join(args):
  continue
 if any(arg.endswith('/scripts/run_experiments.py') or arg=='tune.py' for arg in args):
  targets.append((int(proc.name),args))
for pid,args in targets:
 print('Stopping experiment process',pid,args,flush=True)
 os.kill(pid,signal.SIGTERM)
