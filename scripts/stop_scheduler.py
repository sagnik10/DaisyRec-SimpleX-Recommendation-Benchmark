import os, signal
from pathlib import Path
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():
  continue
 try:
  args=(proc/'cmdline').read_bytes().decode().split('\0')
 except (OSError,UnicodeDecodeError):
  continue
 if any(arg.endswith('/scripts/run_experiments.py') for arg in args):
  print('Stopping scheduler; current tuning children continue:',proc.name,flush=True)
  os.kill(int(proc.name),signal.SIGTERM)
