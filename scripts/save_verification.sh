set -eu
cd /mnt/c/Users/Administrator/Downloads/DaisyRec-v2.0_Updated/DaisyRec-v2.0
bash scripts/verify_linux.sh > artifacts/linux_verification.log 2>&1
/home/sagniksen/.venvs/daisyrec/bin/python -m pip freeze > artifacts/requirements-linux.lock.txt
/home/sagniksen/.venvs/daisyrec/bin/python -m pip check
nproc
free -h
