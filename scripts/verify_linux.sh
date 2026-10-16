set -eu
cd /mnt/c/Users/Administrator/Downloads/DaisyRec-v2.0_Updated/DaisyRec-v2.0
export DAISY_PROGRESS=0
/home/sagniksen/.venvs/daisyrec/bin/python scripts/check_env.py
/home/sagniksen/.venvs/daisyrec/bin/python tests/test_models.py
/home/sagniksen/.venvs/daisyrec/bin/python -m compileall -q daisy tune.py test.py scripts tests
