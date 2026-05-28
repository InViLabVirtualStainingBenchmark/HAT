#!/bin/bash
CONFIG=${1:-options/train/train_HAT_BCI_HPC.yml}
export PYTHONPATH=/code/HAT/basicsr:/code/HAT:/usr/local/lib64/python3.9/site-packages:$PYTHONPATH
cd /code/HAT
python3 setup.py develop --no_cuda_ext 2>/dev/null || true
python3 hat/train.py \
    -opt $CONFIG \
    2>&1 | tee /output/train_log.txt
