#!/bin/bash
# run_phy_geo.sh

export SCHRODINGER=/home/zdy/schrodinger2021-3
export LD_LIBRARY_PATH=$SCHRODINGER/internal/lib:$SCHRODINGER/internal/lib/ssl:$SCHRODINGER/mmshare-v5.4/lib/Linux-x86_64:$LD_LIBRARY_PATH
export PYTHONPATH=$SCHRODINGER/internal/lib/python3.8/site-packages:$PYTHONPATH

# 激活 conda 环境
conda activate FlexPose

# 运行脚本
python /home/zdy/Project2/data_processing/phy_geo.py --config /home/zdy/Project2/config.yaml
