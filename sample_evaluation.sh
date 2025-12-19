#!/bin/bash

if [ $# -ne 1 ]; then
    echo "try: $0 <run_id>, missing your run_id, please check!"
    exit 1
fi


RUN_ID=$1

python sample.py --run_id "$RUN_ID"
python evaluation.py --datasets Istanbul --task Stat --experiment_comments "$RUN_ID" &


