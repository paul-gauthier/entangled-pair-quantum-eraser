#!/bin/bash

set -e

while true
do
    last_file=$(ls -1 data/*.jsonl | tail -n 1)

    if [ -z "$last_file" ]; then
        echo "No .jsonl files found in data/"
        sleep 10
        continue
    fi

    base_name=$(basename "$last_file")
    timestamp_prefix=${base_name:0:19}

    file_pattern="data/${timestamp_prefix}*.jsonl"

    echo "==> ./plots.py $file_pattern"
    ./plots.py $file_pattern

    echo "==> sleeping 10"
    sleep 10
done
