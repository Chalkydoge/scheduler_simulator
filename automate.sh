#!/bin/bash

# 检查 pcm 命令是否存在
if ! command -v pcm &> /dev/null; then
    echo "错误：pcm 命令未找到，请确保已安装 Intel PCM 工具。"
    exit 1
fi
export PCM_NO_PERF=1
timeout 10s bash -c 'pcm -r > "results.log"'

# -> 'node_output.csv'
python3 log.py 

# -> 'node_output_avg.csv'