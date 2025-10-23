#!/bin/bash
set -e

# 用法: ./run.bash [GPU_ID]
# 也可通过环境变量 GPU_ID 设定: GPU_ID=0 ./run.bash
# 默认使用第二个 GPU（索引 1）
DEFAULT_GPU=1

# 优先使用命令行参数，其次使用外部环境变量 GPU_ID，否则使用 DEFAULT_GPU
GPU_ID="${1:-${GPU_ID:-$DEFAULT_GPU}}"

# 设置 CUDA 可见设备（只暴露指定的 GPU 给程序）
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# 创建并进入 build 目录
mkdir -p build
cd build

# 运行 cmake（如已构建可注释掉以下两行以加速）
cmake ..

# 编译
make -j4

# 运行程序，参数按需调整
./GPU_PoissonRecon ../bunny1.ply ./bunny_recon.ply 10