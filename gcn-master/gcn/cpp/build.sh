#!/bin/bash
# 统一的构建脚本，支持宿主机和 Docker 环境

set -e  # 遇到错误立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 检测环境
if [ -d "/workspace/hlexperience" ]; then
    echo "[Build] 检测到 Docker 环境"
    BUILD_DIR="build_docker"
    export HLEXPERIENCE_ROOT="/workspace/hlexperience"
elif [ -d "/home/wanyu/hlexperience" ]; then
    echo "[Build] 检测到宿主机环境"
    BUILD_DIR="build_host"
    export HLEXPERIENCE_ROOT="/home/wanyu/hlexperience"
else
    echo "[Build] 使用当前目录作为基础路径"
    BUILD_DIR="build"
    # 尝试自动检测
    if [ -d "../.." ]; then
        POSSIBLE_ROOT="$(cd ../.. && pwd)"
        if [ -d "$POSSIBLE_ROOT/ramulator-pim-master" ]; then
            export HLEXPERIENCE_ROOT="$POSSIBLE_ROOT"
            echo "[Build] 自动检测到基础路径: $HLEXPERIENCE_ROOT"
        fi
    fi
fi

echo "[Build] 构建目录: $BUILD_DIR"
echo "[Build] 基础路径: ${HLEXPERIENCE_ROOT:-未设置}"

# 创建构建目录
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# 配置 CMake
echo "[Build] 配置 CMake..."
cmake ..

# 编译
echo "[Build] 开始编译..."
make -j$(nproc)

echo "[Build] 编译完成！"
echo "[Build] 可执行文件: $BUILD_DIR/gcn_cpp"
echo ""
echo "运行方式:"
echo "  cd $BUILD_DIR && ./gcn_cpp --dataset cora"

