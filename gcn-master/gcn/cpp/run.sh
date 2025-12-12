#!/bin/bash
# 统一的运行脚本，自动检测环境并设置路径

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 检测环境并设置基础路径
if [ -d "/workspace/hlexperience" ]; then
    echo "[Run] 检测到 Docker 环境"
    BUILD_DIR="build_docker"
    export HLEXPERIENCE_ROOT="/workspace/hlexperience"
elif [ -d "/home/wanyu/hlexperience" ]; then
    echo "[Run] 检测到宿主机环境"
    BUILD_DIR="build_host"
    export HLEXPERIENCE_ROOT="/home/wanyu/hlexperience"
else
    echo "[Run] 使用当前目录作为基础路径"
    BUILD_DIR="build"
    # 尝试自动检测
    if [ -d "../.." ]; then
        POSSIBLE_ROOT="$(cd ../.. && pwd)"
        if [ -d "$POSSIBLE_ROOT/ramulator-pim-master" ]; then
            export HLEXPERIENCE_ROOT="$POSSIBLE_ROOT"
            echo "[Run] 自动检测到基础路径: $HLEXPERIENCE_ROOT"
        fi
    fi
fi

# 检查可执行文件
if [ ! -f "$BUILD_DIR/gcn_cpp" ]; then
    echo "[Run] 错误: 找不到可执行文件 $BUILD_DIR/gcn_cpp"
    echo "[Run] 请先运行: ./build.sh"
    exit 1
fi

echo "[Run] 使用构建目录: $BUILD_DIR"
echo "[Run] 基础路径: ${HLEXPERIENCE_ROOT:-未设置}"
echo ""

# 运行程序，传递所有参数
cd "$BUILD_DIR"
exec ./gcn_cpp "$@"

