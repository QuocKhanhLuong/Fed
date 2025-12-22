#!/bin/bash
# Quick setup script for macOS with Conda
# Script cài đặt nhanh cho FL-QUIC

set -e

echo "=========================================="
echo "FL-QUIC - Cài Đặt Nhanh"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda chưa được cài đặt!"
    echo "Vui lòng cài đặt Anaconda hoặc Miniconda từ:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Conda đã được cài đặt"

# Environment name
ENV_NAME="fl-quic"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo ""
    echo "⚠️  Môi trường '${ENV_NAME}' đã tồn tại"
    read -p "Bạn có muốn xóa và tạo lại? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Đang xóa môi trường cũ..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Đang sử dụng môi trường hiện có..."
    fi
fi

# Create conda environment
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo ""
    echo "📦 Đang tạo môi trường conda '${ENV_NAME}'..."
    conda create -n ${ENV_NAME} python=3.10 -y
fi

# Activate environment
echo ""
echo "🔄 Đang kích hoạt môi trường..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# Install NumPy via conda
echo ""
echo "📥 Đang cài đặt NumPy..."
conda install numpy -y

# Install PyTorch via conda (for macOS)
echo ""
echo "📥 Đang cài đặt PyTorch..."
if [[ $(uname -m) == "arm64" ]]; then
    # Apple Silicon (M1/M2/M3)
    echo "   Phát hiện Apple Silicon - cài đặt PyTorch cho ARM64..."
    conda install pytorch torchvision -c pytorch -y
else
    # Intel Mac
    echo "   Phát hiện Intel Mac - cài đặt PyTorch cho x86_64..."
    conda install pytorch torchvision -c pytorch -y
fi

# Install other dependencies via pip
echo ""
echo "📥 Đang cài đặt các thư viện còn lại..."

# Core dependencies
pip install lz4 --quiet
echo "   ✓ lz4"

pip install aioquic --quiet || {
    echo "   ⚠️  aioquic cần OpenSSL"
    if command -v brew &> /dev/null; then
        echo "   Đang cài đặt OpenSSL qua Homebrew..."
        brew install openssl 2>/dev/null || true
        LDFLAGS="-L$(brew --prefix openssl)/lib" \
        CPPFLAGS="-I$(brew --prefix openssl)/include" \
        pip install aioquic --quiet
        echo "   ✓ aioquic"
    else
        echo "   ❌ Vui lòng cài đặt OpenSSL thủ công"
    fi
}

pip install flwr --quiet
echo "   ✓ flwr"

pip install timm --quiet
echo "   ✓ timm"

pip install tqdm --quiet
echo "   ✓ tqdm"

pip install tensorboard --quiet
echo "   ✓ tensorboard"

# Verify installation
echo ""
echo "=========================================="
echo "🔍 Kiểm Tra Cài Đặt"
echo "=========================================="

python -c "import numpy; print('✓ NumPy:', numpy.__version__)" || echo "✗ NumPy"
python -c "import lz4; print('✓ LZ4')" || echo "✗ LZ4"
python -c "import torch; print('✓ PyTorch:', torch.__version__)" || echo "✗ PyTorch"
python -c "import aioquic; print('✓ aioquic')" || echo "✗ aioquic"
python -c "import flwr; print('✓ Flower')" || echo "✗ Flower"
python -c "import timm; print('✓ timm:', timm.__version__)" || echo "✗ timm"

# Run demo
echo ""
echo "=========================================="
echo "🧪 Chạy Demo"
echo "=========================================="

if python demo_standalone.py; then
    echo ""
    echo "=========================================="
    echo "✅ CÀI ĐẶT THÀNH CÔNG!"
    echo "=========================================="
    echo ""
    echo "Để sử dụng:"
    echo "  1. Kích hoạt môi trường:"
    echo "     conda activate ${ENV_NAME}"
    echo ""
    echo "  2. Chạy test:"
    echo "     python test_transport.py"
    echo ""
    echo "  3. Xem tài liệu:"
    echo "     cat INSTALL_VI.md"
else
    echo ""
    echo "⚠️  Demo gặp lỗi, nhưng các thư viện đã được cài đặt"
    echo "Vui lòng kiểm tra lại các dependencies"
fi

echo ""
echo "Môi trường: ${ENV_NAME}"
echo "Python: $(python --version)"
