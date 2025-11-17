# Hướng Dẫn Cài Đặt - FL-QUIC-LoRA

## Bước 1: Tạo Conda Environment

```bash
# Tạo môi trường conda mới với Python 3.10
conda create -n fl-quic python=3.10 -y

# Kích hoạt môi trường
conda activate fl-quic
```

## Bước 2: Cài Đặt Dependencies Cơ Bản

```bash
# Cài đặt NumPy và các thư viện cơ bản
conda install numpy -y

# Cài đặt PyTorch (cho macOS)
conda install pytorch torchvision -c pytorch -y

# Cài đặt các thư viện còn lại qua pip
pip install lz4
pip install aioquic
pip install flwr
pip install transformers
pip install peft
pip install tqdm
pip install tensorboard
```

## Bước 3: Kiểm Tra Cài Đặt

```bash
# Chạy demo standalone (chỉ cần NumPy và LZ4)
python demo_standalone.py
```

## Bước 4: Test Transport Layer

```bash
# Sau khi cài đặt đầy đủ
python test_transport.py
```

## Lệnh Nhanh (All-in-One)

```bash
# Tạo và cài đặt tất cả
conda create -n fl-quic python=3.10 numpy -y
conda activate fl-quic
conda install pytorch torchvision -c pytorch -y
pip install lz4 aioquic flwr transformers peft tqdm tensorboard

# Kiểm tra
python demo_standalone.py
```

## Lưu Ý cho macOS

- aioquic yêu cầu OpenSSL, có thể cần cài đặt thêm:
  ```bash
  brew install openssl
  ```

- Nếu gặp lỗi khi cài aioquic, thử:
  ```bash
  LDFLAGS="-L$(brew --prefix openssl)/lib" \
  CPPFLAGS="-I$(brew --prefix openssl)/include" \
  pip install aioquic
  ```

## Kiểm Tra Nhanh

```bash
# Kiểm tra các package đã cài
python -c "import numpy; print('✓ NumPy:', numpy.__version__)"
python -c "import lz4; print('✓ LZ4')"
python -c "import torch; print('✓ PyTorch:', torch.__version__)"
```

## Troubleshooting

### Lỗi: "No module named 'aioquic'"
→ `pip install aioquic` hoặc xem phần cài đặt OpenSSL ở trên

### Lỗi: "No module named 'lz4'"
→ `pip install lz4`

### Lỗi: "No module named 'torch'"
→ `conda install pytorch -c pytorch -y`
