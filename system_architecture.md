# System Architecture & Data Flow - Large Scale Federated Learning

## PHẦN 1: KIẾN TRÚC HỆ THỐNG TỔNG THỂ (SYSTEM ARCHITECTURE)

Hệ thống được thiết kế theo mô hình **Client-Server không đồng bộ (Asynchronous)**, tối ưu hóa cho mạng không ổn định thông qua giao thức QUIC.

### 1. Tầng Máy Chủ (Server-Side Cluster)
Để mở rộng (scale) lên hàng nghìn clients, Server không nên là một file python đơn lẻ.
* **Frontend (Load Balancer):** Một UDP Load Balancer (như Nginx hỗ trợ QUIC) để phân phối các gói tin QUIC đến các node xử lý.
* **FL Server Nodes (Stateless Workers):** Các container chạy code `quic_server.py`. Chúng không lưu trạng thái client trong RAM mà chỉ xử lý logic: Nhận weights $\rightarrow$ Giải nén $\rightarrow$ Đẩy vào hàng đợi.
* **State Store (Redis/Etcd):** Lưu trạng thái phiên làm việc (Session), Metadata của Client (Pin, RTT) và tiến độ vòng lặp (Round progress).
* **Model Store (S3/MinIO):** Lưu Global Model và các Checkpoint. Không lưu trên ổ cứng cục bộ của Worker.
* **Aggregator (Background Worker):** Một tiến trình riêng biệt chuyên đọc các bản cập nhật từ hàng đợi và thực hiện thuật toán `FedProx` hoặc `Contribution-Aware Aggregation`.

### 2. Tầng Client (Edge/IoT Devices)
Thiết kế theo dạng module để dễ dàng thay thế thuật toán mà không sửa core hệ thống.
* **Network Monitor (QUIC Profiler):** Module chạy ngầm, liên tục đo RTT, Packet Loss từ tầng giao vận và tính toán điểm `Network_Score` (0.0 - 1.0).
* **Adaptive Runtime:** Môi trường chạy model có khả năng "biến hình". Dựa vào `Network_Score` để quyết định:
    * Bật/tắt các Expert (trong mô hình MobileViT tùy chỉnh).
    * Quyết định tỷ lệ cắt tỉa Gradient (Pruning Ratio) trước khi gửi.
* **Compression Engine:** Bộ máy nén dữ liệu 3 bước (Quantization $\rightarrow$ Sparse Encoding $\rightarrow$ LZ4).

---

## PHẦN 2: LUỒNG DỮ LIỆU CHI TIẾT (DATA FLOW DESCRIPTION)

### Giai đoạn 1: Khởi tạo & Bắt tay (Handshake & Profiling)
1.  **Client khởi động:** Load cấu hình LoRA và MobileViT cục bộ.
2.  **0-RTT Connection:** Client mở kết nối QUIC đến Server. Nếu là kết nối lại (reconnect), Client gửi dữ liệu ngay lập tức mà không chờ bắt tay (Handshake).
3.  **Metadata Exchange (Stream ID 8):**
    * Client gửi gói tin JSON chứa: `Device_ID`, `Battery_Level`, `CPU_Load`.
    * Server ghi nhận vào **Redis**. Nếu Client yếu (Pin < 20%), Server gửi lệnh `STOP` ngay lập tức để tiết kiệm tài nguyên.

### Giai đoạn 2: Phân phối Model (Downlink)
1.  **Server Broadcast:** Khi đủ số lượng Client sẵn sàng, Server lấy Global Model từ **Model Store**.
2.  **Compression:** Server nén Global Weights (dùng Serializer).
3.  **Multiplexing Send (Stream ID 4):** Server đẩy dữ liệu xuống Client qua luồng Weights. Đồng thời gửi cấu hình `Hyperparams` (Learning Rate, FedProx Mu) qua luồng Control (Stream ID 0).

### Giai đoạn 3: Huấn luyện Thích nghi (Local Adaptive Training) - *Quan trọng nhất*
Đây là logic xử lý tại Client:
1.  **Network Sensing:** Trước khi train, Client hỏi module Network Monitor: *"Mạng hiện tại thế nào?"*. Nhận về `alpha` (0.0 tệ -> 1.0 tốt).
2.  **Dynamic Forward Pass:**
    * Dữ liệu ảnh đi qua MobileViT.
    * Tại các điểm rẽ nhánh (Gated Blocks): Nếu `alpha` thấp, kích hoạt các "Lightweight Conv Expert" để xử lý nhiễu nén. Nếu `alpha` cao, chạy luồng Transformer chuẩn.
3.  **Backward Pass & FedProx:**
    * Tính Gradient.
    * Cộng thêm `Proximal Term` (chênh lệch giữa Local Weight và Global Weight) vào Loss để đảm bảo model không đi quá xa (chống Non-IID).

### Giai đoạn 4: Nén & Tải lên (Uplink)
1.  **Adaptive Pruning:**
    * Dựa vào `alpha` (chất lượng mạng), Client quyết định giữ lại $k\%$ Gradient quan trọng nhất (Top-k).
    * Mạng yếu $\rightarrow$ Giữ 10% (Top-10% magnitude).
    * Mạng khỏe $\rightarrow$ Giữ 100%.
2.  **Encoding:** Chuyển ma trận Gradient thưa (nhiều số 0) sang định dạng CSR (Compressed Sparse Row) hoặc Bitmap.
3.  **Serialization:** Quantization (INT8) $\rightarrow$ LZ4 Compression.
4.  **Streaming:** Gửi dữ liệu nén qua QUIC Stream ID 4.

### Giai đoạn 5: Tổng hợp & Cập nhật (Aggregation)
1.  **Buffering:** Server Worker nhận gói tin, giải nén.
2.  **Quality Check:**
    * Server kiểm tra `Network_Score` lúc gửi của Client.
    * Nếu mạng quá tệ (RTT cao), giảm trọng số đóng góp của Client này (Network-Aware Weighting).
3.  **Aggregation:**
    * Aggregator lấy weights từ Redis/Memory.
    * Thực hiện cộng dồn có trọng số (Weighted Sum).
    * Cập nhật Global Model mới và lưu vào S3/MinIO.

---

## PHẦN 3: HƯỚNG DẪN SỬA ĐỔI CHO LARGE SCALE (SCALING STRATEGY)

### 1. Stateless Server
* **Yêu cầu:** Refactor `quic_server.py` để loại bỏ biến toàn cục `self.clients`. Thay vào đó, hãy dùng một interface `ClientManager` có thể kết nối tới Redis.
* **Lý do:** Để bạn có thể chạy 10 cái Server Containers cùng lúc mà vẫn đồng bộ được trạng thái.

### 2. Buffer Management
* **Yêu cầu:** Trong `quic_protocol.py`, hãy thêm giới hạn kích thước buffer (`MAX_BUFFER_SIZE`). Nếu Client gửi quá nhanh mà Server xử lý không kịp, hãy drop gói tin (Backpressure) thay vì để tràn RAM.

### 3. Asynchronous Aggregation
* **Yêu cầu:** Thay đổi `fl_strategy.py`. Thay vì đợi đủ `min_clients` mới chạy (Synchronous), hãy chuyển sang cơ chế `Time-Window`. Cứ mỗi 5 phút, tổng hợp tất cả updates đang có trong Redis, bất kể số lượng, sau đó cập nhật model.

### 4. Fault Tolerance (Chịu lỗi)
* **Yêu cầu:** Thêm cơ chế `Heartbeat` vào Stream ID 0. Nếu Client không gửi heartbeat trong 30s, Server tự động đánh dấu là Offline và loại khỏi vòng train hiện tại để không bị treo.

---

## Coding Agent Prompts

* **Agent 1 (Network):** "Implement the `NetworkMonitor` class in `transport/` that wraps `aioquic` stats and exposes a `get_network_score()` method."
* **Agent 2 (Model):** "Refactor `MobileViT` to include Gating Mechanisms based on an input scalar `quality_score`. Create a custom `forward(x, quality_score)` method."
* **Agent 3 (Server):** "Rewrite `FLServer` to use Redis for storing client states instead of in-memory dictionaries. Implement `Time-Window` aggregation."
