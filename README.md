# SGG-VC: Scene Graph Generation for Video Captioning

SGG-VC là một hệ thống Video Captioning tiên tiến sử dụng Đồ thị Ngữ cảnh (Scene Graph) để nắm bắt các mối quan hệ thực thể trong không gian và thời gian của video, từ đó tạo ra các mô tả chính xác và giàu ngữ nghĩa hơn.

## 🚀 Kiến trúc Mô hình (Architecture)

Mô hình **SGGClassCap** bao gồm các thành phần chính:

1.  **Feature Extractor**: 
    *   Sử dụng **VideoMAE** để trích xuất các đặc trưng chuyển động (motion features).
    *   Sử dụng **YOLO** để nâng cao đặc trưng từ các khung hình chính (keyframes).
2.  **SGG Module (Scene Graph Generation)**:
    *   Tích hợp từ submodule **SGG-Benchmark** (sử dụng mô hình **REACT**).
    *   Phát hiện các thực thể (objects) và mối quan hệ (relations) để xây dựng đồ thị ngữ cảnh cho từng phân đoạn video.
3.  **Temporal SG Encoder**:
    *   Mã hóa chuỗi thời gian của các đồ thị ngữ cảnh, giúp mô hình hiểu được sự thay đổi của các mối quan hệ theo thời gian.
4.  **Q-Former**:
    *   Cơ chế Query Transformer để trích xuất các tín hiệu thị giác quan trọng nhất từ chuỗi đặc trưng thời gian.
5.  **Caption Head**:
    *   Sử dụng mô hình ngôn ngữ lớn thị giác (VLM) **DeepSeek-VL2-tiny** để tạo ra văn bản mô tả cuối cùng.

## 📁 Cấu trúc Thư mục

```text
SGG-VC/
├── models/         # Chứa định nghĩa các thành phần mô hình
│   ├── sgg/        # Submodule SGG-Benchmark (REACT)
│   ├── model.py    # Kiến trúc tổng thể SGGClassCap
│   ├── qformer.py  # Query Transformer
│   ├── decoder.py  # Language Decoder (DeepSeek-VL2)
│   └── ...
├── dataset/        # Quản lý dữ liệu và DataLoader
├── utils/          # Các hàm hỗ trợ và hàm Loss (total_loss)
├── config.py       # Tệp cấu hình tập trung (Hyperparameters, Paths)
├── train.py        # Script huấn luyện chính
├── validate.py     # Script đánh giá và vẽ biểu đồ kết quả
└── README.md
```

## 🛠️ Cai dat (Installation)

### Yeu cau he thong
- Python 3.10
- CUDA 12.x
- GPU VRAM >= 16GB (recommend 24GB for training)

### 1. Clone repository va submodules
```bash
git clone --recursive https://github.com/tucpd/SGG-VC.git
cd SGG-VC
```

### 2. Tao moi truong Conda
```bash
conda create -n svc python=3.10 -y
conda activate svc
```

### 3. Cai dat PyTorch (CUDA 12.1)
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

### 4. Cai dat dependencies
```bash
pip install -r requirements.txt

# Cai dat pycocoevalcap tu git
pip install git+https://github.com/salaniz/pycocoevalcap.git
```

### 5. Build SGG-Benchmark (CUDA extension)
```bash
cd models/sgg
pip install -e .
cd ../..
```

### 6. Tai Pretrained Weights va GloVe

**GloVe Embeddings** (dat o thu muc goc):
- Tai tu: https://nlp.stanford.edu/projects/glove/
- File can thiet: `glove.6B.200d.txt` hoac `glove.6B.200d.pt`

**REACT Checkpoint** (dat vao `models/sgg/checkpoint/react_VG150/`):
- `best_model_epoch_9.pth` - REACT model weights
- Tai tu: [Link checkpoint]

**YOLOv8m VG150** (dat vao `models/sgg/checkpoint/`):
- `yolov8m_vg150.pt` - YOLO backbone for VG150

**Cau truc thu muc checkpoint:**
```
models/sgg/checkpoint/
├── react_VG150/
│   └── best_model_epoch_9.pth
└── yolov8m_vg150.pt
```

## ⚙️ Cấu hình (Configuration)

Tất cả các tham số được quản lý trong `config.py`. Bạn có thể tùy chỉnh:
- `embed_dim`, `proj_dims`: Kích thước vector đặc trưng.
- `decoder_config`: Cấu hình cho DeepSeek-VL2 (num_beams, max_new_tokens).
- `training`: Batch size, Learning rate, số lượng Epoch và tùy chọn `freeze_sgg`.

## 📈 Sử dụng (Usage)

### Huấn luyện (Training)
Chạy script huấn luyện với các tham số mặc định hoặc tùy chỉnh:
```bash
python train.py --batch_size 8 --output_dir ./checkpoints --dataset msrvtt
```

### Đánh giá (Validation)
Mô hình sẽ tự động chạy đánh giá sau mỗi epoch và lưu kết quả tốt nhất vào `best_model.pt`. Các chỉ số như **BLEU-4**, **CIDEr** sẽ được ghi lại trong `results.csv`.

## 📊 Logging & Visualization
Kết quả huấn luyện (Loss, Metrics) được lưu dưới dạng CSV và biểu đồ đồ thị trong thư mục `--output_dir` giúp dễ dàng theo dõi quá trình hội tụ của mô hình.

---
*Dự án đang trong quá trình phát triển (Phase 1: Freeze SGG).*
