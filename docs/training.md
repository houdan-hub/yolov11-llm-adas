# Model Training Methodology

## 1. Traffic Sign Detection Model

### 1.1 Base Architecture

- **Model**: YOLOv11m (medium variant)
- **Framework**: Ultralytics 8.x
- **Input size**: 640×640
- **Classes**: 57 traffic sign categories

### 1.2 Training Configuration

```python
model = YOLO('yolo11m.yaml')
results = model.train(
    data='./TS/data.yaml',    # Dataset configuration
    imgsz=640,                # Input image size
    epochs=150,               # Total training epochs
    batch=4,                  # Batch size (GPU memory constrained)
    workers=16,               # Data loading workers
    amp=True,                 # Automatic Mixed Precision
    cache='ram',              # Cache images in RAM for speed
    project='runs/V11train',
    name='exp'
)
```

### 1.3 Key Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `imgsz` | 640 | Standard YOLO input, balances speed/accuracy |
| `epochs` | 150 | Sufficient for convergence on custom dataset |
| `batch` | 4 | Limited by GPU VRAM; AMP helps mitigate |
| `amp` | True | Reduces VRAM usage, speeds up training |
| `cache` | 'ram' | Eliminates disk I/O bottleneck |
| `workers` | 16 | Parallel data augmentation |

### 1.4 Dataset

- Custom-labeled Chinese traffic sign dataset
- Annotated using LabelImg in YOLO format (`class x_center y_center width height`)
- Classes organized into prohibitory, informational, mandatory, priority, and warning categories

## 2. Experimental Model Variants

Two YOLOv10-based architectural variants were explored to investigate performance improvements:

### 2.1 AKConv Variant (`yolov10n_AKConv.yaml`)

- Replaces standard `Conv` layers in backbone with **AKConv (Asymmetric Convolution Kernel)**
- AKConv decomposes standard convolution into asymmetric kernel operations, reducing parameters while maintaining receptive field
- Uses `SCDown` (Spatial-Channel Downsampling) for efficient feature map reduction
- Incorporates **PSA (Pixel-wise Spatial Attention)** in the neck

### 2.2 WTConv Variant (`yolov10_WTconv.yaml`)

- Integrates **WTConv (Wavelet Transform Convolution)** into C2f blocks (`C2f_WTConv`)
- Adds **EAMA (Efficient Attention Multi-Scale)** module before SPPF
- Wavelet-based convolutions provide frequency-domain feature decomposition
- Targets improved small-object detection (relevant for distant traffic signs)

### 2.3 Comparison Framework

Both variants use:
- Same training pipeline and hyperparameters
- Same dataset
- YOLOv10 detection head (`v10Detect`) with dual-label assignment
- `C2fCIB` in the large-object detection head

## 3. Person/Vehicle Detection Model

- **Base**: YOLOv11 fine-tuned on person/car classes
- **Classes**: 2 (`person`, `car`)
- Used alongside the traffic sign model for dual-model parallel detection

## 4. Inference Optimization

### 4.1 Device Management

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
```

Automatic GPU/CPU fallback with runtime GPU utilization monitoring.

### 4.2 Tensor Preprocessing

Frames are converted to PyTorch tensors once and shared between both models:

```python
frame_tensor = torch.from_numpy(frame).to(device).float() / 255.0
frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC → BCHW
```

This avoids redundant H2D (host-to-device) memory transfers.

### 4.3 Post-Processing

- Confidence threshold: user-adjustable (default 0.50)
- IoU threshold for NMS: user-adjustable (default 0.45)
- Results extracted as numpy arrays: `boxes.xyxy`, `boxes.conf`, `boxes.cls`

## 5. Evaluation Metrics

The following metrics were tracked during training and validation:

- **mAP@0.5**: Mean Average Precision at IoU=0.5
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds 0.5-0.95
- **Precision / Recall**: Per-class and macro-averaged
- **Loss components**: box loss, cls loss, dfl loss
- **Inference FPS**: Measured on both GPU and CPU

## 6. Challenges & Solutions

| Challenge | Solution |
|---|---|
| GPU VRAM insufficient for large batch | AMP + batch=4 + gradient accumulation |
| Dataset class imbalance | Data augmentation + stratified sampling |
| Small traffic signs missed at distance | Higher input resolution + multi-scale training |
| Inference latency in GUI thread | Moved to QThreadPool with frame queue |
| Overfitting on limited data | Transfer learning from COCO pretrained weights |
