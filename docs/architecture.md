# System Architecture

## 1. High-Level Design

The application follows a **layered, event-driven architecture** built on PyQt5's signal/slot mechanism. The core principle is strict separation between the GUI thread and all computationally expensive operations (video I/O, model inference, LLM API calls, TTS).

## 2. Threading Model

### 2.1 Thread Inventory

| Thread | Type | Responsibility |
|---|---|---|
| **Main (GUI)** | `QApplication` main loop | Widget rendering, user input, signal dispatch |
| **DetectionThread** | `QThread` | Frame capture from video/camera, frame queue management, FPS control |
| **DetectionWorker ×4** | `QRunnable` in `QThreadPool` | YOLOv11 inference on queued frames |
| **AIWorker** | `QRunnable` in global pool | Async LLM API calls (scene, Q&A, report) |
| **SpeechWorker** | `threading.Thread` (daemon) | pyttsx3 TTS queue consumption |

### 2.2 Frame Pipeline

```
VideoCapture → Frame Queue (maxsize=10) → DetectionWorker → results_ready signal
     ↑              ↑                              ↓
     │         bounded buffer              process_detection_results()
     │         (backpressure)                    ↓
     └── FPS control                    last_detections (thread-safe lock)
                                               ↓
                                        display_frame() draws bboxes
```

The producer-consumer pattern with a bounded queue ensures:
- The GUI never blocks on inference
- Frames are dropped gracefully if inference can't keep up (queue full → skip put)
- Memory usage is bounded

### 2.3 Thread Safety

- `self.last_detections`: guarded by `threading.Lock()`; written by `DetectionWorker`, read by `display_frame()`
- `self.speech_queue`: `queue.Queue()` (thread-safe) for TTS dispatch
- `self.ai_busy`: boolean flag to serialize LLM requests
- `self.pause_event`: `threading.Event()` for efficient pause/resume

## 3. Detection Subsystem

### 3.1 Dual-Model Parallel Inference

Each `DetectionWorker.run()` executes both models on the same frame tensor:

```python
frame_tensor = torch.from_numpy(frame).to(device).float() / 255.0
frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC → BCHW

traffic_results = traffic_model(frame_tensor, conf=conf_thres, iou=iou_thres)
person_car_results = person_car_model(frame_tensor, conf=conf_thres, iou=iou_thres)
```

Both models share the same preprocessed tensor, avoiding redundant GPU memory transfers.

### 3.2 Class Taxonomy

**Traffic Signs (57 classes)**: Organized into 4 categories:
- `forb_*`: Prohibitory signs (speed limits, no overtaking, no stopping, etc.)
- `info_*`: Informational signs (bus station, crosswalk, highway, parking, etc.)
- `mand_*`: Mandatory signs (turn left, roundabout, straight, etc.)
- `prio_*`: Priority signs (give way, priority road, stop)
- `warn_*`: Warning signs (children, construction, slippery, traffic light, etc.)

**Person/Vehicle (2 classes)**: `person`, `car`

## 4. LLM Integration Subsystem

### 4.1 Prompt Engineering

Three prompt types are constructed dynamically from detection results:

1. **Scene Description**: "根据以下检测结果，用自然语言描述当前交通场景" + detected objects with confidence
2. **Driving Advice**: "根据以下交通场景描述，给出具体的驾驶建议" + scene description
3. **Q&A / Report**: User question or accumulated advice log

### 4.2 Async Execution

```python
class AIWorker(QRunnable):
    def run(self):
        for p in self.prompts:
            rsp = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": p}],
                max_tokens=400
            )
            answers.append(rsp.choices[0].message.content.strip())
        self.signals.result_ready.emit({"task": self.task, "data": answers})
```

All LLM calls are non-blocking from the GUI perspective. Results are delivered via `pyqtSignal(dict)`.

### 4.3 Auto-Analysis Timer

A `QTimer` fires every 1000ms, but actual LLM calls are gated by:
- Non-empty detection results
- 5-second minimum interval since last analysis (`analysis_interval`)
- `ai_busy` flag (no concurrent LLM requests)

## 5. GUI Layout

```
┌──────────────────────────────────────────────────────────────┐
│  [Video Display Area 800x500]          │ [Model Weights]     │
│  FPS: 24.3  GPU: 67%                   │  [Traffic Sign]     │
│                                         │  [Person/Vehicle]   │
│  ┌─── Tab Widget ────────────────────┐  │ [Source Type]       │
│  │ Scene │ Log │ Q&A │ Report │ Stats│  │  [Img][Vid][Cam]   │
│  │ [scene description text]          │  │  [path input]      │
│  │ [driving advice text]             │  │  [Start][Pause]    │
│  │ [Speak Advice Button]             │  │  [Resume][Stop]    │
│  └────────────────────────────────────┘  │ [Frame-by-frame]   │
│                                          │ [Conf/IoU sliders] │
│                                          │ [Detection Tables] │
└──────────────────────────────────────────────────────────────┘
```

The layout uses `QSplitter` for resizable left/right panels, with `QTabWidget` for the multi-function analysis panel.

## 6. Error Handling

- Model loading failures → `QMessageBox.critical()` + graceful degradation
- Video/camera open failures → error signal → GUI notification + auto-stop
- LLM API errors → `AIWorker.signals.error` → GUI notification, detection continues
- Thread cleanup on window close → `stop_detection()` disconnects signals, waits for thread, releases capture
