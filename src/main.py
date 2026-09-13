"""
Multi-Dimensional Road Condition Intelligent Driving Assistance System
Based on YOLOv11 and Large Language Model (LLM)

This project implements an advanced driver assistance system (ADAS) prototype
that combines YOLOv11 object detection with LLM-based scene understanding.

Features:
- Dual-model parallel detection: traffic signs (57 classes) + pedestrians/vehicles
- Multi-source input: image, video file, real-time camera
- Multi-threaded architecture with frame queue and FPS control
- LLM-powered scene description, driving advice, Q&A, and report generation
- Real-time statistics visualization with matplotlib
- Voice broadcast via pyttsx3
- GPU utilization monitoring

Author: Tianbo Hou (Peter)
Internship: Neusoft Education Technology Group, Jun-Jul 2025
"""

# traffic_sign_with_guijiliudong(4).py
import sys
import base64
import cv2
import torch
import time
import os
import queue
import concurrent.futures
import threading
import matplotlib
import GPUtil

matplotlib.use('Agg')  # 使用非GUI后端
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pyttsx3
from collections import defaultdict

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QSlider, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QGroupBox, QTextEdit, QMessageBox, QMenu, QAction, QProgressBar,
    QTabWidget, QHeaderView, QSplitter, QLineEdit, QFrame
)
from PyQt5.QtGui import (
    QPixmap, QImage, QFont, QPalette, QColor, QIcon, QCursor, QBrush, QPainter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QThreadPool, QRunnable, QObject

from ultralytics import YOLO
from openai import OpenAI
# ========= 为 OpenAI 异步调用新增 =========
import copy
from PyQt5.QtCore import QRunnable, QThreadPool, QObject, pyqtSignal
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class AISignals(QObject):
    result_ready = pyqtSignal(dict)   # {"task":str, "data":list[str]}
    error       = pyqtSignal(str)
    finished    = pyqtSignal()

class AIWorker(QRunnable):
    """
    后台线程：顺序跑 prompt_list，全部拿到答案后一次性回传
    task  用来标识是“scene / qa / report”
    """
    def __init__(self, client, prompt_list: list[str], task: str, llm_model: str = "deepseek-ai/DeepSeek-V3"):
        super().__init__()
        self.client    = client
        self.prompts   = copy.deepcopy(prompt_list)
        self.task      = task
        self.llm_model = llm_model
        self.signals   = AISignals()

    def run(self):
        try:
            answers = []
            for p in self.prompts:
                rsp = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": p}],
                    max_tokens=400
                )
                answers.append(rsp.choices[0].message.content.strip())
            self.signals.result_ready.emit({"task": self.task,
                                            "data": answers})
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


# ====================== 背景基类 ======================
class BackgroundWidget(QWidget):
    """自动把指定图片绘制到窗口底层（等比扩展并居中）。"""

    def __init__(self, bg_path: str, parent=None):
        super().__init__(parent)
        self._bg = QPixmap(bg_path)

    def paintEvent(self, event):
        if self._bg and not self._bg.isNull():
            painter = QPainter(self)
            scaled = self._bg.scaled(
                self.size(),
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation
            )
            # 居中
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        # 继续让 Qt 绘制子控件
        super().paintEvent(event)


# ====================================================
# 交通标志类别标签
TRAFFIC_CLASS_LABELS = [
    "forb_ahead", "forb_left", "forb_overtake", "forb_right", "forb_speed_over_10",
    "forb_speed_over_100", "forb_speed_over_130", "forb_speed_over_20", "forb_speed_over_30",
    "forb_speed_over_40", "forb_speed_over_5", "forb_speed_over_50", "forb_speed_over_60",
    "forb_speed_over_70", "forb_speed_over_80", "forb_speed_over_90", "forb_stopping",
    "forb_trucks", "forb_u_turn", "forb_weight_over_3-5t", "forb_weight_over_7-5t",
    "info_bus_station", "info_crosswalk", "info_highway", "info_one_way_traffic",
    "info_parking", "info_taxi_parking", "mand_bike_lane", "mand_left",
    "mand_left_right", "mand_pass_left", "mand_pass_left_right", "mand_pass_right",
    "mand_right", "mand_roundabout", "mand_straigh_left", "mand_straight",
    "mand_straight_right", "prio_give_way", "prio_priority_road", "prio_stop",
    "warn_children", "warn_construction", "warn_crosswalk", "warn_cyclists",
    "warn_domestic_animals", "warn_other_dangers", "warn_poor_road_surface",
    "warn_roundabout", "warn_slippery_road", "warn_speed_bumper", "warn_traffic_light",
    "warn_tram", "warn_two_way_traffic", "warn_wild_animals"
]

# 行人车辆类别标签
PERSON_CAR_CLASS_LABELS = ["person", "car"]


# ====================== 检测线程 ======================
class DetectionWorker(QRunnable):
    def __init__(self, traffic_model, person_car_model, frame, conf_thres, iou_thres, device):
        super().__init__()
        self.traffic_model = traffic_model
        self.person_car_model = person_car_model
        self.frame = frame
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.signals = WorkerSignals()

    def run(self):
        try:
            # 将帧转换为PyTorch张量并移动到GPU
            frame_tensor = torch.from_numpy(self.frame).to(self.device).float() / 255.0
            frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC to BCHW

            # 处理交通标志检测
            traffic_results = self.traffic_model(
                frame_tensor, conf=self.conf_thres, iou=self.iou_thres
            )

            # 处理行人车辆检测
            person_car_results = self.person_car_model(
                frame_tensor, conf=self.conf_thres, iou=self.iou_thres
            )

            self.signals.results_ready.emit(traffic_results, person_car_results, self.frame)
            self.signals.frame_ready.emit(self.frame.copy())  # 新增

        except Exception as e:
            self.signals.error.emit(str(e))


class WorkerSignals(QObject):
    results_ready = pyqtSignal(object, object, object)
    frame_ready   = pyqtSignal(object)                  #
    error = pyqtSignal(str)


class DetectionThread(QThread):
    results_ready = pyqtSignal(object, object, object)  # (traffic_results, person_car_results, image)
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal()
    frame_ready = pyqtSignal(object)  # 用于逐帧播放的信号
    error_occurred = pyqtSignal(str)  # 错误信号
    fps_updated = pyqtSignal(float)  # FPS更新信号
    gpu_usage_updated = pyqtSignal(float)  # GPU使用率更新信号

    def __init__(self, traffic_model, person_car_model, source, source_type,
                 conf_thres, iou_thres, device):
        super().__init__()
        # 业务相关
        self.traffic_model    = traffic_model
        self.person_car_model = person_car_model
        self.source           = source
        self.source_type      = source_type
        self.conf_thres       = conf_thres
        self.iou_thres        = iou_thres
        self.device           = device
        # 控制变量
        # 运行-暂停控制
        self.running = True  # while 循环总开关
        self.pause_event = threading.Event()  # wait() 用
        self.pause_event.clear()  # ← 默认“暂停”状态
        # ← 一启动就处于运行状态
        self.frame_by_frame       = False
        self.next_frame_requested = False
        # 资源
        self.cap          = None
        self.thread_pool  = QThreadPool()
        self.thread_pool.setMaxThreadCount(4)
        self.frame_queue  = queue.Queue(maxsize=10)
        # 统计
        self.frame_count     = 0
        self.start_time      = time.time()
        self.last_frame_time = 0
        self.target_fps      = 30
        self.last_gpu_check  = time.time()
    def run(self):
        print("paused?", self.pause_event.is_set())

        try:
            if self.source_type == 'image':
                self.process_image()
            elif self.source_type == 'video':
                self.process_video()
            elif self.source_type == 'camera':
                self.process_camera()
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            # 留给自己收尾，不会阻塞 GUI
            self.thread_pool.waitForDone()
            self.finished.emit()


    def process_image(self):
        image = cv2.imread(self.source)
        if image is None:
            return

        # 使用线程池处理检测
        worker = DetectionWorker(
            self.traffic_model,
            self.person_car_model,
            image,
            self.conf_thres,
            self.iou_thres,
            self.device
        )
        worker.signals.results_ready.connect(self.results_ready)
        worker.signals.frame_ready.connect(self.frame_ready)  # 新增

        worker.signals.error.connect(self.error_occurred)
        self.thread_pool.start(worker)

    def process_video(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            self.error_occurred.emit(f"无法打开视频: {self.source}")
            return
        total_frames  = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_orig      = self.cap.get(cv2.CAP_PROP_FPS)
        self.target_fps = fps_orig if fps_orig and fps_orig < 30 else 30
        frame_interval = 1.0 / self.target_fps
        # 逐帧模式从暂停状态开始
        if self.frame_by_frame:
            self.pause_event.clear()
        self.start_time      = time.time()
        self.last_frame_time = time.time()
        frame_id             = 0
        while self.running and self.cap.isOpened():
            self.pause_event.wait()                # ****** 阻塞式暂停 ******
            # 控制帧率
            now = time.time()
            dt  = now - self.last_frame_time
            if dt < frame_interval:
                time.sleep(frame_interval - dt)
            self.last_frame_time = time.time()
            # 逐帧：未点“下一帧”时继续暂停
            if self.frame_by_frame and not self.next_frame_requested:
                self.pause_event.clear()
                continue
            self.next_frame_requested = False
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_id += 1
            progress = int(frame_id / total_frames * 100) if total_frames else 0
            self.progress_updated.emit(progress)
            # FPS 统计
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed:
                self.fps_updated.emit(self.frame_count / elapsed)
            # 入队
            if not self.frame_queue.full():
                self.frame_queue.put(frame.copy())
            if not self.frame_queue.empty():
                frm = self.frame_queue.get()
                worker = DetectionWorker(self.traffic_model, self.person_car_model,
                                         frm, self.conf_thres, self.iou_thres, self.device)
                worker.signals.results_ready.connect(self.results_ready)
                worker.signals.frame_ready.connect(self.frame_ready)
                worker.signals.error.connect(self.error_occurred)
                self.thread_pool.start(worker)
            # GPU 使用率
            if time.time() - self.last_gpu_check > 1.0:
                self.last_gpu_check = time.time()
                self.check_gpu_usage()
            # 逐帧：处理完一帧后再次暂停
            if self.frame_by_frame:
                self.pause_event.clear()
        if self.cap:
            self.cap.release()

    def process_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.error_occurred.emit("无法打开摄像头")
            return
        self.target_fps    = 20
        frame_interval     = 1.0 / self.target_fps
        self.start_time    = time.time()
        self.last_frame_time = time.time()
        while self.running and self.cap.isOpened():
            self.pause_event.wait()                # ****** 阻塞式暂停 ******
            now = time.time()
            dt  = now - self.last_frame_time
            if dt < frame_interval:
                time.sleep(frame_interval - dt)
            self.last_frame_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed:
                self.fps_updated.emit(self.frame_count / elapsed)
            if not self.frame_queue.full():
                self.frame_queue.put(frame.copy())
            if not self.frame_queue.empty():
                frm = self.frame_queue.get()
                worker = DetectionWorker(self.traffic_model, self.person_car_model,
                                         frm, self.conf_thres, self.iou_thres, self.device)
                worker.signals.results_ready.connect(self.results_ready)
                worker.signals.frame_ready.connect(self.frame_ready)
                worker.signals.error.connect(self.error_occurred)
                self.thread_pool.start(worker)
            if time.time() - self.last_gpu_check > 1.0:
                self.last_gpu_check = time.time()
                self.check_gpu_usage()
    def check_gpu_usage(self):
        """检查并发送GPU使用率"""
        if self.device == 'cuda':
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = gpus[0].load * 100  # 第一个GPU的使用率
                    self.gpu_usage_updated.emit(gpu_usage)
            except Exception:
                pass  # 忽略GPU监控错误

    # ----- 2) 暂停 ----------------------------------------------------------
    def pause(self):
        """挂起检测（阻塞在 wait()）"""
        self.pause_event.clear()
    # -----------------------------------------------------------------------
    # ----- 3) 恢复 ----------------------------------------------------------
    def resume(self):
        """恢复检测（解除 wait 阻塞）"""
        self.pause_event.set()
        self.last_frame_time = time.time()        # 避免恢复后一帧被限速
    # -----------------------------------------------------------------------
    # ----- 4) 停止 ----------------------------------------------------------
    def stop(self):
        """请求线程安全退出"""
        if not self.running:
            return
        self.running = False
        self.pause_event.set()                    # 万一正阻塞在 wait()
        if self.cap:
            self.cap.release()
        self.thread_pool.setMaxThreadCount(0)
    # 设置逐帧播放模式
    def set_frame_by_frame(self, enabled):
        self.frame_by_frame = enabled

    # 请求下一帧
    def request_next_frame(self):
        if self.frame_by_frame:
            self.next_frame_requested = True
            self.pause_event.set()                 # 唤醒一次
    # -----------------------------------------------------------

class TrafficSignApp(BackgroundWidget):  # 继承 BackgroundWidget
    def __init__(self):
        super().__init__("button/hero.png")  # 传入背景图片
        self.setWindowTitle("交通标志与行人车辆识别系统")
        self.setGeometry(100, 100, 1400, 900)
        self.last_update_time = time.time()
        self.update_interval = 0.1  # 更新间隔200ms

        # 优化样式表
        self.setStyleSheet("""
            QWidget {
                font-family: "SimHei", "WenQuanYi Micro Hei", "Heiti TC";
                font-size: 14px;
                color: #333;
                background: transparent;
            }
            QGroupBox {
                border: 1px solid #ccc;
                border-radius: 5px;
                margin-top: 10px;
                background: rgba(255, 255, 255, 200);
                color: #333;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
                background: transparent;
                color: #333;
            }
            QPushButton {
                background: rgba(76, 175, 80, 0.7);
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                background: rgba(76, 175, 80, 0.9);
            }
            QPushButton:disabled {
                background: rgba(204, 204, 204, 0.7);
                color: #666666;
            }
            QTableWidget {
                gridline-color: #ddd;
                background: rgba(255, 255, 255, 200);
                color: #333;
                alternate-background-color: rgba(240, 240, 240, 200);
            }
            QTableWidget::item {
                padding: 4px;
                color: #333;
                background: transparent;
            }
            QHeaderView::section {
                background: rgba(220, 220, 220, 200);
                padding: 4px;
                border: 1px solid #ddd;
                font-weight: bold;
                color: #333;
            }
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: rgba(220, 220, 220, 200);
                height: 10px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #eee, stop:1 #ccc);
                border: 1px solid #777;
                width: 18px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QProgressBar {
                border: 1px solid grey;
                border-radius: 3px;
                text-align: center;
                background: rgba(220, 220, 220, 200);
                color: #333;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 1px;
            }
            QLineEdit {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 5px;
                background: rgba(255, 255, 255, 200);
                color: #333;
            }
            QTextEdit {
                background: rgba(255, 255, 255, 200);
                border: 1px solid #ccc;
                border-radius: 4px;
                color: #333;
            }
            QTabWidget::pane {
                border: 1px solid #ccc;
                background: transparent;
            }
            QTabBar::tab {
                background: rgba(220, 220, 220, 200);
                border: 1px solid #ccc;
                padding: 8px 16px;
                color: #333;
            }
            QTabBar::tab:selected {
                background: rgba(76, 175, 80, 0.3);
                color: #333;
            }
            QLabel {
                color: #333;
                background: transparent;
            }
        """)

        # 模型与检测相关变量
        self.traffic_model = None
        self.person_car_model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.current_source = None
        self.current_source_type = None
        self.detection_thread = None
        self.detected_objects = {"traffic_signs": [], "persons_cars": []}
        self.last_analysis_time = 0
        self.analysis_interval = 5
        self.log_messages = []
        self.frame_by_frame_mode = False
        self.current_fps = 0
        self.gpu_usage = 0

        # 类别统计
        self.class_counts = defaultdict(int)
        self.class_labels = TRAFFIC_CLASS_LABELS + PERSON_CAR_CLASS_LABELS

        # 语音引擎
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # 语速
        self.engine.setProperty('volume', 1.0)  # 音量
        self.is_speaking = False
        self.speech_queue = queue.Queue()
        self.ai_pool = QThreadPool.globalInstance()
        self.ai_busy = False  # 只允许一次 AI 请求并发，省钱也防碰撞
        self.last_detections = []  # 存放最近一次检测结果
        self.det_lock = threading.Lock()
        self.init_ui()
        self.add_transparency_tab()
        self.add_statistics_tab()  # 新增统计标签页

        # 自动分析定时器
        self.auto_analysis_timer = QTimer(self)
        self.auto_analysis_timer.timeout.connect(self.auto_analyze_scene)
        self.auto_analysis_timer.start(1000)

        # 语音处理线程
        self.speech_thread = threading.Thread(target=self.speech_worker, daemon=True)
        self.speech_thread.start()

        # GPT 客户端
        self.client = OpenAI(
            api_key=os.environ.get('LLM_API_KEY', ''),
            base_url=os.environ.get('LLM_BASE_URL', 'https://api.siliconflow.cn')
        )
        self.llm_model = os.environ.get('LLM_MODEL', 'deepseek-ai/DeepSeek-V3')

    def init_ui(self):
        main_layout = QHBoxLayout()
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(5)

        # ---------- 左侧 -----------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)

        self.image_label = QLabel("请加载模型权重和预测文件")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 500)
        self.image_label.setStyleSheet("""
            border: 1px solid #ccc; 
            border-radius: 5px;
            background-color: rgba(240, 240, 240, 200);
        """)
        left_layout.addWidget(self.image_label)

        # FPS和GPU显示标签
        info_layout = QHBoxLayout()
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setAlignment(Qt.AlignLeft)
        self.fps_label.setStyleSheet(
            "color: green; font-weight: bold; background-color: rgba(0,0,0,0.5); padding: 2px;")

        self.gpu_label = QLabel("GPU: 0%")
        self.gpu_label.setAlignment(Qt.AlignLeft)
        self.gpu_label.setStyleSheet(
            "color: blue; font-weight: bold; background-color: rgba(0,0,0,0.5); padding: 2px;")

        info_layout.addWidget(self.fps_label)
        info_layout.addWidget(self.gpu_label)
        left_layout.addLayout(info_layout)

        self.tabs = QTabWidget()
        # ===== 路况分析 Tab =====
        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout(analysis_widget)
        self.scene_description = QTextEdit()
        self.scene_description.setReadOnly(True)
        self.driving_advice = QTextEdit()
        self.driving_advice.setReadOnly(True)
        self.scene_description.setPlaceholderText("这条路可真路啊...")
        self.driving_advice.setPlaceholderText("除非你开的是QQ飞车，否则我们都能给出良好的建议...")
        analysis_layout.addWidget(QLabel("当前路况描述:"))
        analysis_layout.addWidget(self.scene_description)
        analysis_layout.addWidget(QLabel("驾驶建议:"))
        analysis_layout.addWidget(self.driving_advice)

        # 语音播报按钮
        self.speech_btn = QPushButton("语音播报建议")
        self.speech_btn.setIcon(QIcon("button/语音.png"))
        self.speech_btn.clicked.connect(self.speak_current_advice)
        analysis_layout.addWidget(self.speech_btn)

        self.tabs.addTab(analysis_widget, "路况分析")

        # ===== 建议日志 Tab =====
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setPlaceholderText("这是一本岁月史书...")
        log_layout.addWidget(self.log_display)
        self.tabs.addTab(log_widget, "建议日志")

        # ===== 问答 Tab =====
        qa_widget = QWidget()
        qa_layout = QVBoxLayout(qa_widget)
        self.question_input = QTextEdit()
        self.question_input.setMaximumHeight(80)
        self.question_input.setPlaceholderText("请输入您的问题...")
        self.answer_output = QTextEdit()
        self.answer_output.setReadOnly(True)
        self.answer_output.setPlaceholderText("未来有无限可能孩子...")
        self.ask_btn = QPushButton("提问")
        self.ask_btn.clicked.connect(self.answer_question)
        qa_layout.addWidget(QLabel("你哪来的这么多问题:"))
        qa_layout.addWidget(self.question_input)
        qa_layout.addWidget(self.ask_btn)
        qa_layout.addWidget(QLabel("神会指引你:"))
        qa_layout.addWidget(self.answer_output)

        self.tabs.addTab(qa_widget, "路况万事通")

        # ===== 报告 Tab =====
        report_widget = QWidget()
        report_layout = QVBoxLayout(report_widget)
        self.generate_report_btn = QPushButton("生成道路建议报告")
        self.generate_report_btn.clicked.connect(self.generate_road_report)
        self.report_display = QTextEdit()
        self.report_display.setReadOnly(True)
        self.report_display.setPlaceholderText("我们的报告总是如此完美，无可挑剔...")
        report_layout.addWidget(self.generate_report_btn)
        report_layout.addWidget(self.report_display)
        self.tabs.addTab(report_widget, "道路建议报告")

        left_layout.addWidget(self.tabs)
        splitter.addWidget(left_widget)

        # ---------- 右侧（模型/参数/表格）-----------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)

        # ------ 模型组 ------
        model_group = QGroupBox("模型权重")
        model_layout = QVBoxLayout()
        self.load_traffic_btn = QPushButton("交通标识模型")
        self.load_person_car_btn = QPushButton("行人车辆模型")
        self.load_traffic_btn.clicked.connect(lambda: self.load_model("traffic"))
        self.load_person_car_btn.clicked.connect(lambda: self.load_model("person_car"))
        self.traffic_model_label = QLabel("未加载交通标识模型")
        self.person_car_model_label = QLabel("未加载行人车辆模型")
        model_layout.addWidget(self.load_traffic_btn)
        model_layout.addWidget(self.traffic_model_label)
        model_layout.addWidget(self.load_person_car_btn)
        model_layout.addWidget(self.person_car_model_label)
        model_group.setLayout(model_layout)

        # ------ 数据源组 ------
        source_group = QGroupBox("预测文件类型")
        source_layout = QVBoxLayout()
        source_type_layout = QHBoxLayout()

        icon_btn_style = """
            QPushButton {background: transparent; border:none; padding:0px;}
            QPushButton:checked, QPushButton:hover {background:#e8f5e9; border:2px solid #4CAF50;}
        """
        self.image_btn = QPushButton()
        self.image_btn.setCheckable(True)
        self.image_btn.setIcon(QIcon("button/图片1.png"))
        self.image_btn.setIconSize(QSize(48, 48))
        self.image_btn.setFixedSize(80, 80)
        self.image_btn.setStyleSheet(icon_btn_style)
        self.image_btn.clicked.connect(lambda: self.set_source_type("image"))

        self.video_btn = QPushButton()
        self.video_btn.setCheckable(True)
        self.video_btn.setIcon(QIcon("button/视频.png"))
        self.video_btn.setIconSize(QSize(48, 48))
        self.video_btn.setFixedSize(80, 80)
        self.video_btn.setStyleSheet(icon_btn_style)
        self.video_btn.clicked.connect(lambda: self.set_source_type("video"))

        self.camera_btn = QPushButton()
        self.camera_btn.setCheckable(True)
        self.camera_btn.setIcon(QIcon("button/摄像头.png"))
        self.camera_btn.setIconSize(QSize(48, 48))
        self.camera_btn.setFixedSize(80, 80)
        self.camera_btn.setStyleSheet(icon_btn_style)
        self.camera_btn.clicked.connect(lambda: self.set_source_type("camera"))

        source_type_layout.addWidget(self.image_btn)
        source_type_layout.addWidget(self.video_btn)
        source_type_layout.addWidget(self.camera_btn)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)

        path_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setReadOnly(True)
        self.path_input.setPlaceholderText("请输入文件路径或选择文件")
        self.select_source_btn = QPushButton("选择文件")
        self.select_source_btn.clicked.connect(self.select_source)
        self.select_source_btn.setEnabled(False)

        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.select_source_btn)

        self.source_path_label = QLabel("未选择文件")
        self.start_btn = QPushButton("开始检测")
        self.start_btn.setEnabled(False)
        self.pause_btn = QPushButton("暂停检测")
        self.pause_btn.setEnabled(False)
        self.resume_btn = QPushButton("恢复检测")
        self.resume_btn.setEnabled(False)
        self.stop_btn = QPushButton("停止检测")
        self.stop_btn.setEnabled(False)

        # 新增：逐帧播放按钮
        self.frame_by_frame_btn = QPushButton("逐帧播放")
        self.frame_by_frame_btn.setEnabled(False)
        self.frame_by_frame_btn.setCheckable(True)
        self.frame_by_frame_btn.clicked.connect(self.toggle_frame_by_frame)

        # 新增：下一帧按钮
        self.next_frame_btn = QPushButton("下一帧")
        self.next_frame_btn.setEnabled(False)
        self.next_frame_btn.clicked.connect(self.next_frame)

        self.start_btn.clicked.connect(self.start_detection)
        self.pause_btn.clicked.connect(self.pause_detection)
        self.resume_btn.clicked.connect(self.resume_detection)
        self.stop_btn.clicked.connect(self.stop_detection)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        # 按钮布局
        button_row1 = QHBoxLayout()
        button_row1.addWidget(self.start_btn)
        button_row1.addWidget(self.pause_btn)
        button_row1.addWidget(self.resume_btn)
        button_row1.addWidget(self.stop_btn)

        button_row2 = QHBoxLayout()
        button_row2.addWidget(self.frame_by_frame_btn)
        button_row2.addWidget(self.next_frame_btn)

        source_layout.addLayout(source_type_layout)
        source_layout.addWidget(line)
        source_layout.addLayout(path_layout)
        source_layout.addWidget(self.source_path_label)
        source_layout.addLayout(button_row1)
        source_layout.addLayout(button_row2)
        source_layout.addWidget(self.progress_bar)
        source_group.setLayout(source_layout)

        # ------ 参数组 ------
        param_group = QGroupBox("检测参数")
        param_layout = QVBoxLayout()
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(50)
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(0, 100)
        self.iou_slider.setValue(45)
        self.conf_slider.valueChanged.connect(self.update_conf_label)
        self.iou_slider.valueChanged.connect(self.update_iou_label)
        self.conf_label = QLabel(f"置信度阈值: {self.conf_slider.value() / 100:.2f}")
        self.iou_label = QLabel(f"IOU阈值: {self.iou_slider.value() / 100:.2f}")
        param_layout.addWidget(self.conf_label)
        param_layout.addWidget(self.conf_slider)
        param_layout.addWidget(self.iou_label)
        param_layout.addWidget(self.iou_slider)
        param_group.setLayout(param_layout)

        # ------ 表格 ------
        table_layout = QHBoxLayout()
        self.traffic_table = QTableWidget(0, 6)
        self.traffic_table.setHorizontalHeaderLabels(["交通标志", "置信度", "x1", "y1", "x2", "y2"])
        self.traffic_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.traffic_table.setAlternatingRowColors(True)

        self.person_car_table = QTableWidget(0, 6)
        self.person_car_table.setHorizontalHeaderLabels(["行人/车辆", "置信度", "x1", "y1", "x2", "y2"])
        self.person_car_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.person_car_table.setAlternatingRowColors(True)

        table_layout.addWidget(self.traffic_table)
        table_layout.addWidget(self.person_car_table)

        # ------ 右侧整体布局 ------
        right_layout.addWidget(model_group)
        right_layout.addWidget(source_group)
        right_layout.addWidget(param_group)
        right_layout.addLayout(table_layout)
        splitter.addWidget(right_widget)

        splitter.setSizes([700, 300])
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def add_transparency_tab(self):
        transparency_widget = QWidget()
        transparency_layout = QVBoxLayout(transparency_widget)

        # 背景透明度滑块
        self.bg_opacity_slider = QSlider(Qt.Horizontal)
        self.bg_opacity_slider.setRange(0, 100)
        self.bg_opacity_slider.setValue(100)
        self.bg_opacity_slider.valueChanged.connect(self.update_background_opacity)

        bg_label = QLabel("背景图透明度")
        bg_label.setAlignment(Qt.AlignCenter)

        # 控件透明度滑块
        self.widget_opacity_slider = QSlider(Qt.Horizontal)
        self.widget_opacity_slider.setRange(0, 100)
        self.widget_opacity_slider.setValue(100)
        self.widget_opacity_slider.valueChanged.connect(self.update_widget_opacity)

        widget_label = QLabel("界面控件背景透明度")
        widget_label.setAlignment(Qt.AlignCenter)

        # 右侧框透明度滑块
        self.right_box_opacity_slider = QSlider(Qt.Horizontal)
        self.right_box_opacity_slider.setRange(0, 100)
        self.right_box_opacity_slider.setValue(100)
        self.right_box_opacity_slider.valueChanged.connect(self.update_right_box_opacity)

        right_box_label = QLabel("右侧框透明度")
        right_box_label.setAlignment(Qt.AlignCenter)

        # 显示右侧框透明度数值的标签
        self.right_box_opacity_label = QLabel(f"右侧框透明度: {self.right_box_opacity_slider.value()}%")
        self.right_box_opacity_label.setAlignment(Qt.AlignCenter)

        transparency_layout.addWidget(bg_label)
        transparency_layout.addWidget(self.bg_opacity_slider)
        transparency_layout.addWidget(widget_label)
        transparency_layout.addWidget(self.widget_opacity_slider)
        transparency_layout.addWidget(right_box_label)
        transparency_layout.addWidget(self.right_box_opacity_slider)
        transparency_layout.addWidget(self.right_box_opacity_label)

        self.tabs.addTab(transparency_widget, "透明度设置")

    def add_statistics_tab(self):
        """添加类别统计标签页"""
        statistics_widget = QWidget()
        statistics_layout = QVBoxLayout(statistics_widget)

        # 创建Matplotlib画布
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)

        # 添加更新按钮
        self.update_stats_btn = QPushButton("更新统计图")
        self.update_stats_btn.clicked.connect(self.update_statistics)

        # 添加布局
        statistics_layout.addWidget(QLabel("检测类别统计:"))
        statistics_layout.addWidget(self.canvas)
        statistics_layout.addWidget(self.update_stats_btn)

        self.tabs.addTab(statistics_widget, "类别统计")

    def update_statistics(self):
        """更新类别统计图"""
        # 清空图形
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # 准备数据
        labels = []
        counts = []
        for label in self.class_labels:
            if self.class_counts[label] > 0:
                labels.append(label)
                counts.append(self.class_counts[label])

        if not labels:
            ax.text(0.5, 0.5, "暂无检测数据",
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=15)
            self.canvas.draw()
            return

        # 创建柱状图
        bars = ax.bar(labels, counts, color='#4CAF50')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}', ha='center', va='bottom')

        # 设置图表属性
        ax.set_title('检测类别统计')
        ax.set_xlabel('类别')
        ax.set_ylabel('检测次数')
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, linestyle='--', alpha=0.6)

        # 调整布局
        self.figure.tight_layout()

        # 重绘画布
        self.canvas.draw()

    def update_background_opacity(self, value):
        """设置背景图透明度"""
        self.setWindowOpacity(value / 100)

    def update_widget_opacity(self, value):
        """设置控件背景透明度（通过改样式实现）"""
        opacity_value = value / 100
        stylesheet = f"""
            QWidget {{ background-color: rgba(255, 255, 255, {opacity_value}); }}
            QTextEdit, QLineEdit, QLabel {{ background-color: rgba(255, 255, 255, {opacity_value}); }}
        """
        self.setStyleSheet(stylesheet)

    def update_right_box_opacity(self, value):
        """设置右侧框透明度"""
        opacity_value = value / 100
        stylesheet = f"""
            QGroupBox, QTableWidget {{
                background-color: rgba(255, 255, 255, {opacity_value});
            }}
        """
        self.setStyleSheet(stylesheet)
        self.right_box_opacity_label.setText(f"右侧框透明度: {value}%")

    def set_source_type(self, source_type):
        """设置数据源类型"""
        # 重置所有按钮状态
        self.image_btn.setChecked(False)
        self.video_btn.setChecked(False)
        self.camera_btn.setChecked(False)

        # 设置当前选中的按钮
        if source_type == 'image':
            self.image_btn.setChecked(True)
        elif source_type == 'video':
            self.video_btn.setChecked(True)
        elif source_type == 'camera':
            self.camera_btn.setChecked(True)

        self.current_source_type = source_type
        self.select_source_btn.setEnabled(source_type != 'camera')

        # 更新逐帧按钮状态
        self.frame_by_frame_btn.setEnabled(source_type == 'video')
        self.next_frame_btn.setEnabled(False)

        # 更新路径输入框状态
        if source_type == 'camera':
            self.path_input.setText("")
            self.path_input.setReadOnly(True)
            self.current_source = None
            self.source_path_label.setText("使用摄像头作为数据源")
        else:
            self.path_input.setReadOnly(False)

        self.check_start_enabled()

    def select_source(self):
        """选择数据源文件"""
        if self.current_source_type == 'image':
            file_path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "图像文件 (*.jpg *.jpeg *.png)")
        elif self.current_source_type == 'video':
            file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov)")
            if file_path:
                # 读取视频的第一帧
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        # 转换颜色空间
                        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb_image.shape
                        qimg = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
                        self.image_label.setPixmap(QPixmap.fromImage(qimg).scaled(
                            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    cap.release()
        else:
            file_path = None

        if file_path:
            self.current_source = file_path
            self.path_input.setText(file_path)
            display_path = file_path if len(file_path) < 30 else "..." + file_path[-30:]
            self.source_path_label.setText(f"数据源: {display_path}")
            self.check_start_enabled()

    def load_model(self, model_type):
        file_path, _ = QFileDialog.getOpenFileName(self, f"选择{model_type}模型", "", "模型文件 (*.pt)")
        if not file_path:
            return

        try:
            model = YOLO(file_path)
            model.to(self.device)  # 确保模型在正确的设备上

            if model_type == "traffic":
                self.traffic_model = model
                self.traffic_model_label.setText(f"已加载: {os.path.basename(file_path)}")
            else:
                self.person_car_model = model
                self.person_car_model_label.setText(f"已加载: {os.path.basename(file_path)}")

            self.check_start_enabled()
            QMessageBox.information(self, "成功", f"{model_type}模型加载成功")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")

    def check_start_enabled(self):
        """检查是否可以开始检测"""
        if self.traffic_model and self.person_car_model and self.current_source_type:
            if self.current_source_type == 'camera':
                self.start_btn.setEnabled(True)
            else:
                self.start_btn.setEnabled(bool(self.current_source))
        else:
            self.start_btn.setEnabled(False)

    def update_conf_label(self):
        conf = self.conf_slider.value() / 100
        self.conf_label.setText(f"置信度阈值: {conf:.2f}")

    def update_iou_label(self):
        iou = self.iou_slider.value() / 100
        self.iou_label.setText(f"IOU阈值: {iou:.2f}")

    def toggle_frame_by_frame(self):
        """切换逐帧播放模式"""
        if self.detection_thread is None:
            return

        if self.frame_by_frame_btn.isChecked():
            # 进入逐帧模式
            self.frame_by_frame_mode = True
            self.detection_thread.set_frame_by_frame(True)
            self.next_frame_btn.setEnabled(True)
            self.frame_by_frame_btn.setText("退出逐帧")
            self.detection_thread.pause()
        else:
            # 退出逐帧模式
            self.frame_by_frame_mode = False
            self.detection_thread.set_frame_by_frame(False)
            self.next_frame_btn.setEnabled(False)
            self.frame_by_frame_btn.setText("逐帧播放")
            self.detection_thread.resume()

    def next_frame(self):
        """处理下一帧请求"""
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.request_next_frame()

    def start_detection(self):
        self.image_label.clear()
        if not self.traffic_model or not self.person_car_model:
            QMessageBox.warning(self, "警告", "请先加载两个模型")
            return

        if self.current_source_type in ['image', 'video'] and not self.current_source:
            QMessageBox.warning(self, "警告", "请选择数据源")
            return

        # 重置状态

        self.stop_detection()
        self.auto_analysis_timer.start(1000)
        self.traffic_table.setRowCount(0)
        self.person_car_table.setRowCount(0)
        self.scene_description.clear()
        self.driving_advice.clear()
        self.answer_output.clear()
        self.detected_objects = {"traffic_signs": [], "persons_cars": []}
        self.log_messages = []  # 清空日志
        self.log_display.clear()
        self.current_fps = 0
        self.fps_label.setText("FPS: 0.0")
        self.gpu_label.setText("GPU: 0%")
        with self.det_lock:
            self.last_detections.clear()
        self.image_label.clear()  # 把旧画面擦掉

        # 重置统计
        self.class_counts = defaultdict(int)
        self.update_statistics()

        # 重置逐帧模式
        self.frame_by_frame_mode = False
        self.frame_by_frame_btn.setChecked(False)
        self.frame_by_frame_btn.setText("逐帧播放")
        self.next_frame_btn.setEnabled(False)

        # 获取用户输入的路径
        if self.current_source_type in ['image', 'video']:
            user_path = self.path_input.text().strip()
            if user_path:
                self.current_source = user_path
                display_path = user_path if len(user_path) < 30 else "..." + user_path[-30:]
                self.source_path_label.setText(f"数据源: {display_path}")

        # 启动检测线程
        self.detection_thread = DetectionThread(
            self.traffic_model,
            self.person_car_model,
            self.current_source,
            self.current_source_type,
            self.conf_slider.value() / 100,
            self.iou_slider.value() / 100,
            self.device
        )

        self.detection_thread.results_ready.connect(self.process_detection_results)
        self.detection_thread.progress_updated.connect(self.update_progress)
        self.detection_thread.finished.connect(self.detection_finished)
        self.detection_thread.frame_ready.connect(self.display_frame)
        self.detection_thread.error_occurred.connect(self.handle_error)
        self.detection_thread.fps_updated.connect(self.update_fps)
        self.detection_thread.gpu_usage_updated.connect(self.update_gpu_usage)

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # 视频模式下启用逐帧按钮
        if self.current_source_type == 'video':
            self.frame_by_frame_btn.setEnabled(True)

        self.detection_thread.start()
        self.detection_thread.resume()

    def display_frame(self, frame):
        if frame is None:
            return

        # 拿一份最新的检测结果
        with self.det_lock:
            dets = list(self.last_detections)  # 浅拷贝即可

        # 在当前帧上画所有框
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            label = d["label"]
            conf = d["confidence"]
            color = d["color"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 把画好框的帧显示出来
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def pause_detection(self):
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.pause()
            self.pause_btn.setEnabled(False)
            self.resume_btn.setEnabled(True)

    def resume_detection(self):
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.resume()
            self.pause_btn.setEnabled(True)
            self.resume_btn.setEnabled(False)

    def stop_detection(self):
        # 1) 请求后台线程停止
        if self.detection_thread and self.detection_thread.isRunning():
            # 断开信号
            try:
                self.detection_thread.results_ready.disconnect(self.process_detection_results)
                self.detection_thread.frame_ready.disconnect(self.display_frame)
                self.detection_thread.finished.disconnect(self.detection_finished)
            except TypeError:
                pass  # 已断开就会抛 TypeError，忽略
            # 发送停止指令
            self.detection_thread.stop()
            self.detection_thread.wait(1000)
        self.detection_thread = None  # 先把指针清 0

        # 2) 立即清空检测相关缓存，防止自动分析/叠框
        with self.det_lock:
            self.last_detections.clear()
        self.detected_objects = {"traffic_signs": [], "persons_cars": []}

        # 3) 关掉自动分析定时器
        self.auto_analysis_timer.stop()

        # 4) 停掉正在播放的语音并清空队列
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                break
        if self.is_speaking:
            try:
                self.engine.stop()
            except Exception:
                pass
        self.is_speaking = False

        # 5) 复位界面按钮
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.frame_by_frame_btn.setEnabled(False)
        self.next_frame_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_fps(self, fps):
        self.current_fps = fps
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def update_gpu_usage(self, usage):
        self.gpu_usage = usage
        self.gpu_label.setText(f"GPU: {usage:.1f}%")

    def detection_finished(self):
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.frame_by_frame_btn.setEnabled(False)
        self.next_frame_btn.setEnabled(False)
        if self.current_source_type == 'video':
            self.progress_bar.setValue(100)

    def process_detection_results(self, traffic_results, person_car_results, image):
        """处理检测结果并更新UI"""
        # 限制更新频率
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
        self.last_update_time = current_time
        detections_this_frame = []
        # 清空表格
        self.traffic_table.setRowCount(0)
        self.person_car_table.setRowCount(0)

        # 重置检测对象列表
        self.detected_objects = {"traffic_signs": [], "persons_cars": []}

        # 处理交通标志检测结果
        for result in traffic_results:
            boxes = result.boxes
            if boxes is None:
                continue

            boxes_data = boxes.xyxy.cpu().numpy()
            conf_data = boxes.conf.cpu().numpy()
            cls_data = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes_data)):
                x1, y1, x2, y2 = map(int, boxes_data[i])
                conf = conf_data[i]
                cls_id = cls_data[i]

                if 0 <= cls_id < len(TRAFFIC_CLASS_LABELS):
                    label = TRAFFIC_CLASS_LABELS[cls_id]

                    # 更新类别统计
                    self.class_counts[label] += 1
                    detections_this_frame.append(
                        {"bbox": (x1, y1, x2, y2),
                         "label": label,
                         "confidence": conf,
                         "color": (0, 255, 0)})

                    # 绘制边界框
                    # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # 添加到表格
                    row = self.traffic_table.rowCount()
                    self.traffic_table.insertRow(row)
                    self.traffic_table.setItem(row, 0, QTableWidgetItem(label))
                    self.traffic_table.setItem(row, 1, QTableWidgetItem(f"{conf:.2f}"))
                    self.traffic_table.setItem(row, 2, QTableWidgetItem(str(x1)))
                    self.traffic_table.setItem(row, 3, QTableWidgetItem(str(y1)))
                    self.traffic_table.setItem(row, 4, QTableWidgetItem(str(x2)))
                    self.traffic_table.setItem(row, 5, QTableWidgetItem(str(y2)))

                    # 保存检测到的对象信息
                    self.detected_objects["traffic_signs"].append({
                        "label": label,
                        "confidence": conf,
                        "coordinates": (x1, y1, x2, y2)
                    })

        # 处理行人车辆检测结果
        for result in person_car_results:
            boxes = result.boxes
            if boxes is None:
                continue

            boxes_data = boxes.xyxy.cpu().numpy()
            conf_data = boxes.conf.cpu().numpy()
            cls_data = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes_data)):
                x1, y1, x2, y2 = map(int, boxes_data[i])
                conf = conf_data[i]
                cls_id = cls_data[i]

                if 0 <= cls_id < len(PERSON_CAR_CLASS_LABELS):
                    label = PERSON_CAR_CLASS_LABELS[cls_id]

                    # 更新类别统计
                    self.class_counts[label] += 1
                    detections_this_frame.append(
                        {"bbox": (x1, y1, x2, y2),
                         "label": label,
                         "confidence": conf,
                         "color": (0, 0, 255)})

                    # 绘制边界框
                    # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # 添加到表格
                    row = self.person_car_table.rowCount()
                    self.person_car_table.insertRow(row)
                    self.person_car_table.setItem(row, 0, QTableWidgetItem(label))
                    self.person_car_table.setItem(row, 1, QTableWidgetItem(f"{conf:.2f}"))
                    self.person_car_table.setItem(row, 2, QTableWidgetItem(str(x1)))
                    self.person_car_table.setItem(row, 3, QTableWidgetItem(str(y1)))
                    self.person_car_table.setItem(row, 4, QTableWidgetItem(str(x2)))
                    self.person_car_table.setItem(row, 5, QTableWidgetItem(str(y2)))

                    # 保存检测到的对象信息
                    self.detected_objects["persons_cars"].append({
                        "label": label,
                        "confidence": conf,
                        "coordinates": (x1, y1, x2, y2)
                    })
        with self.det_lock:
            self.last_detections = detections_this_frame
        # 显示处理后的图像
        # rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # h, w, ch = rgb_image.shape
        # qimg = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        # self.image_label.setPixmap(QPixmap.fromImage(qimg).scaled(
        #     self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def auto_analyze_scene(self):
        # 原来的三条判断保持
        now = time.time()
        if (not self.detected_objects["traffic_signs"] and
            not self.detected_objects["persons_cars"]) or \
                (now - self.last_analysis_time < self.analysis_interval):
            return
        if self.ai_busy:  # 正在跑别的 AI 请求
            return
        self.ai_busy = True
        self.last_analysis_time = now

        prompt_scene = self.build_scene_prompt()
        prompt_advice = self.build_advice_prompt(prompt_scene)

        worker = AIWorker(self.client,
                          [prompt_scene, prompt_advice],
                          task="scene",
                          llm_model=self.llm_model)
        worker.signals.result_ready.connect(self.on_ai_result)
        worker.signals.error.connect(self.handle_error)
        worker.signals.finished.connect(lambda: setattr(self, "ai_busy", False))
        self.ai_pool.start(worker)

    def build_scene_prompt(self):
        """完全搬自你原来的 generate_scene_description，但结尾不再请求 AI，直接 return prompt 字符串"""
        prompt = "根据以下检测结果，用自然语言描述当前交通场景：\n"
        if self.detected_objects["traffic_signs"]:
            prompt += "检测到的交通标志：\n"
            for sign in self.detected_objects["traffic_signs"]:
                prompt += f"- {sign['label']} (置信度: {sign['confidence']:.2f})\n"
        if self.detected_objects["persons_cars"]:
            prompt += "检测到的行人和车辆：\n"
            for obj in self.detected_objects["persons_cars"]:
                prompt += f"- {obj['label']} (置信度: {obj['confidence']:.2f})\n"
        prompt += "请用简洁明了的中文描述当前场景，如：'前方有行人穿越马路，路口有红灯，请减速停车。'"
        return prompt

    def build_advice_prompt(self, scene_desc_prompt):
        return f"根据以下交通场景描述，给出具体的驾驶建议：\n{scene_desc_prompt}\n" \
               "请输出简洁、实用的驾驶建议，如：'建议左转绕行'、'前方拥堵，请减速'等。"
    def speak_current_advice(self):
        """语音播报当前驾驶建议"""
        advice = self.driving_advice.toPlainText().strip()
        if advice:
            self.speak(advice)

    def speak(self, text):
        """将文本添加到语音队列"""
        self.speech_queue.put(text)

    def speech_worker(self):
        """语音处理线程"""
        while True:
            if not self.speech_queue.empty():
                text = self.speech_queue.get()
                try:
                    self.is_speaking = True
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception as e:
                    print(f"语音播报错误: {e}")
                finally:
                    self.is_speaking = False
            time.sleep(0.1)

    def answer_question(self):
        question = self.question_input.toPlainText().strip()
        if not question:
            QMessageBox.warning(self, "警告", "请输入问题")
            return
        if self.ai_busy:
            QMessageBox.information(self, "提示", "AI 正忙，请稍候")
            return
        self.ai_busy = True
        self.answer_output.setText("思考中，请稍候...")

        prompt = f"用户的问题是：{question}\n" \
                 "根据以下检测结果回答问题：\n"
        if self.detected_objects["traffic_signs"]:
            prompt += "交通标志：\n"
            for sign in self.detected_objects["traffic_signs"]:
                prompt += f"- {sign['label']}\n"
        if self.detected_objects["persons_cars"]:
            prompt += "行人和车辆：\n"
            for obj in self.detected_objects["persons_cars"]:
                prompt += f"- {obj['label']}\n"
        prompt += "请基于以上信息，用中文简洁明了地回答用户的问题。"

        worker = AIWorker(self.client, [prompt], task="qa", llm_model=self.llm_model)
        worker.signals.result_ready.connect(self.on_ai_result)
        worker.signals.error.connect(self.handle_error)
        worker.signals.finished.connect(lambda: setattr(self, "ai_busy", False))
        self.ai_pool.start(worker)

    def generate_road_report(self):
        if not self.log_messages:
            QMessageBox.warning(self, "警告", "建议日志为空，无法生成报告。")
            return
        if self.ai_busy:
            QMessageBox.information(self, "提示", "AI 正忙，请稍候")
            return
        self.ai_busy = True
        self.report_display.setText("正在生成报告，请稍候...")

        log_text = "\n".join(self.log_messages)
        prompt = f"根据以下建议日志生成一篇道路建议报告：\n{log_text}\n" \
                 "报告应包含对整体交通状况的总结、建议的重点和对未来驾驶的展望等内容。"

        worker = AIWorker(self.client, [prompt], task="report", llm_model=self.llm_model)
        worker.signals.result_ready.connect(self.on_ai_result)
        worker.signals.error.connect(self.handle_error)
        worker.signals.finished.connect(lambda: setattr(self, "ai_busy", False))
        self.ai_pool.start(worker)

    def on_ai_result(self, payload: dict):
        task = payload["task"]
        data = payload["data"]  # list[str]

        if task == "scene":
            desc, advice = data
            self.scene_description.setText(desc)
            self.driving_advice.setText(advice)
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            self.log_messages.append(f"[{ts}] {advice}")
            self.log_display.setText("\n".join(self.log_messages))
            self.speak(advice)  # 有语音就播

        elif task == "qa":
            self.answer_output.setText(data[0])

        elif task == "report":
            self.report_display.setText(data[0])

    def handle_error(self, error_msg):
        """处理检测线程中的错误"""
        QMessageBox.critical(self, "检测错误", f"发生错误: {error_msg}")
        self.stop_detection()

    def closeEvent(self, event):
        """关闭窗口时停止检测线程"""
        self.stop_detection()
        # 等待线程完全停止
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.wait(1000)
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 确保中文显示正常
    font = app.font()
    font.setFamily("SimHei")
    app.setFont(font)
    window = TrafficSignApp()
    window.show()
    sys.exit(app.exec_())