import sys
import collections
import sqlite3
import configparser
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout,
    QWidget, QFileDialog, QTextEdit, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QSlider, QHBoxLayout, QGroupBox, QSizePolicy,
    QTabWidget, QComboBox, QCheckBox, QMessageBox, QMenuBar, QMenu, QAction, QDialog
)
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QWaitCondition, QTimer
import cv2
import torch
from ultralytics import YOLO
import time
from openai import OpenAI
from googletrans import Translator
import warnings
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)

# 在导入matplotlib后添加字体配置
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- Matplotlib 中文显示设置 ---
# 查找系统中可用的中文字体
import matplotlib.font_manager as fm

font_paths = [
    'C:/Windows/Fonts/simhei.ttf', # Windows
    'C:/Windows/Fonts/msyh.ttc' # Windows 微软雅黑
]

found_font_path = None
for fp in font_paths:
    if os.path.exists(fp):
        found_font_path = fp
        break

if found_font_path:
    plt.rcParams['font.sans-serif'] = [fm.FontProperties(fname=found_font_path).get_name()]
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题
    print(f"Matplotlib 已设置为使用字体: {found_font_path}")
else:
    print("警告: 未找到合适的中文字体，图表中的中文可能无法正常显示。")
    print("请确保您的系统中安装了上述路径中的中文字体，或手动指定一个正确的字体路径。")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial'] # 备用英文字体
    plt.rcParams['axes.unicode_minus'] = False
# --- Matplotlib 中文显示设置结束 ---

# --- 配置文件 ---
config = configparser.ConfigParser()
config.read('config.ini')

# --- 数据库设置 ---
DB_FILE = 'traffic_signs.db'


def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  source TEXT,
                  class_name TEXT,
                  confidence REAL,
                  x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS settings
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  conf_threshold REAL,
                  iou_threshold REAL,
                  language TEXT,
                  chart_type TEXT)''')
    conn.commit()
    conn.close()


init_db()


# --- 交通标志检测器 ---
class TrafficSignDetector:
    def __init__(self, weights_path='D:\Eastsoft\code\Python\day12\new_best.pt'):
        print(f"正在从 {weights_path} 加载模型...")
        try:
            self.model = YOLO(weights_path)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)

            # 尝试使用OpenCV GPU加速
            try:
                cv2.cuda.setDevice(0)
                print("已启用OpenCV GPU加速")
            except:
                print("无法启用OpenCV GPU加速，使用CPU模式")

            print(f"YOLOv11 模型已成功加载到 {self.device}")
        except Exception as e:
            print(f"加载模型时出错: {e}")
            sys.exit(1)

    def detect(self, source_frame, imgsz=640, conf_thres=0.25, iou_thres=0.7):
        results = self.model.predict(
            source=source_frame,
            imgsz=imgsz,
            conf=conf_thres,
            iou=iou_thres,
            verbose=False
        )
        return results


# --- 视频处理线程 ---
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    detection_data_signal = pyqtSignal(list)
    detection_class_names_signal = pyqtSignal(list)
    ar_display_changed = pyqtSignal(bool)

    def __init__(self, source_path, detector_model, conf_thres, iou_thres):
        super().__init__()
        self._run_flag = True
        self.source_path = source_path
        self.detector = detector_model
        self.imgsz = 640
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.pause_mutex = QMutex()
        self.pause_condition = QWaitCondition()
        self._paused = False
        self._step_mode = False
        self._show_ar = True
        self._show_fps = True
        self.frame_count = 0
        self.start_time = time.time()

    def update_thresholds(self, conf_thres, iou_thres):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def set_ar_display(self, show):
        self._show_ar = show
        self.ar_display_changed.emit(show)

    def set_step_mode(self, enabled):
        self._step_mode = enabled
        if enabled:
            self.pause()

    def step_frame(self):
        if self._step_mode and self._paused:
            self.unpause()
            QTimer.singleShot(100, self.pause)

    def run(self):
        if isinstance(self.source_path, int):
            cap = cv2.VideoCapture(self.source_path)
        else:
            cap = cv2.VideoCapture(self.source_path)

        if not cap.isOpened():
            print(f"错误：无法打开视频源 {self.source_path}")
            self._run_flag = False
            return

        while self._run_flag and cap.isOpened():
            self.pause_mutex.lock()
            if self._paused:
                self.pause_condition.wait(self.pause_mutex)
            self.pause_mutex.unlock()

            ret, frame = cap.read()
            if ret:
                self.frame_count += 1
                results = self.detector.detect(frame, self.imgsz, self.conf_thres, self.iou_thres)

                # 添加AR效果
                if self._show_ar:
                    annotated_frame = self._add_ar_effects(frame.copy(), results)
                else:
                    annotated_frame = results[0].plot()

                # 添加FPS显示
                if self._show_fps:
                    elapsed_time = time.time() - self.start_time
                    fps = self.frame_count / elapsed_time
                    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                detected_info = []
                detected_class_names = []
                if hasattr(results[0], 'boxes') and results[0].boxes:
                    for i, r_box in enumerate(results[0].boxes):
                        class_id = int(r_box.cls)
                        conf = float(r_box.conf)
                        if class_id < len(self.detector.model.names):
                            class_name = self.detector.model.names[class_id]
                            x1, y1, x2, y2 = map(int, r_box.xyxy[0].tolist())
                            detected_info.append({
                                'id': i + 1,
                                'class': class_name,
                                'confidence': f"{conf:.2f}",
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                            })
                            detected_class_names.append(class_name)

                self.detection_data_signal.emit(detected_info)
                self.detection_class_names_signal.emit(detected_class_names)

                # 转换为Qt图像并发射信号
                rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.change_pixmap_signal.emit(convert_to_qt_format)
            else:
                break

        cap.release()
        print("视频流已结束")

    def _add_ar_effects(self, frame, results):
        if not results[0].boxes:
            return frame

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            class_id = int(box.cls)
            class_name = self.detector.model.names[class_id]

            # 置信度颜色编码
            color = (0, 255, 0) if conf > 0.7 else (0, 255, 255) if conf > 0.4 else (0, 0, 255)

            # AR边框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 文字标签
            label = f"{class_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 中心点标记
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 5, color, -1)

            # 危险标志的警示效果
            if "危险" in class_name or "stop" in class_name.lower():
                x_mid = (x1 + x2) // 2
                y_mid = (y1 + y2) // 2
                radius = max((x2 - x1) // 2, (y2 - y1) // 2)
                cv2.circle(frame, (x_mid, y_mid), radius, (0, 0, 255), 2)
                cv2.line(frame, (x_mid - radius, y_mid), (x_mid + radius, y_mid), (0, 0, 255), 2)
                cv2.line(frame, (x_mid, y_mid - radius), (x_mid, y_mid + radius), (0, 0, 255), 2)

        return frame

    def stop(self):
        self._run_flag = False
        self.unpause()
        self.wait()

    def pause(self):
        self.pause_mutex.lock()
        self._paused = True
        self.pause_mutex.unlock()

    def unpause(self):
        self.pause_mutex.lock()
        self._paused = False
        self.pause_mutex.unlock()
        self.pause_condition.wakeAll()
# --- 语音播报线程 ---
class VoiceThread(QThread):
    voice_finished_signal = pyqtSignal(str)  # 播报完成信号
    voice_error_signal = pyqtSignal(str)      # 错误信号

    def __init__(self, text, language):
        super().__init__()
        self.text = text
        self.language = language
        def __init__(self, text, language):
            super().__init__()
            self.text = text
            self.language = language

        def run(self):
            try:
                import pyttsx3
                engine = pyttsx3.init()
                voices = engine.getProperty('voices')

                # 打印所有可用语音（用于调试）
                print("可用语音列表:")
                for voice in voices:
                    print(f"ID: {voice.id}, 名称: {voice.name}, 语言: {voice.languages}")

                # 根据语言选择发音人
                if self.language == "en":
                    english_voice = None
                    # 优先选择美式或英式英语
                    for voice in voices:
                        if 'english-us' in voice.id.lower() or 'english (united states)' in voice.id.lower():
                            english_voice = voice
                            break
                        elif 'english-uk' in voice.id.lower() or 'english (united kingdom)' in voice.id.lower():
                            english_voice = voice
                            break

                    # 如果没找到美式/英式，尝试其他英文语音
                    if not english_voice:
                        for voice in voices:
                            if 'english' in voice.id.lower() or 'en-us' in voice.id.lower() or 'en-gb' in voice.id.lower():
                                english_voice = voice
                                break

                    if english_voice:
                        engine.setProperty('voice', english_voice.id)
                        print(f"已设置英文语音: {english_voice.name}")
                    else:
                        print("警告: 未找到英文语音，使用默认语音")

                    engine.setProperty('rate', 180)  # 英文语速
                else:
                    chinese_voice = None
                    # 优先选择普通话
                    for voice in voices:
                        if 'chinese (mandarin)' in voice.id.lower() or 'zh-cn' in voice.id.lower():
                            chinese_voice = voice
                            break

                    if chinese_voice:
                        engine.setProperty('voice', chinese_voice.id)
                        print(f"已设置中文语音: {chinese_voice.name}")
                    else:
                        print("警告: 未找到中文语音，使用默认语音")

                    engine.setProperty('rate', 150)  # 中文语速

                # 语音播报
                engine.say(self.text)
                engine.runAndWait()

                # 发送完成信号
                finish_msg = "Voice broadcast completed" if self.language == "en" else "语音播报完成"
                self.voice_finished_signal.emit(finish_msg)

            except ImportError:
                error_msg = "语音播报失败: 未安装pyttsx3库，请运行 'pip install pyttsx3'"
                self.voice_error_signal.emit(error_msg)
            except Exception as e:
                error_msg = f"语音播报失败: {str(e)}"
                self.voice_error_signal.emit(error_msg)
# --- LLM线程 ---
class LLMThread(QThread):
    llm_advice_signal = pyqtSignal(str)  # 发送AI建议到主界面
    llm_error_signal = pyqtSignal(str)  # 发送错误信息到主界面
    request_llm_signal = pyqtSignal(list)  # 请求LLM生成建议

    def __init__(self):
        super().__init__()
        self.llm_client = None
        self._init_llm_client()
        self.translator = Translator()  # 假设使用googletrans或其他翻译库
        self.current_language = "zh"  # 默认语言为中文
        self._run_flag = True  # 线程运行标志
        self.llm_call_interval = 10  # LLM调用间隔(秒)
        self.last_llm_call_time = 0  # 上次调用时间
        self.prompt_queue = collections.deque()  # LLM请求队列
        self.history = []  # 对话历史

    def _init_llm_client(self):
        """初始化LLM客户端"""
        try:
            # 使用本地API配置
            self.local_api_base_url = "https://api.siliconflow.cn"
            self.local_api_key = "sk-lodvqzdtgmuxmtaeyxoxyyhoxivmrqdyermakelixedfwmig"

            self.llm_client = OpenAI(
                api_key=self.local_api_key,
                base_url=self.local_api_base_url
            )
            print("本地LLM客户端初始化成功")
        except Exception as e:
            print(f"初始化本地LLM客户端失败: {e}")
            self.llm_client = None
    def stop(self):
        """停止线程运行"""
        self._run_flag = False
        self.wait()

    def set_language(self, lang):
        """设置语言"""
        self.current_language = lang
        print(f"LLM线程语言已切换为: {lang}")

    def enqueue_prompt(self, prompt):
        """将提示添加到处理队列"""
        self.prompt_queue.append(prompt)

    def _process_llm_request(self, prompt):
        """处理LLM请求"""
        try:
            # 构建对话历史
            messages = [
                {"role": "system", "content": "你是一个智能驾驶助手，根据交通标志提供驾驶建议。"},
                {"role": "user", "content": prompt}
            ]

            # 调用LLM生成回复
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",  # 使用你本地服务支持的模型
                messages=messages,
                temperature=0.7
            )

            # 获取回复文本
            advice = response.choices[0].message.content.strip()

            # 如果需要翻译（非中文环境）
            if self.current_language != "zh":
                try:
                    # 翻译回复
                    translation = self.translator.translate(
                        advice,
                        src="zh",
                        dest=self.current_language
                    )
                    advice = translation.text
                except Exception as e:
                    print(f"翻译失败: {e}")
                    # 翻译失败时保留原文
                    advice += f"\n\n(Translation failed: {str(e)})"

            # 发送建议到主界面
            self.llm_advice_signal.emit(advice)

            # 更新历史
            self.history.append({"user": prompt, "assistant": advice})

        except Exception as e:
            error_msg = f"获取LLM建议失败: {str(e)}"
            self.llm_error_signal.emit(error_msg)
            print(error_msg)

    def _send_advice(self, advice_text):
        """发送建议文本（用于语言切换时重新翻译）"""
        if self.current_language != "zh":
            try:
                # 翻译文本
                translation = self.translator.translate(
                    advice_text,
                    src="zh",
                    dest=self.current_language
                )
                advice_text = translation.text
            except Exception as e:
                print(f"翻译失败: {e}")
                # 翻译失败时保留原文
                advice_text += f"\n\n(Translation failed: {str(e)})"

        # 发送到主界面
        self.llm_advice_signal.emit(advice_text)

    def _update_ui_texts(self, lang):
        """根据目标语言更新所有UI文本"""
        # 按钮文本
        self.upload_image_btn.setText("Upload Image" if lang == "en" else "上传图片")
        self.upload_video_btn.setText("Upload Video" if lang == "en" else "上传视频")
        self.webcam_btn.setText("Webcam" if lang == "en" else "摄像头")
        self.stop_stream_btn.setText("Stop" if lang == "en" else "停止")
        self.replay_btn.setText("History" if lang == "en" else "历史回放")
        self.export_btn.setText("Export Results" if lang == "en" else "导出检测结果")
        self.clear_stats_btn.setText("Clear Stats" if lang == "en" else "清除统计数据")
        self.voice_advice_btn.setText("Voice Broadcast" if lang == "en" else "语音播报")

        # 面板标题
        self.findChild(QGroupBox, "控制面板").setTitle("Control Panel" if lang == "en" else "控制面板")
        self.findChild(QGroupBox, "检测参数").setTitle("Detection Parameters" if lang == "en" else "检测参数")
        self.findChild(QGroupBox, "实时视图").setTitle("Live View" if lang == "en" else "实时视图")
        self.findChild(QGroupBox, "AI驾驶建议").setTitle("AI Driving Advice" if lang == "en" else "AI驾驶建议")

        # 表格标题
        self.results_table.setHorizontalHeaderLabels(
            ['ID', 'Class', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'] if lang == "en" else
            ['ID', '类别', '置信度', 'X1', 'Y1', 'X2', 'Y2']
        )

        # 状态栏初始文本
        self.status_bar.showMessage("System ready" if lang == "en" else "系统就绪")
    def run(self):
        while self._run_flag:
            current_time = time.time()
            if self.prompt_queue and (current_time - self.last_llm_call_time >= self.llm_call_interval):
                detected_class_names = self.prompt_queue.popleft()

                if not detected_class_names:
                    self._send_advice("未检测到交通标志。请继续安全驾驶，并留意周围环境。")
                    self.last_llm_call_time = current_time
                    time.sleep(1)
                    continue

                self.last_llm_call_time = current_time
                self.history.extend(detected_class_names)

                # 上下文感知提示
                context = ", ".join(self.history[-5:])  # 使用最近5个检测作为上下文
                prompt = (f"你是一位专业的驾驶安全专家，根据以下检测到的交通标志序列，"
                          f"提供具体的驾驶建议和注意事项。请以简洁明了的中文给出建议：\n\n"
                          f"当前检测标志: {', '.join(detected_class_names)}\n"
                          f"近期标志序列: {context}\n\n"
                          f"建议应关注:")

                try:
                    if self.llm_client:
                        response = self.llm_client.chat.completions.create(
                            model=config.get('LLM', 'model', fallback="deepseek-ai/DeepSeek-V3"),
                            messages=[
                                {"role": "system",
                                 "content": "你是一位专业的驾驶安全专家，根据交通标志提供具体的驾驶建议和注意事项。"},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.7,
                        )
                        advice = response.choices[0].message.content
                        self._send_advice(advice)
                    else:
                        self.llm_error_signal.emit("LLM客户端未初始化，请检查API配置。")
                except Exception as e:
                    error_msg = f"获取AI建议时出错: {e}\n请检查网络连接或API Key。"
                    self.llm_error_signal.emit(error_msg)
            else:
                time.sleep(0.5)

    def _send_advice(self, advice_text):
        if self.current_language != "zh":
            try:
                advice_text = self.translator.translate(advice_text, dest=self.current_language).text
            except:
                pass
        self.llm_advice_signal.emit(advice_text)

    def request_advice(self, detected_class_names):
        self.prompt_queue.append(detected_class_names)

    def stop(self):
        self._run_flag = False
        self.wait()


# --- 主窗口 ---
class TrafficSignApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "高级交通标志识别与驾驶建议系统"
        self.setGeometry(100, 100, 1600, 1000)
        self.setWindowTitle(self.title)

        # 初始化模型
        weights_path = config.get('Model', 'weights_path', fallback='runs/V11train/exp/weights/best.pt')
        self.detector = TrafficSignDetector(weights_path)

        # 初始化设置
        self.current_conf_thres = 0.25
        self.current_iou_thres = 0.7
        self.current_language = "zh"
        self.chart_type = "bar"
        self.show_ar = True
        self.show_fps = True
        self.is_dark_mode = False

        # 统计数据
        self.total_detection_counts = collections.defaultdict(int)
        self.detection_history = []

        # 线程
        self.video_thread = None
        self.llm_thread = LLMThread()

        self.initUI()
        self._start_llm_thread()
        self._load_settings()

    def initUI(self):
        # 主部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # 菜单栏
        self._create_menu_bar()

        # 控制面板
        control_panel = QGroupBox("控制面板")
        control_layout = QHBoxLayout()

        # 数据源按钮
        self.upload_image_btn = QPushButton("上传图片", self)
        self.upload_image_btn.clicked.connect(self.upload_image)
        control_layout.addWidget(self.upload_image_btn)

        self.upload_video_btn = QPushButton("上传视频", self)
        self.upload_video_btn.clicked.connect(self.upload_video)
        control_layout.addWidget(self.upload_video_btn)

        self.webcam_btn = QPushButton("摄像头", self)
        self.webcam_btn.clicked.connect(self.start_webcam)
        control_layout.addWidget(self.webcam_btn)

        self.stop_stream_btn = QPushButton("停止", self)
        self.stop_stream_btn.clicked.connect(self.stop_stream)
        self.stop_stream_btn.setEnabled(False)
        control_layout.addWidget(self.stop_stream_btn)

        # 回放按钮
        self.replay_btn = QPushButton("历史回放")
        self.replay_btn.clicked.connect(self.show_history_dialog)
        control_layout.addWidget(self.replay_btn)

        control_panel.setLayout(control_layout)
        main_layout.addWidget(control_panel)

        # 阈值控制面板
        threshold_panel = QGroupBox("检测参数")
        threshold_layout = QHBoxLayout()

        # 置信度控制
        conf_label = QLabel("置信度:")
        threshold_layout.addWidget(conf_label)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(int(self.current_conf_thres * 100))
        self.conf_slider.valueChanged.connect(self.update_conf_thres_value)
        threshold_layout.addWidget(self.conf_slider)

        self.conf_value_label = QLabel(f"{self.current_conf_thres:.2f}")
        self.conf_value_label.setFixedWidth(40)
        threshold_layout.addWidget(self.conf_value_label)

        # IOU控制
        iou_label = QLabel("IoU:")
        threshold_layout.addWidget(iou_label)

        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(0, 100)
        self.iou_slider.setValue(int(self.current_iou_thres * 100))
        self.iou_slider.valueChanged.connect(self.update_iou_thres_value)
        threshold_layout.addWidget(self.iou_slider)

        self.iou_value_label = QLabel(f"{self.current_iou_thres:.2f}")
        self.iou_value_label.setFixedWidth(40)
        threshold_layout.addWidget(self.iou_value_label)

        # 显示选项
        self.ar_checkbox = QCheckBox("AR显示")
        self.ar_checkbox.setChecked(self.show_ar)
        self.ar_checkbox.toggled.connect(self.toggle_ar_display)
        threshold_layout.addWidget(self.ar_checkbox)

        self.fps_checkbox = QCheckBox("显示FPS")
        self.fps_checkbox.setChecked(self.show_fps)
        self.fps_checkbox.toggled.connect(self.toggle_fps_display)
        threshold_layout.addWidget(self.fps_checkbox)

        threshold_panel.setLayout(threshold_layout)
        main_layout.addWidget(threshold_panel)

        # 主要内容区域
        content_layout = QHBoxLayout()

        # 左侧：视频显示区域
        video_panel = QGroupBox("实时视图")
        video_layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid black; background-color: lightgray;")
        self.image_label.setMinimumSize(640, 480)
        video_layout.addWidget(self.image_label)

        # 视频控制按钮
        control_btn_layout = QHBoxLayout()
        self.pause_btn = QPushButton("暂停", self)
        self.pause_btn.clicked.connect(self.toggle_pause)
        control_btn_layout.addWidget(self.pause_btn)

        self.step_btn = QPushButton("单帧步进", self)
        self.step_btn.clicked.connect(self.step_frame)
        self.step_btn.setEnabled(False)
        control_btn_layout.addWidget(self.step_btn)

        self.step_mode_check = QCheckBox("步进模式")
        self.step_mode_check.toggled.connect(self.toggle_step_mode)
        control_btn_layout.addWidget(self.step_mode_check)

        video_layout.addLayout(control_btn_layout)
        video_panel.setLayout(video_layout)
        content_layout.addWidget(video_panel)

        # 右侧：信息面板（标签页）
        self.info_tabs = QTabWidget()
        self.info_tabs.setTabPosition(QTabWidget.West)

        # 检测结果标签页
        detection_tab = QWidget()
        detection_layout = QVBoxLayout(detection_tab)

        self.results_table = QTableWidget(self)
        self.results_table.setColumnCount(7)
        self.results_table.setHorizontalHeaderLabels(['ID', '类别', '置信度', 'X1', 'Y1', 'X2', 'Y2'])
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSortingEnabled(True)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setMinimumWidth(500)
        detection_layout.addWidget(self.results_table)

        self.export_btn = QPushButton("导出检测结果")
        self.export_btn.clicked.connect(self.export_detections)
        detection_layout.addWidget(self.export_btn)

        detection_tab.setLayout(detection_layout)
        self.info_tabs.addTab(detection_tab, "检测结果")

        # 统计图表标签页
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)

        self.create_chart_controls(stats_layout)

        self.figure, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        stats_layout.addWidget(self.canvas)

        self.clear_stats_btn = QPushButton("清除统计数据")
        self.clear_stats_btn.clicked.connect(self.clear_statistics)
        stats_layout.addWidget(self.clear_stats_btn)

        stats_tab.setLayout(stats_layout)
        self.info_tabs.addTab(stats_tab, "统计图表")

        # 标志知识库标签页
        knowledge_tab = QWidget()
        knowledge_layout = QVBoxLayout(knowledge_tab)

        self.sign_search = QTextEdit()
        self.sign_search.setPlaceholderText("搜索交通标志...")
        self.sign_search.textChanged.connect(self.search_signs)
        knowledge_layout.addWidget(self.sign_search)

        self.sign_info_display = QTextEdit()
        self.sign_info_display.setReadOnly(True)
        self.sign_info_display.setStyleSheet("background-color: #f8f9fa;")
        knowledge_layout.addWidget(self.sign_info_display)

        knowledge_tab.setLayout(knowledge_layout)
        self.info_tabs.addTab(knowledge_tab, "标志知识库")

        content_layout.addWidget(self.info_tabs)
        main_layout.addLayout(content_layout)

        # AI建议面板
        self.create_advice_panel(main_layout)

        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("系统就绪")

        # 初始化图表
        self.update_statistics_chart()

    def _create_menu_bar(self):
        menu_bar = self.menuBar()

        # 文件菜单
        file_menu = menu_bar.addMenu("文件")

        export_action = QAction("导出设置", self)
        export_action.triggered.connect(self.export_settings)
        file_menu.addAction(export_action)

        import_action = QAction("导入设置", self)
        import_action.triggered.connect(self.import_settings)
        file_menu.addAction(import_action)

        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 视图菜单
        view_menu = menu_bar.addMenu("视图")

        ar_action = QAction("增强现实(AR)显示", self, checkable=True)
        ar_action.setChecked(self.show_ar)
        ar_action.triggered.connect(self.toggle_ar_display)
        view_menu.addAction(ar_action)

        dark_mode_action = QAction("深色模式", self, checkable=True)
        dark_mode_action.triggered.connect(self.toggle_dark_mode)
        view_menu.addAction(dark_mode_action)

        # 语言菜单
        lang_menu = menu_bar.addMenu("Language" if self.current_language == "en" else "语言")

        zh_action = QAction("中文", self, checkable=True)
        zh_action.setChecked(self.current_language == "zh")
        zh_action.triggered.connect(lambda: self.set_language("zh"))
        lang_menu.addAction(zh_action)

        en_action = QAction("English", self, checkable=True)
        en_action.setChecked(self.current_language == "en")
        en_action.triggered.connect(lambda: self.set_language("en"))
        lang_menu.addAction(en_action)

        # 帮助菜单
        help_menu = menu_bar.addMenu("帮助")

        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_chart_controls(self, layout):
        chart_control_layout = QHBoxLayout()

        chart_type_label = QLabel("图表类型:")
        chart_control_layout.addWidget(chart_type_label)

        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(["柱状图", "饼图", "折线图"])
        self.chart_type_combo.setCurrentText(
            {"bar": "柱状图", "pie": "饼图", "line": "折线图"}.get(self.chart_type, "柱状图"))
        self.chart_type_combo.currentTextChanged.connect(self.change_chart_type)
        chart_control_layout.addWidget(self.chart_type_combo)

        self.time_range_combo = QComboBox()
        self.time_range_combo.addItems(["全部", "最近10分钟", "最近1小时", "最近24小时"])
        self.time_range_combo.currentTextChanged.connect(self.update_statistics_chart)
        chart_control_layout.addWidget(self.time_range_combo)

        layout.addLayout(chart_control_layout)

    def create_advice_panel(self, layout):
        advice_panel = QGroupBox("AI驾驶建议")
        advice_layout = QVBoxLayout()

        self.advice_text = QTextEdit()
        self.advice_text.setReadOnly(True)
        self.advice_text.setStyleSheet("background-color: #e8f4ea; font-size: 14px;")
        self.advice_text.setMinimumHeight(100)
        advice_layout.addWidget(self.advice_text)

        advice_control_layout = QHBoxLayout()

        self.voice_advice_btn = QPushButton("语音播报")
        self.voice_advice_btn.clicked.connect(self.text_to_speech)
        advice_control_layout.addWidget(self.voice_advice_btn)

        self.advice_lang_combo = QComboBox()
        self.advice_lang_combo.addItems(["中文", "English"])
        self.advice_lang_combo.setCurrentText("中文" if self.current_language == "zh" else "English")
        self.advice_lang_combo.currentTextChanged.connect(self.change_advice_language)
        advice_control_layout.addWidget(self.advice_lang_combo)

        advice_layout.addLayout(advice_control_layout)
        advice_panel.setLayout(advice_layout)
        layout.addWidget(advice_panel)

    def _start_llm_thread(self):
        self.llm_thread.llm_advice_signal.connect(self.update_ai_advice)
        self.llm_thread.llm_error_signal.connect(self.show_llm_error)
        self.llm_thread.start()

    def _load_settings(self):
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute(
                "SELECT conf_threshold, iou_threshold, language, chart_type FROM settings ORDER BY id DESC LIMIT 1")
            result = c.fetchone()
            conn.close()

            if result:
                conf, iou, lang, chart = result
                self.current_conf_thres = conf
                self.current_iou_thres = iou
                self.current_language = lang
                self.chart_type = chart

                # 更新UI
                self.conf_slider.setValue(int(conf * 100))
                self.conf_value_label.setText(f"{conf:.2f}")
                self.iou_slider.setValue(int(iou * 100))
                self.iou_value_label.setText(f"{iou:.2f}")
        except Exception as e:
            print(f"加载设置时出错: {e}")

    def upload_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp *.gif);;所有文件 (*)", options=options
        )

        if not file_name:
            self.status_bar.showMessage("未选择图片文件")
            return

        try:
            # 停止当前正在运行的视频流
            self.stop_stream()
            self.status_bar.showMessage(f"正在加载图片: {file_name}")

            # 读取图片
            image = cv2.imread(file_name)
            if image is None:
                raise ValueError(f"无法读取图片，请检查文件路径或文件是否损坏: {file_name}")

            # 保存原始图片尺寸用于显示
            original_height, original_width = image.shape[:2]
            self.status_bar.showMessage(f"图片尺寸: {original_width}x{original_height}")

            # 检测图片中的交通标志
            self._detect_and_display(image, source_path=file_name, is_image=True)

        except Exception as e:
            self.status_bar.showMessage(f"处理图片时出错: {str(e)}")
            QMessageBox.critical(self, "图片处理错误", f"无法处理所选图片:\n\n{str(e)}")
            self._reset_ui_state()

    def _detect_and_display(self, frame, source_path=None, is_image=False):
        """
        通用方法：检测图像中的交通标志并显示结果
        """
        try:
            # 确保frame不为空
            if frame is None or frame.size == 0:
                raise ValueError("检测图像为空或无效")

            # 执行交通标志检测 (修正参数顺序和类型)
            results = self.detector.detect(
                source_frame=frame,
                imgsz=640,  # 固定为640，或根据需要调整
                conf_thres=self.current_conf_thres,
                iou_thres=self.current_iou_thres
            )

            # 处理检测结果
            if not results or not hasattr(results[0], 'boxes') or not results[0].boxes:
                self.status_bar.showMessage("未检测到交通标志")
                # 显示原始图像
                display_frame = frame
            else:
                # 绘制检测框和标签
                display_frame = self._annotate_frame(frame.copy(), results)

                # 更新检测结果表格和统计信息
                detected_info = self._process_detection_results(results)
                self.update_detection_results(detected_info)

                # 提取检测到的类别名称并更新统计
                detected_classes = [item['class'] for item in detected_info]
                self.update_detection_classes(detected_classes)

                # 显示检测摘要
                summary = f"检测到 {len(detected_classes)} 个交通标志"
                self.status_bar.showMessage(summary)

            # 显示结果图像
            self._display_image(display_frame)

            # 如果是处理图片，保存检测结果
            if is_image and source_path:
                self._save_detection_history(source_path, detected_info)

        except Exception as e:
            error_msg = f"检测过程中出错: {str(e)}"
            self.status_bar.showMessage(error_msg)
            raise  # 重新抛出异常以便上层处理

    def _annotate_frame(self, frame, results):
        """在图像上绘制检测框和标签"""
        # 使用YOLO内置的绘图函数
        annotated_frame = results[0].plot()

        # 可以在这里添加额外的自定义标注逻辑
        return annotated_frame

    def _process_detection_results(self, results):
        """处理检测结果并返回格式化的数据"""
        detected_info = []
        if hasattr(results[0], 'boxes') and results[0].boxes:
            for i, r_box in enumerate(results[0].boxes):
                class_id = int(r_box.cls)
                conf = float(r_box.conf)
                if class_id < len(self.detector.model.names):
                    class_name = self.detector.model.names[class_id]
                    x1, y1, x2, y2 = map(int, r_box.xyxy[0].tolist())
                    detected_info.append({
                        'id': i + 1,
                        'class': class_name,
                        'confidence': f"{conf:.2f}",
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                    })
        return detected_info

    def _display_image(self, image):
        """将OpenCV图像显示在界面上"""
        # 转换颜色空间 (BGR -> RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 转换为Qt图像
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 缩放并显示
        scaled_pixmap = QPixmap.fromImage(q_img).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def _save_detection_history(self, source_path, detected_info):
        """保存检测历史到数据库"""
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()

            for item in detected_info:
                c.execute(
                    """INSERT INTO detections (source, class_name, confidence, x1, y1, x2, y2) 
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (source_path, item['class'], float(item['confidence']),
                     item['x1'], item['y1'], item['x2'], item['y2'])
                )

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"保存检测历史失败: {str(e)}")

    def _reset_ui_state(self):
        """重置UI状态到初始状态"""
        # 清空检测结果表格
        self.results_table.setRowCount(0)

        # 显示默认提示图像
        default_pixmap = QPixmap()
        self.image_label.setPixmap(default_pixmap)
        self.image_label.setText("请上传图片或启动摄像头")

        # 更新状态栏
        self.status_bar.showMessage("系统就绪")
    def upload_video(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)", options=options
        )
        if file_name:
            self.start_video_stream(file_name)

    def start_webcam(self):
        self.start_video_stream(0)  # 0表示默认摄像头

    def start_video_stream(self, source):
        self.stop_stream()

        # 创建并启动视频线程
        self.video_thread = VideoThread(source, self.detector, self.current_conf_thres, self.current_iou_thres)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.detection_data_signal.connect(self.update_detection_results)
        self.video_thread.detection_class_names_signal.connect(self.update_detection_classes)
        self.video_thread.start()

        # 更新UI状态
        self.upload_image_btn.setEnabled(False)
        self.upload_video_btn.setEnabled(False)
        self.webcam_btn.setEnabled(False)
        self.stop_stream_btn.setEnabled(True)
        self.step_mode_check.setEnabled(True)

        # 显示源信息
        source_name = "摄像头" if isinstance(source, int) else source.split('/')[-1]
        self.status_bar.showMessage(f"正在处理: {source_name}")

    def stop_stream(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread = None

        # 更新UI状态
        self.upload_image_btn.setEnabled(True)
        self.upload_video_btn.setEnabled(True)
        self.webcam_btn.setEnabled(True)
        self.stop_stream_btn.setEnabled(False)
        self.step_mode_check.setChecked(False)
        self.step_mode_check.setEnabled(False)
        self.step_btn.setEnabled(False)

        self.status_bar.showMessage("系统就绪")

    def update_image(self, image):
        scaled_pixmap = QPixmap.fromImage(image).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def update_detection_results(self, detection_data):
        self.results_table.setRowCount(len(detection_data))

        for row, item in enumerate(detection_data):
            self.results_table.setItem(row, 0, QTableWidgetItem(str(item['id'])))
            self.results_table.setItem(row, 1, QTableWidgetItem(item['class']))
            self.results_table.setItem(row, 2, QTableWidgetItem(item['confidence']))
            self.results_table.setItem(row, 3, QTableWidgetItem(str(item['x1'])))
            self.results_table.setItem(row, 4, QTableWidgetItem(str(item['y1'])))
            self.results_table.setItem(row, 5, QTableWidgetItem(str(item['x2'])))
            self.results_table.setItem(row, 6, QTableWidgetItem(str(item['y2'])))

        # 自动调整列宽
        self.results_table.resizeColumnsToContents()

    def update_detection_classes(self, class_names):
        # 更新统计数据
        for name in class_names:
            self.total_detection_counts[name] += 1
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            self.detection_history.append((timestamp, name))

        # 更新数据库
        if class_names:
            source = "摄像头" if (self.video_thread and isinstance(self.video_thread.source_path, int)) else "文件"
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            for name in class_names:
                # 假设x1, y1, x2, y2为0，实际应用中应从检测结果获取
                c.execute(
                    "INSERT INTO detections (source, class_name, confidence, x1, y1, x2, y2) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (source, name, 0.0, 0, 0, 0, 0))
            conn.commit()
            conn.close()

        # 请求AI建议
        self.llm_thread.request_advice(class_names)

        # 更新图表
        self.update_statistics_chart()

    def update_ai_advice(self, advice):
        self.advice_text.setText(advice)
        self.status_bar.showMessage("AI建议已更新")

    def show_llm_error(self, error_msg):
        self.status_bar.showMessage(error_msg)
        QMessageBox.warning(self, "LLM错误", error_msg)

    def update_conf_thres_value(self, value):
        self.current_conf_thres = value / 100.0
        self.conf_value_label.setText(f"{self.current_conf_thres:.2f}")
        if self.video_thread:
            self.video_thread.update_thresholds(self.current_conf_thres, self.current_iou_thres)

    def update_iou_thres_value(self, value):
        self.current_iou_thres = value / 100.0
        self.iou_value_label.setText(f"{self.current_iou_thres:.2f}")
        if self.video_thread:
            self.video_thread.update_thresholds(self.current_conf_thres, self.current_iou_thres)

    def toggle_ar_display(self, state):
        self.show_ar = state
        if self.video_thread:
            self.video_thread.set_ar_display(state)

    def toggle_fps_display(self, state):
        self.show_fps = state
        if self.video_thread:
            self.video_thread._show_fps = state

    def toggle_pause(self):
        if not self.video_thread:
            return

        if self.video_thread._paused:
            self.video_thread.unpause()
            self.pause_btn.setText("暂停")
            self.status_bar.showMessage("视频已恢复")
        else:
            self.video_thread.pause()
            self.pause_btn.setText("继续")
            self.status_bar.showMessage("视频已暂停")

    def toggle_step_mode(self, enabled):
        if not self.video_thread:
            return

        self.video_thread.set_step_mode(enabled)
        self.step_btn.setEnabled(enabled)
        self.status_bar.showMessage("步进模式已" + ("启用" if enabled else "禁用"))

    def step_frame(self):
        if self.video_thread and self.video_thread._step_mode:
            self.video_thread.step_frame()
            self.status_bar.showMessage("已前进一帧")

    def change_chart_type(self, chart_type_text):
        chart_types = {"柱状图": "bar", "饼图": "pie", "折线图": "line"}
        self.chart_type = chart_types.get(chart_type_text, "bar")
        self.update_statistics_chart()

    def update_statistics_chart(self):
        self.ax.clear()

        if not self.total_detection_counts:
            self.ax.text(0.5, 0.5, '暂无检测数据', ha='center', va='center', transform=self.ax.transAxes)
            self.canvas.draw()
            return

        # 根据时间范围筛选数据
        time_range = self.time_range_combo.currentText()
        recent_history = self.detection_history

        if time_range != "全部":
            current_time = time.time()
            time_threshold = 0

            if time_range == "最近10分钟":
                time_threshold = current_time - 600
            elif time_range == "最近1小时":
                time_threshold = current_time - 3600
            elif time_range == "最近24小时":
                time_threshold = current_time - 86400

            recent_history = []
            for timestamp, name in self.detection_history:
                ts = time.mktime(time.strptime(timestamp, "%Y-%m-%d %H:%M:%S"))
                if ts >= time_threshold:
                    recent_history.append((timestamp, name))

        # 统计筛选后的数据
        filtered_counts = collections.defaultdict(int)
        for _, name in recent_history:
            filtered_counts[name] += 1

        if not filtered_counts:
            self.ax.text(0.5, 0.5, f'在{time_range}内无检测数据', ha='center', va='center', transform=self.ax.transAxes)
            self.canvas.draw()
            return

        # 绘制图表
        labels = list(filtered_counts.keys())
        values = list(filtered_counts.values())

        if self.chart_type == "bar":
            self.ax.bar(labels, values)
            self.ax.set_xlabel('交通标志类别')
            self.ax.set_ylabel('检测次数')
            self.ax.set_title(f'交通标志检测统计 ({time_range})')
            self.ax.tick_params(axis='x', rotation=45)
        elif self.chart_type == "pie":
            self.ax.pie(values, labels=labels, autopct='%1.1f%%')
            self.ax.set_title(f'交通标志分布 ({time_range})')
        elif self.chart_type == "line":
            # 按时间排序
            time_data = collections.defaultdict(list)
            for timestamp, name in recent_history:
                time_data[name].append(timestamp)

            for name, timestamps in time_data.items():
                timestamps_sorted = sorted(timestamps)
                counts = list(range(1, len(timestamps_sorted) + 1))
                self.ax.plot(timestamps_sorted, counts, label=name)

            self.ax.set_xlabel('时间')
            self.ax.set_ylabel('累计检测次数')
            self.ax.set_title(f'交通标志检测趋势 ({time_range})')
            self.ax.legend()
            self.ax.tick_params(axis='x', rotation=45)

        self.figure.tight_layout()
        self.canvas.draw()

    def clear_statistics(self):
        self.total_detection_counts.clear()
        self.detection_history.clear()
        self.update_statistics_chart()
        self.status_bar.showMessage("统计数据已清除")

    def export_detections(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "导出检测结果", "", "CSV文件 (*.csv);;文本文件 (*.txt);;所有文件 (*)", options=options
        )

        if file_name:
            try:
                with open(file_name, 'w', encoding='utf-8') as f:
                    # 写入表头
                    f.write("ID,类别,置信度,X1,Y1,X2,Y2\n")

                    # 写入数据
                    for row in range(self.results_table.rowCount()):
                        row_data = []
                        for col in range(self.results_table.columnCount()):
                            item = self.results_table.item(row, col)
                            if item is not None:
                                row_data.append(item.text())
                            else:
                                row_data.append('')
                        f.write(','.join(row_data) + '\n')

                self.status_bar.showMessage(f"检测结果已导出到: {file_name}")
            except Exception as e:
                self.status_bar.showMessage(f"导出失败: {str(e)}")
                QMessageBox.critical(self, "导出失败", f"无法导出文件: {str(e)}")

    def search_signs(self):
        # 简单示例：实际应用中应连接到交通标志知识库
        search_text = self.sign_search.toPlainText().strip().lower()

        if not search_text:
            self.sign_info_display.setText("请输入要搜索的交通标志关键词...")
            return

        # 模拟搜索结果
        signs_database = {
            "限速": "限速标志表示车辆在此路段行驶的最高速度限制。违反限速规定可能导致罚款或扣分。",
            "禁止通行": "禁止通行标志表示禁止所有车辆和行人通行。违反此标志可能导致严重事故。",
            "注意行人": "注意行人标志提醒驾驶员前方可能有行人横穿道路，需减速慢行并注意观察。",
            "转弯": "转弯标志提示驾驶员前方道路将转弯，需提前减速并做好转向准备。",
            "施工": "施工标志表示前方道路正在施工，可能存在道路变窄、路面不平整等情况，需谨慎驾驶。"
        }

        results = []
        for keyword, description in signs_database.items():
            if search_text in keyword.lower() or search_text in description.lower():
                results.append(f"### {keyword}\n{description}\n\n")

        if results:
            self.sign_info_display.setText("".join(results))
        else:
            self.sign_info_display.setText(f"未找到与 '{search_text}' 相关的交通标志信息。")

    def export_settings(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "导出设置", "", "配置文件 (*.ini);;所有文件 (*)", options=options
        )

        if file_name:
            try:
                config = configparser.ConfigParser()
                config['Detection'] = {
                    'conf_threshold': self.current_conf_thres,
                    'iou_threshold': self.current_iou_thres,
                    'chart_type': self.chart_type
                }

                with open(file_name, 'w') as f:
                    config.write(f)

                self.status_bar.showMessage(f"设置已导出到: {file_name}")
            except Exception as e:
                self.status_bar.showMessage(f"导出失败: {str(e)}")
                QMessageBox.critical(self, "导出失败", f"无法导出设置: {str(e)}")

    def import_settings(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "导入设置", "", "配置文件 (*.ini);;所有文件 (*)", options=options
        )

        if file_name:
            try:
                config = configparser.ConfigParser()
                config.read(file_name)

                if 'Detection' in config:
                    detection = config['Detection']

                    if 'conf_threshold' in detection:
                        self.current_conf_thres = float(detection['conf_threshold'])
                        self.conf_slider.setValue(int(self.current_conf_thres * 100))
                        self.conf_value_label.setText(f"{self.current_conf_thres:.2f}")

                    if 'iou_threshold' in detection:
                        self.current_iou_thres = float(detection['iou_threshold'])
                        self.iou_slider.setValue(int(self.current_iou_thres * 100))
                        self.iou_value_label.setText(f"{self.current_iou_thres:.2f}")

                    if 'chart_type' in detection:
                        self.chart_type = detection['chart_type']
                        chart_text = {"bar": "柱状图", "pie": "饼图", "line": "折线图"}.get(self.chart_type, "柱状图")
                        self.chart_type_combo.setCurrentText(chart_text)

                    # 更新数据库
                    conn = sqlite3.connect(DB_FILE)
                    c = conn.cursor()
                    c.execute(
                        "INSERT INTO settings (conf_threshold, iou_threshold, language, chart_type) VALUES (?, ?, ?, ?)",
                        (self.current_conf_thres, self.current_iou_thres, self.current_language, self.chart_type))
                    conn.commit()
                    conn.close()

                    self.status_bar.showMessage(f"设置已从 {file_name} 导入")
                else:
                    QMessageBox.warning(self, "导入失败", "配置文件格式不正确，缺少'Detection'部分")
            except Exception as e:
                self.status_bar.showMessage(f"导入失败: {str(e)}")
                QMessageBox.critical(self, "导入失败", f"无法导入设置: {str(e)}")

    def toggle_dark_mode(self):
        self.is_dark_mode = not self.is_dark_mode

        if self.is_dark_mode:
            self.setStyleSheet("""
                QMainWindow, QWidget, QGroupBox, QTabWidget::pane {
                    background-color: #2a2a2a;
                    color: #ffffff;
                }
                QLabel, QPushButton, QCheckBox, QSlider, QComboBox, QTableWidget {
                    color: #ffffff;
                }
                QPushButton {
                    background-color: #4a4a4a;
                    border: 1px solid #6a6a6a;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #5a5a5a;
                }
                QTableWidget {
                    background-color: #3a3a3a;
                }
                QHeaderView::section {
                    background-color: #4a4a4a;
                    color: white;
                    padding: 4px;
                    border: 1px solid #6a6a6a;
                }
            """)
        else:
            self.setStyleSheet("")  # 恢复默认样式

    def set_language(self, lang):
        self.current_language = lang
        self.llm_thread.set_language(lang)
        self.advice_lang_combo.setCurrentText("中文" if lang == "zh" else "English")

        # 更新菜单项状态
        for action in self.menuBar().actions():
            if action.text() == "语言":
                for lang_action in action.menu().actions():
                    lang_action.setChecked(lang_action.text().startswith(lang == "zh" and "中文" or "English"))

        self.status_bar.showMessage(f"语言已切换为: {lang}")

    def show_about(self):
        QMessageBox.about(self, "关于", """
            高级交通标志识别与驾驶建议系统

            版本: 1.0.0
            描述: 基于深度学习的交通标志实时识别系统，提供驾驶建议和安全提示
            技术支持: YOLOv11, OpenAI API
        """)

    def change_advice_language(self, lang_text):
        lang = "zh" if lang_text == "中文" else "en"
        self.set_language(lang)

    def text_to_speech(self):
        advice_text = self.advice_text.toPlainText().strip()
        if not advice_text:
            self.status_bar.showMessage("没有可播报的内容")
            return

        try:
            # 初始化语音引擎
            import pyttsx3
            engine = pyttsx3.init()

            # 根据当前语言设置语音
            if self.current_language == "zh":
                # 设置中文语音（选择系统中的中文发音人）
                voices = engine.getProperty('voices')
                for voice in voices:
                    # 匹配中文语音（不同系统名称可能不同，需根据实际情况调整）
                    if 'chinese' in voice.id.lower() or 'china' in voice.id.lower():
                        engine.setProperty('voice', voice.id)
                        break
                engine.setProperty('rate', 150)  # 语速（中文适中）
            else:
                # 英文默认语音
                engine.setProperty('rate', 180)  # 语速（英文稍快）

            # 播放语音
            engine.say(advice_text)
            engine.runAndWait()
            self.status_bar.showMessage("语音播报完成")

        except ImportError:
            self.status_bar.showMessage("未安装pyttsx3，请运行 'pip install pyttsx3'")
            QMessageBox.warning(self, "依赖缺失", "语音播报需要安装pyttsx3库：\npip install pyttsx3")
        except Exception as e:
            error_msg = f"语音播报失败: {str(e)}"
            self.status_bar.showMessage(error_msg)
            QMessageBox.warning(self, "语音错误", error_msg)

    def show_history_dialog(self):
        # 创建历史记录对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("历史检测记录")
        dialog.setMinimumSize(800, 600)

        layout = QVBoxLayout(dialog)

        # 创建表格显示历史记录
        history_table = QTableWidget()
        history_table.setColumnCount(4)
        history_table.setHorizontalHeaderLabels(["时间", "来源", "标志类别", "置信度"])
        history_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        history_table.setAlternatingRowColors(True)
        history_table.setSortingEnabled(True)
        history_table.horizontalHeader().setStretchLastSection(True)

        # 从数据库加载历史记录
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("SELECT timestamp, source, class_name, confidence FROM detections ORDER BY timestamp DESC")
            rows = c.fetchall()
            conn.close()

            history_table.setRowCount(len(rows))

            for row_idx, row_data in enumerate(rows):
                timestamp, source, class_name, confidence = row_data
                history_table.setItem(row_idx, 0, QTableWidgetItem(str(timestamp)))
                history_table.setItem(row_idx, 1, QTableWidgetItem(str(source)))
                history_table.setItem(row_idx, 2, QTableWidgetItem(str(class_name)))
                history_table.setItem(row_idx, 3, QTableWidgetItem(f"{confidence:.2f}" if confidence else ""))

            history_table.resizeColumnsToContents()
        except Exception as e:
            QMessageBox.critical(dialog, "错误", f"加载历史记录失败: {str(e)}")

        layout.addWidget(history_table)

        # 添加关闭按钮
        close_button = QPushButton("关闭")
        close_button.clicked.connect(dialog.close)
        layout.addWidget(close_button)

        # 显示对话框
        dialog.exec_()
    def closeEvent(self, event):
        self.stop_stream()
        if self.llm_thread and self.llm_thread.isRunning():
            self.llm_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置中文字体支持
    font = app.font()
    font.setFamily("SimHei")
    app.setFont(font)

    window = TrafficSignApp()
    window.show()

    # 启动应用程序事件循环
    sys.exit(app.exec_())