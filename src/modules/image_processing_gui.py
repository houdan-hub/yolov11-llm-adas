import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("多功能图像处理工具")

        self.original_image = None
        self.processed_image = None
        self.current_effect = tk.StringVar(value="none") # 灰度, canny, adaptive_threshold

        self.setup_ui()

    def setup_ui(self):
        # 菜单栏
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="打开图片", command=self.open_image)
        file_menu.add_command(label="保存结果", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        menubar.add_cascade(label="文件", menu=file_menu)

        # 主内容区域
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 图片显示区
        self.image_display_frame = ttk.Frame(main_frame, relief=tk.SUNKEN, borderwidth=2)
        self.image_display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.image_label = tk.Label(self.image_display_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # 控制面板
        control_panel = ttk.Frame(main_frame, width=250)
        control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        # 效果选择
        effect_frame = ttk.LabelFrame(control_panel, text="图像处理效果")
        effect_frame.pack(pady=10, padx=5, fill=tk.X)

        tk.Radiobutton(effect_frame, text="原始图片", variable=self.current_effect, value="none", command=self.apply_effect).pack(anchor=tk.W)
        tk.Radiobutton(effect_frame, text="灰度转换", variable=self.current_effect, value="grayscale", command=self.apply_effect).pack(anchor=tk.W)
        tk.Radiobutton(effect_frame, text="Canny 边缘检测", variable=self.current_effect, value="canny", command=self.apply_effect).pack(anchor=tk.W)
        tk.Radiobutton(effect_frame, text="自适应阈值二值化", variable=self.current_effect, value="adaptive_threshold", command=self.apply_effect).pack(anchor=tk.W)

        # 参数调整区 (根据效果动态显示)
        self.param_frame = ttk.LabelFrame(control_panel, text="参数调整")
        self.param_frame.pack(pady=10, padx=5, fill=tk.X)
        self.create_param_widgets()

    def create_param_widgets(self):
        # 清空之前的参数控件
        for widget in self.param_frame.winfo_children():
            widget.destroy()

        effect = self.current_effect.get()

        if effect == "canny":
            ttk.Label(self.param_frame, text="Canny 阈值1:").pack(anchor=tk.W)
            self.canny_threshold1_scale = ttk.Scale(self.param_frame, from_=0, to_=255, orient=tk.HORIZONTAL, command=self.apply_effect)
            self.canny_threshold1_scale.set(50)
            self.canny_threshold1_scale.pack(fill=tk.X)

            ttk.Label(self.param_frame, text="Canny 阈值2:").pack(anchor=tk.W)
            self.canny_threshold2_scale = ttk.Scale(self.param_frame, from_=0, to_=255, orient=tk.HORIZONTAL, command=self.apply_effect)
            self.canny_threshold2_scale.set(150)
            self.canny_threshold2_scale.pack(fill=tk.X)

        elif effect == "adaptive_threshold":
            ttk.Label(self.param_frame, text="块大小 (奇数):").pack(anchor=tk.W)
            self.block_size_scale = ttk.Scale(self.param_frame, from_=3, to_=51, orient=tk.HORIZONTAL, command=self.apply_effect)
            self.block_size_scale.set(11) # 默认值
            self.block_size_scale.pack(fill=tk.X)

            ttk.Label(self.param_frame, text="常数 C:").pack(anchor=tk.W)
            self.c_value_scale = ttk.Scale(self.param_frame, from_=-10, to_=10, orient=tk.HORIZONTAL, command=self.apply_effect)
            self.c_value_scale.set(2) # 默认值
            self.c_value_scale.pack(fill=tk.X)

        self.apply_effect() # 重新应用效果以更新参数

    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            try:
                # OpenCV 读取图片是 BGR 格式
                img = cv2.imread(file_path)
                if img is None:
                    raise ValueError("无法读取图片，请检查文件路径或格式。")
                self.original_image = img
                self.apply_effect() # 打开图片后立即应用当前效果
            except Exception as e:
                tk.messagebox.showerror("错误", f"加载图片失败: {e}")

    def display_image(self, img_cv):
        if img_cv is None:
            self.image_label.config(image=None)
            self.image_label.image = None
            return

        # OpenCV (BGR) 转换为 PIL (RGB)
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

        # 调整图片大小以适应显示区域
        img_width, img_height = img_pil.size
        display_width = self.image_display_frame.winfo_width() - 20 # 预留一些边距
        display_height = self.image_display_frame.winfo_height() - 20

        if display_width > 0 and display_height > 0:
            ratio = min(display_width / img_width, display_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)

        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk # 保持引用，防止图片被垃圾回收

    def apply_effect(self, event=None):
        if self.original_image is None:
            return

        current_effect_name = self.current_effect.get()
        self.create_param_widgets() # 重新创建参数控件以匹配当前效果

        if current_effect_name == "none":
            self.processed_image = self.original_image
        elif current_effect_name == "grayscale":
            self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            # 灰度图可能只有一个通道，需要转回 BGR 或直接显示
            if len(self.processed_image.shape) == 2:
                self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR)
        elif current_effect_name == "canny":
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            threshold1 = int(self.canny_threshold1_scale.get())
            threshold2 = int(self.canny_threshold2_scale.get())
            edges = cv2.Canny(gray_image, threshold1, threshold2)
            self.processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) # 转换为BGR以便显示
        elif current_effect_name == "adaptive_threshold":
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            block_size = int(self.block_size_scale.get())
            # 确保 block_size 是奇数且大于1
            if block_size % 2 == 0:
                block_size += 1 # 自动调整为最近的奇数
                self.block_size_scale.set(block_size) # 更新滑块显示

            c_value = int(self.c_value_scale.get())
            _, binary_image = cv2.adaptiveThreshold(gray_image, 255,
                                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY,
                                                    block_size, c_value)
            self.processed_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR) # 转换为BGR以便显示

        self.display_image(self.processed_image)

    def save_image(self):
        if self.processed_image is None:
            tk.messagebox.showwarning("保存失败", "没有图片可供保存！")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if file_path:
            try:
                cv2.imwrite(file_path, self.processed_image)
                tk.messagebox.showinfo("保存成功", f"图片已保存到: {file_path}")
            except Exception as e:
                tk.messagebox.showerror("保存失败", f"保存图片失败: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()