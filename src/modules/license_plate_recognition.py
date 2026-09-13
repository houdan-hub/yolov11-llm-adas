import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
# 读取图像
image = cv2.imread(r'./img_2.png')

# 将图像从BGR颜色空间转换到HSV颜色空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义蓝色物体在HSV颜色空间中的范围
# 这里需要根据实际情况调整蓝色范围的上下限
blue_lower = np.array([100, 50, 50])
blue_upper = np.array([120, 255, 255])

# 创建掩码，只保留蓝色物体
mask = cv2.inRange(hsv_image, blue_lower, blue_upper)

# 通过位运算提取蓝色物体
# 定义结构元
kernel = np.ones((9,9), np.uint8)
kernel2 = np.ones((21,21), np.uint8)
# 进行开闭运算清除噪点填充空洞
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)
contours_fist, _ = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
mask2 = np.zeros_like(mask)
for contour in contours_fist:
    area = cv2.contourArea(contour)
    if area < 200:
        continue
    epsilon = 0.08 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 4:
        cv2.drawContours(mask2, [approx], -1, (255, 255, 255), -1)
blue_objects = cv2.bitwise_and(image, image, mask=mask2)
print(blue_objects)
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
result = ocr.ocr(blue_objects, cls=True)
print(result[0][0][1][0])