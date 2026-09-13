import cv2
import imutils
import numpy as np
import time

def object_tracking_and_rotation_correction():
    # 1. 打开摄像头或读取视频文件
    # 如果要读取视频文件，将参数改为视频文件路径，例如 'your_video.mp4'
    cap = cv2.VideoCapture(0) # 0 代表默认摄像头

    if not cap.isOpened():
        print("错误：无法打开摄像头或视频文件。请检查设备是否连接或文件路径是否正确。")
        return

    # 初始化第一帧用于帧差法
    first_frame = None
    last_detected_angle = 0 # 用于平滑角度变化

    print("开始目标追踪和旋转校正。按 'q' 键退出。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频流结束或无法读取帧。")
            break

        # 将帧转换为灰度图像并进行高斯模糊以减少噪音
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # 如果第一帧尚未初始化，则将其设置为当前帧并继续
        if first_frame is None:
            first_frame = gray
            continue

        # 2. 检测运动区域（使用帧差法）
        # 计算当前帧和第一帧之间的差异
        frame_delta = cv2.absdiff(first_frame, gray)
        # 对差异图像进行阈值处理，将小于某个值的像素设为0，大于某个值的像素设为255
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        # 对阈值图像进行膨胀操作，填充孔洞
        thresh = cv2.dilate(thresh, None, iterations=2)

        # 找到阈值图像中的所有轮廓
        # OpenCV 4.x 版本 findContours 返回两个值：contours 和 hierarchy
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours) # 使用 imutils 统一不同 OpenCV 版本的返回值

        detected_objects = []
        for c in contours:
            # 忽略过小的轮廓（可能是噪音）
            if cv2.contourArea(c) < 500: # 设定最小面积阈值
                continue

            # 获取轮廓的边界框
            (x, y, w, h) = cv2.boundingRect(c)
            detected_objects.append(((x, y, w, h), c))
            # 在原始帧上绘制边界框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 3. 对目标进行角度检测并使用 imutils.rotate_bound 进行旋转矫正
        if detected_objects:
            # 找到最大的运动区域作为主要追踪目标
            main_object_box, main_object_contour = max(detected_objects, key=lambda item: cv2.contourArea(item[1]))
            x, y, w, h = main_object_box

            # 绘制目标轮廓的最小外接矩形（带旋转）
            rect = cv2.minAreaRect(main_object_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(frame, [box], 0, (0, 0, 255), 2) # 红色矩形

            # 获取旋转角度
            # rect[2] 是角度，OpenCV 返回的角度范围是 [-90, 0) 或 [0, 90]
            # 并且根据矩形的长宽比决定。为了旋转校正，我们需要一个统一的角度。
            angle = rect[2]
            if w < h: # 如果高度大于宽度，则角度需要调整
                angle = angle + 90

            # 为了平滑角度变化，我们可以对角度进行一些限制或平均
            # 这里简单地使用一个平滑因子
            smoothed_angle = last_detected_angle * 0.8 + angle * 0.2
            last_detected_angle = smoothed_angle

            # 裁剪出目标区域以便旋转
            # 扩大裁剪区域以确保旋转后目标完整显示
            # 注意：这里直接对整个帧进行旋转可能更简单，因为旋转矫正通常是指整个图像。
            # 如果只想旋转目标本身，需要先裁剪，然后旋转，再放回原位。
            # 实验要求是对目标进行角度检测并对目标区域进行旋转校正
            # 所以我们裁剪出目标区域进行旋转。

            # 计算一个稍微大一点的边界框以包含旋转后的内容
            center_x, center_y = int(x + w / 2), int(y + h / 2)
            # 计算新图像的尺寸，需要足够大以容纳旋转后的图像
            # 假设最大旋转 90 度，新图像的对角线是原图的对角线
            # 这里为了简单，直接裁剪出包含原始检测框的区域，并将其中心作为旋转中心
            # 实际应用中，如果目标很大会裁剪出图像外，需要更复杂的逻辑或对整个帧进行旋转。

            # 裁剪出目标区域
            # 增加一些填充，以防旋转后内容溢出边界
            padding = 50
            start_x = max(0, x - padding)
            start_y = max(0, y - padding)
            end_x = min(frame.shape[1], x + w + padding)
            end_y = min(frame.shape[0], y + h + padding)

            object_roi = frame[start_y:end_y, start_x:end_x].copy()

            # 计算裁剪区域内的旋转中心
            roi_center_x = center_x - start_x
            roi_center_y = center_y - start_y

            if object_roi.shape[0] > 0 and object_roi.shape[1] > 0:
                # 使用 imutils.rotate_bound 进行旋转矫正
                # imutils.rotate_bound 会自动调整图像大小以包含所有内容
                rotated_object = imutils.rotate_bound(object_roi, -smoothed_angle) # 负角度表示逆时针旋转以校正

                # 将旋转后的目标显示在一个单独的窗口中
                cv2.imshow("Rotated Object", rotated_object)
        else:
            # 如果没有检测到目标，隐藏旋转目标窗口（如果存在）
            try:
                cv2.destroyWindow("Rotated Object")
            except cv2.error:
                pass # 窗口可能不存在

        # 4. 使用 imutils.resize 规范输出显示尺寸
        # 原始帧
        display_frame = imutils.resize(frame, width=800) # 将宽度调整为 800 像素
        # 帧差图
        display_thresh = imutils.resize(thresh, width=800)


        cv2.imshow("Original Frame with Detections", display_frame)
        cv2.imshow("Motion Detection (Threshold)", display_thresh)

        # 更新第一帧，以便下一帧可以与当前帧进行比较（实现连续运动检测）
        # 否则，如果 `first_frame` 始终是固定的第一帧，那么它只检测相对于初始状态的运动。
        # 如果需要持续检测“新”运动，每次迭代都更新 `first_frame` 更合适。
        # 这里我们更新 `first_frame` 来实现连续的背景减除效果，
        # 但为了避免积累性的运动残留，可以考虑更复杂的背景建模（如MOG2）
        # 或者在一段时间后重新初始化 first_frame。
        # 对于简单的运动检测，可以每隔N帧更新一次或平滑更新。
        # 为了演示帧差法，我们在此示例中保持 `first_frame` 不变以检测相对于初始背景的运动。
        # 如果需要更鲁棒的背景减除，请使用 cv2.createBackgroundSubtractorMOG2()。
        # 这里为了符合“帧差法”的简单实现，我们只在初始化时设置 first_frame。
        # 如果要追踪持续运动，需要更新 first_frame 为当前帧的平滑版本。
        # 例如：first_frame = cv2.addWeighted(gray, 0.5, first_frame, 0.5, 0)
        # 或者每N帧重置一次 first_frame = None


        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    object_tracking_and_rotation_correction()