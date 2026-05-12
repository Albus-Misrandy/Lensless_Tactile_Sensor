"""
Author: Albus.Misrandy
"""
import os
import cv2
import numpy as np

# camera_matrix = np.array([
#     [822.10936238, 0, 193.29282218],
#     [0, 819.87918358, 228.53925193],
#     [0, 0, 1.0]
# ], dtype=np.float32)

# dist_coeffs = np.array([1.40490240e-01, -2.21689136e+00, -8.16683470e-04, 6.87888386e-05, 6.78923874e+00], dtype=np.float32)

# W, H = 3840, 2160

# 获取最优内参矩阵（alpha=0表示剪裁掉黑色边框，alpha=1表示保留所有像素）
# new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (W, H), 0, (W, H))
# # 创建映射表
# mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (W, H), cv2.CV_32FC1)

# 摄像头编号
# 一般笔记本自带摄像头是 0
# 外接摄像头可能是 1、2、3
# 打开摄像头
cap = cv2.VideoCapture(2)

dark = cv2.imread("./Lensless_images/dark.jpg").astype(np.float32)

# 判断摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头，请检查摄像头编号或连接状态")
    exit()

# =========================
# 设置目标格式
# =========================
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

# 给摄像头一点时间切换模式
for _ in range(10):
    cap.read()

# 创建保存图片的文件夹
save_dir = "Lensless_images"
os.makedirs(save_dir, exist_ok=True)

print("摄像头已打开")
print("按 s 拍照保存")
print("按 q 退出程序")

count = 0

while True:
    # 读取一帧图像
    ret, frame = cap.read()

    if not ret:
        print("读取摄像头画面失败")
        break

    # 使用预计算好的 map 进行重采样，速度极快
    # undistorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

    frame = frame.astype(np.float32)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dark_gray = cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # corrected = frame - dark
    # corrected = np.clip(corrected, 0, None)

    # gray_correct = gray - dark_gray
    # gray_correct = np.clip(gray_correct, 0, None)

    h, w= gray.shape
    if w > 1600:
        scale = 1280 / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        # show_frame = cv2.resize(frame, (new_w, new_h))
        show_frame = cv2.resize(frame, (new_w, new_h))
        show_gray = cv2.resize(gray, (new_w, new_h))

    show_frame = show_frame.astype(np.uint8)
    show_gray = show_gray.astype(np.uint8)
    
    # 显示摄像头画面
    cv2.imshow("Camera", show_frame)
    cv2.imshow("Gray", show_gray)

    # 等待键盘输入
    key = cv2.waitKey(1) & 0xFF

    # 按 s 拍照
    if key == ord('s'):
        # 用当前时间作为图片名，避免覆盖
        # count += 1
        filename = f"ref.jpg"
        filepath = os.path.join(save_dir, filename)

        # 保存图片
        cv2.imwrite(filepath, frame)
        # cv2.imwrite(filepath, corrected)
        print(f"照片已保存：{filepath}")

    # 按 q 退出
    elif key == ord('q'):
        print("退出程序")
        break

# 释放摄像头
cap.release()

# 关闭所有窗口
cv2.destroyAllWindows()