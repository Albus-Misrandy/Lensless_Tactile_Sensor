"""
Author: Albus.Misrandy
"""
import os
import cv2
import numpy as np

save_dir = "Lensless"
os.makedirs(save_dir, exist_ok=True)

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

# def remove_periodic_noise(gray_img, threshold_sigma=5, r=10):
#     """
#     自适应频域滤波函数
#     gray_img: 输入的灰度图 (float32)
#     threshold_sigma: 自动识别噪声点的阈值，越大越保守
#     r: 滤波半径，越大去除越彻底，但可能影响细节
#     """
#     rows, cols = gray_img.shape
#     # 1. FFT 变换
#     f = np.fft.fft2(gray_img)
#     fshift = np.fft.fftshift(f)
    
#     # 2. 计算幅度谱用于寻找噪声点
#     magnitude_spectrum = np.abs(fshift)
#     mag_log = np.log(magnitude_spectrum + 1)
    
#     # 3. 自动寻找异常高亮波峰 (除了中心低频区域)
#     # 计算平均值和标准差来定位离群点
#     mean, std = np.mean(mag_log), np.std(mag_log)
#     thresh = mean + threshold_sigma * std
    
#     mask = np.ones((rows, cols), np.uint8)
#     crow, ccol = rows // 2, cols // 2
    
#     # 屏蔽中心区域（保留低频主体图像信息）
#     ignore_r = 20
    
#     # 寻找可能的噪声亮点
#     locations = np.where(mag_log > thresh)
#     for r_idx, c_idx in zip(locations[0], locations[1]):
#         # 如果亮点不在中心区域，就认为是周期性条纹噪声
#         if abs(r_idx - crow) > ignore_r or abs(c_idx - ccol) > ignore_r:
#             cv2.circle(mask, (c_idx, r_idx), r, 0, -1)
            
#     # 4. 应用遮罩并反变换
#     fshift_filtered = fshift * mask
#     f_ishift = np.fft.ifftshift(fshift_filtered)
#     img_back = np.fft.ifft2(f_ishift)
    
#     return np.abs(img_back).astype(np.float32)

def process_and_visualize_fft(gray_img, threshold_sigma=6, r=8, show_spectrum=True):
    rows, cols = gray_img.shape
    # 1. FFT 变换
    f = np.fft.fft2(gray_img)
    fshift = np.fft.fftshift(f)
    
    # 2. 计算幅度谱用于显示和寻找噪声点
    magnitude_spectrum = np.abs(fshift)
    # log变换方便肉眼观察
    mag_log = np.log(magnitude_spectrum + 1)
    
    # 归一化到 0-255 用于 cv2.imshow 显示
    view_spectrum = cv2.normalize(mag_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 3. 自动识别并滤除噪声点
    mean, std = np.mean(mag_log), np.std(mag_log)
    thresh = mean + threshold_sigma * std
    mask = np.ones((rows, cols), np.uint8)
    crow, ccol = rows // 2, cols // 2
    ignore_r = 30 # 保护中心低频区域
    
    locations = np.where(mag_log > thresh)
    for r_idx, c_idx in zip(locations[0], locations[1]):
        if abs(r_idx - crow) > ignore_r or abs(c_idx - ccol) > ignore_r:
            # 在频谱图上画圈，让你能看到滤掉了哪里
            cv2.circle(view_spectrum, (c_idx, r_idx), r + 2, 255, 1) 
            cv2.circle(mask, (c_idx, r_idx), r, 0, -1)
            
    # 4. 反变换回到图像
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.abs(np.fft.ifft2(f_ishift))
    
    return img_back.astype(np.float32), view_spectrum

while True:
    # 读取一帧图像
    ret, frame = cap.read()

    if not ret:
        print("读取摄像头画面失败")
        break

    frame = frame.astype(np.float32)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dark_gray = cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY).astype(np.float32)

    corrected = frame - dark
    corrected = np.clip(corrected, 0, None)

    gray_correct = gray - dark_gray
    gray_correct = np.clip(gray_correct, 0, None)

    denoised_gray, spectrum_img = process_and_visualize_fft(gray)

    h, w= gray.shape
    if w > 1600:
        scale = 1280 / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        # show_frame = cv2.resize(frame, (new_w, new_h))
        show_frame = cv2.resize(corrected, (new_w, new_h))
        # 映射回 0-255
        show_gray = cv2.normalize(denoised_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        show_gray = cv2.resize(show_gray, (new_w, new_h))
        spectrum_img = cv2.resize(spectrum_img, (new_w, new_h))

    show_frame = show_frame.astype(np.uint8)
    show_gray = show_gray.astype(np.uint8)

    # 显示摄像头画面
    cv2.imshow("Camera", show_frame)
    cv2.imshow("Gray", show_gray)
    cv2.imshow("Real-time FFT Spectrum (Frequency Domain)", spectrum_img)

    # 等待键盘输入
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("退出程序")
        break

    elif key == ord('s'):
        filename = f"{123}.jpg"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, spectrum_img)

# 释放摄像头
cap.release()

# 关闭所有窗口
cv2.destroyAllWindows()