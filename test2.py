"""
Author: Albus.Misrandy
"""
import os
import cv2
import numpy as np

# 摄像头编号
# 一般笔记本自带摄像头是 0
# 外接摄像头可能是 1、2、3
# 打开摄像头
cap = cv2.VideoCapture(2)

dark = cv2.imread("./Lensless_images/dark.jpg").astype(np.float32)
ref = cv2.imread("./Lensless_images/ref.jpg").astype(np.float32)

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

def smooth_1d(v, ksize=81):
    """
    对一维曲线做平滑，只保留慢变化趋势
    """
    if ksize % 2 == 0:
        ksize += 1
    v2 = v.reshape(-1, 1).astype(np.float32)
    v2 = cv2.GaussianBlur(v2, (1, ksize), 0)
    return v2[:, 0]


def estimate_rank1_background(img, smooth_ksize=81, eps=1e-6):
    """
    慢变背景估计：
    bg ≈ a b^T / mean(a)

    a: 每一行的代表亮度
    b: 每一列的代表亮度
    """
    img = img.astype(np.float32)

    # 1) 统计每一行的大致亮度
    a = np.median(img, axis=1).astype(np.float32)

    # 2) 统计每一列的大致亮度
    b = np.median(img, axis=0).astype(np.float32)

    # 3) 平滑，只保留慢变趋势
    a_s = smooth_1d(a, ksize=smooth_ksize)
    b_s = smooth_1d(b, ksize=smooth_ksize)

    # 4) 用行趋势 × 列趋势，拼成二维背景图
    bg = np.outer(a_s, b_s) / (np.mean(a_s) + eps)

    return bg, a_s, b_s

def normalize_for_display(img):
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    vis = (img - mn) / (mx - mn)
    vis = (vis * 255).astype(np.uint8)
    return vis

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
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)

    corrected = frame - dark
    corrected = np.clip(corrected, 0, None)

    gray_correct = gray - dark_gray
    gray_correct = np.clip(gray_correct, 0, None)

    # =========================
    # 4) 做参考归一化
    # =========================
    eps = 1e-6
    norm_img = gray_correct / (ref_gray + eps)

    bg_img, a_s, b_s = estimate_rank1_background(
    norm_img,
    smooth_ksize=81,
    eps=1e-6
    )

    coding_like = norm_img - bg_img
    show_coding = normalize_for_display(coding_like)
    save_code = show_coding
    show_bg = normalize_for_display(bg_img)

    # =========================
    # 6) 为了便于观察，做一个可视化版本
    #    这里只是显示，不是后续计算真实值
    # =========================
    mn = norm_img.min()
    mx = norm_img.max()

    # 归一化图只用于显示，要拉伸一下
    norm_min = norm_img.min()
    norm_max = norm_img.max()
    if norm_max - norm_min < 1e-6:
        show_norm = np.zeros_like(norm_img, dtype=np.uint8)
    else:
        show_norm = (norm_img - norm_min) / (norm_max - norm_min)
        show_norm = (show_norm * 255).astype(np.uint8)

    h, w= gray.shape
    if w > 1600:
        scale = 1280 / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        # show_frame = cv2.resize(frame, (new_w, new_h))
        show_frame = cv2.resize(corrected, (new_w, new_h))
        show_gray = cv2.resize(gray_correct, (new_w, new_h))
        show_norm = cv2.resize(show_norm, (new_w, new_h))
        show_coding = cv2.resize(show_coding, (new_w, new_h))
        show_bg = cv2.resize(show_bg, (new_w, new_h))

    show_frame = show_frame.astype(np.uint8)
    show_gray = show_gray.astype(np.uint8)
    
    # 显示摄像头画面
    cv2.imshow("Camera", show_frame)
    cv2.imshow("Gray", show_gray)
    cv2.imshow("Normalized", show_norm)
    cv2.imshow("Coding_Like", show_coding)
    cv2.imshow("Estimated_Background", show_bg)

    # 等待键盘输入
    key = cv2.waitKey(1) & 0xFF

    # 按 s 拍照
    if key == ord('s'):
        # 用当前时间作为图片名，避免覆盖
        count += 1
        filename = f"{count:03d}.jpg"
        filepath = os.path.join(save_dir, filename)

        # 保存图片
        # cv2.imwrite(filepath, frame)
        # cv2.imwrite(filepath, corrected)
        cv2.imwrite(filepath, save_code)
        print(f"照片已保存：{filepath}")

    # 按 q 退出
    elif key == ord('q'):
        print("退出程序")
        break

# 释放摄像头
cap.release()

# 关闭所有窗口
cv2.destroyAllWindows()