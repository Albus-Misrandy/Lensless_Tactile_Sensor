import cv2
import numpy as np

def normalize_for_display(img):
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    vis = (img - mn) / (mx - mn)
    vis = (vis * 255).astype(np.uint8)
    return vis


coding_nopress = cv2.imread("./Lensless_images/001.jpg").astype(np.float32)
coding_press = cv2.imread("./Lensless_images/003.jpg").astype(np.float32)

# =========================
# 2) 尺寸检查
# =========================
if coding_nopress.shape != coding_press.shape:
    print("尺寸不一致")
    print("coding_nopress.shape =", coding_nopress.shape)
    print("coding_press.shape   =", coding_press.shape)
    exit()

# 3) 做差分
#    delta > 0 / < 0 表示两种状态差异
# =========================
delta = coding_press - coding_nopress

# =========================
# 4) 因为你目前按压区域通常表现为“变暗”
#    所以取反，把按压响应变成正值
# =========================
response = -delta

# =========================
# 5) 只保留正响应
# =========================
response = np.clip(response, 0, None)

# =========================
# 6) 做一点平滑，减轻残余碎噪声
# =========================
response_smooth = cv2.GaussianBlur(response, (0, 0), 3)

# =========================
# 8) 保存显示图
# =========================
recon_vis = normalize_for_display(response_smooth)
cv2.imwrite("reconstruction_baseline.png", recon_vis)
