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

response = cv2.imread("./reconstruction_baseline.png", cv2.IMREAD_GRAYSCALE)

if response is None:
    print("读取图片失败：./reconstruction_baseline.png")
    exit()

response = response.astype(np.float32)

# =========================
# 2) 转成适合显示和阈值分割的 uint8 图
# =========================
response_vis = normalize_for_display(response)

# 3) 阈值分割
#    这里先用 Otsu 自动阈值
# =========================
# _, binary = cv2.threshold(
#     response_vis, 0, 255,
#     cv2.THRESH_BINARY + cv2.THRESH_OTSU
# )
thresh_value = 40
_, binary = cv2.threshold(response_vis, thresh_value, 255, cv2.THRESH_BINARY_INV)

# =========================
# 4) 形态学开运算：去掉零碎噪点
# =========================
kernel = np.ones((5, 5), np.uint8)
binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# =========================
# 5) 找轮廓
# =========================
contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 彩色画布，方便画结果
result = cv2.cvtColor(response_vis, cv2.COLOR_GRAY2BGR)

# =========================
# 6) 彩色热力图
#    因为你要“越黑越红”，所以先取反
# =========================
press_strength = 255 - response_vis
heatmap = cv2.applyColorMap(press_strength, cv2.COLORMAP_JET)

if len(contours) > 0:
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    # 画轮廓
    cv2.drawContours(result, [largest], -1, (0, 255, 0), 2)

    # 外接矩形
    x, y, w, h = cv2.boundingRect(largest)
    cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 质心
    M = cv2.moments(largest)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
        print("接触中心:", (cx, cy))
    else:
        print("质心计算失败")

    print("接触面积(像素):", area)
    print("外接框: x={}, y={}, w={}, h={}".format(x, y, w, h))
else:
    print("没有检测到明显接触区域")

# =========================
# 6) 保存结果
# =========================
cv2.imwrite("response_vis.png", response_vis)
cv2.imwrite("response_binary.png", binary)
cv2.imwrite("response_binary_clean.png", binary_clean)
cv2.imwrite("response_result.png", result)
cv2.imwrite("Colormap.png", heatmap)