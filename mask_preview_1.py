import numpy as np
from PIL import Image, ImageDraw

# -----------------------------
# 你已确定的参数
# -----------------------------
L_GLASS_MM = 10.0          # 整片玻璃 10x10 mm
L_ACTIVE_MM = 8.0          # 有效图案 8x8 mm
CELL_UM = 20.0             # 单元 20 um
K = 400                    # 8mm / 20um = 400

# 版图渲染：为了预览，设定每个mask格子在预览图里对应多少像素（仅预览，不代表加工精度）
PX_PER_CELL = 3

# 输出文件名
OUT_FULL = "mask_full_10x10_preview.png"
OUT_ACTIVE_EXACT = "mask_active_only_400x400.png"
OUT_ACTIVE_PREVIEW = "mask_active_only_preview.png"

# 颜色约定（按 ThinTact 常见振幅铬膜理解）
# M=1 -> 透光开孔（transparent/open） -> 输出图用白色 255 表示
# M=0 -> 铬膜遮光（opaque/chrome）    -> 输出图用黑色 0 表示
# 如果加工厂要求反过来，把 INVERT_OUTPUT 改成 True 即可
INVERT_OUTPUT = False

# -----------------------------
# 生成 phi：平衡 + 限制最大连续段（减少大条纹风险）
# -----------------------------
def gen_phi_balanced(K, seed=7, max_run=25, max_tries=4000):
    """
    生成长度K的±1序列 phi：
    - +1/-1 各一半（整体开孔率更稳定）
    - 限制最长连续段<=max_run（避免出现大条纹）
    """
    rng = np.random.default_rng(seed)
    phi = np.ones(K, dtype=np.int8)
    phi[:K//2] = -1

    def max_consecutive_run(arr):
        run = best = 1
        for i in range(1, len(arr)):
            if arr[i] == arr[i - 1]:
                run += 1
                best = max(best, run)
            else:
                run = 1
        return best

    for _ in range(max_tries):
        rng.shuffle(phi)
        if max_consecutive_run(phi) <= max_run:
            return phi
    # 实在找不到就返回一个平衡随机序列（一般也能用）
    return phi

def mask_from_phi(phi):
    """
    可分离二值mask：
    M_ij = 1 if phi_i == phi_j else 0
    """
    return (phi[:, None] == phi[None, :]).astype(np.uint8)

# 生成 mask
phi = gen_phi_balanced(K, seed=7, max_run=25)
M = mask_from_phi(phi)  # 0/1，按上面的约定：1=透光，0=铬

# -----------------------------
# 1) 输出“只包含有效图案”的两张图
# -----------------------------
# (a) 精确版：400x400（每个格子=1像素）
active_exact = Image.fromarray((M * 255).astype(np.uint8), mode="L")
if INVERT_OUTPUT:
    active_exact = Image.fromarray((255 - np.array(active_exact)).astype(np.uint8), mode="L")
active_exact.save(OUT_ACTIVE_EXACT)

# (b) 预览版：放大方便看
active_preview = active_exact.resize((K * PX_PER_CELL, K * PX_PER_CELL), resample=Image.NEAREST)
active_preview.save(OUT_ACTIVE_PREVIEW)

# -----------------------------
# 2) 输出“整片10x10mm预览图”：居中放8x8mm有效图案 + 外框 + 方向三角（无十字）
# -----------------------------
active_px = K * PX_PER_CELL
glass_px = int(round(active_px * (L_GLASS_MM / L_ACTIVE_MM)))  # 按比例放大到10mm画布
border_px = (glass_px - active_px) // 2

# 白底画布
canvas = Image.new("L", (glass_px, glass_px), 255)
canvas.paste(active_preview, (border_px, border_px))

draw = ImageDraw.Draw(canvas)

# 外框：画在有效区周围（黑色线）
frame_w = max(2, PX_PER_CELL)
draw.rectangle(
    [border_px - 2, border_px - 2, border_px + active_px + 1, border_px + active_px + 1],
    outline=0,
    width=frame_w
)

# 方向标记：左下角边框内一个实心三角（黑色）
# 为了更直观，我们用物理坐标（mm）定义，再转换到像素坐标
def mm_to_px(x_mm, y_mm):
    """
    物理坐标系：玻璃中心(0,0)，+x右，+y上
    图像坐标：左上角(0,0)，y向下，所以需要翻转y
    """
    px = int(round((x_mm / L_GLASS_MM + 0.5) * glass_px))
    py = int(round((0.5 - y_mm / L_GLASS_MM) * glass_px))
    return px, py

# 放在左下角边框区域（不要压到有效8x8mm区域）
tri_mm = [(-4.8, -4.8), (-4.2, -4.8), (-4.8, -4.2)]
tri_px = [mm_to_px(x, y) for x, y in tri_mm]
draw.polygon(tri_px, fill=0)

# 如果整体反相输出（给某些加工厂），最后再反相一次
if INVERT_OUTPUT:
    canvas = Image.fromarray((255 - np.array(canvas)).astype(np.uint8), mode="L")

canvas.save(OUT_FULL)

print("Saved:")
print(" -", OUT_FULL)
print(" -", OUT_ACTIVE_EXACT)
print(" -", OUT_ACTIVE_PREVIEW)