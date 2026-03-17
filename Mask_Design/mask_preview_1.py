import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.colors import black, white

# =========================
# 参数（你已确定）
# =========================
L_GLASS_MM = 10.0  # 玻璃 10x10 mm
L_ACTIVE_MM = 8.0  # 有效图案 8x8 mm
K = 400  # 8mm / 20um = 400
CELL_MM = L_ACTIVE_MM / K  # 0.02 mm
ACTIVE_OFFSET_MM = (L_GLASS_MM - L_ACTIVE_MM) / 2.0  # 1mm 边框

SEED = 7
MAX_RUN = 25

# PNG预览缩放（仅预览，不代表加工精度）
PX_PER_CELL = 3

# 输出文件
OUT_PDF = Path("mask_full_10x10mm_vector.pdf")                     # 1:1 加工版
OUT_PDF_VIEW = Path("mask_full_10x10mm_vector_VIEWx20.pdf")        # 20x 预览版（新增）
OUT_DXF = Path("mask_full_10x10mm_R12_SOLID.dxf")                  # AutoCAD稳版DXF

OUT_FULL_PNG = Path("mask_full_10x10_preview.png")
OUT_ACTIVE_EXACT = Path("mask_active_only_400x400.png")
OUT_ACTIVE_PREVIEW = Path("mask_active_only_preview.png")

# 方向标记（三角形）位置：左下角边框内（玻璃中心坐标 -5..+5）
TRI_CENTER = [(-4.8, -4.8), (-4.2, -4.8), (-4.8, -4.2)]

# 约定（按 ThinTact 常见振幅铬膜逻辑）：
# M=1 -> 透光开孔（white）
# M=0 -> 铬膜遮光（black）
# PDF/DXF里我们画的是“铬膜区域”（黑色）

# =========================
# 生成 phi & 可分离 mask
# =========================
def gen_phi_balanced(K, seed=7, max_run=25, max_tries=4000):
    rng = np.random.default_rng(seed)
    phi = np.ones(K, dtype=np.int8)
    phi[:K // 2] = -1

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
    return phi

def mask_from_phi(phi):
    # M_ij = 1 if phi_i == phi_j else 0
    return (phi[:, None] == phi[None, :]).astype(np.uint8)

def build_runs(phi):
    # 连续同号区间 runs: (start, end, sign)
    runs = []
    start = 0
    cur = int(phi[0])
    for i in range(1, len(phi)):
        if int(phi[i]) != cur:
            runs.append((start, i, cur))
            start = i
            cur = int(phi[i])
    runs.append((start, len(phi), cur))
    return runs

phi = gen_phi_balanced(K, seed=SEED, max_run=MAX_RUN)
M = mask_from_phi(phi)  # 1=open, 0=chrome
runs = build_runs(phi)

# =========================
# 生成“铬膜区域”的矩形块（关键：Y方向要和PNG一致）
# PNG里 row=0 在顶部，所以在几何坐标中要把它映射到 active 区域的“上边”
# 我们用 y = offset + (K - index)*cell 来实现翻转
# =========================
chrome_rects = []
for rs, re, rsign in runs:
    # rs/re 是行区间 [rs,re)
    # 让 rs=0 对应 active 的最上面 -> y 要用 (K - re) 到 (K - rs)
    y1 = ACTIVE_OFFSET_MM + (K - re) * CELL_MM
    y2 = ACTIVE_OFFSET_MM + (K - rs) * CELL_MM

    for cs, ce, csign in runs:
        if rsign == csign:
            continue  # open，不画

        x1 = ACTIVE_OFFSET_MM + cs * CELL_MM
        x2 = ACTIVE_OFFSET_MM + ce * CELL_MM
        chrome_rects.append((x1, y1, x2, y2))

# =========================
# (A) 输出三张PNG（保持不变）
# =========================
# 1) 纯有效图案：400x400（每格=1像素）
active_exact = Image.fromarray((M * 255).astype(np.uint8), mode="L")  # 白=open, 黑=chrome
active_exact.save(OUT_ACTIVE_EXACT)

# 2) 纯有效图案：放大预览
active_preview = active_exact.resize((K * PX_PER_CELL, K * PX_PER_CELL), resample=Image.NEAREST)
active_preview.save(OUT_ACTIVE_PREVIEW)

# 3) 整片10x10预览：居中贴8x8 + 外框 + 方向三角
active_px = K * PX_PER_CELL
glass_px = int(round(active_px * (L_GLASS_MM / L_ACTIVE_MM)))
border_px = (glass_px - active_px) // 2

canvas_png = Image.new("L", (glass_px, glass_px), 255)
canvas_png.paste(active_preview, (border_px, border_px))

draw = ImageDraw.Draw(canvas_png)
frame_w = max(2, PX_PER_CELL)
draw.rectangle(
    [border_px - 2, border_px - 2, border_px + active_px + 1, border_px + active_px + 1],
    outline=0,
    width=frame_w
)

def mm_to_px(x_mm, y_mm):
    px = int(round((x_mm / L_GLASS_MM + 0.5) * glass_px))
    py = int(round((0.5 - y_mm / L_GLASS_MM) * glass_px))
    return px, py

tri_px = [mm_to_px(x, y) for (x, y) in TRI_CENTER]
draw.polygon(tri_px, fill=0)
canvas_png.save(OUT_FULL_PNG)

# =========================
# (B) 输出矢量PDF（reportlab）—— 1:1 + 20x预览
# =========================
MM_TO_PT = 72.0 / 25.4
def mm(v): return v * MM_TO_PT

def export_pdf(out_path: Path, scale: float):
    c = rl_canvas.Canvas(str(out_path), pagesize=(mm(L_GLASS_MM * scale), mm(L_GLASS_MM * scale)))

    # 白底
    c.setFillColor(white)
    c.rect(0, 0, mm(L_GLASS_MM * scale), mm(L_GLASS_MM * scale), stroke=0, fill=1)

    # 铬膜区域：黑色填充
    c.setFillColor(black)
    for (x1, y1, x2, y2) in chrome_rects:
        c.rect(mm(x1 * scale), mm(y1 * scale),
               mm((x2 - x1) * scale), mm((y2 - y1) * scale),
               stroke=0, fill=1)

    # 外框与有效区框线（细线）
    c.setLineWidth(0.2)
    c.setStrokeColor(black)
    c.rect(0, 0, mm(L_GLASS_MM * scale), mm(L_GLASS_MM * scale), stroke=1, fill=0)
    c.rect(mm(ACTIVE_OFFSET_MM * scale), mm(ACTIVE_OFFSET_MM * scale),
           mm(L_ACTIVE_MM * scale), mm(L_ACTIVE_MM * scale), stroke=1, fill=0)

    # 方向三角（中心坐标转到 0..10）
    tri_bl = [(x + 5.0, y + 5.0) for (x, y) in TRI_CENTER]
    path = c.beginPath()
    path.moveTo(mm(tri_bl[0][0] * scale), mm(tri_bl[0][1] * scale))
    path.lineTo(mm(tri_bl[1][0] * scale), mm(tri_bl[1][1] * scale))
    path.lineTo(mm(tri_bl[2][0] * scale), mm(tri_bl[2][1] * scale))
    path.close()
    c.drawPath(path, stroke=0, fill=1)

    c.showPage()
    c.save()

export_pdf(OUT_PDF, scale=1.0)
export_pdf(OUT_PDF_VIEW, scale=20.0)

# =========================
# (C) 输出DXF（R12兼容：SOLID + LINE，AutoCAD最稳）
# =========================
def dxf_r12_header():
    # AC1009 = AutoCAD R12
    return "\n".join([
        "0", "SECTION", "2", "HEADER",
        "9", "$ACADVER", "1", "AC1009",
        "0", "ENDSEC",
        "0", "SECTION", "2", "ENTITIES"
    ]) + "\n"

def dxf_r12_footer():
    return "\n".join(["0", "ENDSEC", "0", "EOF"]) + "\n"

def dxf_line(x1, y1, x2, y2, layer="0"):
    return "\n".join([
        "0", "LINE",
        "8", layer,
        "10", f"{x1:.6f}", "20", f"{y1:.6f}",
        "11", f"{x2:.6f}", "21", f"{y2:.6f}",
    ]) + "\n"

def dxf_solid_rect(x1, y1, x2, y2, layer="0"):
    return "\n".join([
        "0", "SOLID",
        "8", layer,
        "10", f"{x1:.6f}", "20", f"{y1:.6f}",
        "11", f"{x2:.6f}", "21", f"{y1:.6f}",
        "12", f"{x2:.6f}", "22", f"{y2:.6f}",
        "13", f"{x1:.6f}", "23", f"{y2:.6f}",
    ]) + "\n"

entities = []

# 铬膜区域：SOLID
for (x1, y1, x2, y2) in chrome_rects:
    entities.append(dxf_solid_rect(x1, y1, x2, y2, layer="CHROME"))

# 玻璃外框：LINE
entities += [
    dxf_line(0, 0, L_GLASS_MM, 0, layer="OUTLINE"),
    dxf_line(L_GLASS_MM, 0, L_GLASS_MM, L_GLASS_MM, layer="OUTLINE"),
    dxf_line(L_GLASS_MM, L_GLASS_MM, 0, L_GLASS_MM, layer="OUTLINE"),
    dxf_line(0, L_GLASS_MM, 0, 0, layer="OUTLINE"),
]

# 有效区框：LINE
x0 = ACTIVE_OFFSET_MM
x1 = ACTIVE_OFFSET_MM + L_ACTIVE_MM
y0 = ACTIVE_OFFSET_MM
y1 = ACTIVE_OFFSET_MM + L_ACTIVE_MM
entities += [
    dxf_line(x0, y0, x1, y0, layer="OUTLINE"),
    dxf_line(x1, y0, x1, y1, layer="OUTLINE"),
    dxf_line(x1, y1, x0, y1, layer="OUTLINE"),
    dxf_line(x0, y1, x0, y0, layer="OUTLINE"),
]

# 方向三角：LINE
tri_bl = [(x + 5.0, y + 5.0) for (x, y) in TRI_CENTER]
entities += [
    dxf_line(tri_bl[0][0], tri_bl[0][1], tri_bl[1][0], tri_bl[1][1], layer="MARK"),
    dxf_line(tri_bl[1][0], tri_bl[1][1], tri_bl[2][0], tri_bl[2][1], layer="MARK"),
    dxf_line(tri_bl[2][0], tri_bl[2][1], tri_bl[0][0], tri_bl[0][1], layer="MARK"),
]

OUT_DXF.write_text(dxf_r12_header() + "".join(entities) + dxf_r12_footer(), encoding="ascii")

# =========================
# 打印绝对路径（避免你找不到生成文件）
# =========================
def abspath(p: Path) -> str:
    return str(p.resolve())

print("Saved:")
print(" -", abspath(OUT_PDF))
print(" -", abspath(OUT_PDF_VIEW))
print(" -", abspath(OUT_DXF))
print(" -", abspath(OUT_FULL_PNG))
print(" -", abspath(OUT_ACTIVE_EXACT))
print(" -", abspath(OUT_ACTIVE_PREVIEW))
print("Chrome rectangles:", len(chrome_rects))
print("Working directory:", Path.cwd().resolve())