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
OUT_PDF = Path("mask_full_10x10mm_vector.pdf")
OUT_DXF = Path("mask_full_10x10mm.dxf")
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

# 利用可分离结构压缩：只画“铬膜区域”的大矩形块
chrome_rects = []
for rs, re, rsign in runs:
    y1 = ACTIVE_OFFSET_MM + rs * CELL_MM
    y2 = ACTIVE_OFFSET_MM + re * CELL_MM
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
# (B) 输出矢量PDF（reportlab）
# =========================
MM_TO_PT = 72.0 / 25.4


def mm(v): return v * MM_TO_PT


c = rl_canvas.Canvas(str(OUT_PDF), pagesize=(mm(L_GLASS_MM), mm(L_GLASS_MM)))

# 白底
c.setFillColor(white)
c.rect(0, 0, mm(L_GLASS_MM), mm(L_GLASS_MM), stroke=0, fill=1)

# 铬膜区域：黑色填充（无描边）
c.setFillColor(black)
for (x1, y1, x2, y2) in chrome_rects:
    c.rect(mm(x1), mm(y1), mm(x2 - x1), mm(y2 - y1), stroke=0, fill=1)

# 外框与有效区框线（细线）
c.setLineWidth(0.2)
c.setStrokeColor(black)
c.rect(0, 0, mm(L_GLASS_MM), mm(L_GLASS_MM), stroke=1, fill=0)
c.rect(mm(ACTIVE_OFFSET_MM), mm(ACTIVE_OFFSET_MM), mm(L_ACTIVE_MM), mm(L_ACTIVE_MM), stroke=1, fill=0)

# 方向三角：把中心坐标(-5..+5)转成(0..10)
tri_bl = [(x + 5.0, y + 5.0) for (x, y) in TRI_CENTER]
path = c.beginPath()
path.moveTo(mm(tri_bl[0][0]), mm(tri_bl[0][1]))
path.lineTo(mm(tri_bl[1][0]), mm(tri_bl[1][1]))
path.lineTo(mm(tri_bl[2][0]), mm(tri_bl[2][1]))
path.close()
c.drawPath(path, stroke=0, fill=1)

c.showPage()
c.save()


# =========================
# (C) 输出DXF（ASCII LWPOLYLINE）
# =========================
def dxf_header():
    return "\n".join([
        "0", "SECTION", "2", "HEADER",
        "9", "$INSUNITS", "70", "4",  # 4=mm
        "0", "ENDSEC",
        "0", "SECTION", "2", "TABLES",
        "0", "TABLE", "2", "LAYER", "70", "3",
        "0", "LAYER", "2", "CHROME", "70", "0", "62", "7", "6", "CONTINUOUS",
        "0", "LAYER", "2", "OUTLINE", "70", "0", "62", "7", "6", "CONTINUOUS",
        "0", "LAYER", "2", "MARK", "70", "0", "62", "7", "6", "CONTINUOUS",
        "0", "ENDTAB",
        "0", "ENDSEC",
        "0", "SECTION", "2", "ENTITIES"
    ]) + "\n"


def dxf_footer():
    return "\n".join(["0", "ENDSEC", "0", "EOF"]) + "\n"


def lwpolyline(points, layer="0", closed=True):
    lines = ["0", "LWPOLYLINE", "8", layer, "90", str(len(points)), "70", ("1" if closed else "0")]
    for (x, y) in points:
        lines += ["10", f"{x:.6f}", "20", f"{y:.6f}"]
    return "\n".join(lines) + "\n"


entities = []

# 铬膜矩形块
for (x1, y1, x2, y2) in chrome_rects:
    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    entities.append(lwpolyline(pts, layer="CHROME", closed=True))

# 玻璃外框
entities.append(lwpolyline([(0, 0), (L_GLASS_MM, 0), (L_GLASS_MM, L_GLASS_MM), (0, L_GLASS_MM)],
                           layer="OUTLINE", closed=True))
# 有效区框
entities.append(lwpolyline([(ACTIVE_OFFSET_MM, ACTIVE_OFFSET_MM),
                            (ACTIVE_OFFSET_MM + L_ACTIVE_MM, ACTIVE_OFFSET_MM),
                            (ACTIVE_OFFSET_MM + L_ACTIVE_MM, ACTIVE_OFFSET_MM + L_ACTIVE_MM),
                            (ACTIVE_OFFSET_MM, ACTIVE_OFFSET_MM + L_ACTIVE_MM)],
                           layer="OUTLINE", closed=True))

# 方向三角
entities.append(lwpolyline(tri_bl, layer="MARK", closed=True))

OUT_DXF.write_text(dxf_header() + "".join(entities) + dxf_footer(), encoding="ascii")

print("Saved:")
print(" -", OUT_PDF)
print(" -", OUT_DXF)
print(" -", OUT_FULL_PNG)
print(" -", OUT_ACTIVE_EXACT)
print(" -", OUT_ACTIVE_PREVIEW)
print("Chrome rectangles:", len(chrome_rects))
