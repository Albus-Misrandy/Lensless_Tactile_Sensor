import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.colors import black, white

# =========================
# 参数（你已确定）
# =========================
L_GLASS_MM = 10.0
L_ACTIVE_MM = 8.0
K = 400
CELL_MM = L_ACTIVE_MM / K
ACTIVE_OFFSET_MM = (L_GLASS_MM - L_ACTIVE_MM) / 2.0

SEED = 7
MAX_RUN = 25
PX_PER_CELL = 3

# 输出文件
OUT_PDF = Path("mask_full_10x10mm_vector.pdf")  # 1:1 加工版
OUT_PDF_VIEW = Path("mask_full_10x10mm_vector_VIEWx20.pdf")  # 20x 预览版

OUT_DXF_SOLID = Path("mask_full_10x10mm_R12_SOLID.dxf")  # 生产用DXF（发厂）
OUT_DXF_VIEW_SOLID = Path("mask_full_10x10mm_VIEWx20_SOLID.dxf")  # 查看用DXF（放大20x，仍是SOLID填充）

OUT_FULL_PNG = Path("mask_full_10x10_preview.png")
OUT_ACTIVE_EXACT = Path("mask_active_only_400x400.png")
OUT_ACTIVE_PREVIEW = Path("mask_active_only_preview.png")

TRI_CENTER = [(-4.8, -4.8), (-4.2, -4.8), (-4.8, -4.2)]


# 约定：
# M=1 -> 白 -> 透光
# M=0 -> 黑 -> 铬膜遮光
# PDF/DXF 都是画 M=0 的区域（铬膜区域）

# =========================
# 生成 phi & mask
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
    return (phi[:, None] == phi[None, :]).astype(np.uint8)


def build_runs(phi):
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
# chrome_rects（确保与PNG方向一致：row0在顶部）
# =========================
chrome_rects = []
for rs, re, rsign in runs:
    y1 = ACTIVE_OFFSET_MM + (K - re) * CELL_MM
    y2 = ACTIVE_OFFSET_MM + (K - rs) * CELL_MM
    for cs, ce, csign in runs:
        if rsign == csign:
            continue
        x1 = ACTIVE_OFFSET_MM + cs * CELL_MM
        x2 = ACTIVE_OFFSET_MM + ce * CELL_MM
        chrome_rects.append((x1, y1, x2, y2))

# =========================
# (A) PNG 三张
# =========================
active_exact = Image.fromarray((M * 255).astype(np.uint8), mode="L")
active_exact.save(OUT_ACTIVE_EXACT)

active_preview = active_exact.resize((K * PX_PER_CELL, K * PX_PER_CELL), resample=Image.NEAREST)
active_preview.save(OUT_ACTIVE_PREVIEW)

active_px = K * PX_PER_CELL
glass_px = int(round(active_px * (L_GLASS_MM / L_ACTIVE_MM)))
border_px = (glass_px - active_px) // 2

canvas_png = Image.new("L", (glass_px, glass_px), 255)
canvas_png.paste(active_preview, (border_px, border_px))
draw = ImageDraw.Draw(canvas_png)

frame_w = max(2, PX_PER_CELL)
draw.rectangle([border_px - 2, border_px - 2, border_px + active_px + 1, border_px + active_px + 1],
               outline=0, width=frame_w)


def mm_to_px(x_mm, y_mm):
    px = int(round((x_mm / L_GLASS_MM + 0.5) * glass_px))
    py = int(round((0.5 - y_mm / L_GLASS_MM) * glass_px))
    return px, py


tri_px = [mm_to_px(x, y) for (x, y) in TRI_CENTER]
draw.polygon(tri_px, fill=0)

canvas_png.save(OUT_FULL_PNG)

# =========================
# (B) PDF：1:1 + 20x
# =========================
MM_TO_PT = 72.0 / 25.4


def mm(v): return v * MM_TO_PT


def export_pdf(out_path: Path, scale: float):
    c = rl_canvas.Canvas(str(out_path),
                         pagesize=(mm(L_GLASS_MM * scale), mm(L_GLASS_MM * scale)))

    c.setFillColor(white)
    c.rect(0, 0, mm(L_GLASS_MM * scale), mm(L_GLASS_MM * scale), stroke=0, fill=1)

    c.setFillColor(black)
    for (x1, y1, x2, y2) in chrome_rects:
        c.rect(mm(x1 * scale), mm(y1 * scale),
               mm((x2 - x1) * scale), mm((y2 - y1) * scale),
               stroke=0, fill=1)

    c.setLineWidth(0.2)
    c.setStrokeColor(black)
    c.rect(0, 0, mm(L_GLASS_MM * scale), mm(L_GLASS_MM * scale), stroke=1, fill=0)
    c.rect(mm(ACTIVE_OFFSET_MM * scale), mm(ACTIVE_OFFSET_MM * scale),
           mm(L_ACTIVE_MM * scale), mm(L_ACTIVE_MM * scale), stroke=1, fill=0)

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
# (C) DXF：R12 + SOLID + LINE（顶点顺序修复，避免菱形显示）
# =========================
def dxf_r12_header():
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


def dxf_solid_rect_NO_DIAG(x1, y1, x2, y2, layer="0"):
    """
    关键：点1-点3这条边变成矩形左边界，不再是内部对角线
    点顺序：
      1:(x1,y1) 2:(x2,y1) 3:(x1,y2) 4:(x2,y2)
    """
    return "\n".join([
        "0", "SOLID",
        "8", layer,
        "10", f"{x1:.6f}", "20", f"{y1:.6f}",  # pt1
        "11", f"{x2:.6f}", "21", f"{y1:.6f}",  # pt2
        "12", f"{x1:.6f}", "22", f"{y2:.6f}",  # pt3  (注意这里是x1,y2)
        "13", f"{x2:.6f}", "23", f"{y2:.6f}",  # pt4
    ]) + "\n"


def export_dxf_solid(out_path: Path, scale: float):
    ents = []

    # 铬膜区域：SOLID
    for (x1, y1, x2, y2) in chrome_rects:
        ents.append(dxf_solid_rect_NO_DIAG(x1 * scale, y1 * scale, x2 * scale, y2 * scale, layer="CHROME"))

    # 外框/有效框/方向三角（线）
    Lg = L_GLASS_MM * scale
    La = L_ACTIVE_MM * scale
    off = ACTIVE_OFFSET_MM * scale

    ents += [
        dxf_line(0, 0, Lg, 0, layer="OUTLINE"),
        dxf_line(Lg, 0, Lg, Lg, layer="OUTLINE"),
        dxf_line(Lg, Lg, 0, Lg, layer="OUTLINE"),
        dxf_line(0, Lg, 0, 0, layer="OUTLINE"),
    ]
    ents += [
        dxf_line(off, off, off + La, off, layer="OUTLINE"),
        dxf_line(off + La, off, off + La, off + La, layer="OUTLINE"),
        dxf_line(off + La, off + La, off, off + La, layer="OUTLINE"),
        dxf_line(off, off + La, off, off, layer="OUTLINE"),
    ]

    tri_bl = [(x + 5.0, y + 5.0) for (x, y) in TRI_CENTER]
    tri_bl = [(p[0] * scale, p[1] * scale) for p in tri_bl]
    ents += [
        dxf_line(tri_bl[0][0], tri_bl[0][1], tri_bl[1][0], tri_bl[1][1], layer="MARK"),
        dxf_line(tri_bl[1][0], tri_bl[1][1], tri_bl[2][0], tri_bl[2][1], layer="MARK"),
        dxf_line(tri_bl[2][0], tri_bl[2][1], tri_bl[0][0], tri_bl[0][1], layer="MARK"),
    ]

    out_path.write_text(dxf_r12_header() + "".join(ents) + dxf_r12_footer(), encoding="ascii")


# 生产用（1:1）
export_dxf_solid(OUT_DXF_SOLID, scale=1.0)
# 查看用（放大20x）
export_dxf_solid(OUT_DXF_VIEW_SOLID, scale=20.0)


# 打印绝对路径
def abspath(p: Path) -> str:
    return str(p.resolve())


print("Saved:")
print(" -", abspath(OUT_PDF))
print(" -", abspath(OUT_PDF_VIEW))
print(" -", abspath(OUT_DXF_SOLID))
print(" -", abspath(OUT_DXF_VIEW_SOLID))
print(" -", abspath(OUT_FULL_PNG))
print(" -", abspath(OUT_ACTIVE_EXACT))
print(" -", abspath(OUT_ACTIVE_PREVIEW))
print("Chrome rectangles:", len(chrome_rects))
print("Working directory:", Path.cwd().resolve())
