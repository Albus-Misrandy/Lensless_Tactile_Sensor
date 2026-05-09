"""
Author: Albus.Misrandy
"""
import argparse
import cv2
import numpy as np
from pathlib import Path

USE_DEBAND = False
USE_DIRECTIONAL_DEBAND = False
ROTATE_TO_HORIZONTAL_DEG = 0.0

def build_parser():
    parser = argparse.ArgumentParser(description="Lensless tactile reconstruction.")
    parser.add_argument("--Camera_ID", type=int, default=1)
    return parser

def decode_fourcc(v):
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

def normalize_for_display(img):
    """
    把 float 图像拉伸到 0~255 方便显示
    这里只用于显示，不改变后面真实计算值
    """
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    vis = (img - mn) / (mx - mn)
    vis = (vis * 255).astype(np.uint8)
    return vis

def smooth_1d(v, ksize=81):
    """
    对一维按行曲线做平滑，得到慢变趋势
    """
    if ksize % 2 == 0:
        ksize += 1
    v2 = v.reshape(-1, 1).astype(np.float32)
    v2 = cv2.GaussianBlur(v2, (1, ksize), 0)
    return v2[:, 0]

def remove_horizontal_banding_additive(img, margin=80, smooth_ksize=81):
    """
    对横向条纹做加性去条纹

    img: float32 图像
    margin: 用左右边缘区域估计每一行亮度
    smooth_ksize: 行方向平滑核大小
    """
    img = img.astype(np.float32)
    h, w = img.shape

    # 如果图像太窄，就退化为整行中位数
    if margin * 2 >= w:
        row_stat = np.median(img, axis=1)
    else:
        side_pixels = np.concatenate([img[:, :margin], img[:, -margin:]], axis=1)
        row_stat = np.median(side_pixels, axis=1)

    # 平滑后的曲线 = 慢变趋势
    row_trend = smooth_1d(row_stat, ksize=smooth_ksize)

    # 条纹项 = 实际每行统计 - 慢变趋势
    row_banding = row_stat - row_trend

    # 从每一行减掉这个条纹项
    corrected = img - row_banding[:, None]

    return corrected, row_stat, row_trend, row_banding

def estimate_rank1_background(img, smooth_ksize=81, eps=1e-6):
    """
    慢变低频背景估计：
    Y_o ≈ a b^T / mean(a)

    a: 每行统计
    b: 每列统计
    """
    img = img.astype(np.float32)

    a = np.median(img, axis=1).astype(np.float32)
    b = np.median(img, axis=0).astype(np.float32)

    a_s = smooth_1d(a, ksize=smooth_ksize)
    b_s = smooth_1d(b, ksize=smooth_ksize)

    bg = np.outer(a_s, b_s) / (np.mean(a_s) + eps)
    return bg, a_s, b_s

def rotate_image(img, angle_deg):
    """
    旋转图像，边界用反射填充，避免黑边
    """
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    rotated = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    return rotated

def remove_directional_banding_additive(img, rotate_to_horizontal_deg=35.0, margin=80, smooth_ksize=81):
    """
    先把斜向条纹旋转到接近水平，再调用按行去条纹，最后旋转回来
    """
    # 1) 先旋转，让条纹更接近水平
    rotated = rotate_image(img, rotate_to_horizontal_deg)

    # 2) 在旋转后的图上做原来的“按行去条纹”
    corrected_rot, row_stat, row_trend, row_banding = remove_horizontal_banding_additive(
        rotated,
        margin=margin,
        smooth_ksize=smooth_ksize
    )

    # 3) 把估计出来的条纹项也做成一张图，便于显示
    h_rot, w_rot = rotated.shape
    banding_map_rot = np.tile(row_banding[:, None], (1, w_rot))

    # 4) 旋转回原方向
    corrected = rotate_image(corrected_rot, -rotate_to_horizontal_deg)
    banding_map = rotate_image(banding_map_rot, -rotate_to_horizontal_deg)

    return corrected, rotated, corrected_rot, banding_map

def main():
    parser = build_parser()
    arg = parser.parse_args()

    cap = cv2.VideoCapture(arg.Camera_ID)

    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
     # =========================
    # 设置目标格式：MJPG 4000x3000 15fps
    # =========================
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

    # 给摄像头一点时间切换模式
    for _ in range(10):
        cap.read()
    
    # 读取实际生效的参数
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))

    print("当前实际参数：")
    print(f"FOURCC = {actual_fourcc}")
    print(f"Width  = {actual_width}")
    print(f"Height = {actual_height}")
    print(f"FPS    = {actual_fps}")
    print("AUTO_EXPOSURE =", cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))

    # =========================
    # 读取参考图
    # =========================
    ref_path = Path("translucent_ref/reference.npy")
    if not ref_path.exists():
        print("没有找到参考图:reference.npy")
        print("请先运行上一步，采集参考帧。")
        cap.release()
        return
    
    reference = np.load(ref_path).astype(np.float32)
    eps = 1e-6

    print("已加载参考图：", ref_path)
    print("reference shape =", reference.shape)
    print("按 q 键退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取画面")
            break

        # =========================
        # 直接取红通道
        # =========================
        # red = frame[:, :, 2].astype(np.float32)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 尺寸检查
        # if red.shape != reference.shape:
        #     print("当前红通道尺寸和参考图尺寸不一致")
        #     print("red.shape =", red.shape)
        #     print("reference.shape =", reference.shape)
        #     break
        if gray.shape != reference.shape:
            print("当前灰度图尺寸和参考图尺寸不一致")
            print("gray.shape =", gray.shape)
            print("reference.shape =", reference.shape)
            break

        # =========================
        # 参考归一化
        # =========================
        # norm_img = red / (reference + eps)
        norm_img = gray / (reference + eps)

        # =========================
        # 第二步：可选去条纹
        # =========================
        if USE_DEBAND:
            if USE_DIRECTIONAL_DEBAND:
                debanded, rotated_norm, corrected_rot, banding_map = remove_directional_banding_additive(
                    norm_img,
                    rotate_to_horizontal_deg=ROTATE_TO_HORIZONTAL_DEG,
                    margin=80,
                    smooth_ksize=81
                )
            else:
                debanded, row_stat, row_trend, row_banding = remove_horizontal_banding_additive(
                    norm_img,
                    margin=80,
                    smooth_ksize=81
                )
                banding_map = np.tile(row_banding[:, None], (1, norm_img.shape[1]))

            stage2_img = debanded
        else:
            debanded = None
            banding_map = None
            stage2_img = norm_img

        # =========================
        # 第三步：慢变背景去除
        # =========================
        bg_img, a_s, b_s = estimate_rank1_background(
            stage2_img,
            smooth_ksize=81,
            eps=eps
        )

        coding_like = stage2_img - bg_img

        h, w = gray.shape
        # h, w = red.shape
        print(
            f"frame=({h},{w}), "
            # f"red mean={red.mean():.2f}"
            f"gray mean={gray.mean():.2f}, "
            f"norm mean={norm_img.mean():.3f}, "
            f"stage2 mean={stage2_img.mean():.3f}, "
            f"coding_like mean={coding_like.mean():.3f}"
        )

        # =========================
        # 显示图像
        # =========================
        show_frame = frame
        show_gray = np.clip(gray, 0, 255).astype(np.uint8)
        # show_red = np.clip(red, 0, 255).astype(np.uint8)
        show_norm = normalize_for_display(norm_img)
        show_bg = normalize_for_display(bg_img)
        show_coding = normalize_for_display(coding_like)

        if USE_DEBAND:
            show_stage2 = normalize_for_display(debanded)
            show_banding = normalize_for_display(banding_map)
        else:
            show_stage2 = normalize_for_display(stage2_img)
            show_banding = np.zeros_like(show_stage2, dtype=np.uint8)

        if w > 1600:
            scale = 1280 / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            show_frame = cv2.resize(frame, (new_w, new_h))
            # show_red = cv2.resize(show_red, (new_w, new_h))
            show_gray = cv2.resize(show_gray, (new_w, new_h))
            show_norm = cv2.resize(show_norm, (new_w, new_h))
            show_stage2 = cv2.resize(show_stage2, (new_w, new_h))
            show_bg = cv2.resize(show_bg, (new_w, new_h))
            show_coding = cv2.resize(show_coding, (new_w, new_h))
            show_banding = cv2.resize(show_banding, (new_w, new_h))

        # print("min =", gray.min(), "max =", gray.max(), "mean =", gray.mean())
        cv2.imshow("Camera", show_frame)
        # cv2.imshow("Red_Channel", show_red)
        cv2.imshow("Gray", show_gray)
        cv2.imshow("Normalized", show_norm)

        if USE_DEBAND:
            cv2.imshow("Debanded", show_stage2)
            cv2.imshow("Estimated_Banding", show_banding)
        else:
            cv2.imshow("Stage2_NoDeband", show_stage2)

        cv2.imshow("Estimated_Background", show_bg)
        cv2.imshow("Coding_Like", show_coding)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
