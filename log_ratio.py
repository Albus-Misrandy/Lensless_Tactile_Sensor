"""
Author: Albus.Misrandy
"""
import argparse
import time
import cv2
import numpy as np
from pathlib import Path

OUT_DIR = Path("out_camera_log_ratio")
OUT_DIR.mkdir(exist_ok=True)

def build_parser():
    parser = argparse.ArgumentParser(description="Lensless tactile reconstruction.")
    parser.add_argument("--Camera_ID", type=int, default=2)
    parser.add_argument("--FRAME_WIDTH", type=int, default=1920)
    parser.add_argument("--FRAME_HEIGHT", type=int, default=1080)
    parser.add_argument("--FPS", type=int, default=30)
    parser.add_argument("--CHANNEL", type=str, default="gray", help="choose color channels.")
    # parser.add_argument()
    return parser

def decode_fourcc(v):
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])


def extract_channel(frame_bgr: np.ndarray, channel: str) -> np.ndarray:
    """
    从 BGR 图像中提取指定通道，返回 float32。
    """
    b, g, r = cv2.split(frame_bgr)

    if channel == "gray":
        out = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    elif channel == "r":
        out = r
    elif channel == "g":
        out = g
    elif channel == "b":
        out = b
    else:
        raise ValueError("CHANNEL 只能是 'gray', 'r', 'g', 'b'")

    return out.astype(np.float32)


def robust_normalize_for_display(x: np.ndarray, p_low=1, p_high=99) -> np.ndarray:
    """
    把任意 float 图像稳健归一化到 0~255，方便保存和显示。
    """
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)

    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.uint8)

    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0, 1)

    return (y * 255).astype(np.uint8)


def remove_low_frequency_background(x: np.ndarray, sigma: float = 35) -> np.ndarray:
    """
    去掉低频背景。
    你的灰色涂层图像中心偏暗、边缘偏亮，这一步就是专门处理这个问题。
    """
    bg = cv2.GaussianBlur(x, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    highpass = x - bg
    return highpass

def compute_log_ratio_response(current: np.ndarray, reference: np.ndarray):
    """
    核心算法：
    1. log-ratio
    2. 去低频背景
    """
    delta = np.log((current + 1.0) / (reference + 1.0))
    delta_hp = remove_low_frequency_background(delta, 45)
    return delta, delta_hp

def make_colormap(x_uint8: np.ndarray) -> np.ndarray:
    """
    把灰度响应图转成伪彩色，方便观察。
    """
    return cv2.applyColorMap(x_uint8, cv2.COLORMAP_TURBO)

def capture_reference(cap: cv2.VideoCapture, n_frames: int, channel: str) -> np.ndarray:
    """
    采集无按压 reference。
    为了减少噪声，连续采 n_frames 帧取平均。
    """
    print(f"\n开始采集 reference，请保持无按压状态，采集 {n_frames} 帧...")

    frames = []

    # 先丢掉几帧，让相机稳定
    for _ in range(5):
        cap.read()

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("采集 reference 时读取摄像头失败")

        img = extract_channel(frame, channel)
        frames.append(img)

        cv2.imshow("capturing reference", frame)
        cv2.waitKey(1)

    cv2.destroyWindow("capturing reference")

    ref = np.mean(frames, axis=0).astype(np.float32)

    print("reference 采集完成")
    print(f"ref min/max/mean: {ref.min():.2f}, {ref.max():.2f}, {ref.mean():.2f}")

    return ref

def save_current_results(frame_bgr, current, reference, delta, delta_hp):
    """
    保存当前结果。
    """
    t = time.strftime("%Y%m%d_%H%M%S")

    cv2.imwrite(str(OUT_DIR / f"{t}_raw_bgr.png"), frame_bgr)

    current_vis = robust_normalize_for_display(current)
    reference_vis = robust_normalize_for_display(reference)
    delta_vis = robust_normalize_for_display(delta, 1, 99)
    delta_hp_vis = robust_normalize_for_display(delta_hp, 1, 99)
    delta_hp_color = make_colormap(delta_hp_vis)

    cv2.imwrite(str(OUT_DIR / f"{t}_current_channel.png"), current_vis)
    cv2.imwrite(str(OUT_DIR / f"{t}_reference.png"), reference_vis)
    cv2.imwrite(str(OUT_DIR / f"{t}_log_ratio.png"), delta_vis)
    cv2.imwrite(str(OUT_DIR / f"{t}_log_ratio_highpass.png"), delta_hp_vis)
    cv2.imwrite(str(OUT_DIR / f"{t}_log_ratio_highpass_color.png"), delta_hp_color)

    # 同时保存 float 数据，后面做算法会用到
    np.save(str(OUT_DIR / f"{t}_current.npy"), current)
    np.save(str(OUT_DIR / f"{t}_reference.npy"), reference)
    np.save(str(OUT_DIR / f"{t}_delta.npy"), delta)
    np.save(str(OUT_DIR / f"{t}_delta_hp.npy"), delta_hp)

    print("\n已保存当前结果到:", OUT_DIR.resolve())
    print("重点看:")
    print(f" - {t}_log_ratio_highpass.png")
    print(f" - {t}_log_ratio_highpass_color.png")

def main():
    parser = build_parser()
    arg = parser.parse_args()

    cap = cv2.VideoCapture(arg.Camera_ID)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, arg.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, arg.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, arg.FPS)

    print("===== Camera Settings =====")
    print("Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("FPS:", cap.get(cv2.CAP_PROP_FPS))
    print("FOURCC:", decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC)))
    print("AUTO_EXPOSURE:", cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
    print("EXPOSURE:", cap.get(cv2.CAP_PROP_EXPOSURE))
    print("GAIN:", cap.get(cv2.CAP_PROP_GAIN))
    print("===========================")

    reference = None

    while True:
        ret, frame = cap.read()

        if not ret:
            print("读取摄像头失败")
            break

        current = extract_channel(frame, arg.CHANNEL)
        # 原始画面
        show_raw = frame.copy()

        # 显示当前通道
        current_vis = robust_normalize_for_display(current)

        if reference is None:
            cv2.putText(
                show_raw,
                "Press 'r' to capture REST reference",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

            cv2.imshow("raw camera", show_raw)
            cv2.imshow("current channel", current_vis)

        else:
            delta, delta_hp = compute_log_ratio_response(current, reference)

            delta_vis = robust_normalize_for_display(delta, 1, 99)
            delta_hp_vis = robust_normalize_for_display(delta_hp, 1, 99)
            delta_hp_color = make_colormap(delta_hp_vis)

            # 显示统计信息
            text1 = f"cur mean={current.mean():.1f}, min={current.min():.0f}, max={current.max():.0f}"
            text2 = f"delta_hp std={delta_hp.std():.6f}"

            cv2.putText(show_raw, text1, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(show_raw, text2, (30, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("raw camera", show_raw)
            cv2.imshow("current channel", current_vis)
            cv2.imshow("log-ratio", delta_vis)
            cv2.imshow("log-ratio highpass", delta_hp_vis)
            cv2.imshow("log-ratio highpass color", delta_hp_color)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            reference = capture_reference(cap, 30, arg.CHANNEL)

        elif key == ord("s"):
            if reference is None:
                print("还没有 reference，请先按 r 采集无按压基线。")
            else:
                delta, delta_hp = compute_log_ratio_response(current, reference)
                save_current_results(frame, current, reference, delta, delta_hp)

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()