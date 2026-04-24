"""
Author: Albus.Misrandy
"""
import argparse
import cv2
import numpy as np
from pathlib import Path

def build_parser():
    parser = argparse.ArgumentParser(description="Lensless tactile reconstruction.")
    parser.add_argument("--Camera_ID", type=int, default=1)
    # parser.add_argument("ref_path", type=str, default="cabil", help="Reference_Path")
    return parser

def decode_fourcc(v):
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

def main():
    parser = build_parser()
    arg = parser.parse_args()

    cap = cv2.VideoCapture(arg.Camera_ID)

    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # =========================
    # 设置目标格式
    # =========================
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    cap.set(cv2.CAP_PROP_FPS, 25)
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
    # 参考帧保存目录
    # =========================
    save_dir = Path("translucent_ref")
    save_dir.mkdir(exist_ok=True)

    ref_num_frames = 30   # 参考帧数量

    print("按 q 键退出")
    print("按 c 键采集无接触参考帧（红通道 + 彩色参考图）")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取画面")
            break

        # =========================
        # 改这里：直接取红色通道
        # OpenCV 的顺序是 BGR，所以红色是 [:, :, 2]
        # =========================
        # red = frame[:, :, 2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # h, w = red.shape
        h, w = gray.shape
        # print(f"frame shape = ({h}, {w}), min = {red.min()}, max = {red.max()}, mean = {red.mean():.2f}")
        print(f"frame shape = ({h}, {w}), min = {gray.min()}, max = {gray.max()}, mean = {gray.mean():.2f}")

        # 原图太大时，缩小显示，避免窗口太卡
        show_frame = frame
        # show_red = red
        show_gray = gray

        if w > 1600:
            scale = 1280 / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            show_frame = cv2.resize(frame, (new_w, new_h))
            # show_red = cv2.resize(red, (new_w, new_h))
            show_gray = cv2.resize(gray, (new_w, new_h))

        cv2.imshow("Camera", show_frame)
        # cv2.imshow("Red_Channel", show_red)
        cv2.imshow("Gray", show_gray)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # =========================
        # 按 c 采集参考帧
        # =========================
        elif key == ord('c'):
            print(f"开始采集 {ref_num_frames} 张参考帧，请保持无接触、光照稳定...")

            ref_frames = []
            ref_color_frames = []

            for i in range(ref_num_frames):
                ret, frame = cap.read()
                if not ret:
                    print("中途读取失败，采集终止")
                    ref_frames = []
                    ref_color_frames = []
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                ref_frames.append(gray)

                ref_color_frames.append(frame.astype(np.float32))

                vis = np.clip(gray, 0, 255).astype(np.uint8)
                cv2.putText(
                    vis,
                    f"Collecting ref {i+1}/{ref_num_frames}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    255,
                    2
                )
                cv2.imshow("Gray", vis)
                cv2.waitKey(30)

            if len(ref_frames) == ref_num_frames:
                reference = np.mean(np.stack(ref_frames, axis=0), axis=0).astype(np.float32)

                np.save(save_dir / "reference.npy", reference)

                ref_vis = np.clip(reference, 0, 255).astype(np.uint8)
                cv2.imwrite(str(save_dir / "reference.png"), ref_vis)

                # =========================
                # 2) 彩色参考图（给你查看用）
                # =========================
                reference_color = np.mean(np.stack(ref_color_frames, axis=0), axis=0)
                reference_color = np.clip(reference_color, 0, 255).astype(np.uint8)
                cv2.imwrite(str(save_dir / "reference_color.png"), reference_color)

                print("参考帧保存完成：")
                print(save_dir / "reference.npy")
                print(save_dir / "reference.png")
                print(save_dir / "reference_color.png")
            else:
                print("参考帧采集失败，没有保存")
        # elif key == ord('c'):
        #     print(f"开始采集 {ref_num_frames} 张参考帧，请保持无接触、光照稳定...")

        #     ref_red_frames = []
        #     ref_color_frames = []

        #     for i in range(ref_num_frames):
        #         ret, frame = cap.read()
        #         if not ret:
        #             print("中途读取失败，采集终止")
        #             ref_red_frames = []
        #             ref_color_frames = []
        #             break

        #         # 红通道作为后面算法输入
        #         red = frame[:, :, 2].astype(np.float32)
        #         ref_red_frames.append(red)

        #         # 彩色帧也存起来，后面求平均后保存一张彩色参考图
        #         ref_color_frames.append(frame.astype(np.float32))

        #         vis = np.clip(red, 0, 255).astype(np.uint8)
        #         cv2.putText(
        #             vis,
        #             f"Collecting ref {i+1}/{ref_num_frames}",
        #             (20, 40),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             1.0,
        #             255,
        #             2
        #         )
        #         cv2.imshow("Red_Channel", vis)
        #         cv2.waitKey(30)

        #     if len(ref_red_frames) == ref_num_frames:
        #         # =========================
        #         # 1) 红通道参考图（给算法用）
        #         # =========================
        #         reference = np.mean(np.stack(ref_red_frames, axis=0), axis=0).astype(np.float32)
        #         np.save(save_dir / "reference.npy", reference)

        #         ref_vis = np.clip(reference, 0, 255).astype(np.uint8)
        #         cv2.imwrite(str(save_dir / "reference.png"), ref_vis)

        #         # =========================
        #         # 2) 彩色参考图（给你查看用）
        #         # =========================
        #         reference_color = np.mean(np.stack(ref_color_frames, axis=0), axis=0)
        #         reference_color = np.clip(reference_color, 0, 255).astype(np.uint8)
        #         cv2.imwrite(str(save_dir / "reference_color.png"), reference_color)

        #         print("参考帧保存完成：")
        #         print(save_dir / "reference.npy")
        #         print(save_dir / "reference.png")
        #         print(save_dir / "reference_color.png")
        #     else:
        #         print("参考帧采集失败，没有保存")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()