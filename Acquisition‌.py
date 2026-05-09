"""
Author: Albus.Misrandy
"""
import os
import cv2

# 摄像头编号
# 一般笔记本自带摄像头是 0
# 外接摄像头可能是 1、2、3
# 打开摄像头
cap = cv2.VideoCapture(2)

# 判断摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头，请检查摄像头编号或连接状态")
    exit()

# 设置摄像头分辨率，可根据需要修改
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

# 创建保存图片的文件夹
save_dir = "Lensless_images"
os.makedirs(save_dir, exist_ok=True)

print("摄像头已打开")
print("按 s 拍照保存")
print("按 q 退出程序")

count = 0

while True:
    # 读取一帧图像
    ret, frame = cap.read()

    if not ret:
        print("读取摄像头画面失败")
        break

    # 显示摄像头画面
    cv2.imshow("Camera", frame)

    # 等待键盘输入
    key = cv2.waitKey(1) & 0xFF

    # 按 s 拍照
    if key == ord('s'):
        # 用当前时间作为图片名，避免覆盖
        count += 1
        filename = f"{count:03d}.jpg"
        filepath = os.path.join(save_dir, filename)

        # 保存图片
        cv2.imwrite(filepath, frame)
        print(f"照片已保存：{filepath}")

    # 按 q 退出
    elif key == ord('q'):
        print("退出程序")
        break

# 释放摄像头
cap.release()

# 关闭所有窗口
cv2.destroyAllWindows()