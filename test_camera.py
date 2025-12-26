#! /usr/bin/env python3
import cv2, sys
import numpy as np
from franka_env.camera.rs_capture import RSCapture  # 确保文件名正确

camera_serial = ['230422272349', '419122270589', '130322274099']


def main():
    # TODO: 把你的相机序列号填进去
    serial = camera_serial[int(sys.argv[1])]

    cap = RSCapture(
        name="cam0",
        serial_number=serial,
        dim=(1280, 720),
        fps=15,
        depth=True,       # 如只想看彩色图，改为 False
        exposure=40000
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Frame grab failed")
                continue

            # 仅彩色：H x W x 3
            if frame.ndim == 3 and frame.shape[2] == 3:
                color = frame
                # 确保是 uint8
                if color.dtype != np.uint8:
                    color = cv2.convertScaleAbs(color)
                cv2.imshow("Color", color)

            # 彩色 + 深度：H x W x 4
            elif frame.ndim == 3 and frame.shape[2] == 4:
                # 分离彩色与深度
                color = frame[:, :, :3]  # 可能是 uint16
                depth_raw = frame[:, :, 3]  # uint16

                # 1) 彩色图转成 uint8
                if color.dtype != np.uint8:
                    # convertScaleAbs 会做缩放并转换到 uint8，比直接 astype 更安全
                    color_vis = cv2.convertScaleAbs(color)
                else:
                    color_vis = color

                # 2) 深度图归一化并伪彩色
                # 深度是 16 位，先归一化到 0-255 再转 uint8
                min_depth = 0   # 0.3m
                max_depth = 2000  # 2.0m
                depth_raw = np.clip(depth_raw, min_depth, max_depth)
                depth_vis = (depth_raw - min_depth) / (max_depth - min_depth) * 255.0
                depth_vis = depth_vis.astype(np.uint8)

                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

                # 3) 保证尺寸完全一致（一般对齐后本来就一致，这里是保险做法）
                xx = 0
                yy = 0
                xxx = xx + 512
                yyy = yy + 512
                color_vis = color_vis[xx:xxx, yy:yyy, :]
                depth_vis = depth_vis[xx:xxx, yy:yyy, :]

                # 4) 横向拼接显示
                combine = cv2.hconcat([color_vis, depth_vis])
                cv2.imshow("Color + Depth", combine)

            else:
                # 不符合预期的情况，输出一下信息用于调试
                print(f"Unexpected frame shape: {frame.shape}, dtype: {frame.dtype}")
                continue

            # 按 q 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass

    finally:
        cap.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

