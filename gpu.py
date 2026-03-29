#!/usr/bin/env python3
"""
让 GPU 持续保持约 10% 利用率的脚本。
原理：通过交替执行短暂的 GPU 计算和 sleep，控制占空比来达到目标利用率。
用法：python gpu_keep_10.py [--target 10] [--gpu 0]
"""

import argparse
import time
import torch


def main():
    parser = argparse.ArgumentParser(description="维持 GPU 在指定利用率附近")
    parser.add_argument("--target", type=float, default=10.0, help="目标 GPU 利用率 (%%，默认 10)")
    parser.add_argument("--gpu", type=int, default=0, help="使用的 GPU 编号 (默认 0)")
    parser.add_argument("--matrix-size", type=int, default=1024, help="矩阵大小 (默认 1024)")
    args = parser.parse_args()

    target = args.target / 100.0  # 转为 0~1 比例
    device = torch.device(f"cuda:{args.gpu}")

    print(f"🚀 目标 GPU 利用率: {args.target:.0f}%  |  GPU: {args.gpu}  |  矩阵大小: {args.matrix_size}")
    print("按 Ctrl+C 停止\n")

    # 预分配矩阵
    a = torch.randn(args.matrix_size, args.matrix_size, device=device)
    b = torch.randn(args.matrix_size, args.matrix_size, device=device)

    # 占空比参数
    work_time = 0.01  # 每轮计算时间 (秒)
    sleep_time = work_time * (1.0 - target) / max(target, 0.01)

    print(f"⚙️  计算 {work_time*1000:.1f}ms / 休眠 {sleep_time*1000:.1f}ms  (占空比 ≈ {target*100:.0f}%)")

    try:
        while True:
            # --- 计算阶段 ---
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < work_time:
                torch.mm(a, b)
            torch.cuda.synchronize(device)

            # --- 休眠阶段 ---
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n✅ 已停止")


if __name__ == "__main__":
    main()
