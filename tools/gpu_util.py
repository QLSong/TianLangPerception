#!/usr/bin/env python
"""采样 nvidia-smi，统计 GPU 利用率的平均值。Ctrl+C 结束并打印结果。"""
import argparse
import subprocess
import time


def query_gpus():
    out = subprocess.check_output(
        [
            'nvidia-smi',
            '--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total',
            '--format=csv,noheader,nounits',
        ],
        text=True,
    )
    rows = []
    for line in out.strip().splitlines():
        index, util, mem_util, mem_used, mem_total = [part.strip() for part in line.split(',')]
        rows.append({
            'index': int(index),
            'util': float(util),
            'mem_util': float(mem_util),
            'mem_used': float(mem_used),
            'mem_total': float(mem_total),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description='统计 GPU 平均利用率')
    parser.add_argument('--interval', type=float, default=1.0, help='采样间隔，秒')
    parser.add_argument('--duration', type=float, default=0.0, help='采样时长，秒。0 表示一直采样到 Ctrl+C')
    parser.add_argument('--gpu', type=int, default=None, help='只统计这一张卡，默认全部')
    args = parser.parse_args()

    sums = {}
    count = 0
    start = time.time()
    print('index  util%  mem_util%  mem_used_MiB  samples  avg_util%')
    try:
        while args.duration <= 0 or time.time() - start < args.duration:
            for row in query_gpus():
                if args.gpu is not None and row['index'] != args.gpu:
                    continue
                slot = sums.setdefault(row['index'], {'util': 0.0, 'mem_util': 0.0, 'mem_used': 0.0, 'n': 0})
                slot['util'] += row['util']
                slot['mem_util'] += row['mem_util']
                slot['mem_used'] += row['mem_used']
                slot['n'] += 1
                slot['mem_total'] = row['mem_total']
                print(
                    f"{row['index']:5d}  {row['util']:5.1f}  {row['mem_util']:9.1f}  "
                    f"{row['mem_used']:12.0f}  {slot['n']:7d}  {slot['util'] / slot['n']:8.1f}",
                    flush=True,
                )
            count += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass

    if not sums:
        print('没有采样到 GPU')
        return
    print('---- average ----')
    for index in sorted(sums):
        slot = sums[index]
        n = slot['n']
        print(
            f"GPU {index}: avg_util {slot['util'] / n:.1f}%  "
            f"avg_mem_util {slot['mem_util'] / n:.1f}%  "
            f"avg_mem {slot['mem_used'] / n:.0f}/{slot['mem_total']:.0f} MiB  "
            f"samples {n}"
        )


if __name__ == '__main__':
    main()
