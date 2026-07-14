#!/bin/bash
# 为 D2 造灾难现场数据
set -e
echo "🎬 准备灾难现场..."

# 1. data/raw/ 假装放了珍贵标注(后面要保护它,reset 绝不能碰)
mkdir -p data/raw/precious_dataset/images data/raw/precious_dataset/labels
for i in $(seq 1 200); do
    echo "fake image $i" > "data/raw/precious_dataset/images/img_${i}.jpg"
    echo "0 0.5 0.5 0.3 0.4" > "data/raw/precious_dataset/labels/img_${i}.txt"
done
echo "  ✅ data/raw/precious_dataset/ — 400 个文件(模拟珍贵标注)"

# 2. data/processed/ 里造一份"划分后的派生数据集"(reset 的清理目标之一)
mkdir -p data/processed/demo/train data/processed/demo/val data/processed/demo/test
for i in $(seq 1 120); do echo "img $i" > "data/processed/demo/train/img_${i}.txt"; done
for i in $(seq 1 20);  do echo "img $i" > "data/processed/demo/val/img_${i}.txt";   done
for i in $(seq 1 20);  do echo "img $i" > "data/processed/demo/test/img_${i}.txt";  done
echo "  ✅ data/processed/demo/ — 160 个文件(模拟划分后的派生数据集)"

# 3. runs/ 里造一个 2GB 稀疏文件(删它时会跑文件系统)
mkdir -p runs/exp_2026_05_10
dd if=/dev/zero of=runs/exp_2026_05_10/best.pt bs=1 count=0 seek=2G 2>/dev/null
echo "  ✅ runs/exp_2026_05_10/best.pt — 2 GB(稀疏文件)"

# 4. runs/ 里造 5000 个小文件(大量 inode,删起来真的慢)
mkdir -p runs/exp_2026_05_10/tb_logs
for i in $(seq 1 5000); do
    echo "step $i loss 0.${i}" > "runs/exp_2026_05_10/tb_logs/event.${i}"
done
echo "  ✅ runs/exp_2026_05_10/tb_logs/ — 5000 个小文件"

# 5. models/trained/ 造几个"训练产出的权重"(可再生产物,reset 会清)
mkdir -p models/trained
for i in $(seq 1 3); do echo "fake weights $i" > "models/trained/model_v${i}.pt"; done
echo "  ✅ models/trained/ — 3 个训练权重(模拟可再生产物)"

# 6. 一些已存在的日志(撞墙⑤的舞台)
mkdir -p apps/platform/logging/training/2026-05-10
for i in $(seq 1 50); do
    echo "training run $i log" > "apps/platform/logging/training/2026-05-10/run-${i}.log"
done
echo "  ✅ apps/platform/logging/ — 50 份训练日志"

echo "🎬 灾难现场准备就绪(总文件约 5810,名义约 2 GB)。"
du -sh data/raw data/processed runs models/trained apps/platform/logging 2>/dev/null