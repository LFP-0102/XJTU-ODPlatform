# 为 D2 造灾难现场数据
Write-Host "🎬 准备灾难现场..."

New-Item -ItemType Directory -Force -Path "data/raw/precious_dataset/images" | Out-Null
New-Item -ItemType Directory -Force -Path "data/raw/precious_dataset/labels" | Out-Null
1..200 | ForEach-Object {
    "fake image $_" | Set-Content "data/raw/precious_dataset/images/img_$_.jpg"
    "0 0.5 0.5 0.3 0.4" | Set-Content "data/raw/precious_dataset/labels/img_$_.txt"
}
Write-Host "  ✅ data/raw/precious_dataset/ — 400 个文件"

New-Item -ItemType Directory -Force -Path "data/processed/demo/train" | Out-Null
New-Item -ItemType Directory -Force -Path "data/processed/demo/val" | Out-Null
New-Item -ItemType Directory -Force -Path "data/processed/demo/test" | Out-Null
1..120 | ForEach-Object { "img $_" | Set-Content "data/processed/demo/train/img_$_.txt" }
1..20  | ForEach-Object { "img $_" | Set-Content "data/processed/demo/val/img_$_.txt" }
1..20  | ForEach-Object { "img $_" | Set-Content "data/processed/demo/test/img_$_.txt" }
Write-Host "  ✅ data/processed/demo/ — 160 个文件(模拟划分后的派生数据集)"

New-Item -ItemType Directory -Force -Path "runs/exp_2026_05_10" | Out-Null
fsutil file createnew "runs/exp_2026_05_10/best.pt" 2147483648 | Out-Null
fsutil sparse setflag "runs/exp_2026_05_10/best.pt" | Out-Null
Write-Host "  ✅ runs/exp_2026_05_10/best.pt — 2 GB"

New-Item -ItemType Directory -Force -Path "runs/exp_2026_05_10/tb_logs" | Out-Null
1..5000 | ForEach-Object { "step $_ loss" | Set-Content "runs/exp_2026_05_10/tb_logs/event.$_" }
Write-Host "  ✅ runs/exp_2026_05_10/tb_logs/ — 5000 个文件"

New-Item -ItemType Directory -Force -Path "models/trained" | Out-Null
1..3 | ForEach-Object { "fake weights $_" | Set-Content "models/trained/model_v$_.pt" }
Write-Host "  ✅ models/trained/ — 3 个训练权重(模拟可再生产物)"

New-Item -ItemType Directory -Force -Path "apps/platform/logging/training/2026-05-10" | Out-Null
1..50 | ForEach-Object { "training run $_" | Set-Content "apps/platform/logging/training/2026-05-10/run-$_.log" }
Write-Host "  ✅ apps/platform/logging/ — 50 份日志"

Write-Host "🎬 灾难现场准备就绪。"